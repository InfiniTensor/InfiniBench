#!/usr/bin/env python3
"""
Executor - Universal Test Execution Framework
"""

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import subprocess

from infinibench.adapter import BaseAdapter
from infinibench.common.constants import ErrorCode, TEST_CATEGORIES
from infinibench.utils.hardware_detector import HardwareDetector
from infinibench.utils.path_utils import sanitize_filename

logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """
    Standardized test result structure.

    Used throughout the execution lifecycle and returned to Dispatcher.

    Note:
        result_code: 0 = success, non-zero = error code (following Linux convention)
    """

    run_id: str
    testcase: str
    result_code: int  # 0 = success, non-zero = error code
    result_file: Optional[str] = None
    skipped: bool = False
    config: Optional[Dict[str, Any]] = None
    duration: float = 0.0
    error_msg: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to lightweight dictionary format for Dispatcher aggregation."""
        return {
            "run_id": self.run_id,
            "testcase": self.testcase,
            "result_code": self.result_code,
            "result_file": self.result_file,
            "skipped": self.skipped,
            "config": self.config,
            "duration": self.duration,
            "error_msg": self.error_msg,
        }


class Executor:
    """
    Universal test executor for all test types.

    Responsibilities:
        1. Manage adapter lifecycle (setup -> process -> teardown)
        2. Save results to disk
        3. Return result summary
    """

    def __init__(self, payload: Dict[str, Any], adapter: BaseAdapter):
        """
        Initialize executor.

        Args:
            payload: Test payload with testcase, config, etc.
            adapter: Configured adapter instance
        """
        self.payload = payload
        self.adapter = adapter
        self.testcase = payload.get("testcase", "unknown")
        self.run_id = payload.get("run_id", "")
        self.test_input = None

        # Setup output directory from config
        config = payload.get("config", {})
        self.output_dir = Path(config.get("output_dir", "./output"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.debug(f"Executor initialized: testcase={self.testcase}")

    def setup(self) -> None:
        """
        Setup phase - initialize adapter.

        This should be called before execute().
        """
        config = self.payload.get("config", {})

        # Inject testcase, run_id, and other metadata into config
        config["_testcase"] = self.payload.get("testcase", "")
        config["_run_id"] = self.payload.get("run_id", "")
        config["_time"] = self.payload.get("time", None)

        # Initialize test_input from payload
        self.test_input = self.payload

        self.adapter.setup(config)

        logger.debug(f"Executor: Setup complete for {self.testcase}")

    def teardown(self, result: Any) -> str:
        """
        Teardown phase - cleanup adapter, collect metrics, and save results.

        This should be called after process() completes.

        Args:
            result:

        Returns:
            Path to saved result file
        """
        # Always cleanup adapter
        try:
            self.adapter.teardown()
        except Exception as teardown_error:
            logger.warning(
                f"Executor: Teardown failed for {self.testcase}: {teardown_error}"
            )

        # TODO: Add metrics calculation method

        # Save result to disk
        result_file = self._save_result(result)

        logger.debug(f"Executor: Teardown complete for {self.testcase}")
        return result_file

    def execute(self) -> TestResult:
        """
        Execute the complete test with proper lifecycle management.

        Lifecycle:
            1. adapter.setup(config)
            2. adapter.process(payload)
            3. adapter.teardown() - includes saving results
            4. Return TestResult

        Returns:
            TestResult object with result_code and file path.
        """
        logger.info(f"Executor: Running {self.testcase}")

        start_time = time.time()
        config = self.payload.get("config", {})
        test_result = TestResult(
            run_id=self.run_id,
            testcase=self.testcase,
            result_code=0,
            result_file=None,
            config=config,
            duration=0.0,
            error_msg=None,
        )

        response = {}

        try:
            self.setup()
            response = self.adapter.process(self.test_input)

            # Enrich environment if missing
            if isinstance(response, dict) and "environment" not in response:
                response = self._enrich_environment(response)

            result_file = self.teardown(response)
            test_result.result_file = result_file
            test_result.duration = time.time() - start_time

            logger.info(
                f"Executor: {self.testcase} completed in {test_result.duration:.2f}s"
            )
            return test_result

        except Exception as e:
            return self._handle_error(e, start_time, test_result)

    def _enrich_environment(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich response with environment information."""
        env = self._build_environment(response)

        # Rebuild ordered dict
        ordered = {}
        for key in [
            "run_id",
            "time",
            "testcase",
            "success",
            "environment",
            "result_code",
            "config",
            "metrics",
        ]:
            if key == "environment":
                ordered["environment"] = env
            elif key in response:
                ordered[key] = response[key]

        for key, value in response.items():
            if key not in ordered:
                ordered[key] = value

        return ordered

    def _handle_error(
        self, error: Exception, start_time: float, test_result: TestResult
    ) -> TestResult:
        """Handle different types of errors."""
        duration = time.time() - start_time
        test_result.duration = duration
        error_msg = str(error)

        # Set error message
        test_result.error_msg = error_msg

        # Determine error type and result code
        if isinstance(error, subprocess.TimeoutExpired):
            test_result.result_code = ErrorCode.TIMEOUT
            logger.error(f"Executor: Timeout after {duration:.2f}s: {error_msg[:300]}")
        elif isinstance(error, ValueError):
            test_result.result_code = ErrorCode.CONFIG
            logger.warning(f"Executor: Configuration error: {error_msg[:300]}")
        elif isinstance(error, RuntimeError):
            if any(
                kw in error_msg.lower() for kw in ["memory", "oom", "out of memory"]
            ):
                test_result.result_code = ErrorCode.SYSTEM
                logger.error(f"Executor: Memory error: {error_msg[:300]}")
            else:
                test_result.result_code = ErrorCode.GENERIC
                logger.warning(f"Executor: Runtime error: {error_msg[:300]}")
        else:
            test_result.result_code = ErrorCode.GENERIC
            logger.error(f"Executor: Unexpected error: {error}", exc_info=True)

        # Build error response
        response = self._build_error_response(error_msg, test_result.result_code)

        # Save result
        try:
            if not test_result.result_file:
                test_result.result_file = self._save_result(response)
        except Exception as e:
            logger.error(f"Executor: Failed to save result: {e}")

        return test_result

    def _build_error_response(self, error_msg: str, result_code: int) -> Dict[str, Any]:
        """
        Build a response dict containing error information for saving to disk.

        Args:
            error_msg: Error message string
            result_code: Error result code

        Returns:
            Dictionary with basic test info and error details
        """
        config = self.payload.get("config", {})

        # Create a cleaned config without injected metadata
        cleaned_config = {
            k: v
            for k, v in config.items()
            if not k.startswith("_")  # Skip _testcase, _run_id, _time
        }

        # Extract device information
        resolved = self._extract_device_info(config)

        return {
            "run_id": self.run_id,
            "testcase": self.testcase,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "result_code": result_code,
            "error_msg": error_msg,
            "success": 1,  # 1 = failure
            "config": cleaned_config,
            "resolved": resolved,
        }

    def _extract_device_info(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract device information from config."""
        device_used = 0
        gpus_per_node = 0
        nodes = 1

        # Try device_involved
        if "device_involved" in config:
            try:
                device_used = int(config.get("device_involved", 0) or 0)
            except (ValueError, TypeError):
                device_used = 0

        # Try single_node config
        if isinstance(config.get("single_node"), dict):
            single_node = config["single_node"]
            device_ids = single_node.get("device_ids", [])
            if device_ids:
                device_used = len(device_ids)
            gpus_per_node = device_used
        else:
            gpus_per_node = device_used

        # Try multi_node config
        if "multi_node" in config:
            try:
                nodes = int(config.get("multi_node", {}).get("num_nodes", 1) or 1)
            except (ValueError, TypeError):
                nodes = 1

        return {
            "nodes": nodes,
            "gpus_per_node": gpus_per_node,
            "device_used": device_used,
        }

    def _build_environment(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build a unified environment block
        """
        resolved = response.get("resolved", {}) if isinstance(response, dict) else {}
        nodes = int(resolved.get("nodes", 1) or 1)
        device_used = int(resolved.get("device_used", 0) or 0)
        gpn = int(resolved.get("gpus_per_node", 0) or 0)

        # Fallback to config hints if adapter didn't provide
        cfg = self.payload.get("config", {}) or {}

        accel_type = (
            (cfg.get("accelerator_type") or cfg.get("device_type") or "")
            .strip()
            .lower()
        )  # optional
        device_ids = cfg.get("device_ids")

        if device_ids is None and isinstance(cfg.get("single_node"), dict):
            device_ids = cfg["single_node"].get("device_ids")

        if device_used <= 0:
            try:
                device_used = int(cfg.get("device_involved", 0) or 0)
            except Exception:
                device_used = 0

        if nodes <= 1:
            topo = f"{device_used}x1 ring mesh"
        else:
            topo = f"{nodes}x{(gpn or max(1, device_used // nodes))} ring mesh"

        hw = {
            "cpu_model": "Unknown",
            "memory_gb": 0,
            "gpu_model": "Unknown",
            "gpu_memory_gb": 0,
            "driver_version": "Unknown",
            "cuda_version": "Unknown",
            "accelerator_type": accel_type or "generic",
        }

        framework_info = resolved.get("framework", {})
        if not framework_info:
            framework_info = {"name": "unknown", "version": "unknown"}

        return {
            "cluster_scale": nodes,
            "topology": topo,
            "cluster": [
                {
                    "machine": {
                        "cpu_model": hw.get("cpu_model", "Unknown"),
                        "memory_gb": hw.get("memory_gb", 0),
                        "accelerators": [
                            {
                                "model": hw.get("gpu_model", "Unknown"),
                                "count": device_used,
                                "memory_gb_per_card": hw.get("gpu_memory_gb", 0),
                                "driver": hw.get("driver_version", "Unknown"),
                                "cuda": hw.get("cuda_version", "Unknown"),
                                # reserved: type of platform
                                "type": hw.get("accelerator_type", "generic"),
                            }
                        ],
                    },
                    "framework": [framework_info],
                }
            ],
        }

    def _save_result(self, result: Dict[str, Any]) -> str:
        """
        Save detailed result to disk as JSON.

        Args:
            result: Complete result dict with data and metrics

        Returns:
            Absolute path to saved file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 1) Prefer run_id in result (adapter may generate a richer run_id)
        run_id = None
        if isinstance(result, dict):
            run_id = result.get("run_id") or result.get("raw_result", {}).get("run_id")

        # 2) Fallback to payload run_id
        if not run_id:
            run_id = self.payload.get("run_id") or self.run_id

        # 3) Determine output subdirectory based on testcase
        testcase = self.payload.get("testcase", "")

        # Extract category from testcase (e.g., "hardware.cudaUnified.Comprehensive" -> "hardware")
        category = "other"  # default fallback
        for prefix, subdir in TEST_CATEGORIES.items():
            if testcase.startswith(prefix):
                category = subdir
                break

        if run_id:
            safe_run_id = sanitize_filename(run_id)

            # Put results in category-specific subdirectory
            category_dir = self.output_dir / category
            category_dir.mkdir(parents=True, exist_ok=True)

            filename = f"{safe_run_id}_results.json"
            output_file = category_dir / filename
        else:
            # Final fallback to old naming (in root output_dir)
            safe_name = self.testcase.replace(".", "_").replace("/", "_")
            filename = f"{safe_name}_{timestamp}_results.json"
            output_file = self.output_dir / filename

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        logger.debug(f"Executor: Results saved to {output_file}")
        return str(output_file)
