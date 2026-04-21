#!/usr/bin/env python3
"""Dispatcher - Test Orchestration Framework"""

import logging
import json
from typing import Dict, Any, List
from pathlib import Path
from datetime import datetime

from infinibench.adapter import BaseAdapter
from infinibench.executor import Executor, TestResult
from infinibench.common.constants import TestCategory

logger = logging.getLogger(__name__)

# Adapter registry: maps (test_type, framework) -> adapter factory
# test_type must use TestCategory enum (not string literals)
_ADAPTER_REGISTRY = {
    (TestCategory.OPERATOR, "infinicore"): lambda: _create_infinicore_adapter(),
    (TestCategory.HARDWARE, "cudaunified"): lambda: _create_hardware_adapter(),
    (TestCategory.COMM, "nccltest"): lambda: _create_nccltests_adapter(),
    (TestCategory.INFER, "infinilm"): lambda: _create_inference_adapter(),
    (TestCategory.INFER, "vllm"): lambda: _create_inference_adapter(),
    (TestCategory.TRAIN, "megatron"): lambda: _create_training_adapter(),
    (TestCategory.TRAIN, "infinitrain"): lambda: _create_training_adapter(),
}


def _create_inference_adapter():
    """Create inference adapter"""
    from infinibench.inference.inference_adapter import InferenceAdapter

    return InferenceAdapter()


def _create_infinicore_adapter():
    """Create InfiniCore adapter (lazy import)."""
    from infinibench.operators.infinicore_adapter import InfiniCoreAdapter

    return InfiniCoreAdapter()


def _create_nccltests_adapter():
    """Create NCCL communication adapter."""
    from infinibench.communication.nccl_adapter import NcclTestsAdapter

    return NcclTestsAdapter()


def _create_hardware_adapter():
    """Create Hardware adapter (lazy import)."""
    from infinibench.hardware.hardware_adapter import HardwareTestAdapter

    return HardwareTestAdapter()


def _create_training_adapter():
    """Create training adapter (lazy import)."""
    from infinibench.training.training_adapter import TrainingAdapter

    return TrainingAdapter()


class Dispatcher:
    """Test orchestration dispatcher for managing test executions."""

    def validate_input(self, test_input: Dict[str, Any]) -> bool:
        """
        Validate configuration. Override this method to add custom validation.

        Returns:
            True if test_input is valid, False otherwise
        """
        return "testcase" in test_input

    def dispatch(self, inputs: Any) -> Dict[str, Any]:
        """
        Route payloads to appropriate adapters and execute tests.

        Two-phase execution:
        1. Validation phase: Create all adapters and validate inputs
        2. Execution phase: Execute tests with valid adapters

        Args:
            inputs: Single dict or list of dicts with 'testcase' field

        Returns:
            Aggregated results dictionary
        """
        # Normalize to list
        if isinstance(inputs, dict):
            inputs = [inputs]
        elif not isinstance(inputs, list):
            raise ValueError(f"Invalid inputs type: {type(inputs)}")

        # Filter valid inputs
        valid_test_inputs = [
            test_input
            for test_input in inputs
            if isinstance(test_input, dict) and self.validate_input(test_input)
        ]
        skipped = len(inputs) - len(valid_test_inputs)
        logger.info(
            f"Processing {len(valid_test_inputs)} valid inputs (skipped {skipped} invalid)"
        )

        # Phase 1: Validation - Create all adapters
        valid_executions = []
        skipped_results = []

        for test_input in valid_test_inputs:
            testcase = test_input["testcase"]

            try:
                test_type, framework = self._parse_testcase(testcase)
                adapter = self._create_adapter(test_type, framework)
                valid_executions.append((test_input, adapter))
                logger.debug(f"Validated {testcase} - adapter ready")
            except ValueError as e:
                logger.error(f"Skipping {testcase}: {e}")
                skipped_results.append(
                    TestResult(
                        run_id=test_input.get("run_id", "unknown"),
                        testcase=testcase,
                        result_code=1,  # non-zero = error code
                        result_file=None,
                        skipped=True,
                    ).to_dict()
                )

        logger.info(
            f"Validation complete: {len(valid_executions)} valid, {len(skipped_results)} skipped"
        )

        # Phase 2: Execution - Run all valid tests
        all_results = []
        for idx, (test_input, adapter) in enumerate(valid_executions, 1):
            testcase = test_input["testcase"]
            logger.info(f"[{idx}/{len(valid_executions)}] Executing {testcase}")

            executor = Executor(test_input, adapter)
            test_result = executor.execute()
            all_results.append(test_result.to_dict())

        # Add skipped results
        all_results.extend(skipped_results)

        # Aggregate and save
        aggregated = self._aggregate_results(all_results)
        self._save_summary(aggregated)
        return aggregated

    def _create_adapter(self, test_type: str, framework: str) -> BaseAdapter:
        """Create adapter based on test type and framework."""
        key = (test_type, framework)
        if key in _ADAPTER_REGISTRY:
            adapter_factory = _ADAPTER_REGISTRY[key]
            return adapter_factory()

        raise ValueError(
            f"Adapter not registered: test_type={test_type}, framework={framework}"
        )

    def _parse_testcase(self, testcase: str) -> tuple[str, str]:
        """
        Parse testcase to extract test_type and framework.

        Args:
            testcase: Test case name in format 'test_type.Framework.Something'

        Returns:
            Tuple of (test_type, framework)

        Examples:
            'infer.InfiniLM.Direct' -> ('infer', 'infinilm')
            'operator.InfiniCore.Conv' -> ('operator', 'infinicore')
        """
        parts = testcase.split(".")

        if len(parts) < 2:
            logger.warning(f"Invalid testcase format: {testcase}, using defaults")
            return TestCategory.OPERATOR.value, "infinicore"

        # First part is test_type - validate against TestCategory enum
        test_type_str = parts[0].lower()
        try:
            test_type = TestCategory(test_type_str)
        except ValueError:
            valid_types = ", ".join(c.value for c in TestCategory)
            raise ValueError(
                f"Invalid test_type '{test_type_str}' in '{testcase}'. "
                f"Must be one of: {valid_types}"
            )

        # Second part is framework
        framework = parts[1].lower()

        return test_type.value, framework

    def _aggregate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate results from executors."""
        total = len(results)
        successful = sum(1 for r in results if r["result_code"] == 0)
        failed = total - successful

        # 计算总duration
        total_duration = 0

        aggregated = {
            "total_tests": total,
            "successful_tests": successful,
            "failed_tests": failed,
            "results": [],
            "timestamp": datetime.now().isoformat(),
        }

        for r in results:
            # 提取duration（如果存在）
            duration = r.get("duration", 0)
            total_duration += duration

            aggregated["results"].append(
                {
                    "run_id": r["run_id"],
                    "testcase": r["testcase"],
                    "result_code": r["result_code"],
                    "result_file": r["result_file"],
                    "skipped": r.get("skipped", False),
                    "duration": duration,  # 添加duration
                }
            )

        aggregated["total_duration_seconds"] = total_duration

        # 添加失败详情
        failed_details = []
        for r in results:
            if r["result_code"] != 0:
                failed_details.append(
                    {
                        "testcase": r["testcase"],
                        "run_id": r["run_id"],
                        "result_code": r["result_code"],
                        "error_msg": r.get("error_msg", "Unknown error"),
                        "result_file": r.get("result_file"),
                    }
                )

        if failed_details:
            aggregated["failed_tests_details"] = failed_details

        return aggregated

    def _save_summary(self, aggregated: Dict[str, Any]) -> None:
        """Save aggregated results summary to disk with Git and CI information."""
        # 获取Git信息
        from infinibench.utils.git_utils import get_git_info, get_ci_environment_info

        git_info = get_git_info()
        ci_info = get_ci_environment_info()

        # 构建增强的汇总数据（保留原有的所有字段）
        enhanced_summary = {
            **aggregated,  # 这里包含了 results, total_tests 等所有原有字段
            "git": git_info,
            "ci_environment": ci_info,
        }

        # 计算总运行时长
        total_duration = sum(
            r.get("duration", 0) for r in aggregated.get("results", [])
        )
        enhanced_summary["total_duration_seconds"] = total_duration

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"dispatcher_summary_{timestamp}.json"

        # Save summary to a separate directory
        summary_dir = Path("./summary_output")
        summary_dir.mkdir(parents=True, exist_ok=True)

        # Save summary to a separate directory
        summary_dir = Path("./summary_output")
        summary_dir.mkdir(parents=True, exist_ok=True)

        with open(summary_dir / filename, "w", encoding="utf-8") as f:
            json.dump(enhanced_summary, f, indent=2, ensure_ascii=False)

        logger.info(f"Summary saved to {summary_dir / filename}")
        if git_info.get("is_git_repo"):
            logger.info(
                f"Git info: commit={git_info.get('short_commit')}, branch={git_info.get('branch')}"
            )
        else:
            logger.info("Git info: not in a Git repository")
