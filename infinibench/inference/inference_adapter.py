#!/usr/bin/env python3
"""Inference adapter"""

import os
import logging
import json
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

from infinibench.adapter import BaseAdapter
from infinibench.input import TestInput
from infinibench.common.testcase_utils import generate_run_id

logger = logging.getLogger(__name__)


class InferenceAdapter(BaseAdapter):
    """Main adapter for inference tests"""

    def __init__(self):
        super().__init__()
        self.config = None
        self.mode = "direct"
        self.framework = "infinilm"
        self._testcase = None

    def setup(self, config: Dict[str, Any]) -> None:
        """Initialize inference resources"""
        # Get testcase and run_id from injected fields
        testcase = config.get("_testcase", "")
        run_id = config.get("_run_id", "")

        # Store testcase for error reporting
        self._testcase = testcase

        # Parse mode and framework
        if "service" in testcase.lower():
            self.mode = "service"
        if "direct" in testcase.lower():
            self.mode = "direct"

        if "vllm" in testcase.lower():
            self.framework = "vllm"
        elif "infinilm" in testcase.lower():
            self.framework = "infinilm"

        # Create configuration object
        self.config = self._create_inference_config(config)

        logger.info(
            f"Inference adapter setup: mode={self.mode}, framework={self.framework}"
        )

    def process(self, test_input) -> Dict[str, Any]:
        """Execute inference test"""
        try:
            # Execute inference
            if self.mode == "direct":
                from .direct import DirectInferenceEngine

                engine = DirectInferenceEngine(self.config)
            else:
                from .service import ServiceInferenceEngine

                engine = ServiceInferenceEngine(self.config)

            result_data = engine.execute()

            # Write results
            from .result_writer import InferenceResultWriter

            writer = InferenceResultWriter(self.config, result_data)
            result_payload = writer.generate_result()

            # Generate resolved info
            dev = getattr(self.config, "device", None)
            device_ids = []
            if isinstance(dev, dict):
                device_ids = dev.get("device_ids") or []
            else:
                device_ids = getattr(dev, "device_ids", []) or []

            # Optional fallback
            if not device_ids:
                cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
                if cvd:
                    device_ids = [x for x in cvd.split(",") if x.strip() != ""]

            resolved = {
                "nodes": 1,
                "gpus_per_node": len(device_ids),
                "device_used": len(device_ids),
            }

            # Return result
            return {
                "result_code": 0,
                "success": 0,
                "metrics": result_payload.get("metrics", []),
                "run_id": result_payload.get("run_id", ""),
                "testcase": result_payload.get("testcase", ""),
                "time": result_payload.get(
                    "time", datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                ),
                "config": result_payload.get("config", {}),
                "resolved": resolved,
            }

        except Exception as e:
            # Log error with context, then re-raise for Executor to handle
            logger.error(
                f"InferenceAdapter: Test failed for {self._testcase}\n"
                f"  Mode: {self.mode}\n"
                f"  Framework: {self.framework}\n"
                f"  Error: {str(e)}",
                exc_info=True,
            )
            raise

    def teardown(self) -> None:
        """Cleanup resources"""
        logger.info("Inference adapter teardown complete")

    def _create_inference_config(self, config: Dict[str, Any]):
        """Create inference configuration object"""
        # Get information from injected fields
        testcase = config.get("_testcase", "")
        user_run_id = config.get("_run_id", "")

        final_run_id = generate_run_id(testcase, user_run_id)

        # Parse mode and framework
        mode = "direct"
        framework = "infinilm"
        if "service" in testcase.lower():
            mode = "service"
        if "direct" in testcase.lower():
            mode = "direct"

        if "vllm" in testcase.lower():
            framework = "vllm"
        elif "infinilm" in testcase.lower():
            framework = "infinilm"

        class SimpleInferenceConfig:
            def __init__(self, config_dict, testcase, run_id, mode, framework):
                self.testcase = testcase
                self.run_id = run_id
                self.mode = mode
                self.framework = framework

                # Basic configuration
                self.model = config_dict.get("model", "")
                self.model_path = config_dict.get("model_path", "")
                self.output_dir = config_dict.get("output_dir", "./output")

                # Device configuration
                self.device = config_dict.get("device", {})

                # infer_args - get from config or create default dict
                self.infer_args = config_dict.get("infer_args", {})

                # Ensure infer_args contains all required fields
                defaults = {
                    "static_batch_size": 1,
                    "prompt_token_num": 128,
                    "output_token_num": 128,
                    "max_seq_len": 4096,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "top_k": 50,
                    "parallel": {"dp": 1, "tp": 1, "pp": 1, "sp": 1},
                }

                # Merge default values with user configuration
                for key, default_value in defaults.items():
                    if key not in self.infer_args:
                        self.infer_args[key] = default_value

                # Execution parameters
                self.warmup_iterations = config_dict.get("warmup_iterations", 10)
                self.measured_iterations = config_dict.get("measured_iterations", 100)

        return SimpleInferenceConfig(config, testcase, final_run_id, mode, framework)
