#!/usr/bin/env python3
"""InfiniCore Operator Adapter"""

import logging
import copy
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Union

from infinicore.test.framework import TestManager
from infinicore.test.framework.devices import InfiniDeviceNames

from infinibench.adapter import BaseAdapter
from infinibench.common.constants import (
    InfiniBenchJson,
    InfiniCoreRequest,
    OperatorConfig,
    TensorSpec,
    InfiniCoreResult,
    DEVICE_CPU,
    DEVICE_NVIDIA,
    PERF_HOST,
    PERF_DEVICE,
    PLATFORM_INFINICORE,
    DEFAULT_TOLERANCE,
)
from infinibench.operators.flops_calculator import (
    FLOPSCalculator,
    calculate_bandwidth,
)

logger = logging.getLogger(__name__)


class InfiniCoreAdapter(BaseAdapter):
    """Adapter for InfiniCore operator tests (Add, MatMul, etc.)."""

    # Metric name constants
    METRIC_LATENCY = "operator.latency"
    METRIC_ACCURACY = "operator.tensor_accuracy"
    METRIC_FLOPS = "operator.flops"
    METRIC_BANDWIDTH = "operator.bandwidth"

    # Device names from InfiniCore (uppercase for matching)
    DEVICE_NAMES = [v.upper() for v in InfiniDeviceNames.values()]

    def __init__(self):
        """Initialize adapter."""
        self.req_metrics_template = []

    def process(self, test_input: Union[Dict[str, Any], Any]) -> Dict[str, Any]:
        """Execute the operator test."""
        # Normalize test input to dict format
        test_input = self._normalize_test_input(test_input)
        if not test_input:
            raise ValueError(f"Invalid test_input type: {type(test_input)}")

        testcase = test_input.get(InfiniBenchJson.TESTCASE, "unknown")
        logger.info(f"InfiniCoreAdapter: Processing {testcase}")

        try:
            core_req = self._convert_to_request(test_input)
            core_resp = self._execute_backend(core_req)
            result = self._convert_from_response(core_resp, test_input)
            return result

        except Exception as e:
            # Log error with context, then re-raise for Executor to handle
            config = test_input.get(InfiniBenchJson.CONFIG, {})
            operator = config.get("operator", "unknown")
            device = config.get("device", "unknown")

            logger.error(
                f"InfiniCoreAdapter: Operator test failed for {testcase}\n"
                f"  Operator: {operator}\n"
                f"  Device: {device}\n"
                f"  Error: {str(e)}",
                exc_info=True,
            )
            raise

    def _convert_to_request(self, legacy_json: dict) -> list:
        """Convert legacy JSON format to InfiniCore request format."""
        config = legacy_json.get(InfiniBenchJson.CONFIG, {})
        self.req_metrics_template = legacy_json.get(InfiniBenchJson.METRICS, [])

        operator = config.get(OperatorConfig.OPERATOR, "").capitalize()
        # TODO: need to also support torch.nn and infinicore.nn case
        infinicore_op = config.get(
            OperatorConfig.INFINICORE_OP,
            f"infinicore.{config.get(OperatorConfig.OPERATOR, '')}",
        )
        torch_op = config.get(
            OperatorConfig.TORCH_OP, f"torch.{config.get(OperatorConfig.OPERATOR, '')}"
        )

        run_args = self._parse_runtime_args(config)
        device_str = (
            config.get(OperatorConfig.DEVICE, DEVICE_NVIDIA).upper()
            if config.get(OperatorConfig.DEVICE)
            else DEVICE_CPU
        )

        # Set device flags
        for device in self.DEVICE_NAMES:
            run_args[device.lower()] = device in device_str

        # Get base directory for relative path resolution
        data_base_dir = config.get("data_base_dir", ".")

        # Build inputs
        infinicore_inputs = []
        for inp in config.get(OperatorConfig.INPUTS, []):
            base_spec = {
                k: inp[k]
                for k in (
                    TensorSpec.NAME,
                    TensorSpec.DTYPE,
                    TensorSpec.SHAPE,
                    TensorSpec.STRIDES,
                    TensorSpec.INIT_MODE,
                )
                if k in inp
            }

            if TensorSpec.FILE_PATH in inp:
                file_path = Path(inp[TensorSpec.FILE_PATH])
                if not file_path.is_absolute():
                    file_path = Path(data_base_dir) / file_path
                base_spec[TensorSpec.FILE_PATH] = str(file_path)

            infinicore_inputs.append(base_spec)

        # Build kwargs
        infinicore_kwargs = {
            attr[TensorSpec.NAME]: attr[TensorSpec.VALUE]
            for attr in config.get(OperatorConfig.ATTRIBUTES, [])
        }

        outputs = config.get(OperatorConfig.OUTPUTS, [])
        if outputs:
            out_cfg = outputs[0]
            if TensorSpec.INPLACE in out_cfg:
                infinicore_kwargs["out"] = out_cfg[TensorSpec.INPLACE]
            else:
                arg_name = out_cfg.get("arg_name", "out")
                infinicore_kwargs[arg_name] = {
                    k: out_cfg[k]
                    for k in (TensorSpec.NAME, TensorSpec.SHAPE, TensorSpec.DTYPE)
                    if k in out_cfg
                }
                if TensorSpec.STRIDES in out_cfg:
                    infinicore_kwargs[arg_name][TensorSpec.STRIDES] = out_cfg[
                        TensorSpec.STRIDES
                    ]

        if "op_kwargs" in config:
            infinicore_kwargs.update(config["op_kwargs"])

        request = [
            {
                InfiniCoreRequest.OPERATOR: operator,
                InfiniCoreRequest.DEVICE: device_str,
                InfiniCoreRequest.TORCH_OP: torch_op,
                InfiniCoreRequest.INFINICORE_OP: infinicore_op,
                InfiniCoreRequest.ARGS: run_args,
                InfiniCoreRequest.TESTCASES: [
                    {
                        InfiniCoreRequest.DESCRIPTION: f"Auto-Gen: {operator}",
                        InfiniCoreRequest.INPUTS: infinicore_inputs,
                        InfiniCoreRequest.KWARGS: infinicore_kwargs,
                        InfiniCoreRequest.TOLERANCE: config.get(
                            OperatorConfig.TOLERANCE, DEFAULT_TOLERANCE
                        ),
                        InfiniCoreRequest.RESULT: None,
                    }
                ],
            }
        ]

        return request

    def _parse_runtime_args(self, config: dict) -> dict:
        """Parse runtime arguments from config."""
        args = {
            "bench": "both",
            "num_prerun": config.get("warmup_iterations", 5),
            "num_iterations": config.get("measured_iterations", 100),
            "verbose": False,
            "debug": False,
            "eq_nan": False,
            "save": "test_report.json",
            **{device.lower(): False for device in self.DEVICE_NAMES},
        }

        return args

    def _handle_latency(self, metric: dict, context: dict):
        """Handle latency metric."""
        avg_latency_sec = context.get("avg_latency_sec")
        if avg_latency_sec is not None and avg_latency_sec > 0:
            metric.update(
                {
                    "value": round(avg_latency_sec * 1000, 6),  # Convert to ms
                    "type": "scalar",
                    "raw_data_url": "",
                    "unit": "ms",
                }
            )

    def _handle_accuracy(self, metric: dict, context: dict):
        """Handle accuracy metric (mock)."""
        metric.update({"value": "PASS", "unit": ""})

    def _handle_flops(self, metric: dict, context: dict, config: dict = None):
        """Handle FLOPS metric."""
        value = 0.0
        avg_latency_sec = context.get("avg_latency_sec")

        if avg_latency_sec and avg_latency_sec > 0:
            inputs = config.get(OperatorConfig.INPUTS, [])
            outputs = config.get(OperatorConfig.OUTPUTS, [])
            operator = config.get(OperatorConfig.OPERATOR, "").lower()

            flops = FLOPSCalculator.get_flops(operator, inputs, outputs)

            if flops > 0:
                tflops = (flops / avg_latency_sec) / 1e12
                value = tflops if tflops < 0.0001 else round(tflops, 4)

        metric.update(
            {"value": value, "type": "scalar", "raw_data_url": "", "unit": "TFLOPS"}
        )

    def _handle_bandwidth(self, metric: dict, context: dict, config: dict = None):
        """Handle bandwidth metric."""
        value = 0.0
        avg_latency_sec = context.get("avg_latency_sec")

        if avg_latency_sec and avg_latency_sec > 0:
            inputs = config.get(OperatorConfig.INPUTS, [])
            outputs = config.get(OperatorConfig.OUTPUTS, [])

            bandwidth_info = calculate_bandwidth(inputs, outputs)

            if bandwidth_info["total_bytes"] > 0:
                bandwidth_gbs = (bandwidth_info["total_bytes"] / avg_latency_sec) / 1e9
                value = (
                    bandwidth_gbs if bandwidth_gbs < 0.0001 else round(bandwidth_gbs, 4)
                )

        metric.update(
            {"value": value, "type": "scalar", "raw_data_url": "", "unit": "GB/s"}
        )

    def _convert_from_response(self, saved_files: list, original_req: dict) -> dict:
        """Convert InfiniCore saved file content to InfiniBench format."""
        final_json = copy.deepcopy(original_req)
        final_json[InfiniBenchJson.TIME] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        try:
            if not saved_files:
                raise ValueError("No saved files returned from TestManager")

            with open(saved_files[0], "r") as f:
                saved_data = json.load(f)

            tc_result = saved_data[0][InfiniCoreResult.TESTCASES][0][
                InfiniCoreResult.RESULT
            ]
            is_success = tc_result.get(InfiniCoreResult.STATUS, {}).get(
                InfiniCoreResult.SUCCESS, False
            )
            final_json[InfiniBenchJson.RESULT_CODE] = 0 if is_success else 1

            if not is_success:
                final_json[InfiniBenchJson.ERROR_MSG] = tc_result.get(
                    InfiniCoreResult.STATUS, {}
                ).get(InfiniCoreResult.ERROR, "Unknown")
                return final_json

            perf_data = tc_result.get(InfiniCoreResult.PERF_MS, {}).get(
                PLATFORM_INFINICORE, {}
            )

            # CPU devices use "host", accelerator devices use "device"
            device_type = saved_data[0].get("device", DEVICE_CPU).upper()
            latency_field = PERF_HOST if device_type == DEVICE_CPU else PERF_DEVICE

            # Calculate average latency per iteration
            # InfiniCore returns total time for all iterations
            total_latency_ms = perf_data.get(latency_field)
            args_data = saved_data[0].get("args", {})
            num_iterations = max(args_data.get("num_iterations", 1), 1)

            avg_latency_sec = None
            if total_latency_ms and total_latency_ms > 0:
                avg_latency_sec = (total_latency_ms / num_iterations) / 1000.0

            context = {
                "avg_latency_sec": avg_latency_sec,
                "tflops": tc_result.get(InfiniCoreResult.METRICS, {}).get("tflops"),
                "bandwidth_gbs": tc_result.get(InfiniCoreResult.METRICS, {}).get(
                    "bandwidth_gbs"
                ),
            }

            metric_handlers = {
                self.METRIC_LATENCY: self._handle_latency,
                self.METRIC_ACCURACY: self._handle_accuracy,
                self.METRIC_FLOPS: self._handle_flops,
                self.METRIC_BANDWIDTH: self._handle_bandwidth,
            }

            config = final_json.get(InfiniBenchJson.CONFIG, {})

            if InfiniBenchJson.METRICS in final_json and self.req_metrics_template:
                for i, metric in enumerate(
                    final_json[InfiniBenchJson.METRICS][
                        : len(self.req_metrics_template)
                    ]
                ):
                    handler = metric_handlers.get(metric.get("name"))
                    if handler:
                        # Pass config for FLOPS and bandwidth calculation
                        if metric.get("name") in [
                            self.METRIC_FLOPS,
                            self.METRIC_BANDWIDTH,
                        ]:
                            handler(metric, context, config)
                        else:
                            handler(metric, context)

        except Exception as e:
            logger.error(f"[InfiniCoreAdapter] Parsing Error: {e}")
            final_json[InfiniBenchJson.RESULT_CODE] = 1
            final_json[InfiniBenchJson.ERROR_MSG] = str(e)

        return final_json

    def _execute_backend(self, infinicore_req: list) -> list:
        """Execute InfiniCore backend API using TestManager."""
        test_manager = TestManager(verbose=False, bench_mode=True)
        all_passed, saved_files = test_manager.test(
            target_ops=None, json_cases_list=infinicore_req, global_exec_args=None
        )
        return saved_files
