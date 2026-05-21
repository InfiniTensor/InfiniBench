#!/usr/bin/env python3
"""Hardware Test Adapter for cross-platform bandwidth benchmarks.

Routes to platform-specific implementations:
- NVIDIA / MetaX / Iluvatar / Hygon / Moore → CudaPlatform
- Cambricon MLU → CambriconPlatform
- Ascend NPU → AscendPlatform
"""

import logging
import subprocess
import re
from pathlib import Path
from typing import Any, Dict, Optional, List

from infinimetrics.adapter import BaseAdapter
from infinimetrics.common.csv_utils import save_csv, create_timeseries_metric
from infinimetrics.common.command_builder import build_command_from_config
from infinimetrics.common.constants import (
    TEST_TYPE_MAP,
    MEMORY_DIRECTIONS,
    STREAM_OPERATIONS,
    MEMORY_CSV_FIELDS,
    L1_CACHE_CSV_FIELDS,
    L2_CACHE_CSV_FIELDS,
    L1_CACHE_PATTERN,
    L2_CACHE_PATTERN,
    CACHE_TEST_TIMEOUT,
    DEFAULT_TEST_TIMEOUT,
    METRIC_PREFIX_MEM_SWEEP,
    InfiniMetricsJson,
)
from infinimetrics.hardware.platforms import create_platform_handler
from infinimetrics.utils.time_utils import get_timestamp

logger = logging.getLogger(__name__)

# Platform name aliases for routing
_CUDA_COMPATIBLE_PLATFORMS = {
    "nvidia", "cuda", "metax", "metax_gpu", "iluvatar", "iluvatar_gpu",
    "hygon", "sugon_dcu", "moore", "moore_gpu",
}
_CAMBRICON_PLATFORMS = {"cambricon", "cambricon_mlu"}
_ASCEND_PLATFORMS = {"ascend", "ascend_npu"}


def _resolve_platform(config: Dict[str, Any]) -> str:
    """Determine the hardware platform from config.

    Checks in order:
    1. config["platform"]
    2. config["gpu_platform"]
    3. Falls back to "cuda" (CUDA-compatible default)
    """
    for key in ("platform", "gpu_platform", "accelerator_type"):
        val = config.get(key, "").lower().strip()
        if val:
            return val
    return "cuda"


class HardwareTestAdapter(BaseAdapter):
    """Adapter for hardware performance tests across multiple platforms."""

    def __init__(
        self,
        cuda_perf_path: str = None,
        output_dir: str = "./output",
    ):
        self.cuda_perf_path = cuda_perf_path or str(
            Path(__file__).parent
            / "cuda-memory-benchmark"
            / "build"
            / "cuda_perf_suite"
        )
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.build_dir = Path(__file__).parent / "cuda-memory-benchmark"
        self.build_script = self.build_dir / "build.sh"
        self._platform_handler = None

    def setup(self, config: Dict[str, Any]) -> None:
        """Initialize resources before running tests."""
        device = config.get("device", "cuda").lower()
        if device == "cpu":
            return

        platform_name = _resolve_platform(config)
        output_dir = Path(config.get("output_dir", "./output")) / "hardware"

        # Create platform handler via registry
        self._platform_handler = create_platform_handler(
            platform_name, output_dir, config
        )
        self._platform_handler.setup()

    def process(self, test_input: Any) -> Dict[str, Any]:
        """Process test input and return results."""
        # Normalize test input to dict format
        test_input = self._normalize_test_input(test_input)
        if not test_input:
            raise ValueError(f"Invalid test_input type: {type(test_input)}")

        testcase = test_input.get(InfiniMetricsJson.TESTCASE, "unknown")
        config = test_input.get(InfiniMetricsJson.CONFIG, {})
        run_id = test_input.get(InfiniMetricsJson.RUN_ID, "unknown")

        logger.info(f"HardwareTestAdapter: Processing {testcase}")

        # Put CSV files in hardware/ subdirectory to match JSON location
        self.output_dir = Path(config.get("output_dir", "./output")) / "hardware"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        device = config.get("device", "cuda").lower()
        test_type = config.get("test_type", "comprehensive")

        try:
            if device == "cpu":
                logger.info(
                    "CPU mode: Skipping hardware tests (not supported on CPU), returning empty results"
                )
                metrics = []
                command = None
            else:
                platform_name = _resolve_platform(config)
                logger.info(
                    "GPU mode (device=%s, platform=%s): Executing hardware tests",
                    device,
                    platform_name,
                )

                # Ensure platform handler output_dir is up to date
                if self._platform_handler is not None:
                    self._platform_handler.output_dir = self.output_dir

                # Recreate platform handler if output_dir changed
                self._platform_handler = create_platform_handler(
                    platform_name, self.output_dir, config
                )
                self._platform_handler.setup()

                command = self._platform_handler.get_command(config)
                metrics = self._platform_handler.run_test(test_type, config)

            # Add command to config for traceability
            result_config = config.copy()
            if command:
                result_config["command"] = command

            return {
                InfiniMetricsJson.RESULT_CODE: 0,
                InfiniMetricsJson.TIME: get_timestamp(),
                InfiniMetricsJson.RUN_ID: run_id,
                InfiniMetricsJson.TESTCASE: testcase,
                InfiniMetricsJson.CONFIG: result_config,
                InfiniMetricsJson.METRICS: metrics,
            }

        except Exception as e:
            # Log error with context, then re-raise for Executor to handle
            logger.error(
                f"HardwareTestAdapter: Test failed for {testcase}\n"
                f"  Device: {device}\n"
                f"  Test Type: {test_type}\n"
                f"  Error: {str(e)}",
                exc_info=True,
            )
            raise
