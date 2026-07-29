#!/usr/bin/env python3
"""Hardware Test Adapter for CUDA Unified Benchmark Suite"""

import logging
import subprocess
import re
import shutil
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
from infinimetrics.utils.time_utils import get_timestamp

logger = logging.getLogger(__name__)


def detect_platform() -> str:
    """Detect GPU platform: 'metax', 'corex', 'hygon', or 'cuda'."""
    # Check MetaX first (cucc/mxcc compiler)
    maca_path = Path("/opt/maca")
    cucc_path = maca_path / "tools" / "cu-bridge" / "bin" / "cucc"
    if cucc_path.exists() or shutil.which("cucc") or shutil.which("mxcc"):
        return "metax"
    # Check Hygon DCU (DTK / hipcc)
    dtk_path = Path("/opt/dtk")
    if (dtk_path / "bin" / "hipcc").exists() or shutil.which("hipcc"):
        # DTK also ships rocm-smi / hy-smi
        if (dtk_path / "bin" / "hy-smi").exists() or shutil.which("hy-smi"):
            return "hygon"
        # hipcc available but not necessarily Hygon — could be AMD ROCm.
        # Still treat as hygon if DTK path exists.
        if dtk_path.exists():
            return "hygon"
    # Check Iluvatar CoreX (ixsmi or clang++)
    corex_path = Path("/usr/local/corex")
    if (corex_path / "bin" / "ixsmi").exists() or (corex_path / "bin" / "clang++").exists():
        return "corex"
    return "cuda"


class HardwareTestAdapter(BaseAdapter):
    """Adapter for CUDA Unified hardware performance tests.

    Works on NVIDIA GPUs and CUDA-compatible domestic GPUs
    (Moore Threads, Iluvatar CoreX, MetaX, etc.) via CUDA compatibility layers.
    """

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

    def setup(self, config: Dict[str, Any]) -> None:
        """Initialize resources before running tests."""
        device = config.get("device", "cuda").lower()
        if (
            device == "cpu"
            or config.get("skip_build", False)
            or Path(self.cuda_perf_path).exists()
        ):
            return
        self._build_cuda_project()

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
                logger.info("GPU mode (device=%s): Executing CUDA tests", device)
                cmd = self._build_command(config)
                command = " ".join(cmd)  # Store command as string
                output = self._execute_test(cmd, test_type)
                metrics = self._parse_output(output, test_type, run_id)

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

    def _build_cuda_project(self) -> None:
        """Build CUDA project if needed."""
        if not self.build_dir.exists():
            raise FileNotFoundError(
                f"CUDA benchmark directory not found: {self.build_dir}"
            )
        if not self.build_script.exists():
            raise FileNotFoundError(f"Build script not found: {self.build_script}")
        platform = detect_platform()
        logger.info(
            "Building CUDA project in: %s (platform: %s)", self.build_dir, platform
        )
        try:
            result = subprocess.run(
                ["bash", str(self.build_script), "--platform", platform],
                cwd=str(self.build_dir),
                capture_output=True,
                text=True,
                timeout=300,
            )
            if result.returncode != 0:
                raise RuntimeError(f"Failed to build CUDA project:\n{result.stderr}")
            logger.info(
                "CUDA project build completed successfully (platform: %s)", platform
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("CUDA project build timed out after 5 minutes")

    def _build_command(self, config: Dict[str, Any]) -> List[str]:
        """Build command for CUDA test suite."""
        test_type = config.get("test_type", "all")
        cuda_test_type = TEST_TYPE_MAP.get(test_type, test_type.lower())

        base_command = [self.cuda_perf_path, f"--{cuda_test_type}"]

        # Use command builder for optional parameters
        param_mappings = [
            ("device_id", "--device"),
            ("iterations", "--iterations"),
            ("array_size", "--array-size"),
        ]

        return build_command_from_config(base_command, config, param_mappings)

    def _execute_test(self, cmd: List[str], test_type: str) -> str:
        """Execute CUDA test and return output."""
        if not Path(self.cuda_perf_path).exists():
            raise RuntimeError(f"cuda_perf_suite not found: {self.cuda_perf_path}")
        logger.info("Executing: %s", " ".join(cmd))

        timeout = (
            CACHE_TEST_TIMEOUT if test_type.lower() == "cache" else DEFAULT_TEST_TIMEOUT
        )
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True, timeout=timeout
        )
        return result.stdout

    def _parse_output(self, output: str, test_type: str, run_id: str) -> List[Dict]:
        """Parse test output based on test type."""
        if test_type == "Comprehensive":
            return (
                self._parse_memory_bandwidth(output, run_id, METRIC_PREFIX_MEM_SWEEP)
                + self._parse_stream_benchmark(output)
                + self._parse_cache_bandwidth(output, run_id)
            )

        # Single test type
        metric_map = {
            "MemSweep": (METRIC_PREFIX_MEM_SWEEP, self._parse_memory_bandwidth),
            "Stream": (None, self._parse_stream_benchmark),
            "Cache": (None, self._parse_cache_bandwidth),
        }

        if test_type in metric_map:
            prefix, parser = metric_map[test_type]
            return parser(output, run_id, prefix) if prefix else parser(output, run_id)
        return []

    def _parse_memory_bandwidth(
        self, output: str, run_id: str, metric_prefix: str
    ) -> List[Dict]:
        """Parse memory bandwidth test output."""
        metrics = []
        is_sweep = "sweep" in metric_prefix

        for direction_label, key in MEMORY_DIRECTIONS:
            csv_data = self._parse_bandwidth_data(output, direction_label)

            if not csv_data:
                continue

            metric_name = f"{metric_prefix}_{key}"

            if is_sweep:
                metrics.append(
                    self._create_timeseries_metric(
                        metric_name,
                        csv_data,
                        f"mem_sweep_{key}_{run_id}",
                        MEMORY_CSV_FIELDS,
                    )
                )
            else:
                max_bw = max(row["bandwidth_gbps"] for row in csv_data)
                metrics.append(
                    {
                        "name": metric_name,
                        "value": round(max_bw, 2),
                        "type": "scalar",
                        "unit": "GB/s",
                    }
                )

        return metrics

    def _parse_bandwidth_data(self, output: str, direction: str) -> List[Dict]:
        """Parse bandwidth data for a specific direction."""
        csv_data = []

        # Sweep format - match from direction header to next section
        # The pattern stops at: ==== (next section), Direction:, STREAM:, or end of string
        sweep_pattern = rf"{direction}.*?Size \(MB\)\s+Time \(ms\)Bandwidth \(GB/s\)\s+CV \(%\)\s*-+\s*(.*?)\s*(?=\n=+|Direction:|STREAM:|\Z)"
        sweep_match = re.search(sweep_pattern, output, re.DOTALL)

        if sweep_match:
            data_block = sweep_match.group(1)
            for line in data_block.strip().split("\n"):
                line = line.strip()
                if line and not line.startswith("-"):
                    result = self._parse_sweep_line(line)
                    if result:
                        csv_data.append(result)

        return csv_data

    @staticmethod
    def _parse_sweep_line(line: str) -> Optional[Dict]:
        """Parse a line from sweep format output."""
        parts = line.split()
        if len(parts) >= 3:
            try:
                return {"size_mb": float(parts[0]), "bandwidth_gbps": float(parts[2])}
            except (ValueError, IndexError):
                pass
        return None

    def _create_timeseries_metric(
        self,
        name: str,
        data: List[Dict],
        base_filename: str,
        fields: List[str],
        unit: str = "GB/s",
    ) -> Dict:
        """Create a timeseries metric with CSV file."""
        return create_timeseries_metric(
            output_dir=self.output_dir,
            metric_name=name,
            data=data,
            base_filename=base_filename,
            fields=fields,
            unit=unit,
        )

    def _parse_stream_benchmark(self, output: str, run_id: str = None) -> List[Dict]:
        """Parse STREAM benchmark output."""
        metrics = []
        for op in STREAM_OPERATIONS:
            match = re.search(rf"STREAM_{op.capitalize()}\s+(\d+\.\d+)", output)
            if match:
                metrics.append(
                    {
                        "name": f"hardware.stream_{op}",
                        "value": float(match.group(1)),
                        "type": "scalar",
                        "unit": "GB/s",
                    }
                )
        return metrics

    def _parse_cache_bandwidth(self, output: str, run_id: str) -> List[Dict]:
        """Parse cache bandwidth sweep test output."""
        metrics = []

        # Parse L1
        l1_match = re.search(L1_CACHE_PATTERN, output, re.DOTALL)
        if l1_match:
            l1_data = self._parse_cache_lines(l1_match.group(1), cache_level="l1")
            if l1_data:
                metrics.append(
                    self._create_timeseries_metric(
                        "hardware.gpu_cache_l1",
                        l1_data,
                        f"cache_l1_bandwidth_{run_id}",
                        L1_CACHE_CSV_FIELDS,
                    )
                )

        # Parse L2
        l2_match = re.search(L2_CACHE_PATTERN, output, re.DOTALL)
        if l2_match:
            l2_data = self._parse_cache_lines(l2_match.group(1), cache_level="l2")
            if l2_data:
                metrics.append(
                    self._create_timeseries_metric(
                        "hardware.gpu_cache_l2",
                        l2_data,
                        f"cache_l2_bandwidth_{run_id}",
                        L2_CACHE_CSV_FIELDS,
                    )
                )

        return metrics

    def _parse_cache_lines(self, text: str, cache_level: str) -> List[Dict]:
        """
        Parse cache metrics from text lines.

        Args:
            text: Text containing cache data lines
            cache_level: Either 'l1' or 'l2'

        Returns:
            List of parsed cache metric dictionaries
        """
        csv_data = []

        for line in text.strip().split("\n"):
            if parsed := self._parse_cache_line(line, cache_level):
                csv_data.append(parsed)

        return csv_data

    def _parse_cache_line(self, line: str, cache_level: str) -> Optional[Dict]:
        """
        Parse a single cache line.

        Args:
            line: Line of text containing cache metrics
            cache_level: Either 'l1' or 'l2'

        Returns:
            Dictionary with parsed metrics or None if parsing fails
        """
        parts = line.split()

        if cache_level == "l1" and len(parts) >= 5:
            try:
                return {
                    "data_set": f"{parts[0]} {parts[1]}",
                    "_sort_key": float(parts[0]),
                    "exec_time": parts[2],
                    "spread": parts[3],
                    "eff_bw": float(parts[4].removesuffix("GB/s")),
                }
            except (ValueError, IndexError):
                pass

        elif cache_level == "l2" and len(parts) >= 7:
            try:
                return {
                    "data_set": f"{parts[0]} {parts[1]}",
                    "exec_data": f"{parts[2]} {parts[3]}",
                    "_sort_key": float(parts[2].replace("kB", "")),
                    "exec_time": parts[4],
                    "spread": parts[5],
                    "eff_bw": float(parts[6].removesuffix("GB/s")),
                }
            except (ValueError, IndexError):
                pass

        return None
