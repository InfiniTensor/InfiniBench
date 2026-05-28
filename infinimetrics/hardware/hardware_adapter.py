#!/usr/bin/env python3
"""Hardware Test Adapter for Unified Benchmark Suite (CUDA / Cambricon MLU)"""

import logging
import subprocess
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

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

# Per-device benchmark configuration
_DEVICE_CONFIGS = {
    "cuda": {
        "binary_name": "cuda_perf_suite",
        "benchmark_subdir": "cuda-memory-benchmark",
        "build_script": "build.sh",
        "has_cache_test": True,
    },
    "cambricon": {
        "binary_name": "mlu_perf_suite",
        "benchmark_subdir": "cambricon-memory-benchmark",
        "build_script": "build.sh",
        "has_cache_test": False,
    },
    "ascend": {
        "binary_name": "npu_perf_suite",
        "benchmark_subdir": "ascend-memory-benchmark",
        "build_script": "build.sh",
        "has_cache_test": False,
    },
}


class HardwareTestAdapter(BaseAdapter):
    """Adapter for hardware performance tests. Supports CUDA and Cambricon MLU."""

    def __init__(
        self,
        perf_binary_path: str = None,
        output_dir: str = "./output",
    ):
        self.perf_binary_path = perf_binary_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.hardware_dir = Path(__file__).parent

    def setup(self, config: Dict[str, Any]) -> None:
        """Build benchmark binary if not found."""
        device = self._get_device_type(config)
        if (
            device == "cpu"
            or config.get("skip_build", False)
            or Path(self._get_binary_path(device)).exists()
        ):
            return
        self._build_project(device)

    def process(self, test_input: Any) -> Dict[str, Any]:
        """Process test input and return results."""
        test_input = self._normalize_test_input(test_input)
        if not test_input:
            raise ValueError(f"Invalid test_input type: {type(test_input)}")

        testcase = test_input.get(InfiniMetricsJson.TESTCASE, "unknown")
        config = test_input.get(InfiniMetricsJson.CONFIG, {})
        run_id = test_input.get(InfiniMetricsJson.RUN_ID, "unknown")

        logger.info(f"HardwareTestAdapter: Processing {testcase}")

        self.output_dir = Path(config.get("output_dir", "./output")) / "hardware"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        device = self._get_device_type(config)
        test_type = config.get("test_type", "comprehensive")

        try:
            if device == "cpu":
                logger.info("CPU mode: Skipping hardware tests")
                metrics = []
                command = None
            else:
                logger.info("Device mode (device=%s): Executing tests", device)
                cmd = self._build_command(config, device)
                command = " ".join(cmd)
                output = self._execute_test(cmd, device, test_type)
                metrics = self._parse_output(output, test_type, run_id, device)

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
            logger.error(
                f"HardwareTestAdapter: Test failed for {testcase}\n"
                f"  Device: {device}\n"
                f"  Test Type: {test_type}\n"
                f"  Error: {str(e)}",
                exc_info=True,
            )
            raise

    # ------------------------------------------------------------------
    # Device detection
    # ------------------------------------------------------------------

    def _get_device_type(self, config: Dict[str, Any]) -> str:
        """Determine device type from config. Falls back to testcase framework."""
        device = config.get("device", "").lower()
        if device in _DEVICE_CONFIGS:
            return device
        # Auto-detect from testcase: hardware.cambricon.* → cambricon
        testcase = config.get("_testcase", "")
        if "cambricon" in testcase.lower():
            return "cambricon"
        if "ascend" in testcase.lower():
            return "ascend"
        if device == "cpu":
            return "cpu"
        return "cuda"

    def _get_device_config(self, device: str) -> Dict:
        cfg = _DEVICE_CONFIGS.get(device)
        if not cfg:
            raise ValueError(f"Unknown device type: {device}")
        return cfg

    def _get_binary_path(self, device: str) -> str:
        if self.perf_binary_path:
            return self.perf_binary_path
        dev_cfg = self._get_device_config(device)
        return str(
            self.hardware_dir / dev_cfg["benchmark_subdir"] / "build" / dev_cfg["binary_name"]
        )

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _build_project(self, device: str) -> None:
        dev_cfg = self._get_device_config(device)
        build_dir = self.hardware_dir / dev_cfg["benchmark_subdir"]
        build_script = build_dir / dev_cfg["build_script"]

        if not build_dir.exists():
            raise FileNotFoundError(f"Benchmark directory not found: {build_dir}")
        if not build_script.exists():
            raise FileNotFoundError(f"Build script not found: {build_script}")

        logger.info("Building %s project in: %s", device, build_dir)
        try:
            result = subprocess.run(
                ["bash", str(build_script)],
                cwd=str(build_dir),
                capture_output=True,
                text=True,
                timeout=300,
            )
            if result.returncode != 0:
                raise RuntimeError(f"Build failed:\n{result.stderr}")
            logger.info("Build completed successfully")
        except subprocess.TimeoutExpired:
            raise RuntimeError("Build timed out after 5 minutes")

    # ------------------------------------------------------------------
    # Command building & execution
    # ------------------------------------------------------------------

    def _build_command(self, config: Dict[str, Any], device: str) -> List[str]:
        test_type = config.get("test_type", "all")
        cli_flag = TEST_TYPE_MAP.get(test_type, test_type.lower())

        binary = self._get_binary_path(device)
        base_command = [binary, f"--{cli_flag}"]

        param_mappings = [
            ("device_id", "--device"),
            ("iterations", "--iterations"),
            ("array_size", "--array-size"),
        ]
        return build_command_from_config(base_command, config, param_mappings)

    def _execute_test(self, cmd: List[str], device: str, test_type: str) -> str:
        binary = self._get_binary_path(device)
        if not Path(binary).exists():
            raise RuntimeError(f"Benchmark binary not found: {binary}")
        logger.info("Executing: %s", " ".join(cmd))

        timeout = CACHE_TEST_TIMEOUT if test_type.lower() == "cache" else DEFAULT_TEST_TIMEOUT
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True, timeout=timeout
        )
        return result.stdout

    # ------------------------------------------------------------------
    # Output parsing
    # ------------------------------------------------------------------

    def _parse_output(
        self, output: str, test_type: str, run_id: str, device: str
    ) -> List[Dict]:
        dev_cfg = self._get_device_config(device)
        metric_prefix = f"hardware.{device}"

        if test_type == "Comprehensive":
            parts = [self._parse_memory_bandwidth(output, run_id, metric_prefix)]
            parts.append(self._parse_stream_benchmark(output, metric_prefix))
            if dev_cfg["has_cache_test"]:
                parts.append(self._parse_cache_bandwidth(output, run_id))
            return [m for part in parts for m in part]

        parsers = {
            "MemSweep": lambda: self._parse_memory_bandwidth(
                output, run_id, metric_prefix
            ),
            "Stream": lambda: self._parse_stream_benchmark(output, metric_prefix),
            "Cache": lambda: self._parse_cache_bandwidth(output, run_id)
            if dev_cfg["has_cache_test"]
            else [],
        }

        parser = parsers.get(test_type)
        return parser() if parser else []

    # -- Memory bandwidth --

    def _parse_memory_bandwidth(
        self, output: str, run_id: str, metric_prefix: str
    ) -> List[Dict]:
        metrics = []
        is_sweep = "sweep" in metric_prefix or True  # always sweep for now

        for direction_label, key in MEMORY_DIRECTIONS:
            csv_data = self._parse_bandwidth_data(output, direction_label)
            if not csv_data:
                continue

            metric_name = f"{metric_prefix}.mem_sweep_{key}"

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
        csv_data = []

        # Match from direction header to next section (====, Direction:, STREAM, or end)
        pattern = rf"{direction}.*?Size \(MB\).*?-+\s*\n(.*?)(?=\n\s*=+|\nDirection:|\nSTREAM|\Z)"
        match = re.search(pattern, output, re.DOTALL)

        if match:
            for line in match.group(1).strip().split("\n"):
                line = line.strip()
                if not line or line.startswith("-") or line.startswith("NOTE"):
                    continue
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        csv_data.append(
                            {"size_mb": float(parts[0]), "bandwidth_gbps": float(parts[2])}
                        )
                    except (ValueError, IndexError):
                        pass

        return csv_data

    # -- STREAM --

    def _parse_stream_benchmark(self, output: str, metric_prefix: str) -> List[Dict]:
        metrics = []
        for op in STREAM_OPERATIONS:
            match = re.search(rf"STREAM_{op.capitalize()}\s+(\d+\.\d+)", output)
            if match:
                metrics.append(
                    {
                        "name": f"{metric_prefix}.stream_{op}",
                        "value": float(match.group(1)),
                        "type": "scalar",
                        "unit": "GB/s",
                    }
                )
        return metrics

    # -- Cache (CUDA only) --

    def _parse_cache_bandwidth(self, output: str, run_id: str) -> List[Dict]:
        metrics = []

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
        csv_data = []
        for line in text.strip().split("\n"):
            if parsed := self._parse_cache_line(line, cache_level):
                csv_data.append(parsed)
        return csv_data

    def _parse_cache_line(self, line: str, cache_level: str) -> Optional[Dict]:
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

    # -- Helpers --

    def _create_timeseries_metric(
        self,
        name: str,
        data: List[Dict],
        base_filename: str,
        fields: List[str],
        unit: str = "GB/s",
    ) -> Dict:
        return create_timeseries_metric(
            output_dir=self.output_dir,
            metric_name=name,
            data=data,
            base_filename=base_filename,
            fields=fields,
            unit=unit,
        )
