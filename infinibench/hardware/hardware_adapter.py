#!/usr/bin/env python3
"""Hardware test adapter for native and CUDA-compatible accelerators."""

import logging
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from infinibench.adapter import BaseAdapter
from infinibench.common.command_builder import build_command_from_config
from infinibench.common.constants import (
    CACHE_TEST_TIMEOUT,
    DEFAULT_TEST_TIMEOUT,
    L1_CACHE_CSV_FIELDS,
    L1_CACHE_PATTERN,
    L2_CACHE_CSV_FIELDS,
    L2_CACHE_PATTERN,
    MEMORY_CSV_FIELDS,
    MEMORY_DIRECTIONS,
    METRIC_PREFIX_MEM_SWEEP,
    STREAM_OPERATIONS,
    TEST_TYPE_MAP,
    InfiniBenchJson,
)
from infinibench.common.csv_utils import create_timeseries_metric
from infinibench.hardware.constants import PLATFORM_ALIASES, PLATFORM_CONFIGS
from infinibench.utils.time_utils import get_timestamp

logger = logging.getLogger(__name__)


def detect_platform() -> str:
    """Detect the installed accelerator toolchain."""
    if (
        shutil.which("npu-smi")
        or shutil.which("atc")
        or Path("/usr/local/Ascend/ascend-toolkit").exists()
    ):
        return "ascend"
    if shutil.which("cncc") or Path("/usr/local/neuware").exists():
        return "cambricon"
    if shutil.which("mcc") or shutil.which("mthreads-gmi"):
        return "moore"

    maca_path = Path("/opt/maca")
    if (
        (maca_path / "tools" / "cu-bridge" / "bin" / "cucc").exists()
        or shutil.which("cucc")
        or shutil.which("mxcc")
    ):
        return "metax"

    dtk_path = Path("/opt/dtk")
    if (
        (dtk_path / "bin" / "hy-smi").exists()
        or shutil.which("hy-smi")
        or (dtk_path.exists() and shutil.which("hipcc"))
    ):
        return "hygon"

    corex_path = Path("/usr/local/corex")
    if (corex_path / "bin" / "ixsmi").exists() or (
        corex_path / "bin" / "clang++"
    ).exists():
        return "corex"
    return "cuda"


class HardwareTestAdapter(BaseAdapter):
    """Adapter for unified hardware performance tests.

    The existing CUDA constructor and private method signatures remain supported.
    Additional platforms are selected through ``config.device`` or runtime
    toolchain detection.
    """

    def __init__(
        self,
        cuda_perf_path: str = None,
        output_dir: str = "./output",
        perf_binary_path: str = None,
    ):
        self.hardware_dir = Path(__file__).parent
        self.cuda_perf_path = cuda_perf_path or str(
            self.hardware_dir / "cuda-memory-benchmark" / "build" / "cuda_perf_suite"
        )
        self.perf_binary_path = perf_binary_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Preserve attributes used by existing callers.
        self.build_dir = self.hardware_dir / "cuda-memory-benchmark"
        self.build_script = self.build_dir / "build.sh"

    def setup(self, config: Dict[str, Any]) -> None:
        """Build the selected benchmark binary if it is missing."""
        device = self._get_device_type(config)
        if (
            device == "cpu"
            or config.get("skip_build", False)
            or Path(self._get_binary_path(device)).exists()
        ):
            return
        self._build_project(device)

    def process(self, test_input: Any) -> Dict[str, Any]:
        """Process a hardware test input and return normalized metrics."""
        test_input = self._normalize_test_input(test_input)
        if not test_input:
            raise ValueError(f"Invalid test_input type: {type(test_input)}")

        testcase = test_input.get(InfiniBenchJson.TESTCASE, "unknown")
        config = test_input.get(InfiniBenchJson.CONFIG, {})
        run_id = test_input.get(InfiniBenchJson.RUN_ID, "unknown")

        logger.info("HardwareTestAdapter: Processing %s", testcase)
        self.output_dir = Path(config.get("output_dir", "./output")) / "hardware"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        device = self._get_device_type(config)
        test_type = config.get("test_type", "comprehensive")

        try:
            if device == "cpu":
                logger.info(
                    "CPU mode: Skipping hardware tests (not supported on CPU), "
                    "returning empty results"
                )
                metrics = []
                command = None
            else:
                logger.info("Accelerator mode (platform=%s): Executing tests", device)
                cmd = self._build_command(config, device)
                command = " ".join(cmd)
                output = self._execute_test(cmd, test_type, device)
                metrics = self._parse_output(output, test_type, run_id, device)

            result_config = config.copy()
            if command:
                result_config["command"] = command
            if device not in ("cpu", "cuda"):
                result_config.setdefault("platform", device)
            return {
                InfiniBenchJson.RESULT_CODE: 0,
                InfiniBenchJson.TIME: get_timestamp(),
                InfiniBenchJson.RUN_ID: run_id,
                InfiniBenchJson.TESTCASE: testcase,
                InfiniBenchJson.CONFIG: result_config,
                InfiniBenchJson.METRICS: metrics,
            }
        except Exception as exc:
            logger.error(
                "HardwareTestAdapter: Test failed for %s\n"
                "  Device: %s\n"
                "  Test Type: %s\n"
                "  Error: %s",
                testcase,
                device,
                test_type,
                str(exc),
                exc_info=True,
            )
            raise

    def _get_device_type(self, config: Dict[str, Any]) -> str:
        """Resolve an explicit device or the locally installed toolchain."""
        explicit = str(config.get("device", "")).lower().strip()
        if explicit == "cpu":
            return "cpu"
        if explicit:
            # Unknown explicit device values historically used the CUDA suite.
            return PLATFORM_ALIASES.get(explicit, "cuda")

        return detect_platform()

    @staticmethod
    def _get_device_config(device: str) -> Dict[str, Any]:
        config = PLATFORM_CONFIGS.get(device)
        if not config:
            raise ValueError(f"Unknown device type: {device}")
        return config

    def _get_binary_path(self, device: str) -> str:
        if self.perf_binary_path:
            return self.perf_binary_path
        if device in ("cuda", "metax", "corex", "hygon", "moore"):
            return self.cuda_perf_path
        device_config = self._get_device_config(device)
        return str(
            self.hardware_dir
            / device_config["benchmark_subdir"]
            / "build"
            / device_config["binary_name"]
        )

    def _build_cuda_project(self) -> None:
        """Build the detected CUDA-compatible project through the legacy entrypoint."""
        self._build_project(detect_platform())

    def _build_project(self, device: str) -> None:
        device_config = self._get_device_config(device)
        build_dir = self.hardware_dir / device_config["benchmark_subdir"]
        build_script = build_dir / "build.sh"
        if not build_dir.exists():
            raise FileNotFoundError(f"Benchmark directory not found: {build_dir}")
        if not build_script.exists():
            raise FileNotFoundError(f"Build script not found: {build_script}")

        command = ["bash", str(build_script)]
        if device_config["build_platform"]:
            command.extend(["--platform", device_config["build_platform"]])

        logger.info("Building %s project in: %s", device, build_dir)
        try:
            result = subprocess.run(
                command,
                cwd=str(build_dir),
                capture_output=True,
                text=True,
                timeout=300,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"Failed to build {device} hardware project:\n{result.stderr}"
                )
            logger.info("%s hardware project build completed successfully", device)
        except subprocess.TimeoutExpired:
            raise RuntimeError("Hardware project build timed out after 5 minutes")

    def _build_command(self, config: Dict[str, Any], device: str = "cuda") -> List[str]:
        """Build a benchmark command using the existing CUDA CLI contract."""
        test_type = config.get("test_type", "all")
        cli_test_type = TEST_TYPE_MAP.get(test_type, test_type.lower())
        base_command = [self._get_binary_path(device), f"--{cli_test_type}"]
        param_mappings = [
            ("device_id", "--device"),
            ("iterations", "--iterations"),
            ("array_size", "--array-size"),
        ]
        return build_command_from_config(base_command, config, param_mappings)

    def _execute_test(
        self, cmd: List[str], test_type: str, device: str = "cuda"
    ) -> str:
        """Execute a benchmark and return stdout."""
        binary = self._get_binary_path(device)
        if not Path(binary).exists():
            if device == "cuda":
                raise RuntimeError(f"cuda_perf_suite not found: {binary}")
            raise RuntimeError(f"{device} benchmark binary not found: {binary}")
        logger.info("Executing: %s", " ".join(cmd))

        timeout = (
            CACHE_TEST_TIMEOUT if test_type.lower() == "cache" else DEFAULT_TEST_TIMEOUT
        )
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True, timeout=timeout
        )
        return result.stdout

    def _parse_output(
        self,
        output: str,
        test_type: str,
        run_id: str,
        device: str = "cuda",
    ) -> List[Dict]:
        """Parse benchmark output while preserving existing CUDA metric names."""
        cache_parser = self._get_device_config(device)["cache_parser"]
        if test_type == "Comprehensive":
            return (
                self._parse_memory_bandwidth(output, run_id, METRIC_PREFIX_MEM_SWEEP)
                + self._parse_stream_benchmark(output)
                + self._parse_cache_for_platform(output, run_id, cache_parser)
            )

        metric_map = {
            "MemSweep": lambda: self._parse_memory_bandwidth(
                output, run_id, METRIC_PREFIX_MEM_SWEEP
            ),
            "Stream": lambda: self._parse_stream_benchmark(output),
            "Cache": lambda: self._parse_cache_for_platform(
                output, run_id, cache_parser
            ),
        }
        parser = metric_map.get(test_type)
        return parser() if parser else []

    def _parse_memory_bandwidth(
        self, output: str, run_id: str, metric_prefix: str
    ) -> List[Dict]:
        """Parse memory bandwidth sweep output."""
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
        """Parse bandwidth rows for one transfer direction."""
        csv_data = []
        sweep_pattern = (
            rf"{direction}.*?Size \(MB\)\s+Time \(ms\)\s*Bandwidth \(GB/s\)"
            rf"\s+CV \(%\)\s*-+\s*(.*?)\s*"
            rf"(?=\n=+|Direction:|STREAM:|\Z)"
        )
        sweep_match = re.search(sweep_pattern, output, re.DOTALL)
        if sweep_match:
            for line in sweep_match.group(1).strip().split("\n"):
                line = line.strip()
                if line and not line.startswith("-") and not line.startswith("NOTE"):
                    result = self._parse_sweep_line(line)
                    if result:
                        csv_data.append(result)
        return csv_data

    @staticmethod
    def _parse_sweep_line(line: str) -> Optional[Dict]:
        """Parse one memory sweep row."""
        parts = line.split()
        if len(parts) >= 3:
            try:
                return {
                    "size_mb": float(parts[0]),
                    "bandwidth_gbps": float(parts[2]),
                }
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
        """Create a timeseries metric and its CSV file."""
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
        for operation in STREAM_OPERATIONS:
            match = re.search(rf"STREAM_{operation.capitalize()}\s+(\d+\.\d+)", output)
            if match:
                metrics.append(
                    {
                        "name": f"hardware.stream_{operation}",
                        "value": float(match.group(1)),
                        "type": "scalar",
                        "unit": "GB/s",
                    }
                )
        return metrics

    def _parse_cache_for_platform(
        self, output: str, run_id: str, parser_name: str
    ) -> List[Dict]:
        if parser_name == "ascend":
            return self._parse_ascend_d2d_size_sweep(output, run_id)
        if parser_name == "cambricon":
            return self._parse_cambricon_cache(output, run_id)
        if parser_name == "cuda":
            return self._parse_cache_bandwidth(output, run_id)
        raise ValueError(f"Unknown cache parser: {parser_name}")

    def _parse_ascend_d2d_size_sweep(self, output: str, run_id: str) -> List[Dict]:
        """Parse Ascend's ACL D2D memcpy size sweep."""
        match = re.search(
            r"D2D Memcpy Size Sweep Test.*?Eff\. bw\s*-+\s*\n(.*?)(?=\Z)",
            output,
            re.DOTALL,
        )
        if not match:
            return []
        rows = self._parse_cache_lines(match.group(1), "l2")
        if not rows:
            return []
        return [
            self._create_timeseries_metric(
                "hardware.d2d_memcpy_size_sweep",
                rows,
                f"d2d_memcpy_size_sweep_{run_id}",
                L2_CACHE_CSV_FIELDS,
            )
        ]

    def _parse_cambricon_cache(self, output: str, run_id: str) -> List[Dict]:
        """Map Cambricon NRAM and L2 results to the existing cache schema."""
        metrics = []
        nram_match = re.search(
            r"NRAM Bandwidth Test.*?Spread\s*-+\s*\n(.*?)(?=\n\s*=+|" r"\nL2 Cache|\Z)",
            output,
            re.DOTALL,
        )
        if nram_match:
            rows = []
            for line in nram_match.group(1).strip().splitlines():
                parts = line.split()
                if len(parts) >= 6:
                    try:
                        rows.append(
                            {
                                "data_set": f"{parts[0]} {parts[1]}",
                                "_sort_key": float(parts[0]),
                                "exec_time": parts[2],
                                "spread": parts[5],
                                "eff_bw": float(parts[3]),
                            }
                        )
                    except (ValueError, IndexError):
                        pass
            if rows:
                metrics.append(
                    self._create_timeseries_metric(
                        "hardware.gpu_cache_l1",
                        rows,
                        f"cache_l1_bandwidth_{run_id}",
                        L1_CACHE_CSV_FIELDS,
                    )
                )

        l2_match = re.search(
            r"L2 Cache Bandwidth Sweep Test.*?Eff\. bw\s*-+\s*\n(.*?)(?=\Z)",
            output,
            re.DOTALL,
        )
        if l2_match:
            rows = self._parse_cache_lines(l2_match.group(1), "l2")
            if rows:
                metrics.append(
                    self._create_timeseries_metric(
                        "hardware.gpu_cache_l2",
                        rows,
                        f"cache_l2_bandwidth_{run_id}",
                        L2_CACHE_CSV_FIELDS,
                    )
                )
        return metrics

    def _parse_cache_bandwidth(self, output: str, run_id: str) -> List[Dict]:
        """Parse the existing CUDA L1 and L2 cache output."""
        metrics = []
        l1_match = re.search(L1_CACHE_PATTERN, output, re.DOTALL)
        if l1_match:
            l1_data = self._parse_cache_lines(l1_match.group(1), "l1")
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
            l2_data = self._parse_cache_lines(l2_match.group(1), "l2")
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
        """Parse cache metric rows."""
        rows = []
        for line in text.strip().split("\n"):
            parsed = self._parse_cache_line(line, cache_level)
            if parsed:
                rows.append(parsed)
        return rows

    @staticmethod
    def _parse_cache_line(line: str, cache_level: str) -> Optional[Dict]:
        """Parse one CUDA-style cache metric row."""
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
