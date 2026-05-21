#!/usr/bin/env python3
"""CUDA-compatible platform handler for hardware bandwidth tests.

Supports NVIDIA, MetaX, Iluvatar, Hygon, Moore Threads — all platforms
that expose CUDA-compatible APIs and can run CUDA benchmark binaries.
"""

import logging
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from infinimetrics.common.csv_utils import save_csv, create_timeseries_metric
from infinimetrics.common.constants import (
    TEST_TYPE_MAP,
    MEMORY_DIRECTIONS,
    STREAM_OPERATIONS,
    MEMORY_CSV_FIELDS,
    L1_CACHE_CSV_FIELDS,
    L2_CACHE_CSV_FIELDS,
    L1_CACHE_PATTERN,
    L2_CACHE_PATTERN,
)

logger = logging.getLogger(__name__)


class CudaPlatform:
    """Hardware test runner for CUDA-compatible platforms."""

    def __init__(self, output_dir: Path, config: Dict[str, Any] = None):
        self.config = config or {}
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Resolve cuda_perf_suite binary path
        self.cuda_perf_path = self.config.get(
            "cuda_perf_path",
            str(
                Path(__file__).parent.parent
                / "cuda-memory-benchmark"
                / "build"
                / "cuda_perf_suite"
            ),
        )
        self.build_dir = Path(__file__).parent.parent / "cuda-memory-benchmark"
        self.build_script = self.build_dir / "build.sh"

    def setup(self) -> None:
        """Build CUDA project if binary not found."""
        if self.config.get("skip_build", False) or Path(self.cuda_perf_path).exists():
            return
        self._build()

    def run_test(self, test_type: str, config: Dict[str, Any]) -> List[Dict]:
        """Run hardware test and return parsed metrics."""
        cmd = self._build_command(config)
        output = self._execute(cmd, test_type)
        return self._parse_output(output, test_type)

    def get_command(self, config: Dict[str, Any]) -> str:
        """Return the command string for traceability."""
        cmd = self._build_command(config)
        return " ".join(cmd)

    def _build(self) -> None:
        """Build CUDA project."""
        if not self.build_dir.exists():
            raise FileNotFoundError(
                f"CUDA benchmark directory not found: {self.build_dir}"
            )
        if not self.build_script.exists():
            raise FileNotFoundError(f"Build script not found: {self.build_script}")

        logger.info("Building CUDA project in: %s", self.build_dir)
        result = subprocess.run(
            ["bash", str(self.build_script)],
            cwd=str(self.build_dir),
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to build CUDA project:\n{result.stderr}")
        logger.info("CUDA project build completed successfully")

    def _build_command(self, config: Dict[str, Any]) -> List[str]:
        """Build command for CUDA test suite."""
        test_type = config.get("test_type", "all")
        cuda_test_type = TEST_TYPE_MAP.get(test_type, test_type.lower())

        cmd = [self.cuda_perf_path, f"--{cuda_test_type}"]

        for config_key, param_name in [
            ("device_id", "--device"),
            ("iterations", "--iterations"),
            ("array_size", "--array-size"),
        ]:
            if config_key in config:
                cmd.extend([param_name, str(config[config_key])])

        return cmd

    def _execute(self, cmd: List[str], test_type: str) -> str:
        """Execute CUDA test and return output."""
        if not Path(self.cuda_perf_path).exists():
            raise RuntimeError(f"cuda_perf_suite not found: {self.cuda_perf_path}")

        logger.info("Executing: %s", " ".join(cmd))
        timeout = 1800 if test_type.lower() == "cache" else 600
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True, timeout=timeout
        )
        return result.stdout

    def _parse_output(self, output: str, test_type: str) -> List[Dict]:
        """Parse test output based on test type."""
        if test_type == "Comprehensive":
            return (
                self._parse_memory_bandwidth(output, "hardware.mem_sweep")
                + self._parse_stream_benchmark(output)
                + self._parse_cache_bandwidth(output)
            )

        metric_map = {
            "MemSweep": ("hardware.mem_sweep", self._parse_memory_bandwidth),
            "Stream": (None, lambda o, _: self._parse_stream_benchmark(o)),
            "Cache": (None, lambda o, _: self._parse_cache_bandwidth(o)),
        }
        if test_type in metric_map:
            prefix, parser = metric_map[test_type]
            return parser(output, prefix) if prefix else parser(output, prefix)
        return []

    def _parse_memory_bandwidth(self, output: str, prefix: str) -> List[Dict]:
        """Parse memory bandwidth test output."""
        import re

        metrics = []
        is_sweep = "sweep" in prefix

        for direction_label, key in MEMORY_DIRECTIONS:
            csv_data = []
            sweep_pattern = (
                rf"{direction_label}.*?Size \(MB\)\s+Time \(ms\)Bandwidth \(GB/s\)"
                rf"\s+CV \(%\)\s*-+\s*(.*?)\s*(?=\n=+|Direction:|STREAM:|\Z)"
            )
            sweep_match = re.search(sweep_pattern, output, re.DOTALL)
            if sweep_match:
                for line in sweep_match.group(1).strip().split("\n"):
                    parts = line.strip().split()
                    if len(parts) >= 3 and not line.strip().startswith("-"):
                        try:
                            csv_data.append(
                                {
                                    "size_mb": float(parts[0]),
                                    "bandwidth_gbps": float(parts[2]),
                                }
                            )
                        except (ValueError, IndexError):
                            pass

            if not csv_data:
                continue

            metric_name = f"{prefix}_{key}"
            if is_sweep:
                metrics.append(
                    create_timeseries_metric(
                        self.output_dir,
                        metric_name,
                        csv_data,
                        f"mem_sweep_{key}",
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

    def _parse_stream_benchmark(self, output: str) -> List[Dict]:
        """Parse STREAM benchmark output."""
        import re

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

    def _parse_cache_bandwidth(self, output: str) -> List[Dict]:
        """Parse cache bandwidth sweep test output."""
        import re

        metrics = []

        l1_match = re.search(L1_CACHE_PATTERN, output, re.DOTALL)
        if l1_match:
            l1_data = self._parse_cache_lines(l1_match.group(1), "l1")
            if l1_data:
                metrics.append(
                    create_timeseries_metric(
                        self.output_dir,
                        "hardware.gpu_cache_l1",
                        l1_data,
                        "cache_l1_bandwidth",
                        L1_CACHE_CSV_FIELDS,
                    )
                )

        l2_match = re.search(L2_CACHE_PATTERN, output, re.DOTALL)
        if l2_match:
            l2_data = self._parse_cache_lines(l2_match.group(1), "l2")
            if l2_data:
                metrics.append(
                    create_timeseries_metric(
                        self.output_dir,
                        "hardware.gpu_cache_l2",
                        l2_data,
                        "cache_l2_bandwidth",
                        L2_CACHE_CSV_FIELDS,
                    )
                )

        return metrics

    def _parse_cache_lines(self, text: str, cache_level: str) -> List[Dict]:
        """Parse cache metrics from text lines."""
        csv_data = []
        for line in text.strip().split("\n"):
            parts = line.split()
            if cache_level == "l1" and len(parts) >= 5:
                try:
                    csv_data.append(
                        {
                            "data_set": f"{parts[0]} {parts[1]}",
                            "_sort_key": float(parts[0]),
                            "exec_time": parts[2],
                            "spread": parts[3],
                            "eff_bw": float(parts[4].removesuffix("GB/s")),
                        }
                    )
                except (ValueError, IndexError):
                    pass
            elif cache_level == "l2" and len(parts) >= 7:
                try:
                    csv_data.append(
                        {
                            "data_set": f"{parts[0]} {parts[1]}",
                            "exec_data": f"{parts[2]} {parts[3]}",
                            "_sort_key": float(parts[2].replace("kB", "")),
                            "exec_time": parts[4],
                            "spread": parts[5],
                            "eff_bw": float(parts[6].removesuffix("GB/s")),
                        }
                    )
                except (ValueError, IndexError):
                    pass
        return csv_data
