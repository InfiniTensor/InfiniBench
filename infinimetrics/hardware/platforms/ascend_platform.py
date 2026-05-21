#!/usr/bin/env python3
"""Ascend NPU platform handler for hardware bandwidth tests.

Uses msprof profiler and torch-based memory bandwidth test for Ascend NPU.
Falls back to a pure PyTorch bandwidth measurement when native tools
are unavailable.
"""

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class AscendPlatform:
    """Hardware test runner for Ascend NPU platforms."""

    # PyTorch-based bandwidth test script
    _BANDWIDTH_SCRIPT = """\
import torch
import torch_npu  # noqa: F401
import numpy as np
import json
import time

def measure_bw(shape, dtype, direction, device="npu:0", iterations=10):
    data = torch.randn(shape, dtype=dtype, device=device)
    cpu_data = torch.randn(shape, dtype=dtype, device="cpu")

    # Warmup
    for _ in range(3):
        if direction == "h2d":
            _ = cpu_data.to(device)
        elif direction == "d2h":
            _ = data.to("cpu")
        elif direction == "d2d":
            _ = data.clone()
    torch.npu.synchronize()

    start = time.time()
    for _ in range(iterations):
        if direction == "h2d":
            _ = cpu_data.to(device)
        elif direction == "d2h":
            _ = data.to("cpu")
        elif direction == "d2d":
            _ = data.clone()
    torch.npu.synchronize()

    elapsed = (time.time() - start) / iterations
    bytes_moved = shape[0] * dtype.itemsize if len(shape) == 1 else np.prod(shape) * dtype.itemsize
    bw_gbs = bytes_moved / elapsed / 1e9
    return bw_gbs

results = {}
sizes_mb = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
dtype = torch.float32
element_size = dtype.itemsize

for direction in ["h2d", "d2h", "d2d"]:
    bw_list = []
    for size_mb in sizes_mb:
        num_elements = size_mb * 1024 * 1024 // element_size
        try:
            bw = measure_bw((num_elements,), dtype, direction)
            bw_list.append({"size_mb": size_mb, "bandwidth_gbps": round(bw, 2)})
        except Exception as e:
            print(f"Error at {direction} {size_mb}MB: {e}")
    results[direction] = bw_list

# Also measure peak (largest size)
peak = {}
for direction, bw_list in results.items():
    if bw_list:
        peak[direction] = max(bw_list, key=lambda x: x["bandwidth_gbps"])["bandwidth_gbps"]

print("RESULT_JSON:" + json.dumps({"sweep": results, "peak": peak}))
"""

    def __init__(self, output_dir: Path, config: Dict[str, Any] = None):
        self.config = config or {}
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.msprof_path = self.config.get("msprof_path", "msprof")
        self.use_torch_fallback = self.config.get("use_torch_fallback", True)

    def setup(self) -> None:
        """Check for available tools."""
        try:
            result = subprocess.run(
                [self.msprof_path, "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            logger.info("msprof available: %s", result.stdout.strip())
            self.use_torch_fallback = False
        except (FileNotFoundError, subprocess.TimeoutExpired):
            logger.info("msprof not found; using PyTorch-based bandwidth test")

    def run_test(self, test_type: str, config: Dict[str, Any]) -> List[Dict]:
        """Run hardware test and return parsed metrics."""
        metrics = []

        if test_type in ("Comprehensive", "MemSweep"):
            metrics.extend(self._run_memory_bandwidth(config))

        if test_type == "Stream":
            logger.warning("STREAM test uses memory bandwidth measurement on Ascend")
            metrics.extend(self._run_memory_bandwidth(config))

        if test_type == "Cache":
            logger.warning("Cache test not directly supported on Ascend NPU")

        return metrics

    def get_command(self, config: Dict[str, Any]) -> str:
        """Return the command string for traceability."""
        return f"ascend_platform: test_type={config.get('test_type', 'Comprehensive')}"

    def _run_memory_bandwidth(self, config: Dict[str, Any]) -> List[Dict]:
        """Run memory bandwidth measurement using PyTorch."""
        if not self.use_torch_fallback:
            # Try msprof-based approach
            return self._run_msprof_bandwidth(config)

        return self._run_torch_bandwidth(config)

    def _run_torch_bandwidth(self, config: Dict[str, Any]) -> List[Dict]:
        """Run PyTorch-based bandwidth test."""
        import json

        script = self._BANDWIDTH_SCRIPT
        if device_id := config.get("device_id"):
            script = script.replace('device="npu:0"', f'device="npu:{device_id}"')

        logger.info("Running PyTorch-based Ascend bandwidth test")
        try:
            result = subprocess.run(
                ["python3", "-c", script],
                capture_output=True,
                text=True,
                timeout=600,
            )
            if result.returncode != 0:
                logger.error("Ascend bandwidth test failed: %s", result.stderr)
                return []

            # Parse JSON from output
            for line in result.stdout.split("\n"):
                if line.startswith("RESULT_JSON:"):
                    data = json.loads(line[len("RESULT_JSON:"):])
                    return self._parse_torch_results(data)

            logger.warning("No RESULT_JSON found in Ascend bandwidth test output")
            return []

        except subprocess.TimeoutExpired:
            logger.error("Ascend bandwidth test timed out")
            return []
        except json.JSONDecodeError as e:
            logger.error("Failed to parse Ascend bandwidth test output: %s", e)
            return []

    def _run_msprof_bandwidth(self, config: Dict[str, Any]) -> List[Dict]:
        """Run msprof-based bandwidth analysis."""
        metrics = []
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                cmd = [
                    self.msprof_path,
                    "--application=python3 -c 'import torch; import torch_npu; print(torch.npu.memory_stats())'",
                    f"--output={tmpdir}",
                    "--duration=10",
                ]
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=120
                )
                if result.returncode != 0:
                    logger.error("msprof failed: %s", result.stderr)
                    return []

                # Parse msprof output
                logger.info("msprof output saved to %s", tmpdir)
                # msprof output parsing would go here

        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.error("msprof execution failed: %s", e)

        return metrics

    def _parse_torch_results(self, data: Dict[str, Any]) -> List[Dict]:
        """Parse PyTorch bandwidth test results into metrics."""
        from infinimetrics.common.csv_utils import create_timeseries_metric
        from infinimetrics.common.constants import MEMORY_CSV_FIELDS

        metrics = []
        sweep = data.get("sweep", {})
        peak = data.get("peak", {})

        for direction in ["h2d", "d2h", "d2d"]:
            bw_list = sweep.get(direction, [])
            if not bw_list:
                continue

            # Save sweep data as timeseries
            metrics.append(
                create_timeseries_metric(
                    self.output_dir,
                    f"hardware.mem_sweep_{direction}",
                    bw_list,
                    f"ascend_mem_sweep_{direction}",
                    MEMORY_CSV_FIELDS,
                )
            )

            # Add peak as scalar
            if direction in peak:
                metrics.append(
                    {
                        "name": f"hardware.mem_sweep_{direction}_peak",
                        "value": round(peak[direction], 2),
                        "type": "scalar",
                        "unit": "GB/s",
                    }
                )

        return metrics
