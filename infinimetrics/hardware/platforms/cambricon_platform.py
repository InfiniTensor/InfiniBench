#!/usr/bin/env python3
"""Cambricon MLU platform handler for hardware bandwidth tests.

Uses InfiniPerf hardware/bang tools:
- cnrt-memcpy: Memory bandwidth (H2D/D2H/D2D)
- cnvs: CNVS tool for PCIe/MLULink/Memory bandwidth
"""

import logging
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List

from infinimetrics.common.csv_utils import save_csv, create_timeseries_metric

logger = logging.getLogger(__name__)

# Default paths relative to repo root
_REPO_ROOT = Path(__file__).resolve().parents[5]
_BANG_DIR = _REPO_ROOT / "InfiniPerf" / "benchmarks" / "hardware" / "bang"


class CambriconPlatform:
    """Hardware test runner for Cambricon MLU platforms."""

    def __init__(self, output_dir: Path, config: Dict[str, Any] = None):
        self.config = config or {}
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Tool paths
        self.cnrt_memcpy_path = self.config.get(
            "cnrt_memcpy_path",
            str(_BANG_DIR / "cnrt-memcpy" / "cnrt_memcpy"),
        )
        self.cnvs_path = self.config.get(
            "cnvs_path",
            str(_BANG_DIR / "cnvs" / "cnvs"),
        )

    def setup(self) -> None:
        """Verify tools exist."""
        # Cambricon tools are pre-built; just check existence
        if not Path(self.cnrt_memcpy_path).exists():
            logger.warning(
                "cnrt_memcpy not found at %s; memory bandwidth test may fail",
                self.cnrt_memcpy_path,
            )
        if not Path(self.cnvs_path).exists():
            logger.warning(
                "cnvs not found at %s; CNVS tests may fail", self.cnvs_path
            )

    def run_test(self, test_type: str, config: Dict[str, Any]) -> List[Dict]:
        """Run hardware test and return parsed metrics."""
        metrics = []

        if test_type in ("Comprehensive", "MemSweep"):
            metrics.extend(self._run_cnrt_memcpy(config))
        if test_type in ("Comprehensive",) and Path(self.cnvs_path).exists():
            metrics.extend(self._run_cnvs(config))
        if test_type == "Stream":
            logger.warning("STREAM test not natively supported on Cambricon MLU")
        if test_type == "Cache":
            logger.warning("Cache test not natively supported on Cambricon MLU")

        return metrics

    def get_command(self, config: Dict[str, Any]) -> str:
        """Return the command string for traceability."""
        return f"cambricon_platform: test_type={config.get('test_type', 'Comprehensive')}"

    def _run_cnrt_memcpy(self, config: Dict[str, Any]) -> List[Dict]:
        """Run cnrt-memcpy bandwidth test."""
        if not Path(self.cnrt_memcpy_path).exists():
            logger.error("cnrt_memcpy binary not found: %s", self.cnrt_memcpy_path)
            return []

        cmd = [self.cnrt_memcpy_path]
        if device_id := config.get("device_id"):
            cmd.extend(["--device", str(device_id)])

        logger.info("Executing cnrt-memcpy: %s", " ".join(cmd))
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=True, timeout=600
            )
            return self._parse_cnrt_memcpy_output(result.stdout)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.error("cnrt-memcpy failed: %s", e)
            return []

    def _run_cnvs(self, config: Dict[str, Any]) -> List[Dict]:
        """Run CNVS tool for PCIe/MLULink/Memory bandwidth."""
        if not Path(self.cnvs_path).exists():
            logger.error("cnvs binary not found: %s", self.cnvs_path)
            return []

        metrics = []
        for test_name in ["pcie", "mlulink", "memory"]:
            cmd = [self.cnvs_path, "--test", test_name]
            logger.info("Executing cnvs: %s", " ".join(cmd))
            try:
                result = subprocess.run(
                    cmd, capture_output=True, text=True, check=True, timeout=600
                )
                parsed = self._parse_cnvs_output(result.stdout, test_name)
                metrics.extend(parsed)
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                logger.warning("cnvs %s test failed: %s", test_name, e)

        return metrics

    def _parse_cnrt_memcpy_output(self, output: str) -> List[Dict]:
        """Parse cnrt-memcpy output for bandwidth metrics."""
        metrics = []

        # Common patterns in cnrt-memcpy output:
        # H2D: xxx MB/s  or  H2D Bandwidth: xxx GB/s
        direction_map = {
            "h2d": r"(?:H2D|HostToDevice).*?(\d+(?:\.\d+)?)\s*(?:GB/s|MB/s)",
            "d2h": r"(?:D2H|DeviceToHost).*?(\d+(?:\.\d+)?)\s*(?:GB/s|MB/s)",
            "d2d": r"(?:D2D|DeviceToDevice).*?(\d+(?:\.\d+)?)\s*(?:GB/s|MB/s)",
        }

        for key, pattern in direction_map.items():
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                value = float(match.group(1))
                # Convert MB/s to GB/s if needed
                if "MB/s" in match.group(0):
                    value = value / 1024.0
                metrics.append(
                    {
                        "name": f"hardware.mem_sweep_{key}",
                        "value": round(value, 2),
                        "type": "scalar",
                        "unit": "GB/s",
                    }
                )

        return metrics

    def _parse_cnvs_output(self, output: str, test_name: str) -> List[Dict]:
        """Parse CNVS tool output."""
        metrics = []

        # Try to extract bandwidth values
        bw_matches = re.findall(
            rf"(\w+)\s+(?:bandwidth|Bandwidth)\s*:\s*(\d+(?:\.\d+)?)\s*(GB/s|MB/s)",
            output,
            re.IGNORECASE,
        )
        for label, value, unit in bw_matches:
            val = float(value)
            if unit == "MB/s":
                val = val / 1024.0
            metrics.append(
                {
                    "name": f"hardware.cambricon_{test_name}_{label.lower()}",
                    "value": round(val, 2),
                    "type": "scalar",
                    "unit": "GB/s",
                }
            )

        # Fallback: look for any numeric bandwidth value
        if not bw_matches:
            bw_match = re.search(
                r"(\d+(?:\.\d+)?)\s*(GB/s)", output, re.IGNORECASE
            )
            if bw_match:
                metrics.append(
                    {
                        "name": f"hardware.cambricon_{test_name}_bandwidth",
                        "value": round(float(bw_match.group(1)), 2),
                        "type": "scalar",
                        "unit": "GB/s",
                    }
                )

        return metrics
