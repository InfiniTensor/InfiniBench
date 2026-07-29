#!/usr/bin/env python3
"""Hardware Information Collector."""

import logging
import re
import subprocess
from dataclasses import dataclass
from shutil import which
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Hardware probe configurations
PROBE_CONFIGS = {
    "nvidia": {
        "command": [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ],
        "pattern": r"\bGPU\b",
        "default_name": "NVIDIA GPU",
        "parse_output": True,
    },
    "moore": {
        "command": ["mthreads-gmi", "-q"],
        "pattern": r"\bGPU\b|\bProduct\b",
        "default_name": "Moore Threads GPU",
    },
    "amd": {
        "candidates": ["amd-smi", "rocm-smi"],
        "pattern": r"\bGPU\b",
        "default_name": "AMD GPU",
    },
    "ascend": {
        "command": ["npu-smi", "info"],
        "pattern": r"\bNPU\b|\bDevice\b",
        "default_name": "Ascend NPU",
    },
    "cambricon": {
        "command": ["cnmon", "info"],
        "pattern": r"\bMLU\b|\bDevice\b",
        "default_name": "Cambricon MLU",
    },
}


@dataclass
class ProbeResult:
    """Result from probing a hardware type."""

    success: bool
    count: int = 0
    model: str = ""
    driver: str = ""
    memory_gb: int = 0


def _which(cmd: str) -> Optional[str]:
    """Check if command exists in PATH."""
    try:
        return which(cmd)
    except Exception:
        return None


class HardwareCollector:
    """Collects static hardware information (CPU, memory, GPU)."""

    def collect(self, accel_type: str = "", device_ids: Any = None) -> Dict[str, Any]:
        """
        Best-effort static HW collector (CPU/mem/GPU model/driver/CUDA).

        Args:
            accel_type: Hint for accelerator type
            device_ids: Device IDs to query (currently unused)

        Returns:
            Dictionary with hardware information
        """
        hw: Dict[str, Any] = {
            "cpu_model": "Unknown",
            "memory_gb": 0,
            "gpu_model": "Unknown",
            "gpu_count": 0,
            "gpu_memory_gb": 0,
            "driver_version": "Unknown",
            "cuda_version": "Unknown",
            "accelerator_type": "generic",
        }

        self._collect_cpu_info(hw)
        self._collect_memory_info(hw)

        # Determine probe order based on hint
        hint = (accel_type or "").lower().strip()
        probe_order = self._get_probe_order(hint)

        for probe_type in probe_order:
            result = self._probe(probe_type, hw)
            if result.success:
                hw["accelerator_type"] = probe_type
                if probe_type == "nvidia":
                    hw["cuda_version"] = (
                        self._collect_cuda_version() or hw["cuda_version"]
                    )
                elif probe_type == "moore":
                    musa_ver = self._collect_musa_version()
                    if musa_ver:
                        hw["cuda_version"] = f"MUSA {musa_ver}"
                return hw

        return hw

    def _get_probe_order(self, hint: str) -> List[str]:
        """Get probe order based on accelerator type hint."""
        order = ["nvidia", "moore", "amd", "ascend", "cambricon"]
        if hint in order:
            return [hint] + [p for p in order if p != hint]
        return order

    def _collect_cpu_info(self, hw: Dict[str, Any]) -> None:
        """Collect CPU model information."""
        try:
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if "model name" in line:
                        hw["cpu_model"] = line.split(":", 1)[1].strip()
                        break
        except Exception as e:
            logger.warning(f"Failed to collect CPU info: {e}")

    def _collect_memory_info(self, hw: Dict[str, Any]) -> None:
        """Collect total memory information."""
        try:
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if "MemTotal" in line:
                        mem_kb = int(line.split()[1])
                        hw["memory_gb"] = mem_kb // (1024 * 1024)
                        break
        except Exception as e:
            logger.warning(f"Failed to collect memory info: {e}")

    def _probe(self, probe_type: str, hw: Dict[str, Any]) -> ProbeResult:
        """Generic probe dispatcher."""
        probe_methods = {
            "nvidia": self._probe_nvidia,
            "moore": self._probe_mthreads,
            "amd": self._probe_amd,
            "ascend": self._probe_generic_command,
            "cambricon": self._probe_generic_command,
        }
        method = probe_methods.get(probe_type)
        if method:
            return method(probe_type, hw)
        return ProbeResult(success=False)

    def _probe_nvidia(self, probe_type: str, hw: Dict[str, Any]) -> ProbeResult:
        """Probe NVIDIA GPU with special parsing."""
        config = PROBE_CONFIGS["nvidia"]
        try:
            r = subprocess.run(
                config["command"], capture_output=True, text=True, timeout=5
            )
            if r.returncode != 0 or not r.stdout.strip():
                return ProbeResult(success=False)

            lines = [x.strip() for x in r.stdout.strip().splitlines() if x.strip()]
            hw["gpu_count"] = len(lines)

            parts = [x.strip() for x in lines[0].split(",")]
            if len(parts) >= 3:
                hw["gpu_model"] = parts[0]
                hw["driver_version"] = parts[2]
                mem_match = re.search(r"(\d+)\s*MiB", parts[1])
                if mem_match:
                    hw["gpu_memory_gb"] = int(mem_match.group(1)) // 1024
            return ProbeResult(success=True, count=hw["gpu_count"])
        except Exception:
            return ProbeResult(success=False)

    def _probe_amd(self, probe_type: str, hw: Dict[str, Any]) -> ProbeResult:
        """Probe AMD GPU - detect available tool first."""
        config = PROBE_CONFIGS["amd"]
        tool = next((c for c in config["candidates"] if _which(c)), None)
        if not tool:
            return ProbeResult(success=False)

        cmd = [tool, "list"] if tool == "amd-smi" else [tool, "-i"]
        return self._run_probe_command(
            cmd, config["pattern"], config["default_name"], hw
        )

    def _probe_mthreads(self, probe_type: str, hw: Dict[str, Any]) -> ProbeResult:
        """Probe Moore Threads GPU using mthreads-gmi."""
        config = PROBE_CONFIGS["moore"]
        if not _which(config["command"][0]):
            return ProbeResult(success=False)

        try:
            r = subprocess.run(
                config["command"], capture_output=True, text=True, timeout=5
            )
            if r.returncode != 0 or not r.stdout.strip():
                return ProbeResult(success=False)

            gpu_count = 0
            gpu_name = None
            for line in r.stdout.splitlines():
                stripped = line.strip()
                if re.match(r"^GPU\d+\s", stripped):
                    gpu_count += 1
                if stripped.startswith("Product Name") and ":" in stripped:
                    gpu_name = stripped.split(":", 1)[1].strip()

            if gpu_count > 0:
                hw["gpu_count"] = max(hw["gpu_count"], gpu_count)
            if hw["gpu_model"] == "Unknown":
                hw["gpu_model"] = gpu_name or config["default_name"]

            # Try to get MUSA version
            musa_ver = self._collect_musa_version()
            if musa_ver:
                hw["cuda_version"] = f"MUSA {musa_ver}"

            return ProbeResult(success=True, count=hw["gpu_count"])
        except Exception:
            return ProbeResult(success=False)

    def _probe_generic_command(
        self, probe_type: str, hw: Dict[str, Any]
    ) -> ProbeResult:
        """Generic probe for ascend/cambricon using command output."""
        config = PROBE_CONFIGS.get(probe_type)
        if not config or not _which(config["command"][0]):
            return ProbeResult(success=False)

        return self._run_probe_command(
            config["command"], config["pattern"], config["default_name"], hw
        )

    def _run_probe_command(
        self, command: List[str], pattern: str, default_name: str, hw: Dict[str, Any]
    ) -> ProbeResult:
        """Run probe command and parse output."""
        try:
            r = subprocess.run(command, capture_output=True, text=True, timeout=5)
            if r.returncode != 0 or not r.stdout.strip():
                return ProbeResult(success=False)

            count = len([x for x in r.stdout.splitlines() if re.search(pattern, x)])
            if count:
                hw["gpu_count"] = max(hw["gpu_count"], count)
            if hw["gpu_model"] == "Unknown":
                hw["gpu_model"] = default_name
            return ProbeResult(success=True, count=count)
        except Exception:
            return ProbeResult(success=False)

    def _collect_cuda_version(self) -> Optional[str]:
        """Collect CUDA version using nvcc."""
        try:
            r = subprocess.run(
                ["nvcc", "--version"], capture_output=True, text=True, timeout=2
            )
            if r.returncode == 0:
                for line in r.stdout.splitlines():
                    if "release" in line:
                        match = re.search(r"release\s+(\d+\.\d+)", line)
                        if match:
                            return match.group(1)
        except Exception as e:
            logger.debug(f"Failed to collect CUDA version: {e}")
        return None

    def _collect_musa_version(self) -> Optional[str]:
        """Collect MUSA version using mcc."""
        try:
            r = subprocess.run(
                ["mcc", "--version"], capture_output=True, text=True, timeout=2
            )
            if r.returncode == 0:
                for line in r.stdout.splitlines():
                    match = re.search(r"(\d+\.\d+\.\d+)", line)
                    if match:
                        return match.group(1)
        except Exception:
            pass
        return None


# Singleton instance for convenience
_collector = HardwareCollector()


def collect_hardware_info(
    accel_type: str = "", device_ids: Any = None
) -> Dict[str, Any]:
    """Convenience function to collect hardware info."""
    return _collector.collect(accel_type, device_ids)
