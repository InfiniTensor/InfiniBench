#!/usr/bin/env python3
"""Hardware detection utilities for Executor."""

import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional


def _which(cmd: str) -> Optional[str]:
    """Check if command exists in PATH."""
    try:
        from shutil import which

        return which(cmd)
    except Exception:
        return None


class HardwareDetector:
    """Detect hardware information (CPU, GPU, memory)."""

    NVIDIA_SMI_QUERY = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader",
    ]
    AMD_SMI_CANDIDATES = ["amd-smi", "rocm-smi"]
    MTHREADS_SMI_QUERY = ["mthreads-gmi", "-q"]

    @classmethod
    def detect(cls, accel_type_hint: str = "") -> Dict[str, Any]:
        """
        Detect hardware information.

        Args:
            accel_type_hint: Hint for accelerator type (nvidia/amd/ascend/cambricon)

        Returns:
            Dictionary with hardware information
        """
        hw = cls._init_hardware_dict()

        # CPU detection
        cls._detect_cpu(hw)
        # Memory detection
        cls._detect_memory(hw)

        # GPU detection order
        probes = cls._get_probe_order(accel_type_hint)

        for probe in probes:
            if probe == "nvidia" and cls._probe_nvidia(hw):
                hw["accelerator_type"] = "nvidia"
                hw["cuda_version"] = cls._get_cuda_version() or hw["cuda_version"]
                return hw
            if probe == "moore" and cls._probe_mthreads(hw):
                hw["accelerator_type"] = "moore"
                musa_ver = cls._get_musa_version()
                if musa_ver:
                    hw["cuda_version"] = f"MUSA {musa_ver}"
                return hw
            if probe == "amd" and cls._probe_amd(hw):
                hw["accelerator_type"] = "amd"
                return hw
            if probe == "ascend" and cls._probe_ascend(hw):
                hw["accelerator_type"] = "ascend"
                return hw
            if probe == "cambricon" and cls._probe_cambricon(hw):
                hw["accelerator_type"] = "cambricon"
                return hw
            if probe == "generic":
                hw["accelerator_type"] = "generic"
                return hw

        return hw

    @classmethod
    def _init_hardware_dict(cls) -> Dict[str, Any]:
        return {
            "cpu_model": "Unknown",
            "memory_gb": 0,
            "gpu_model": "Unknown",
            "gpu_count": 0,
            "gpu_memory_gb": 0,
            "driver_version": "Unknown",
            "cuda_version": "Unknown",
            "accelerator_type": "generic",
        }

    @classmethod
    def _detect_cpu(cls, hw: Dict[str, Any]) -> None:
        try:
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if "model name" in line:
                        hw["cpu_model"] = line.split(":", 1)[1].strip()
                        break
        except Exception:
            pass

    @classmethod
    def _detect_memory(cls, hw: Dict[str, Any]) -> None:
        try:
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if "MemTotal" in line:
                        mem_kb = int(line.split()[1])
                        hw["memory_gb"] = mem_kb // (1024 * 1024)
                        break
        except Exception:
            pass

    @classmethod
    def _get_probe_order(cls, hint: str) -> List[str]:
        hint = hint.lower().strip()
        valid_types = ("nvidia", "moore", "amd", "ascend", "cambricon", "generic")
        probes = [hint] if hint in valid_types else []
        for p in ["nvidia", "moore", "amd", "ascend", "cambricon", "generic"]:
            if p not in probes:
                probes.append(p)
        return probes

    @classmethod
    def _probe_nvidia(cls, hw: Dict[str, Any]) -> bool:
        try:
            r = subprocess.run(
                cls.NVIDIA_SMI_QUERY, capture_output=True, text=True, timeout=5
            )
            if r.returncode != 0 or not r.stdout.strip():
                return False

            lines = [x.strip() for x in r.stdout.strip().splitlines() if x.strip()]
            hw["gpu_count"] = len(lines)

            p = [x.strip() for x in lines[0].split(",")]
            if len(p) >= 3:
                hw["gpu_model"] = p[0]
                hw["driver_version"] = p[2]
                mm = re.search(r"(\d+)\s*MiB", p[1])
                if mm:
                    hw["gpu_memory_gb"] = int(mm.group(1)) // 1024
            return True
        except Exception:
            return False

    @classmethod
    def _probe_mthreads(cls, hw: Dict[str, Any]) -> bool:
        try:
            if not _which("mthreads-gmi"):
                return False
            r = subprocess.run(
                cls.MTHREADS_SMI_QUERY, capture_output=True, text=True, timeout=5
            )
            if r.returncode != 0 or not r.stdout.strip():
                return False

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
                hw["gpu_model"] = gpu_name or "Moore Threads GPU"
            return True
        except Exception:
            return False

    @classmethod
    def _probe_amd(cls, hw: Dict[str, Any]) -> bool:
        try:
            tool = None
            for c in cls.AMD_SMI_CANDIDATES:
                if _which(c):
                    tool = c
                    break
            if not tool:
                return False

            cmd = ["amd-smi", "list"] if tool == "amd-smi" else ["rocm-smi", "-i"]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            if r.returncode != 0 or not r.stdout.strip():
                return False

            lines = [
                x
                for x in r.stdout.splitlines()
                if re.search(r"\bGPU\b", x, re.IGNORECASE)
            ]
            hw["gpu_count"] = (
                max(hw["gpu_count"], len(lines)) if lines else hw["gpu_count"]
            )
            hw["gpu_model"] = (
                hw["gpu_model"] if hw["gpu_model"] != "Unknown" else "AMD GPU"
            )
            return True
        except Exception:
            return False

    @classmethod
    def _probe_ascend(cls, hw: Dict[str, Any]) -> bool:
        try:
            if not _which("npu-smi"):
                return False
            r = subprocess.run(
                ["npu-smi", "info"], capture_output=True, text=True, timeout=5
            )
            if r.returncode != 0 or not r.stdout.strip():
                return False

            cnt = len(
                [
                    x
                    for x in r.stdout.splitlines()
                    if re.search(r"\bNPU\b|\bDevice\b", x)
                ]
            )
            hw["gpu_count"] = max(hw["gpu_count"], cnt) if cnt else hw["gpu_count"]
            hw["gpu_model"] = (
                hw["gpu_model"] if hw["gpu_model"] != "Unknown" else "Ascend NPU"
            )
            return True
        except Exception:
            return False

    @classmethod
    def _probe_cambricon(cls, hw: Dict[str, Any]) -> bool:
        try:
            if not _which("cnmon"):
                return False
            r = subprocess.run(
                ["cnmon", "info"], capture_output=True, text=True, timeout=5
            )
            if r.returncode != 0 or not r.stdout.strip():
                return False

            cnt = len(
                [
                    x
                    for x in r.stdout.splitlines()
                    if re.search(r"\bMLU\b|\bDevice\b", x)
                ]
            )
            hw["gpu_count"] = max(hw["gpu_count"], cnt) if cnt else hw["gpu_count"]
            hw["gpu_model"] = (
                hw["gpu_model"] if hw["gpu_model"] != "Unknown" else "Cambricon MLU"
            )
            return True
        except Exception:
            return False

    @classmethod
    def _get_cuda_version(cls) -> Optional[str]:
        try:
            r = subprocess.run(
                ["nvcc", "--version"], capture_output=True, text=True, timeout=2
            )
            if r.returncode == 0:
                for line in r.stdout.splitlines():
                    if "release" in line:
                        m = re.search(r"release\s+(\d+\.\d+)", line)
                        if m:
                            return m.group(1)
        except Exception:
            pass
        return None

    @classmethod
    def _get_musa_version(cls) -> Optional[str]:
        try:
            r = subprocess.run(
                ["mcc", "--version"], capture_output=True, text=True, timeout=2
            )
            if r.returncode == 0:
                for line in r.stdout.splitlines():
                    m = re.search(r"(\d+\.\d+\.\d+)", line)
                    if m:
                        return m.group(1)
        except Exception:
            pass
        return None
