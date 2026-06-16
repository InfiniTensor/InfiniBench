#!/usr/bin/env python3
"""
platform_config.py — Multi-platform accelerator configuration registry.

Provides:
  - AcceleratorType enum for all supported platforms
  - Platform registry mapping each accelerator to:
      * device visible env var (e.g. CUDA_VISIBLE_DEVICES)
      * SMI tool command for card count detection
      * display name

Usage:
    from platform_config import AcceleratorType, get_platform_info

    info = get_platform_info(AcceleratorType.CAMBRICON)
    print(info.env_var)          # "MLU_VISIBLE_DEVICES"
    print(info.get_card_count()) # 8
"""

import re
import subprocess
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class AcceleratorType(Enum):
    """All supported accelerator platforms."""
    CPU = "cpu"
    NVIDIA = "nvidia"
    CAMBRICON = "cambricon"
    ASCEND = "ascend"
    METAX = "metax"
    MOORE = "moore"
    ILUVATAR = "iluvatar"
    KUNLUN = "kunlun"
    HYGON = "hygon"
    QY = "qy"
    ALI = "ali"


@dataclass
class PlatformInfo:
    """Platform-specific configuration for one accelerator type."""
    name: str                        # InfiniCore device name (uppercase)
    env_var: str                     # Visible devices env var
    smi_commands: list[str]          # SMI tool commands to try (in order)
    smi_count_pattern: str           # Regex pattern to count cards in SMI output

    def get_card_count(self) -> int:
        """Detect number of available cards via SMI tool.

        Returns:
            Number of cards, or 0 if detection fails.
        """
        for cmd in self.smi_commands:
            try:
                result = subprocess.run(
                    cmd, shell=True,
                    capture_output=True, text=True, timeout=10,
                )
                if result.returncode == 0 and result.stdout.strip():
                    matches = re.findall(
                        self.smi_count_pattern, result.stdout, re.IGNORECASE,
                    )
                    if matches:
                        return len(matches)
            except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
                continue
        return 0


# ---------------------------------------------------------------------------
# Platform Registry
# ---------------------------------------------------------------------------

PLATFORM_REGISTRY: dict[AcceleratorType, PlatformInfo] = {
    AcceleratorType.NVIDIA: PlatformInfo(
        name="NVIDIA",
        env_var="CUDA_VISIBLE_DEVICES",
        smi_commands=["nvidia-smi --query-gpu=name --format=csv,noheader"],
        smi_count_pattern=r".+",   # each line is one GPU
    ),
    AcceleratorType.CAMBRICON: PlatformInfo(
        name="CAMBRICON",
        env_var="MLU_VISIBLE_DEVICES",
        smi_commands=["cnmon info"],
        smi_count_pattern=r"\bMLU\b|\bDevice\b",
    ),
    AcceleratorType.ASCEND: PlatformInfo(
        name="ASCEND",
        env_var="ASCEND_RT_VISIBLE_DEVICES",
        smi_commands=["npu-smi info"],
        smi_count_pattern=r"\bNPU\b|\bDevice\b",
    ),
    AcceleratorType.METAX: PlatformInfo(
        name="METAX",
        env_var="CUDA_VISIBLE_DEVICES",    # Metax uses CUDA-compatible runtime
        smi_commands=["mx-smi"],
        smi_count_pattern=r"\bGPU\b|\bDevice\b",
    ),
    AcceleratorType.MOORE: PlatformInfo(
        name="MOORE",
        env_var="MUSA_VISIBLE_DEVICES",
        smi_commands=["mthreads-gmi"],
        smi_count_pattern=r"\bGPU\b|\bDevice\b",
    ),
    AcceleratorType.ILUVATAR: PlatformInfo(
        name="ILUVATAR",
        env_var="IX_CUDA_VISIBLE_DEVICES",
        smi_commands=["ixsmi"],
        smi_count_pattern=r"\bGPU\b|\bDevice\b",
    ),
    AcceleratorType.KUNLUN: PlatformInfo(
        name="KUNLUN",
        env_var="XPU_VISIBLE_DEVICES",
        smi_commands=["xpu-smi info"],
        smi_count_pattern=r"\bXPU\b|\bDevice\b",
    ),
    AcceleratorType.HYGON: PlatformInfo(
        name="HYGON",
        env_var="HIP_VISIBLE_DEVICES",
        smi_commands=["hygon-smi", "rocm-smi -i"],
        smi_count_pattern=r"\bGPU\b|\bDCU\b|\bDevice\b",
    ),
    AcceleratorType.QY: PlatformInfo(
        name="QY",
        env_var="CUDA_VISIBLE_DEVICES",    # QY uses CUDA-compatible runtime
        smi_commands=["qy-smi"],
        smi_count_pattern=r"\bGPU\b|\bDevice\b",
    ),
    AcceleratorType.ALI: PlatformInfo(
        name="ALI",
        env_var="ALI_PPU_VISIBLE_DEVICES",
        smi_commands=["ppu-smi"],
        smi_count_pattern=r"\bPPU\b|\bDevice\b",
    ),
}


def get_platform_info(accel_type: AcceleratorType) -> PlatformInfo:
    """Get platform info for an accelerator type.

    Args:
        accel_type: The accelerator type enum value.

    Returns:
        PlatformInfo for the given type.

    Raises:
        ValueError: If the accelerator type is not in the registry.
    """
    if accel_type in PLATFORM_REGISTRY:
        return PLATFORM_REGISTRY[accel_type]

    raise ValueError(
        f"Unsupported accelerator: {accel_type}. "
        f"Supported: {[t.value for t in PLATFORM_REGISTRY]}"
    )


def detect_platform() -> Optional[AcceleratorType]:
    """Auto-detect the available accelerator platform.

    Tries each platform's SMI tool in a reasonable order.
    Returns the first platform that reports >= 1 card.

    Returns:
        Detected AcceleratorType, or None if no accelerator found.
    """
    # Detection order: most common first
    detection_order = [
        AcceleratorType.NVIDIA,
        AcceleratorType.CAMBRICON,
        AcceleratorType.ASCEND,
        AcceleratorType.METAX,
        AcceleratorType.MOORE,
        AcceleratorType.ILUVATAR,
        AcceleratorType.KUNLUN,
        AcceleratorType.HYGON,
        AcceleratorType.QY,
        AcceleratorType.ALI,
    ]

    for accel_type in detection_order:
        info = PLATFORM_REGISTRY[accel_type]
        count = info.get_card_count()
        if count > 0:
            logger.info(f"Auto-detected: {accel_type.value} ({count} cards)")
            return accel_type

    return None


def resolve_accelerator_type(device_str: str) -> AcceleratorType:
    """Resolve a device string (case-insensitive) to AcceleratorType.

    Args:
        device_str: Device name string, e.g. "nvidia", "NVIDIA", "cambricon"

    Returns:
        Matching AcceleratorType enum value.

    Raises:
        ValueError: If no match found.
    """
    normalized = device_str.strip().lower()
    for accel_type in AcceleratorType:
        if accel_type.value == normalized:
            return accel_type
    raise ValueError(
        f"Unknown device '{device_str}'. "
        f"Supported: {[t.value for t in AcceleratorType if t != AcceleratorType.CPU]}"
    )


# ---------------------------------------------------------------------------
# CLI helper — used by run_multi_gpu.sh via: python platform_config.py <command> <device>
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python platform_config.py <command> [args...]")
        print("Commands:")
        print("  env-var <device>       Print the visible-devices env var name")
        print("  card-count <device>    Print the number of detected cards")
        print("  detect                 Auto-detect platform and print device name")
        print("  list                   List all supported platforms")
        sys.exit(1)

    command = sys.argv[1]

    if command == "list":
        for accel_type in AcceleratorType:
            if accel_type == AcceleratorType.CPU:
                continue
            info = PLATFORM_REGISTRY.get(accel_type)
            if info:
                print(f"  {accel_type.value:<12} env_var={info.env_var}")
        sys.exit(0)

    if command == "detect":
        detected = detect_platform()
        if detected:
            print(detected.value)
        else:
            print("none")
            sys.exit(1)

    elif command in ("env-var", "card-count"):
        if len(sys.argv) < 3:
            print(f"Usage: python platform_config.py {command} <device>", file=sys.stderr)
            sys.exit(1)

        device_str = sys.argv[2]
        try:
            accel_type = resolve_accelerator_type(device_str)
            info = get_platform_info(accel_type)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

        if command == "env-var":
            print(info.env_var)
        else:
            count = info.get_card_count()
            print(count)

    else:
        print(f"Unknown command: {command}", file=sys.stderr)
        sys.exit(1)
