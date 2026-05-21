#!/usr/bin/env python3
"""Hardware platform implementations for cross-platform bandwidth testing."""

from infinimetrics.hardware.platforms.cuda_platform import CudaPlatform
from infinimetrics.hardware.platforms.cambricon_platform import CambriconPlatform
from infinimetrics.hardware.platforms.ascend_platform import AscendPlatform

PLATFORM_MAP = {
    # CUDA-compatible platforms
    "nvidia": CudaPlatform,
    "metax": CudaPlatform,
    "metax_gpu": CudaPlatform,
    "iluvatar": CudaPlatform,
    "iluvatar_gpu": CudaPlatform,
    "hygon": CudaPlatform,
    "sugon_dcu": CudaPlatform,
    "moore": CudaPlatform,
    "moore_gpu": CudaPlatform,
    "cuda": CudaPlatform,
    # Cambricon
    "cambricon": CambriconPlatform,
    "cambricon_mlu": CambriconPlatform,
    # Ascend
    "ascend": AscendPlatform,
    "ascend_npu": AscendPlatform,
}


def create_platform_handler(platform_name: str, output_dir, config=None):
    """Create the appropriate platform handler based on platform name."""
    platform_key = platform_name.lower().replace(" ", "_")
    platform_cls = PLATFORM_MAP.get(platform_key)
    if platform_cls is None:
        raise ValueError(
            f"Unsupported hardware platform: {platform_name}. "
            f"Supported: {list(PLATFORM_MAP.keys())}"
        )
    return platform_cls(output_dir=output_dir, config=config or {})
