"""Hardware platform aliases and benchmark configuration."""

PLATFORM_ALIASES = {
    "cuda": "cuda",
    "cudaunified": "cuda",
    "nvidia": "cuda",
    "metax": "metax",
    "corex": "corex",
    "iluvatar": "corex",
    "hygon": "hygon",
    "moore": "moore",
    "mthreads": "moore",
    "musa": "moore",
    "cambricon": "cambricon",
    "mlu": "cambricon",
    "ascend": "ascend",
    "npu": "ascend",
}

PLATFORM_CONFIGS = {
    "cuda": {
        "binary_name": "cuda_perf_suite",
        "benchmark_subdir": "cuda-memory-benchmark",
        "build_platform": "cuda",
        "cache_parser": "cuda",
        "compilers": ("nvcc", "/usr/local/cuda/bin/nvcc"),
        "detection_tools": ("nvcc", "nvidia-smi"),
    },
    "metax": {
        "binary_name": "cuda_perf_suite",
        "benchmark_subdir": "cuda-memory-benchmark",
        "build_platform": "metax",
        "cache_parser": "cuda",
        "compilers": (
            "/opt/maca/tools/cu-bridge/bin/cucc",
            "cucc",
            "/opt/maca/mxgpu_llvm/bin/mxcc",
            "mxcc",
        ),
        "detection_tools": (
            "/opt/maca/tools/cu-bridge/bin/cucc",
            "cucc",
            "mxcc",
        ),
    },
    "corex": {
        "binary_name": "cuda_perf_suite",
        "benchmark_subdir": "cuda-memory-benchmark",
        "build_platform": "corex",
        "cache_parser": "cuda",
        "compilers": (
            "/usr/local/corex/bin/clang++",
            "/usr/local/corex/bin/nvcc",
            "nvcc",
        ),
        "detection_tools": (
            "/usr/local/corex/bin/ixsmi",
            "/usr/local/corex/bin/clang++",
        ),
    },
    "hygon": {
        "binary_name": "cuda_perf_suite",
        "benchmark_subdir": "cuda-memory-benchmark",
        "build_platform": "hygon",
        "cache_parser": "cuda",
        "compilers": ("/opt/dtk/bin/hipcc", "hipcc"),
        "detection_tools": ("/opt/dtk/bin/hy-smi", "hy-smi"),
        "conditional_detection_tools": (("/opt/dtk", "hipcc"),),
    },
    "moore": {
        "binary_name": "cuda_perf_suite",
        "benchmark_subdir": "cuda-memory-benchmark",
        "build_platform": "moore",
        "cache_parser": "cuda",
        "compilers": ("/usr/local/musa/bin/mcc", "mcc"),
        "detection_tools": ("mcc", "mthreads-gmi"),
    },
    "cambricon": {
        "binary_name": "mlu_perf_suite",
        "benchmark_subdir": "cambricon-memory-benchmark",
        "build_platform": None,
        "cache_parser": "cambricon",
        "detection_tools": ("cncc",),
        "detection_paths": ("/usr/local/neuware",),
    },
    "ascend": {
        "binary_name": "npu_perf_suite",
        "benchmark_subdir": "ascend-memory-benchmark",
        "build_platform": None,
        "cache_parser": "ascend",
        "detection_tools": ("npu-smi", "atc"),
        "detection_paths": ("/usr/local/Ascend/ascend-toolkit",),
    },
}

PLATFORM_DETECTION_ORDER = (
    "ascend",
    "cambricon",
    "moore",
    "metax",
    "hygon",
    "corex",
)
