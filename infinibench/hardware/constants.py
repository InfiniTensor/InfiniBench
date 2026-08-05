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
    "ascend": "ascend",
    "npu": "ascend",
}

PLATFORM_CONFIGS = {
    "cuda": {
        "binary_name": "cuda_perf_suite",
        "benchmark_subdir": "cuda-memory-benchmark",
        "build_platform": "cuda",
        "cache_parser": "cuda",
    },
    "metax": {
        "binary_name": "cuda_perf_suite",
        "benchmark_subdir": "cuda-memory-benchmark",
        "build_platform": "metax",
        "cache_parser": "cuda",
    },
    "corex": {
        "binary_name": "cuda_perf_suite",
        "benchmark_subdir": "cuda-memory-benchmark",
        "build_platform": "corex",
        "cache_parser": "cuda",
    },
    "hygon": {
        "binary_name": "cuda_perf_suite",
        "benchmark_subdir": "cuda-memory-benchmark",
        "build_platform": "hygon",
        "cache_parser": "cuda",
    },
    "moore": {
        "binary_name": "cuda_perf_suite",
        "benchmark_subdir": "cuda-memory-benchmark",
        "build_platform": "moore",
        "cache_parser": "cuda",
    },
    "ascend": {
        "binary_name": "npu_perf_suite",
        "benchmark_subdir": "ascend-memory-benchmark",
        "build_platform": None,
        "cache_parser": "ascend",
    },
}
