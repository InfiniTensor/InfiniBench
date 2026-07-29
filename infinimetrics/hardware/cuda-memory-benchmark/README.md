# GPU Performance Benchmark Suite

A comprehensive GPU performance testing suite for memory bandwidth, STREAM benchmark, and cache analysis. Supports NVIDIA CUDA and domestic GPU platforms via a unified build system.

## Supported Platforms

| Platform | Flag | Compiler | Notes |
|----------|------|----------|-------|
| NVIDIA GPU | `--platform cuda` | nvcc | Native CUDA |
| MetaX | `--platform metax` | cucc (cu-bridge) | CUDA-compatible via cu-bridge |
| Iluvatar CoreX | `--platform corex` | nvcc (CoreX SDK) | CUDA-compatible via CoreX |
| Hygon DCU | `--platform hygon` | hipcc (DTK) | HIP backend |
| Moore Threads | `--platform moore` | mcc (MUSA SDK) | MUSA backend via `-mtgpu` |

## Features

- **Memory Bandwidth Tests**: Host-to-Device, Device-to-Host, Device-to-Device transfers
- **STREAM Benchmark**: Standard memory bandwidth benchmark (Copy, Scale, Add, Triad operations)
- **Cache Performance Tests**: L1 and L2 cache bandwidth analysis
- **Multi-Platform**: Unified source code, per-platform build via `--platform` flag
- **Modern C++ Design**: RAII patterns, smart pointers, exception safety

## Requirements

Common:
- CMake 3.18 or higher
- C++17 compatible compiler

Platform-specific:
- **NVIDIA / MetaX / CoreX**: CUDA Toolkit 11.0+
- **Hygon**: DTK (HIP)
- **Moore Threads**: MUSA SDK (mcc)

## Building

```bash
cd cuda-memory-benchmark

# NVIDIA
bash build.sh --platform cuda

# MetaX
bash build.sh --platform metax

# Iluvatar CoreX
bash build.sh --platform corex

# Hygon DCU
bash build.sh --platform hygon

# Moore Threads
bash build.sh --platform moore
```

All platforms produce the same output binary: `build/cuda_perf_suite`

### Manual Build (NVIDIA CUDA only)

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### Adjusting CUDA Architecture

Edit `CMakeLists.txt` to specify your GPU architecture:

```cmake
set(CMAKE_CUDA_ARCHITECTURES "80;86;89;90" CACHE STRING "CUDA architectures")
```

Common architectures:
- 80: A100, RTX 3090
- 86: RTX 3080, A30
- 89: RTX 4080, 4090
- 90: H100

## Usage

All platforms share the same CLI interface:

```bash
# Run all tests
./build/cuda_perf_suite --all

# Memory bandwidth tests only
./build/cuda_perf_suite --memory

# STREAM benchmark only
./build/cuda_perf_suite --stream

# Cache performance tests only
./build/cuda_perf_suite --cache

# Specify GPU device
./build/cuda_perf_suite --all --device 1

# More iterations
./build/cuda_perf_suite --all --iterations 20

# Quiet mode
./build/cuda_perf_suite --all --quiet
```

### Environment Variables

Some platforms use environment variables to select GPU devices:

| Platform | Environment Variable | Example |
|----------|---------------------|---------|
| NVIDIA | `CUDA_VISIBLE_DEVICES` | `CUDA_VISIBLE_DEVICES=0 ./build/cuda_perf_suite --all` |
| Moore Threads | `MUSA_VISIBLE_DEVICES` | `MUSA_VISIBLE_DEVICES=1 ./build/cuda_perf_suite --all` |
| Hygon | `HIP_VISIBLE_DEVICES` | `HIP_VISIBLE_DEVICES=0 ./build/cuda_perf_suite --all` |
| MetaX | `MACA_VISIBLE_DEVICES` | `MACA_VISIBLE_DEVICES=0 ./build/cuda_perf_suite --all` |

### Command Line Options

```
Options:
  --all                    Run all tests (default)
  --memory                 Run memory bandwidth tests only
  --stream                 Run STREAM benchmark only
  --cache                  Run cache benchmarks only
  --device <id>            Specify GPU device ID (default: 0)
  --iterations <n>         Number of measurement iterations (default: 10)
  --array-size <size>      Array size for STREAM test (default: 67108864)
  --quiet                  Reduce output verbosity
  --help                   Show help message
```

## Test Descriptions

### 1. Memory Bandwidth Tests

Tests data transfer bandwidth between different memory spaces:

- **Host to Device (Pinned)**: Measures PCIe bandwidth using page-locked memory
- **Device to Host (Pinned)**: Measures PCIe read bandwidth
- **Device to Device**: Measures GPU internal memory bandwidth

**Output**: Transfer time and bandwidth for various buffer sizes (64KB to 1GB)

### 2. STREAM Benchmark

Standard STREAM benchmark with 4 operations:

- **Copy**: `a[i] = b[i]` - 2 bytes per element
- **Scale**: `a[i] = scalar * b[i]` - 2 bytes per element
- **Add**: `a[i] = b[i] + c[i]` - 3 bytes per element
- **Triad**: `a[i] = b[i] + scalar * c[i]` - 3 bytes per element

**Output**: Bandwidth in GB/s for each operation

### 3. Cache Performance Tests

Tests L1 and L2 cache bandwidth by varying working set sizes

**Output**: Bandwidth vs working set size, revealing cache hierarchy characteristics

## Understanding Results

### Bandwidth Metrics

- **Average Time**: Mean execution time
- **Trimmed Mean**: Average excluding min/max values (more robust)
- **Coefficient of Variation (CV)**: Relative standard deviation (lower is better)

## Project Structure

```
cuda-memory-benchmark/
├── include/               # Header files
│   ├── gpu_runtime.h                 # Cross-platform GPU API abstraction
│   ├── cuda_utils.h                 # CUDA utilities (RAII wrappers)
│   ├── performance_test.h           # Base testing framework
│   ├── memory_bandwidth_test.h      # Memory copy tests
│   ├── stream_benchmark.h           # STREAM benchmark
│   └── cache_benchmark.h            # Cache tests
├── src/
│   └── main.cu                      # Main program entry
├── CMakeLists.txt                   # CMake build configuration
├── build.sh                         # Unified build script (all platforms)
├── README.md                        # This file
└── QUICKSTART.md                    # Quick start guide
```
