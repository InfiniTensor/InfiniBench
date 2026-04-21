# Quick Start Guide

## 1. Build

```bash
cd cuda-memory-benchmark

# NVIDIA GPU
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

## 2. Run All Tests

```bash
# NVIDIA
./build/cuda_perf_suite --all

# Moore Threads (specify GPU device)
MUSA_VISIBLE_DEVICES=0 ./build/cuda_perf_suite --all

# MetaX
MACA_VISIBLE_DEVICES=0 ./build/cuda_perf_suite --all

# Hygon
HIP_VISIBLE_DEVICES=0 ./build/cuda_perf_suite --all
```

## 3. Run Individual Test Suites

### Memory Bandwidth Tests
```bash
./build/cuda_perf_suite --memory
```
Tests Host↔Device and Device↔Device transfer bandwidths with multiple buffer sizes.

### STREAM Benchmark
```bash
./build/cuda_perf_suite --stream
```
Standard STREAM benchmark measuring sustainable memory bandwidth.

### Cache Tests
```bash
./build/cuda_perf_suite --cache
```
Tests L1 and L2 cache performance with varying working set sizes.

## 4. Common Usage Patterns

### Quick Performance Check
```bash
./build/cuda_perf_suite --all
```

### Detailed STREAM Benchmark
```bash
./build/cuda_perf_suite --stream --iterations 50
```

### Test Specific GPU
```bash
./build/cuda_perf_suite --all --device 1
```

### Quiet Mode
```bash
./build/cuda_perf_suite --all --quiet
```

## 5. Understanding Output

The tests report:
- **Time (ms)**: Average execution time
- **Bandwidth (GB/s)**: Data transfer rate
- **CV (%)**: Coefficient of Variation (consistency measure)

Lower CV = more consistent results.

## 6. Next Steps

- Read [README.md](README.md) for detailed documentation and platform notes
- Adjust test parameters for your specific use case
- Integrate into your performance testing workflow
