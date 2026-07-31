# Installation Guide

This guide covers the installation and setup of InfiniMetrics.

## Prerequisites

- **Python**: 3.8 or higher
- **Compiler**: GCC 11.3
- **CMake**: 3.20+

## Dependencies Installation

### Core Python Dependencies

Install the core dependencies based on your needs:

```bash
# For InfiniLM adapter
pip install numpy torch

# For vLLM adapter
pip install vllm

# For data processing
pip install pandas
```

### Build Hardware Benchmarks (Optional)

If you plan to use the hardware testing modules, you need to build the CUDA memory benchmark suite:

```bash
cd infinimetrics/hardware/cuda-memory-benchmark
bash build.sh --platform cuda
```

For MetaX, Iluvatar, Hygon, and Moore Threads build and runtime instructions,
see [Hardware Benchmarks](../infinimetrics/hardware/README.md).

**Note**: This requires:
- CUDA toolkit (compatible with your GPU driver)
- C++ compiler with CUDA support
- CMake 3.20 or higher

## Verification

After installation, verify your setup:

```bash
# Run a simple hardware test
python main.py format_input_comprehensive_hardware.json
```

## Platform-Specific Notes

### NVIDIA GPUs
Ensure CUDA_HOME is set:
```bash
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

### Other Accelerators
Refer to your accelerator vendor's documentation for driver and toolkit installation.
