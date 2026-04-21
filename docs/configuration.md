# Configuration Guide

This guide explains how to configure and run tests in InfiniBench.

## Input File Format

Test specifications are provided in JSON format. A typical configuration file looks like this:

```json
{
    "run_id": "unique_run_identifier",
    "testcase": "hardware.cudaUnified.Comprehensive",
    "config": {
        "device": "nvidia",
        "array_size": 67108864,
        "output_dir": "./output"
    },
    "metrics": [
        {"name": "hardware.mem_bw_h2d"},
        {"name": "hardware.mem_bw_d2h"},
        {"name": "hardware.stream_triad"}
    ]
}
```

## Configuration Parameters (Just some examples)

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `run_id` | string | Unique test run identifier | Required |
| `testcase` | string | Test type identifier | Required |
| `config.device` | string | Accelerator type (nvidia/amd/huawei/cambricon) | nvidia |
| `config.array_size` | int | Array size for STREAM tests | 67108864 |
| `config.output_dir` | string | Output directory path | ./output |

## Test Case Naming Convention

Format: `<category>.<framework>.<test_name>`

### Categories
- `hardware` - Hardware-level tests
- `operator` - Operator-level tests
- `infer` - Inference tests
- `comm` - Communication tests

### Frameworks
- `cudaUnified` - CUDA unified memory tests
- `infinicore` - InfiniCore operator tests
- `infinilm` - InfiniLM inference tests
- `vllm` - vLLM inference tests
- `nccltest` - NCCL communication tests

### Examples
- `hardware.cudaUnified.Comprehensive`
- `operator.infinicore.Matmul`
- `infer.infinilm.direct`
- `comm.nccltest.AllReduce`

## Running Tests

### Single Test

```bash
python main.py input.json
```

### Multiple Tests

```bash
# Run all JSON configs in a directory
python main.py ./test_configs/
```

### Verbose Output

```bash
python main.py input.json --verbose
```

### Custom Output Directory

```bash
python main.py input.json --output ./results
```

## Example Configurations

### Hardware Benchmark

```json
{
    "run_id": "hw_test_001",
    "testcase": "hardware.cudaUnified.Comprehensive",
    "config": {
        "device": "nvidia",
        "array_size": 67108864,
        "output_dir": "./output"
    },
    "metrics": [
        {"name": "hardware.mem_sweep_h2d"},
        {"name": "hardware.stream_triad"}
    ]
}
```

### Inference Test

```json
{
    "run_id": "infer_test_001",
    "testcase": "infer.infinilm.direct",
    "config": {
        "model_path": "/path/to/model",
        "batch_size": 32,
        "output_dir": "./output"
    },
    "metrics": [
        {"name": "infer.throughput"},
        {"name": "infer.latency"}
    ]
}
```
