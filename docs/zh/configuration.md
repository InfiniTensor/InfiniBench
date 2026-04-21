# 配置指南

本指南解释如何在 InfiniBench 中配置和运行测试。

## 输入文件格式

测试规格以 JSON 格式提供。典型的配置文件如下所示：

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
        {"name": "hardware.mem_sweep_h2d"},
        {"name": "hardware.stream_triad"}
    ]
}
```

## 配置参数(实例)

| 参数 | 类型 | 描述 | 默认值 |
|------|------|------|--------|
| `run_id` | string | 唯一的测试运行标识符 | 必填 |
| `testcase` | string | 测试类型标识符 | 必填 |
| `config.device` | string | 加速器类型 (nvidia/amd/huawei/cambricon) | nvidia |
| `config.array_size` | int | STREAM 测试的数组大小 | 67108864 |
| `config.output_dir` | string | 输出目录路径 | ./output |

## 测试用例命名约定

格式：`<类别>.<框架>.<测试名称>`

### 类别
- `hardware` - 硬件级测试
- `operator` - 算子级测试
- `infer` - 推理测试
- `comm` - 通信测试

### 框架
- `cudaUnified` - CUDA 统一内存测试
- `infinicore` - InfiniCore 算子测试
- `infinilm` - InfiniLM 推理测试
- `vllm` - vLLM 推理测试
- `nccltest` - NCCL 通信测试

### 示例
- `hardware.cudaUnified.Comprehensive`
- `operator.infinicore.Matmul`
- `infer.infinilm.direct`
- `comm.nccltest.AllReduce`

## 运行测试

### 单个测试

```bash
python main.py input.json
```

### 多个测试

```bash
# 运行目录中的所有 JSON 配置
python main.py ./test_configs/
```

### 详细输出

```bash
python main.py input.json --verbose
```

## 配置示例

### 硬件基准测试

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
        {"name": "hardware.mem_bw_h2d"},
        {"name": "hardware.stream_triad"}
    ]
}
```

### 推理测试

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
