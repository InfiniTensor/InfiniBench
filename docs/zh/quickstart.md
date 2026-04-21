# 快速开始指南

本指南将帮助您在几分钟内开始使用 InfiniBench。

## 前提条件

在开始之前，请确保您具备：
- **Python 3.8+** 已安装
- **Git** 已安装（用于克隆仓库）
- **GPU**（可选，用于硬件测试）

## 安装

### 1. 克隆仓库

```bash
git clone https://github.com/InfiniTensor/InfiniBench.git
cd InfiniBench
```

### 2. 初始化子模块

```bash
git submodule update --init --recursive
```

### 3. 安装依赖

```bash
# 核心依赖
pip install numpy torch pandas

# 可选：支持 vLLM
pip install vllm
```

### 4. 构建硬件基准测试（可选）

如果您计划运行硬件测试：

```bash
cd infinibench/hardware/cuda-memory-benchmark
bash build.sh
```

## 运行第一个测试

### 硬件测试

使用单个命令运行全面的硬件基准测试：

```bash
python main.py format_input_comprehensive_hardware.json
```

这将测试：
- 内存带宽（H2D、D2H、D2D, bidirectional）
- STREAM 基准测试
- GPU 缓存性能

### 推理测试

运行推理基准测试：

```bash
cd infinibench/inference
python infer_main.py --config config.json
```

## 理解输出

### 测试结果位置

结果保存在：
```
./summary_output/
└── dispatcher_summary_YYYYMMDD_HHMMSS.json    # 总体摘要
./output/
└── <test_case_name>/
    ├── seperated_test_result.json              # 分测试结果
    └── metrics.csv                             # 时间序列指标
```

### 示例：硬件测试输出

```json
{
  "total_tests": 1,
  "successful_tests": 1,
  "failed_tests": 0,
  "results": [{
    "run_id": "test_001",
    "testcase": "hardware.cudaUnified.Comprehensive",
    "result_code": 0,
    "result_file": "./output/hardware.cudaUnified.Comprehensive/metrics.json"
  }]
}
```

## 后续步骤

- **配置测试**：请参阅[配置指南](./configuration.md)进行自定义
