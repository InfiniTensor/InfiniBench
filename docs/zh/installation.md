# 安装指南

本指南涵盖 InfiniBench 的安装和设置。

## 前置要求

- **Python**: 3.8 或更高版本
- **编译器**: GCC 11.3
- **CMake**: 3.20+

## 依赖安装

### 核心 Python 依赖

根据需要安装核心依赖：

```bash
# 用于 InfiniLM 适配器
pip install numpy torch

# 用于 vLLM 适配器
pip install vllm

# 用于数据处理
pip install pandas
```

### 构建硬件基准测试（可选）

如果计划使用硬件测试模块，需要构建 CUDA 内存基准测试套件：

```bash
cd infinibench/hardware/cuda-memory-benchmark
bash build.sh
```

**注意**: 这需要：
- CUDA 工具包（与 GPU 驱动兼容）
- 支持 CUDA 的 C++ 编译器
- CMake 3.20 或更高版本

## 验证安装

安装后，验证您的设置：

```bash
# 运行简单的硬件测试
python main.py format_input_comprehensive_hardware.json
```

## 平台特定说明

### NVIDIA GPU
确保 CUDA_HOME 已设置：
```bash
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

### 其他加速器
请参阅加速器供应商的文档以了解驱动和工具包安装。
