# 项目架构

本文档描述 InfiniBench 框架的架构和组织。

## 目录结构

```
InfiniBench/
├── main.py                          # 主入口文件
├── infinibench/
│   ├── adapter.py                  # 基础适配器接口
│   ├── dispatcher.py               # 测试编排调度器
│   ├── executor.py                 # 通用测试执行器
│   ├── input.py                    # 测试输入数据类
│   │
│   ├── common/                     # 共享工具
│   │   ├── constants.py            # 测试类型、指标、枚举
│   │   ├── metrics.py              # 指标定义
│   │   └── testcase_utils.py      # 测试用例工具
│   │
│   ├── hardware/                   # 硬件测试模块
│   │   └── cuda-memory-benchmark/  # CUDA 内存基准测试套件
│   │       ├── include/            # C++ 头文件
│   │       ├── src/                # CUDA/C++ 源代码
│   │       ├── CMakeLists.txt     # 构建配置
│   │       ├── build.sh            # 构建脚本
│   │       └── QUICKSTART.md       # 硬件测试快速开始指南
│   │
│   ├── operators/                  # 算子级测试
│   │   ├── infinicore_adapter.py   # InfiniCore 操作
│   │   └── flops_calculator.py    # FLOPS 计算器
│   │
│   ├── inference/                  # 推理评估
│   │   ├── adapters/
│   │   │   ├── infinilm_adapter.py # InfiniLM 适配器
│   │   │   └── vllm_adapter.py     # vLLM 适配器
│   │   ├── infer_main.py           # 推理入口点
│   │   └── utils/                  # 推理工具
│   │
│   ├── communication/              # 通信测试
│   │   └── nccl_adapter.py         # NCCL 适配器
│   │
│   └── utils/                      # 工具函数
│       └── input_loader.py        # 输入文件加载器
│       └── ...
|
├── submodules/
│   └── nccl-tests/                 # NCCL 测试套件（git 子模块）
│
├── format_input_comprehensive_hardware.json  # 示例配置
└── summary_output/                # 测试结果目录
```

## 架构图

```mermaid
flowchart TB
    subgraph InfiniBench["InfiniBench"]
        Input[输入文件] --> Dispatcher[调度器]
        Dispatcher --> Executor[执行器]
        Executor --> AdapterRegistry["适配器注册表"]

        subgraph Adapters["适配器注册表"]
            Hardware["硬件测试"]
            Operator["算子测试"]
            Inference["推理测试"]
        end

        AdapterRegistry --> HWAdapter["硬件适配器"]
        AdapterRegistry --> OpAdapter["算子适配器"]
        AdapterRegistry --> InfAdapter["推理适配器"]

        subgraph Concrete["具体适配器"]
            HWAdapter["硬件测试<br/>适配器"]
            OpAdapter["InfiniCore<br/>适配器"]
            InfAdapter["Inference<br/>适配器"]
        end

        Concrete --> Metrics["指标系统"]

        Metrics --> Output[(输出目录)]

        style Hardware fill:#e1f5ff
        style Operator fill:#f3f9ff
        style Inference fill:#fafcff
        style HWAdapter fill:#bbdefb
        style OpAdapter fill:#c3e6f5
        style InfAdapter fill:#cce5ff
        style Metrics fill:#e3f2fd
        style Input fill:#fff3e0
        style Dispatcher fill:#ffe0b2
        style Executor fill:#ffccbc
        style AdapterRegistry fill:#ffecb3
        style Output fill:#ffe0b2
    end
```

## 组件概述

### 主入口
- **main.py**: 从命令行运行测试的入口点

### 核心组件
- **adapter.py**: 所有测试适配器必须实现的基础适配器接口
- **dispatcher.py**: 编排测试执行并管理适配器注册表
- **executor.py**: 通过适配器运行测试的通用测试执行器

### 测试类别
- **hardware/**: 硬件级基准测试（内存、缓存、计算）
- **operators/**: 算子级性能测试
- **inference/**: 端到端推理评估
- **communication/**: 加速卡通信测试

### 公共工具
- **common/constants.py**: 测试类型、指标和枚举
- **common/metrics.py**: 指标定义和数据结构
- **common/testcase_utils.py**: 测试用例解析和验证

## 数据流

1. **输入**: JSON 配置文件指定测试参数
2. **调度**: 调度器解析输入并选择适当的适配器
3. **执行**: 执行器通过适配器运行测试
4. **收集**: 指标被收集和聚合
5. **输出**: 结果保存到输出目录

有关配置的更多详情，请参阅[配置指南](./configuration.md)。
