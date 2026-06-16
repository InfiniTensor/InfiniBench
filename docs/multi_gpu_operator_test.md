# 多卡算子测试工具

多卡并行测试工具，用于在多张 GPU 上同时运行相同的算子 benchmark，聚合计算总吞吐量，衡量多卡总算力。

## 文件说明

```
scripts/
├── run_multi_gpu.sh         # 多卡并行启动脚本（主入口）
├── aggregate_multi_gpu.py   # 结果聚合脚本（自动调用）
└── platform_config.py       # 多平台配置注册表（设备→环境变量→SMI命令）
```

## 快速开始

```bash
# 在 2,3 号卡上跑 8192 matmul
bash scripts/run_multi_gpu.sh \
    test_inputs/configs/opbench.matmul.large.8192x8192.float32.json \
    --gpu-ids 2,3

# 在全部 8 张卡上跑
bash scripts/run_multi_gpu.sh \
    test_inputs/configs/opbench.matmul.large.8192x8192.float32.json

# 指定 4 张卡
bash scripts/run_multi_gpu.sh \
    test_inputs/configs/opbench.matmul.large.8192x8192.float32.json \
    --gpu-count 4
```

## 命令参数

```
bash scripts/run_multi_gpu.sh <test_config.json> [options]
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `<test_config.json>` | 测试配置 JSON 文件（必填） | — |
| `--device <name>` | 加速器平台，见下表 | 自动检测 |
| `--gpu-ids <ids>` | 指定 GPU 编号，逗号分隔 | 使用全部卡 |
| `--gpu-count <N>` | 使用前 N 张卡 | 使用全部卡 |
| `--output-dir <dir>` | 输出目录 | `./output` |
| `-h, --help` | 显示帮助 | — |

> `--gpu-ids` 优先级高于 `--gpu-count`。

## 支持的硬件平台

通过 `--device` 指定，不指定时自动检测：

| 平台 | `--device` 值 | 环境变量 | SMI 工具 |
|------|--------------|---------|----------|
| NVIDIA | `nvidia` | `CUDA_VISIBLE_DEVICES` | `nvidia-smi` |
| 寒武纪 | `cambricon` | `MLU_VISIBLE_DEVICES` | `cnmon` |
| 昇腾 | `ascend` | `ASCEND_RT_VISIBLE_DEVICES` | `npu-smi` |
| 沐曦 | `metax` | `CUDA_VISIBLE_DEVICES` | `mx-smi` |
| 摩尔 | `moore` | `MUSA_VISIBLE_DEVICES` | `mthreads-gmi` |
| 天数 | `iluvatar` | `IX_CUDA_VISIBLE_DEVICES` | `ixsmi` |
| 昆仑芯 | `kunlun` | `XPU_VISIBLE_DEVICES` | `xpu-smi` |
| 海光 | `hygon` | `HIP_VISIBLE_DEVICES` | `hygon-smi` |
| 趋域 | `qy` | `CUDA_VISIBLE_DEVICES` | `qy-smi` |
| 阿里 PPU | `ali` | `ALI_PPU_VISIBLE_DEVICES` | `ppu-smi` |

也可以通过 `platform_config.py` 单独查询：

```bash
# 列出所有支持的平台
python scripts/platform_config.py list

# 查询某平台的环境变量名
python scripts/platform_config.py env-var cambricon
# 输出: MLU_VISIBLE_DEVICES

# 检测某平台的卡数
python scripts/platform_config.py card-count nvidia
# 输出: 8

# 自动检测当前平台
python scripts/platform_config.py detect
# 输出: nvidia
```

## 各平台示例

```bash
# NVIDIA — 指定 4 张卡
bash scripts/run_multi_gpu.sh test.json --device nvidia --gpu-ids 0,1,2,3

# 寒武纪 — 使用全部卡
bash scripts/run_multi_gpu.sh test.json --device cambricon

# 昇腾 — 使用前 4 张卡
bash scripts/run_multi_gpu.sh test.json --device ascend --gpu-count 4

# 沐曦 — 指定卡
bash scripts/run_multi_gpu.sh test.json --device metax --gpu-ids 2,3,4,5

# 海光 — 全卡
bash scripts/run_multi_gpu.sh test.json --device hygon
```

> **注意**：在其他平台上跑时，测试 JSON 里的 `"device"` 字段也要对应修改，例如寒武纪需改为 `"device": "cambricon"`，否则 InfiniCore 仍走 NVIDIA 后端。

## 输出说明

### 目录结构

每次运行创建一个带时间戳的输出目录：

```
output/multi_gpu_20260605_082711/
├── gpu_2/
│   ├── run.log                              # 该卡运行日志
│   └── operator/
│       └── opbench.matmul.large.8192x8192.float32_results.json
├── gpu_3/
│   ├── run.log
│   └── operator/
│       └── opbench.matmul.large.8192x8192.float32_results.json
├── gpu_5/
│   └── ...
├── gpu_7/
│   └── ...
└── aggregated_results.json                  # 聚合结果
```

### 终端输出

```
==============================================================================
  Multi-GPU Operator Test Summary  (4 GPUs)
==============================================================================
  GPU   Latency(ms)        TFLOPS      BW(GB/s)    Accuracy
------------------------------------------------------------------------------
GPU 2      8.287140      132.6768       97.1754        PASS
GPU 3      8.469197      129.8248       95.0865        PASS
GPU 5      8.265216      133.0288       97.4332        PASS
GPU 7      8.289915      132.6324       97.1429        PASS
------------------------------------------------------------------------------
TOTAL      8.327867      528.1628      386.8380        PASS
==============================================================================
```

### 聚合结果 JSON（`aggregated_results.json`）

```json
{
  "type": "multi_gpu_aggregation",
  "gpu_count": 4,
  "testcase": "operator.InfiniCore.Matmul",
  "timestamp": "2026-06-05 08:27:15",
  "per_gpu": [
    {
      "gpu_id": 2,
      "latency_ms": 8.28714,
      "tflops": 132.6768,
      "bandwidth_gbs": 97.1754,
      "accuracy": "PASS"
    }
  ],
  "aggregate": {
    "total_tflops": 528.1628,
    "avg_latency_ms": 8.327867,
    "total_bandwidth_gbs": 386.838,
    "accuracy": "PASS",
    "all_passed": true
  }
}
```

## 指标计算公式

### Latency（延迟）

由 InfiniCore 框架返回，为固定次数迭代的**平均单次耗时**：

```
latency = total_time_ms / num_iterations
```

### TFLOPS（算力吞吐）

```
TFLOPS = FLOPS_per_iteration / latency_sec / 1e12
```

其中 matmul 的单次 FLOPS：

```
FLOPS = 2 × M × N × K       （C[M,N] = A[M,K] @ B[K,N]）
```

| 矩阵尺寸 | 单次 FLOPS |
|-----------|-----------|
| 4096×4096 | 2 × 4096³ = **137 GFLOPS** |
| 8192×8192 | 2 × 8192³ = **1100 GFLOPS** |

### BW（有效带宽）

```
BW = total_data_bytes / latency_sec / 1e9
```

其中 `total_data_bytes` = 所有输入张量 + 所有输出张量的字节数。

| 矩阵尺寸 (float32) | 数据量 |
|---------------------|--------|
| 4096×4096 | 3 × 4096² × 4B = **201 MB** |
| 8192×8192 | 3 × 8192² × 4B = **805 MB** |

### 聚合规则

| 指标 | 聚合方式 |
|------|---------|
| 总 TFLOPS | Σ 各卡 TFLOPS（直接求和） |
| 平均延迟 | mean(各卡延迟) |
| 总带宽 | Σ 各卡带宽 |
| 准确率 | 全部 PASS 才算 PASS |

## 工作原理

```
run_multi_gpu.sh
  │
  ├── 解析参数：配置文件、设备平台、GPU 列表
  ├── 通过 platform_config.py 获取环境变量名和卡数
  │
  ├── 并行启动 N 个进程：
  │     进程 0:  ENV_VAR=0  python main.py config.json --output gpu_0/
  │     进程 1:  ENV_VAR=1  python main.py config.json --output gpu_1/
  │     ...
  │     进程 N:  ENV_VAR=N  python main.py config.json --output gpu_N/
  │
  ├── wait 等待所有进程完成
  │
  └── 调用 aggregate_multi_gpu.py 聚合结果
        ├── 扫描 gpu_*/operator/*_results.json
        ├── 提取每卡 metrics
        ├── 计算 Σ TFLOPS、平均延迟等
        ├── 打印汇总表格
        └── 保存 aggregated_results.json
```

## 常见问题

### Q: 为什么 8192×8192 的 TFLOPS 比 4096×4096 提升不大？

这是正常现象。TFLOPS = 计算量 / 耗时，当矩阵大到一定程度后 GPU 计算单元已经饱和（compute-bound），计算量和耗时同比例增长，吞吐率趋于稳定。4096×4096 在 A100 上已经达到 TF32 峰值的 ~80%。

### Q: 矩阵更大后 BW 为什么反而下降？

计算量按 N³ 增长，数据量按 N² 增长。矩阵越大，计算密度（FLOPS/Byte）越高，GPU 花在数据搬运上的时间占比越低，所以有效带宽反而下降。这说明算子更加 compute-bound 了。

### Q: 各卡 TFLOPS 有差异正常吗？

正常。不同卡之间可能有微小的频率波动、缓存状态差异等，2-3% 的偏差在预期范围内。

### Q: 可以只跑聚合脚本吗？

可以，手动指定之前运行的输出目录即可：

```bash
python scripts/aggregate_multi_gpu.py ./output/multi_gpu_20260605_082711/
```
