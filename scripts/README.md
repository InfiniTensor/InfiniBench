# Testing Scripts

Unified test execution scripts for InfiniMetrics.

## Quick Start

```bash
# Run tests with input file(s)
./scripts/run_tests.sh test.json

# Run tests in a directory
./scripts/run_tests.sh test_dir/

# Run multiple input files
./scripts/run_tests.sh test1.json test2.json test3.json
```

## Structure

```
scripts/
├── run_tests.sh                   # Unified test execution script
├── generate_operator_inputs.py    # Operator test input generator
├── aggregate_results.py           # Test results aggregator
└── common/                        # Shared utilities
    ├── install_deps.sh            # Dependency management (check + install)
    └── prepare_env.sh             # Environment preparation functions
```

## Script Organization

### Main Script: `run_tests.sh`

Unified test execution script with automatic dependency management.

**Usage (Direct Execution):**
```bash
./scripts/run_tests.sh [OPTIONS] <input_paths...>
```

**Usage (Source Mode - Environment Variables Persist):**
```bash
source scripts/run_tests.sh
run_tests [OPTIONS] <input_paths...>
```

**Options:**
```bash
--check <types>   Check specific dependencies before running (comma-separated)
                   Types: hardware, operator, all
--no-check        Skip dependency checking
--help, -h        Show help message
```

**Input paths:**
- Can be JSON files or directories

**Examples:**
```bash
# Direct execution (recommended for CI/automation)
./scripts/run_tests.sh test.json
./scripts/run_tests.sh test_dir/
./scripts/run_tests.sh test1.json test2.json
./scripts/run_tests.sh --check hardware test.json

# Source mode (recommended for development)
source scripts/run_tests.sh
run_tests test.json
run_tests --check all test.json
```

### Common Functions (`common/`)

**`install_deps.sh`**: Unified dependency management (check + install)

Can be used standalone or sourced by other scripts.

**Standalone usage:**
```bash
# Install specific component
export INFINICORE_PATH="/path/to/InfiniCore"
source scripts/common/install_deps.sh operator   # Install InfiniCore
source scripts/common/install_deps.sh hardware   # Build CUDA benchmark
source scripts/common/install_deps.sh all        # Install everything
```

**Components:**
- `operator` - InfiniCore (operator testing)
- `hardware` - CUDA memory benchmark (hardware testing)

**Checking functions** (available when sourced):
- `check_cuda` - Check NVIDIA CUDA toolkit
- `check_infinicore` - Check InfiniCore package

**Installation functions** (available when sourced):
- `install_infinicore` - Install InfiniCore from source
- `install_hardware` - Build CUDA memory benchmark

**`prepare_env.sh`**: Environment preparation functions
- `log_test_start` - Log test start message with timestamp
- `log_test_end` - Log test completion with exit code
- `cleanup_on_error` - Error trap handler
- `get_timestamp` - Get current timestamp

---

### Input Generator: `generate_operator_inputs.py`

生成覆盖小、中、大规模张量的标准化算子测试输入，数据由随机种子生成，自动覆盖不同 shape × dtype 组合。输出 InfiniMetrics 兼容的 JSON 配置和 `.npy` 数据文件。

**Usage:**

```bash
python scripts/generate_operator_inputs.py [OPTIONS]
```

**Options:**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--output, -o` | `./operator_test_inputs` | 输出目录 |
| `--operators` | `matmul add sub mul div` | 要测试的算子，可选: `matmul add sub mul div linear` |
| `--dtypes` | `float16 float32 bfloat16` | 数据精度 |
| `--scales` | `small medium large` | 张量规模 |
| `--device` | `nvidia` | 目标设备 |
| `--seed` | `42` | 随机种子（保证可复现） |
| `--warmup` | `10` | 预热迭代次数 |
| `--measured` | `100` | 测量迭代次数 |
| `--dry-run` | - | 仅预览测试用例，不生成文件 |

**Examples:**

```bash
# 生成所有算子、所有规模、所有精度的完整测试输入
python scripts/generate_operator_inputs.py --output ./test_inputs --seed 42

# 只生成 matmul + add，float16 + float32
python scripts/generate_operator_inputs.py --operators matmul add --dtypes float16 float32

# 只生成小规模和中规模
python scripts/generate_operator_inputs.py --scales small medium

# 预览将生成哪些测试用例（不产生文件）
python scripts/generate_operator_inputs.py --dry-run
```

**Output:**

```
operator_test_inputs/
├── configs/                          # 每个测试用例一个 JSON 文件
│   ├── opbench.matmul.small.64x64.float16.json
│   ├── opbench.add.large.4096x4096.float32.json
│   └── ...
├── data/                             # .npy 张量数据文件
│   ├── matmul_small_64x64_float16_a.npy
│   └── ...
├── all_test_inputs.json              # 所有用例合并（可直接传给 main.py）
└── _generation_metadata.json         # 生成参数记录
```

**Run the generated inputs:**

```bash
# 执行单个或全部配置
python main.py ./test_inputs/configs/
python main.py ./test_inputs/all_test_inputs.json

# 也可通过 run_tests.sh 执行
./scripts/run_tests.sh ./test_inputs/configs/
```

**Coverage:**

| 规模 | MatMul / Linear Shapes | Element-wise Shapes |
|------|----------------------|---------------------|
| small | 64×64, 128×128, 256×256 | 64×64, 128×256, 256×512 |
| medium | 512×512, 768×1024, 1024×768 | 512×1024, 1024×1024, 2048×512 |
| large | 1024×1024, 2048×2048, 4096×4096 | 2048×2048, 4096×4096, 8192×1024 |

默认全部组合（5 算子 × 3 规模 × 3 shape × 3 精度 = 135 个测试用例）。

---

### Results Aggregator: `aggregate_results.py`

从 `output/` 目录读取所有结果文件，按算子、规模、精度分类汇总，输出统计表和详细指标。

**Usage:**

```bash
python scripts/aggregate_results.py [OPTIONS]
```

**Options:**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--input, -i` | `./output` | 结果文件所在目录 |
| `--output, -o` | `./aggregated_results.json` | 汇总输出 JSON 路径 |
| `--print` | - | 在终端打印可读的汇总表格 |
| `--filter-operator` | - | 只汇总指定算子 |
| `--filter-scale` | - | 只汇总指定规模 (small/medium/large) |
| `--filter-dtype` | - | 只汇总指定精度 (float16/float32/bfloat16) |

**Examples:**

```bash
# 汇总所有结果并在终端打印表格
python scripts/aggregate_results.py --print

# 指定输入输出路径
python scripts/aggregate_results.py -i ./output -o ./summary.json --print

# 只看 matmul 算子
python scripts/aggregate_results.py --filter-operator matmul --print

# 只看大规模 float16 结果
python scripts/aggregate_results.py --filter-scale large --filter-dtype float16 --print
```

**Output (JSON):**

```json
{
  "total": 135,
  "passed": 120,
  "failed": 15,
  "pass_rate": "88.9%",
  "by_operator": {
    "matmul": { "passed": 27, "failed": 0, "latency_ms": {...}, "tflops": {...}, "bandwidth_gbs": {...} },
    "add":    { "passed": 27, "failed": 0, ... }
  },
  "by_scale":    { "small": {...}, "medium": {...}, "large": {...} },
  "by_dtype":    { "float16": {...}, "float32": {...}, "bfloat16": {...} },
  "failed_details": [ { "run_id": "...", "error_msg": "..." } ],
  "detailed_table": [ { "run_id": "...", "operator": "...", "latency": ..., "flops": ..., ... } ]
}
```

**Console output (with `--print`):**

```
========================================================================
  InfiniMetrics Test Results Summary
========================================================================
  Total  : 135
  Passed : 120
  Failed : 15
  Rate   : 88.9%
========================================================================

── By Operator ──
  Operator     Total Passed Failed    Latency(ms)     TFLOPS   BW(GB/s)
  --------------------------------------------------------------------
  matmul         27     27      0        0.1234     1.2345    12.3456
  add            27     27      0        0.0567     0.5678     5.6789

── By Scale ──
  ...

── Failed Tests (15) ──
  ...

── Passed Tests Detail (120) ──
  ...
```

---

## Output

All test results are saved to:
```
output/
```

## Requirements

- Python 3.10+
- NumPy
- Bash 4.0+
- CUDA toolkit (for CUDA hardware tests)
- InfiniCore source (for operator tests)
