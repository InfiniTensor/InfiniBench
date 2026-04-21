# Quick Start Guide

This guide will help you get up and running with InfiniBench in minutes.

## Prerequisites

Before you begin, ensure you have:
- **Python 3.8+** installed
- **Git** installed (for cloning the repository)
- **GPU** (optional, for hardware tests)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/InfiniTensor/InfiniBench.git
cd InfiniBench
```

### 2. Initialize Submodules

```bash
git submodule update --init --recursive
```

### 3. Install Dependencies

```bash
# Core dependencies
pip install numpy torch pandas

# Optional: For vLLM support
pip install vllm
```

### 4. Build Hardware Benchmarks (Optional)

If you plan to run hardware tests:

```bash
cd infinibench/hardware/cuda-memory-benchmark
bash build.sh
```

## Run Your First Test

### Hardware Test

Run a comprehensive hardware benchmark with a single command:

```bash
python main.py format_input_comprehensive_hardware.json
```

This will test:
- Memory bandwidth (H2D, D2H, D2D, bidirectional)
- STREAM benchmark
- GPU cache performance

### Inference Test

Run an inference benchmark:

```bash
cd infinibench/inference
python infer_main.py --config config.json
```

## Understanding the Output

### Test Results Location

Results are saved in:
```
./summary_output/
└── dispatcher_summary_YYYYMMDD_HHMMSS.json    # Overall summary
./output/
└── <test_case_name>/
    ├── seperated_test_result.json              # Seperated test result
    └── metrics.csv                             # Timeseries metrics
```

### Example: Hardware Test Output

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

## MongoDB Integration (Optional)

InfiniBench supports storing test results in MongoDB for persistent storage and dashboard visualization.

### Prerequisites

```bash
# Install MongoDB dependencies
pip install pymongo watchdog

# Ensure MongoDB is running locally or set connection URL
export MONGO_URI="mongodb://localhost:27017"
```

### CLI Usage

```bash
# Start file watcher (auto-import new results, runs forever)
python -m db.watcher --output-dir ./output --summary-dir ./summary_output

# One-time scan only (import existing files and exit)
python -m db.watcher --scan
```

### Python API

```python
from pathlib import Path
from db.watcher import Watcher

# Create watcher and start monitoring
watcher = Watcher(
    output_dir=Path("./output"),
    summary_dir=Path("./summary_output")
)

# One-time scan
result = watcher.scan()
print(f"Imported: {len(result['imported'])}")

# Or run forever (auto-import new files)
watcher.run_forever()  # Blocks until Ctrl+C
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MONGO_URI` | `mongodb://localhost:27017` | MongoDB connection URL |
| `MONGO_DB_NAME` | `infinibench` | Database name |
| `MONGO_COLLECTION` | `test_runs` | Test results collection |

## Next Steps

- **Configure Tests**: See [Configuration Guide](./configuration.md) for customization
