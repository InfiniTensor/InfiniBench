# Testing Scripts

Unified test execution scripts for InfiniBench.

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
├── run_tests.sh               # Unified test execution script
└── common/                    # Shared utilities
    ├── install_deps.sh        # Dependency management (check + install)
    └── prepare_env.sh         # Environment preparation functions
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

## Output

All test results are saved to:
```
output/
```

## Requirements

- Python 3.10+
- Bash 4.0+
- CUDA toolkit (for CUDA hardware tests)
- InfiniCore source (for operator tests)
