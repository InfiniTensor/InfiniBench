#!/bin/bash
# Unified dependency management script for InfiniBench
# Usage: ./scripts/common/install_deps.sh [component]
#
# Components:
#   operator   - InfiniCore (operator testing)
#   hardware   - CUDA memory benchmark (hardware testing)
#   all        - All components (default)

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    SCRIPT_IS_SOURCED="false"
else
    SCRIPT_IS_SOURCED="true"
fi

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# ========================================
# Environment Variables (always exported)
# ========================================
export INFINI_ROOT="$HOME/.infini"
export LD_LIBRARY_PATH="$INFINI_ROOT/lib:$LD_LIBRARY_PATH"

# Safe exit function that uses return when sourced, exit when executed directly
safe_exit() {
    local exit_code="${1:-1}"
    if [[ "$SCRIPT_IS_SOURCED" == "true" ]]; then
        return "$exit_code"
    else
        exit "$exit_code"
    fi
}

# ========================================
# Dependency Checking Functions
# ========================================

# Check if a command exists
check_command() {
    if command -v "$1" &> /dev/null; then
        return 0
    else
        return 1
    fi
}

# Check Python package
check_python_package() {
    python -c "import $1" 2>/dev/null
}

# Check CUDA
check_cuda() {
    echo -n "  CUDA... "
    if check_command nvcc; then
        local version=$(nvcc --version | grep -oP 'release \K[0-9]+(\.[0-9]+)+' || echo "unknown")
        echo -e "${GREEN}[OK]${NC} (nvcc $version)"
        return 0
    else
        echo -e "${RED}[FAIL]${NC} not found"
        return 1
    fi
}

# Check InfiniCore
check_infinicore() {
    echo -n "  InfiniCore... "

    # Try to import and capture error details
    error_msg=$(python -c "import infinicore" 2>&1)
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}[OK]${NC}"
        return 0
    else
        echo -e "${YELLOW}[WARNING]${NC}"
        echo "    Failed to import infinicore:"
        echo "    ${error_msg}" | head -n 15 | sed 's/^/    /'
        return 1
    fi
}

# ========================================
# InfiniCore (Operator Testing)
# ========================================
install_infinicore() {
    echo ""
    echo "=========================================="
    echo "InfiniCore (Operator)"
    echo "=========================================="
    echo ""

    # 1. Check if INFINICORE_PATH is set
    if [ -z "$INFINICORE_PATH" ]; then
        echo -e "${RED}[ERROR] INFINICORE_PATH environment variable not set${NC}"
        echo ""
        echo "Please set INFINICORE_PATH environment variable:"
        echo "  export INFINICORE_PATH=/path/to/InfiniCore"
        echo ""
        echo "Example:"
        echo "  export INFINICORE_PATH=\"\$HOME/workplace/random_input/InfiniCore\""
        echo "  $0 operator"
        return 1
    fi

    # 2. Display environment
    echo -e "${BLUE}Environment:${NC}"
    echo "  INFINI_ROOT: $INFINI_ROOT"
    echo "  LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
    echo ""

    # 3. Check if InfiniCore is installed
    echo -e "${BLUE}Checking InfiniCore...${NC}"
    if python -c "import infinicore" 2>/dev/null; then
        echo -e "${GREEN}[OK] InfiniCore already installed${NC}"
        return 0
    fi

    echo -e "${YELLOW}[WARNING] InfiniCore not found, installing...${NC}"
    echo ""

    # 4. Install InfiniCore
    if [ ! -d "$INFINICORE_PATH" ]; then
        echo -e "${RED}[ERROR] InfiniCore not found at: $INFINICORE_PATH${NC}"
        return 1
    fi

    echo -e "${BLUE}InfiniCore path: $INFINICORE_PATH${NC}"
    echo ""

    cd "$INFINICORE_PATH" || {
        echo -e "${RED}[ERROR] Failed to change directory to $INFINICORE_PATH${NC}"
        return 1
    }

    echo -e "${BLUE}Step 1: Installing Python dependencies...${NC}"
    if ! python scripts/install.py --nv-gpu=y; then
        echo -e "${RED}[ERROR] Failed to install Python dependencies${NC}"
        return 1
    fi
    echo ""

    echo -e "${BLUE}Step 2: Building InfiniCore...${NC}"
    if ! xmake build _infinicore; then
        echo -e "${RED}[ERROR] Failed to build InfiniCore${NC}"
        return 1
    fi
    echo ""

    echo -e "${BLUE}Step 3: Installing InfiniCore...${NC}"
    if ! xmake install _infinicore; then
        echo -e "${RED}[ERROR] Failed to install InfiniCore${NC}"
        return 1
    fi
    echo ""

    echo -e "${BLUE}Step 4: Installing Python package...${NC}"
    if ! pip install -e .; then
        echo -e "${RED}[ERROR] Failed to install Python package${NC}"
        return 1
    fi
    echo ""

    # 5. Verify installation
    echo -e "${BLUE}Verifying installation...${NC}"
    echo "INFINI_ROOT: $INFINI_ROOT"
    echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

    error_msg=$(python -c "import infinicore" 2>&1)
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}[OK] InfiniCore installation completed successfully!${NC}"
        return 0
    else
        echo -e "${RED}[ERROR] InfiniCore installation verification failed${NC}"
        echo "    Import error:"
        echo "    ${error_msg}" | head -n 15 | sed 's/^/    /'
        echo ""
        echo "    Possible causes:"
        echo "    - Dependencies not installed (run: python scripts/install.py --nv-gpu=y)"
        echo "    - Library not built (run: xmake build _infinicore)"
        echo "    - LD_LIBRARY_PATH not set correctly"
        return 1
    fi
}

# ========================================
# CUDA Memory Benchmark (Hardware Testing)
# ========================================
install_hardware() {
    echo ""
    echo "=========================================="
    echo "CUDA Memory Benchmark (Hardware)"
    echo "=========================================="
    echo ""

    # 1. Initialize environment variables
    local SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    # Find project root by going up until we find the marker directory
    local PROJECT_ROOT="$SCRIPT_DIR"
    while [[ "$PROJECT_ROOT" != "/" && ! -d "$PROJECT_ROOT/infinibench" ]]; do
        PROJECT_ROOT="$(dirname "$PROJECT_ROOT")"
    done
    local BENCHMARK_PATH="$PROJECT_ROOT/infinibench/hardware/cuda-memory-benchmark"

    # 2. Check if already built
    echo -e "${BLUE}Checking CUDA memory benchmark...${NC}"
    if [ -f "$BENCHMARK_PATH/build/cuda_perf_suite" ]; then
        echo -e "${GREEN}[OK] CUDA memory benchmark already built${NC}"
        return 0
    fi

    echo -e "${YELLOW}[WARNING] CUDA memory benchmark not found, building...${NC}"
    echo ""

    # 3. Build
    if [ ! -d "$BENCHMARK_PATH" ]; then
        echo -e "${RED}[ERROR] CUDA memory benchmark not found at: $BENCHMARK_PATH${NC}"
        return 1
    fi

    echo -e "${BLUE}Benchmark path: $BENCHMARK_PATH${NC}"
    echo ""

    cd "$BENCHMARK_PATH" || {
        echo -e "${RED}[ERROR] Failed to change directory to $BENCHMARK_PATH${NC}"
        return 1
    }

    echo -e "${BLUE}Step 1: Cleaning previous build...${NC}"
    rm -rf build 2>/dev/null
    echo ""

    echo -e "${BLUE}Step 2: Building benchmark...${NC}"
    if ! bash build.sh; then
        echo -e "${RED}[ERROR] Build script failed${NC}"
        return 1
    fi
    echo ""

    # 4. Verify
    if [ -f "$BENCHMARK_PATH/build/cuda_perf_suite" ]; then
        echo -e "${GREEN}[OK] CUDA memory benchmark built successfully!${NC}"
        return 0
    else
        echo -e "${RED}[ERROR] CUDA memory benchmark build failed${NC}"
        return 1
    fi
}

# ========================================
# Main
# ========================================
show_usage() {
    echo "Usage: $0 [component]"
    echo ""
    echo "Components:"
    echo "  operator   - InfiniCore (operator testing)"
    echo "  hardware   - CUDA memory benchmark (hardware testing)"
    echo "  all        - All components (default)"
    echo ""
    echo "Environment variables:"
    echo "  INFINICORE_PATH  - Path to InfiniCore source (required for operator)"
    echo ""
    echo "Examples:"
    echo "  export INFINICORE_PATH=\"\$HOME/workplace/InfiniCore\""
    echo "  $0 operator   # Install InfiniCore"
    echo "  $0 hardware   # Build CUDA benchmark"
    echo "  $0 all        # Install everything"
    echo "  $0            # Install everything (default)"
}

main() {
    local COMPONENT="${1:-all}"
    local exit_code=0

    echo "=========================================="
    echo "InfiniBench Dependency Manager"
    echo "=========================================="
    echo ""

    case "$COMPONENT" in
        operator)
            install_infinicore
            exit_code=$?
            ;;
        hardware)
            install_hardware
            exit_code=$?
            ;;
        all)
            install_infinicore
            exit_code=$?
            if [ $exit_code -ne 0 ]; then
                echo -e "${YELLOW}[WARNING] InfiniCore installation failed, skipping hardware${NC}"
                return $exit_code
            fi
            
            install_hardware
            exit_code=$?
            ;;
        --help|-h)
            show_usage
            return 0
            ;;
        *)
            echo -e "${RED}[ERROR] Unknown component: $COMPONENT${NC}"
            echo ""
            show_usage
            return 1
            ;;
    esac

    if [ $exit_code -eq 0 ]; then
        echo ""
        echo "=========================================="
        echo -e "${GREEN}[OK] All operations completed successfully!${NC}"
        echo "=========================================="
    fi

    return $exit_code
}

# Valid component names for manual sourcing
VALID_COMPONENTS=("operator" "hardware" "all" "--help" "-h")

# Run main() if:
# 1. Script is executed directly (./scripts/common/install_deps.sh operator), OR
# 2. Script is sourced manually with a valid component name (source scripts/common/install_deps.sh operator)
# Do NOT run when sourced by other scripts (like run_tests.sh)
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    # Direct execution
    main "$@"
elif [[ $# -gt 0 ]]; then
    # Sourced with arguments - check if first arg is a valid component name
    first_arg="$1"
    for component in "${VALID_COMPONENTS[@]}"; do
        if [[ "$first_arg" == "$component" ]]; then
            # Manual sourcing with valid component
            main "$@"
            break
        fi
    done
fi
