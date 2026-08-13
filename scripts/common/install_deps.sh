#!/bin/bash
# Unified dependency management script for InfiniBench
# Usage: ./scripts/common/install_deps.sh [component]
#
# Components:
#   operator   - InfiniCore (operator testing)
#   hardware   - CUDA memory benchmark (hardware testing)
#   inference  - Inference frameworks (vLLM, InfiniLM)
#   training   - Training frameworks (InfiniTrain, Megatron)
#   comm       - Communication tests (NCCL tests)
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
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

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

# Check vLLM
check_vllm() {
    echo -n "  vLLM... "
    if check_python_package vllm; then
        echo -e "${GREEN}[OK]${NC}"
        return 0
    else
        echo -e "${YELLOW}[WARNING]${NC} not installed"
        return 1
    fi
}

# Check InfiniLM
check_infinilm() {
    echo -n "  InfiniLM... "
    if python -c "import infinilm" 2>/dev/null; then
        local version=$(python -c "import infinilm; print(getattr(infinilm, '__version__', 'unknown'))" 2>/dev/null || echo "unknown")
        echo -e "${GREEN}[OK]${NC} (version $version)"
        return 0
    else
        # Check if directory exists but not in PYTHONPATH
        local INFINILM_PATH="${INFINILM_PATH:-$HOME/InfiniLM}"
        if [ -d "$INFINILM_PATH/python/infinilm" ]; then
            echo -e "${YELLOW}[WARNING]${NC} found at $INFINILM_PATH but not in PYTHONPATH"
            echo "Run: export PYTHONPATH=$INFINILM_PATH/python:\$PYTHONPATH"
        else
            echo -e "${YELLOW}[WARNING]${NC} not installed"
        fi
        return 1
    fi
}

# TODO: Megatron-LM check - currently assumes submodule usage
check_megatron() {
    echo -n "  Megatron-LM... "
    if check_python_package megatron; then
        echo -e "${GREEN}[OK]${NC} (installed as package)"
        return 0
    elif [ -d "$PROJECT_ROOT/submodules/Megatron-LM" ]; then
        echo -e "${GREEN}[OK]${NC} (found as submodule)"
        return 0
    else
        echo -e "${YELLOW}[WARNING]${NC} not found (expected as submodule or Python package)"
        return 1
    fi
}

# TODO: InfiniTrain check - currently not available as Python package
check_infinitrain() {
    echo -n "  InfiniTrain... "
    if [ -d "$PROJECT_ROOT/submodules/InfiniTrain" ]; then
        echo -e "${GREEN}[OK]${NC} (found as submodule)"
        return 0
    else
        echo -e "${YELLOW}[WARNING]${NC} not found (expected as submodule)"
        return 1
    fi
}

# Check NCCL tests
check_nccl_tests() {
    echo -n "  NCCL tests... "
    local NCCL_TESTS_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../submodules/nccl-tests" && pwd)"
    if [ -f "$NCCL_TESTS_PATH/build/all_reduce_perf" ]; then
        echo -e "${GREEN}[OK]${NC}"
        return 0
    else
        echo -e "${YELLOW}[WARNING]${NC} not built"
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
# NCCL Tests (Communication)
# ========================================
install_comm() {
    echo ""
    echo "=========================================="
    echo "NCCL Tests (Communication)"
    echo "=========================================="
    echo ""

    local SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    local PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
    local NCCL_TESTS_PATH="$PROJECT_ROOT/submodules/nccl-tests"

    # 1. Check CUDA
    echo -e "${BLUE}Checking dependencies...${NC}"
    if ! check_cuda; then
        echo -e "${RED}[ERROR] CUDA is required for NCCL tests${NC}"
        return 1
    fi
    echo ""

    # 2. Check if already built
    echo -e "${BLUE}Checking NCCL tests...${NC}"
    if [ -f "$NCCL_TESTS_PATH/build/all_reduce_perf" ]; then
        echo -e "${GREEN}[OK] NCCL tests already built${NC}"
        return 0
    fi

    echo -e "${YELLOW}[WARNING] NCCL tests not found, building...${NC}"
    echo ""

    # 3. Build NCCL tests
    if [ ! -d "$NCCL_TESTS_PATH" ]; then
        echo -e "${RED}[ERROR] NCCL tests not found at: $NCCL_TESTS_PATH${NC}"
        echo "Please ensure submodules are initialized:"
        echo "  git submodule update --init --recursive"
        return 1
    fi

    cd "$NCCL_TESTS_PATH" || {
        echo -e "${RED}[ERROR] Failed to change directory to $NCCL_TESTS_PATH${NC}"
        return 1
    }

    echo -e "${BLUE}Building NCCL tests...${NC}"
    if ! make -j; then
        echo -e "${RED}[ERROR] Failed to build NCCL tests${NC}"
        return 1
    fi
    echo ""

    # 4. Verify
    if [ -f "build/all_reduce_perf" ]; then
        echo -e "${GREEN}[OK] NCCL tests built successfully!${NC}"
        return 0
    else
        echo -e "${RED}[ERROR] NCCL tests build failed${NC}"
        return 1
    fi
}

# ========================================
# Inference Frameworks
# ========================================
install_inference() {
    echo ""
    echo "=========================================="
    echo "Inference Frameworks"
    echo "=========================================="
    echo ""

    local exit_code=0

    # 1. Check InfiniCore dependency (required for InfiniLM)
    echo -e "${BLUE}Checking InfiniCore (required for InfiniLM)...${NC}"
    if ! check_infinicore; then
        echo -e "${YELLOW}[WARNING] InfiniCore not found, InfiniLM may not work${NC}"
        echo "Consider running: $0 operator"
    fi
    echo ""

    # 2. Install vLLM
    echo -e "${BLUE}Installing vLLM...${NC}"
    if check_python_package vllm; then
        echo -e "${GREEN}[OK] vLLM already installed${NC}"
    else
        if pip install vllm; then
            echo -e "${GREEN}[OK] vLLM installed${NC}"
        else
            echo -e "${RED}[ERROR] Failed to install vLLM${NC}"
            exit_code=1
        fi
    fi
    echo ""

    # 3. Install InfiniLM
    echo -e "${BLUE}Installing InfiniLM...${NC}"
    
    # Check if already installed
    if python -c "import infinilm" 2>/dev/null; then
        echo -e "${GREEN}[OK] InfiniLM already installed${NC}"
        return $exit_code
    fi

    # Determine InfiniLM path
    local INFINILM_PATH="${INFINILM_PATH:-}"
    
    # Priority 1: Environment variable
    if [ -n "$INFINILM_PATH" ] && [ -d "$INFINILM_PATH" ]; then
        INFINILM_PATH="$INFINILM_PATH"
    # Priority 2: Home directory
    elif [ -d "$HOME/InfiniLM" ]; then
        INFINILM_PATH="$HOME/InfiniLM"
    # Priority 3: Parent directory
    elif [ -d "$PROJECT_ROOT/../InfiniLM" ]; then
        INFINILM_PATH="$(cd "$PROJECT_ROOT/../InfiniLM" && pwd)"
    fi

    if [ -z "$INFINILM_PATH" ] || [ ! -d "$INFINILM_PATH" ]; then
        echo -e "${YELLOW}[WARNING] InfiniLM directory not found${NC}"
        echo ""
        echo "Please set INFINILM_PATH environment variable:"
        echo "export INFINILM_PATH=/home/sunjinge/InfiniLM"
        echo ""
        echo "Then run this script again"
        return $exit_code
    fi

    echo -e "${BLUE}Found InfiniLM at: $INFINILM_PATH${NC}"

    # Check Python package structure
    local PYTHON_PACKAGE_PATH="$INFINILM_PATH/python"
    
    if [ -d "$PYTHON_PACKAGE_PATH" ]; then
        echo -e "${BLUE}Found Python package at: $PYTHON_PACKAGE_PATH${NC}"
        
        # Add to PYTHONPATH for current session
        export PYTHONPATH="$PYTHON_PACKAGE_PATH:$PYTHONPATH"
        
        # Add to .bashrc for future sessions
        echo ""
        echo -e "${YELLOW}To make InfiniLM permanently available, add these lines to your ~/.bashrc:${NC}"
        echo "  export INFINILM_PATH=\"$INFINILM_PATH\""
        echo "  export PYTHONPATH=\"$PYTHON_PACKAGE_PATH:\$PYTHONPATH\""
        echo ""
        
        # Check if we can import now
        if python -c "import infinilm" 2>/dev/null; then
            echo -e "${GREEN}[OK] InfiniLM successfully configured${NC}"
        else
            echo -e "${YELLOW}[WARNING] InfiniLM found but import failed${NC}"
            echo "Current PYTHONPATH: $PYTHONPATH"
            echo "You may need to install additional dependencies"
        fi
    else
        echo -e "${YELLOW}[WARNING] No Python package found at $PYTHON_PACKAGE_PATH${NC}"
        echo "Contents of $INFINILM_PATH:"
        ls -la "$INFINILM_PATH"
    fi

    return $exit_code
}

# ========================================
# Training Frameworks
# ========================================
install_training() {
    echo ""
    echo "=========================================="
    echo "Training Frameworks"
    echo "=========================================="
    echo ""

    local exit_code=0

    # TODO: Megatron-LM installation - currently assumes submodule usage
    echo -e "${BLUE}Setting up Megatron-LM...${NC}"

    local MEGATRON_PATH=""

    if [ -n "$MEGATRON_PATH" ] && [ -d "$MEGATRON_PATH" ]; then
        MEGATRON_PATH="$MEGATRON_PATH"

    elif [ -d "$HOME/Megatron-LM" ]; then
        MEGATRON_PATH="$HOME/Megatron-LM"

    elif [ -d "$PROJECT_ROOT/submodules/Megatron-LM" ]; then
        MEGATRON_PATH="$PROJECT_ROOT/submodules/Megatron-LM"

    fi

    if [ -z "$MEGATRON_PATH" ]; then
        echo -e "${YELLOW}[WARNING] Megatron-LM not found${NC}"
        echo ""
        echo "Supported locations:"
        echo "  \$MEGATRON_PATH"
        echo "  \$HOME/Megatron-LM"
        echo "  submodules/Megatron-LM"
    else
        echo -e "${GREEN}[OK] Megatron-LM found at $MEGATRON_PATH${NC}"
    fi

    echo ""

    # TODO: InfiniTrain setup - to be implemented
    echo -e "${BLUE}Setting up InfiniTrain...${NC}"
    echo -e "${YELLOW}[INFO] InfiniTrain setup not yet implemented${NC}"
    echo ""

    return $exit_code
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
    echo "  inference  - Inference frameworks (vLLM, InfiniLM)"
    echo "  training   - Training frameworks (Megatron-LM, InfiniTrain)"
    echo "  comm       - Communication tests (NCCL tests)"
    echo "  all        - All components (default)"
    echo ""
    echo "Environment variables:"
    echo "  INFINICORE_PATH  - Path to InfiniCore source (required for operator)"
    echo "  INFINILM_PATH    - Path to InfiniLM source"
    echo "  INFINITRAIN_PATH - Path to InfiniTrain source"
    echo "  MEGATRON_PATH    - Path to Megatron-LM source"
    echo ""
    echo "Examples:"
    echo "  export INFINICORE_PATH=\"\$HOME/workplace/InfiniCore\""
    echo "  export INFINILM_PATH=\"\$HOME/InfiniLM\""
    echo "  $0 operator   # Install InfiniCore"
    echo "  $0 hardware   # Build CUDA benchmark"
    echo "  $0 inference  # Install inference frameworks"
    echo "  $0 comm       # Build NCCL tests"
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
        inference)
            install_inference
            exit_code=$?
            ;;
        training)
            install_training
            exit_code=$?
            ;;
        comm)
            install_comm
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

            install_inference
            exit_code=$?
            
            install_training
            exit_code=$?
            
            install_comm
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
