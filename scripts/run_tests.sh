#!/bin/bash
# Unified test execution script
# Usage: ./scripts/run_tests.sh [OPTIONS] <input_paths...>
#
# Input paths can be:
#   - JSON files (test.json)
#   - Directories (test_dir/)
#
# Options:
#   --check <types>  Check specific dependencies before running (comma-separated)
#   --no-check       Skip dependency checking
#
# Examples:
#   ./scripts/run_tests.sh test.json                    # Run tests without dependency check
#   ./scripts/run_tests.sh test_dir/                   # Run tests in directory
#   ./scripts/run_tests.sh --check all test.json        # Check all dependencies
#   ./scripts/run_tests.sh --check hardware test.json   # Check hardware deps only
#   ./scripts/run_tests.sh --check hardware,operator test.json

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common/install_deps.sh"
source "${SCRIPT_DIR}/common/prepare_env.sh"

# Default values
CHECK_DEPS="false"
SPECIFIC_CHECKS=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --check)
            CHECK_DEPS="true"
            SPECIFIC_CHECKS="$2"
            shift 2
            ;;
        --no-check)
            CHECK_DEPS="false"
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS] <input_paths...>"
            echo ""
            echo "Input paths can be JSON files or directories."
            echo ""
            echo "Options:"
            echo "  --check <types>  Check specific dependencies before running (comma-separated)"
            echo "                   Types: hardware, operator, all"
            echo "  --no-check       Skip dependency checking"
            echo "  --help, -h       Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 test.json                    # Run tests without dependency check"
            echo "  $0 test_dir/                    # Run tests in directory"
            echo "  $0 --check all test.json        # Check all dependencies"
            echo "  $0 --check hardware test.json   # Check hardware deps only"
            echo "  $0 --check hardware,operator test.json"
            exit 0
            ;;
        *)
            break
            ;;
    esac
done

# All remaining arguments are input paths (files or directories)
INPUT_PATHS=("$@")

# Require at least one input path
if [ ${#INPUT_PATHS[@]} -eq 0 ]; then
    echo "[ERROR] Error: At least one input path (file or directory) is required"
    echo ""
    echo "Usage: $0 [OPTIONS] <input_paths...>"
    echo ""
    echo "Input paths can be JSON files or directories."
    echo ""
    echo "Examples:"
    echo "  $0 test.json                    # Run tests without dependency check"
    echo "  $0 test_dir/                    # Run tests in directory"
    echo "  $0 --check all test.json        # Check all dependencies"
    echo "  $0 --check hardware test.json   # Check hardware deps only"
    echo "  $0 --check hardware,operator test.json"
    echo ""
    echo "Run '$0 --help' for more information"
    exit 1
fi

# ========================================
# Dependency checking functions
# ========================================
check_all_deps() {
    local types="$1"

    # Convert comma-separated to array
    IFS=',' read -ra DEP_TYPES <<< "$types"

    for dtype in "${DEP_TYPES[@]}"; do
        dtype=$(echo "$dtype" | xargs)  # trim whitespace
        case "$dtype" in
            hardware)
                echo "Checking hardware dependencies:"
                check_cuda
                ;;
            operator)
                echo "Checking operator dependencies:"
                check_infinicore
                ;;
            all)
                echo "Checking all dependencies:"
                check_cuda
                check_infinicore
                ;;
            *)
                echo "[WARNING] Unknown dependency type: $dtype"
                ;;
        esac
    done
}

# ========================================
# Main execution
# ========================================
main() {
    echo "=========================================="
    echo "InfiniBench Test Runner"
    echo "=========================================="
    echo ""

    # Check dependencies if requested (default: false)
    if [ "$CHECK_DEPS" = "true" ]; then
        if [ -n "$SPECIFIC_CHECKS" ]; then
            check_all_deps "$SPECIFIC_CHECKS"
        else
            # If --check specified without value, check all
            check_all_deps "all"
        fi
        echo ""
    fi

    # Display environment
    echo "Environment: INFINI_ROOT=$INFINI_ROOT"
    echo ""

    # Run tests
    log_test_start "InfiniBench"
    trap 'cleanup_on_error "InfiniBench"' ERR

    echo "Running ${#INPUT_PATHS[@]} input path(s):"
    for path in "${INPUT_PATHS[@]}"; do
        echo "  - $path"
    done
    echo ""

    # Execute main.py
    python main.py "${INPUT_PATHS[@]}" --output "./output/"

    log_test_end "InfiniBench" $?
}

main "$@"
