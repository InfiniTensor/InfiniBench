#!/bin/bash
set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "=========================================="
echo "  NPU Performance Suite - Build Script"
echo "=========================================="
echo ""

# Detect CANN installation
if [ -z "${ASCEND_HOME_PATH}" ] && [ -z "${ASCEND_TOOLKIT_HOME}" ]; then
    # Try common locations
    for dir in /usr/local/Ascend/ascend-toolkit/latest \
               /usr/local/Ascend/ascend-toolkit/latest/*/ascend-toolkit/latest; do
        if [ -d "$dir" ]; then
            export ASCEND_HOME_PATH="$dir"
            break
        fi
    done
    if [ -z "${ASCEND_HOME_PATH}" ]; then
        echo -e "${RED}ERROR: CANN toolkit not found.${NC}"
        echo "Set ASCEND_HOME_PATH or ASCEND_TOOLKIT_HOME environment variable."
        echo "Example: export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit/latest"
        exit 1
    fi
fi

# Check for g++
if ! command -v g++ &> /dev/null; then
    echo -e "${RED}ERROR: g++ not found.${NC}"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"

echo -e "${YELLOW}Creating build directory...${NC}"
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

echo -e "${YELLOW}Configuring with CMake...${NC}"
cmake .. -DCMAKE_BUILD_TYPE=Release

echo -e "${YELLOW}Building...${NC}"
make -j$(nproc)

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}Build succeeded!${NC}"
    echo ""
    echo "Executable: ${BUILD_DIR}/npu_perf_suite"
    echo ""
    echo "Usage:"
    echo "  ${BUILD_DIR}/npu_perf_suite --all"
    echo "  ${BUILD_DIR}/npu_perf_suite --memory"
    echo "  ${BUILD_DIR}/npu_perf_suite --stream"
    echo "  ${BUILD_DIR}/npu_perf_suite --cache"
    echo ""
else
    echo ""
    echo -e "${RED}Build failed!${NC}"
    echo "Please check the error messages above."
    exit 1
fi
