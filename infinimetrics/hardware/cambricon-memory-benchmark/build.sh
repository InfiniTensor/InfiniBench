#!/bin/bash
set -e

RED='\033[0;31m'
GREEN='\03033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "=========================================="
echo "  MLU Performance Suite - Build Script"
echo "=========================================="
echo ""

if ! command -v cncc &> /dev/null; then
    echo -e "${RED}ERROR: cncc not found. Install CNToolkit.${NC}"
    exit 1
fi

if [ -z "${NEUWARE_HOME}" ]; then
    export NEUWARE_HOME="/usr/local/neuware"
fi

# MLU architecture - must be set for device kernel compilation
if [ -z "${MLU_ARCH}" ]; then
    MLU_ARCH="mtp_592"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"

mkdir -p "${BUILD_DIR}"

echo -e "${YELLOW}Compiling (arch=${MLU_ARCH})...${NC}"
cncc "${SCRIPT_DIR}/src/main.mlu" \
    -o "${BUILD_DIR}/mlu_perf_suite" \
    -O3 -std=c++17 \
    --bang-mlu-arch="${MLU_ARCH}" \
    -I"${NEUWARE_HOME}/include" \
    -I"${SCRIPT_DIR}/include" \
    -L"${NEUWARE_HOME}/lib64" \
    -lcnrt -lstdc++ -lm

echo ""
echo -e "${GREEN}Build succeeded!${NC}"
echo ""
echo "Executable: ${BUILD_DIR}/mlu_perf_suite"
echo ""
echo "Usage:"
echo "  ${BUILD_DIR}/mlu_perf_suite --all"
echo "  ${BUILD_DIR}/mlu_perf_suite --memory"
echo "  ${BUILD_DIR}/mlu_perf_suite --stream"
echo "  ${BUILD_DIR}/mlu_perf_suite --cache"
echo "  MLU_ARCH=mtp_592 ./build.sh   # override arch"
echo ""
