#!/bin/bash

# Build script for CUDA Performance Suite
# Usage:
#   bash build.sh --platform cuda    # Build with native CUDA (NVIDIA GPU)
#   bash build.sh --platform metax   # Build with cu-bridge (MetaX GPU)

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Parse arguments
PLATFORM=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --platform)
            PLATFORM="$2"
            shift 2
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Usage: bash build.sh --platform <cuda|metax>"
            exit 1
            ;;
    esac
done

if [[ -z "$PLATFORM" ]]; then
    echo -e "${RED}ERROR: --platform is required.${NC}"
    echo "Usage: bash build.sh --platform <cuda|metax>"
    exit 1
fi

echo "=========================================="
echo "  CUDA Performance Suite - Build Script"
echo "  Platform: ${PLATFORM}"
echo "=========================================="
echo ""

if [[ "$PLATFORM" == "metax" ]]; then
    # ---- MetaX platform: using cu-bridge ----

    export MACA_PATH=${MACA_PATH:-/opt/maca}
    export CUCC_PATH=${CUCC_PATH:-${MACA_PATH}/tools/cu-bridge}
    export PATH=$PATH:${CUCC_PATH}/tools:${CUCC_PATH}/bin
    export CUCC_CMAKE_ENTRY=2
    export CUDA_PATH=${CUCC_PATH}

    # Create nvcc symlink (if it doesn't exist)
    if [ ! -e ${CUCC_PATH}/bin/nvcc ]; then
        ln -s ${CUCC_PATH}/bin/cucc ${CUCC_PATH}/bin/nvcc
    fi

    if ! command -v cucc &> /dev/null; then
        echo -e "${RED}ERROR: cucc not found. Please check cu-bridge installation at ${CUCC_PATH}${NC}"
        exit 1
    fi

    echo -e "${YELLOW}[MetaX] Using cu-bridge: ${CUCC_PATH}${NC}"

    mkdir -p build
    cd build

    echo -e "${YELLOW}Configuring with cmake_maca...${NC}"
    cmake_maca .. -DCMAKE_BUILD_TYPE=Release -DPLATFORM=metax

    echo -e "${YELLOW}Building with make_maca...${NC}"
    make_maca -j$(nproc)

elif [[ "$PLATFORM" == "cuda" ]]; then
    # ---- NVIDIA CUDA platform ----

    if ! command -v nvcc &> /dev/null; then
        echo -e "${RED}ERROR: nvcc not found. Please install CUDA toolkit.${NC}"
        exit 1
    fi

    echo -e "${YELLOW}[CUDA] Using native nvcc: $(which nvcc)${NC}"

    mkdir -p build
    cd build

    echo -e "${YELLOW}Configuring with CMake...${NC}"
    cmake .. -DCMAKE_BUILD_TYPE=Release -DPLATFORM=cuda

    echo -e "${YELLOW}Building...${NC}"
    make -j$(nproc)

else
    echo -e "${RED}ERROR: Unsupported platform '${PLATFORM}'. Use 'cuda' or 'metax'.${NC}"
    exit 1
fi

# Check if build was successful
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}Build completed successfully!${NC}"
    echo ""
    echo "Executable: build/cuda_perf_suite"
    echo ""
    echo "Usage:"
    echo "  ./build/cuda_perf_suite --help        # Show help"
    echo "  ./build/cuda_perf_suite --all         # Run all tests"
    echo "  ./build/cuda_perf_suite --memory      # Run memory bandwidth tests"
    echo "  ./build/cuda_perf_suite --stream      # Run STREAM benchmark only"
    echo "  ./build/cuda_perf_suite --cache       # Run cache tests only"
    echo ""
else
    echo ""
    echo -e "${RED}Build failed!${NC}"
    echo "Please check the error messages above."
    exit 1
fi
