#!/bin/bash

# Build script for SYCL kernels
# This script configures and builds the SYCL project with proper compiler detection

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
BUILD_TYPE="Release"
BUILD_DIR="build"
CLEAN_BUILD=false
ENABLE_INTEL_GPU=true
DEBUG_MODE=false
RUN_TESTS=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --debug)
            DEBUG_MODE=true
            BUILD_TYPE="Debug"
            shift
            ;;
        --clean)
            CLEAN_BUILD=true
            shift
            ;;
        --no-intel-gpu)
            ENABLE_INTEL_GPU=false
            shift
            ;;
        --run-tests)
            RUN_TESTS=true
            shift
            ;;
        --build-dir)
            BUILD_DIR="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --debug          Enable debug mode (no optimization)"
            echo "  --clean          Clean build directory before building"
            echo "  --no-intel-gpu   Disable Intel GPU specific compilation"
            echo "  --run-tests      Run tests after building"
            echo "  --build-dir DIR  Specify build directory (default: build)"
            echo "  -h, --help       Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo -e "${GREEN}=== SYCL Kernel Build Script ===${NC}"
echo ""

# Check if we're in the right directory
if [ ! -f "CMakeLists.txt" ]; then
    echo -e "${RED}Error: CMakeLists.txt not found. Please run this script from the sycl directory.${NC}"
    exit 1
fi

# Clean build directory if requested
if [ "$CLEAN_BUILD" = true ]; then
    echo -e "${YELLOW}Cleaning build directory...${NC}"
    rm -rf $BUILD_DIR
fi

# Create build directory
mkdir -p $BUILD_DIR
cd $BUILD_DIR

# Setup environment for Intel oneAPI
if [ -f "/opt/intel/oneapi/setvars.sh" ]; then
    echo -e "${GREEN}Setting up Intel oneAPI environment...${NC}"
    source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
fi

# Configure with cmake
echo -e "${GREEN}Configuring build with CMake...${NC}"

CMAKE_ARGS=(
    -DCMAKE_BUILD_TYPE=$BUILD_TYPE
    -DSYCL_ARCH_INTEL_GPU=$ENABLE_INTEL_GPU
    -DSYCL_DEBUG=$DEBUG_MODE
)

# Check if SYCL_COMPILER is set in environment
if [ -n "$SYCL_COMPILER" ]; then
    CMAKE_ARGS+=(-DSYCL_COMPILER=$SYCL_COMPILER)
fi

cmake "${CMAKE_ARGS[@]}" ..

# Build
echo -e "${GREEN}Building SYCL kernels...${NC}"
make -j$(nproc)

echo ""
echo -e "${GREEN}=== Build completed successfully! ===${NC}"
echo ""

# List built executables
echo -e "${YELLOW}Built executables:${NC}"
ls -la test_* 2>/dev/null || echo "No test executables found"

# Run tests if requested
if [ "$RUN_TESTS" = true ]; then
    echo ""
    echo -e "${GREEN}=== Running Tests ===${NC}"

    # Run each test
    for test_exe in test_*; do
        if [ -f "$test_exe" ] && [ -x "$test_exe" ]; then
            echo -e "${YELLOW}Running $test_exe...${NC}"
            ./$test_exe
            echo ""
        fi
    done

    # Or use ctest
    echo -e "${YELLOW}Running all tests via CTest...${NC}"
    ctest --verbose
fi

echo ""
echo -e "${GREEN}Done!${NC}"