#!/bin/bash

# SYCL Vector Addition Test Suite
# This script builds and runs comprehensive tests for the SYCL vector addition kernel

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}===================================${NC}"
echo -e "${GREEN}SYCL VectorAdd Test Suite${NC}"
echo -e "${GREEN}===================================${NC}"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"

# Display build directory
echo -e "Build Directory: ${BUILD_DIR}"
echo -e "Time: $(date)"
echo ""

# Check if oneAPI is set up
echo "Checking oneAPI availability..."
if ! command -v icpx &> /dev/null; then
    echo -e "${RED}Error: Intel DPC++/SYCL compiler (icpx) not found!${NC}"
    echo -e "${YELLOW}Please source oneAPI environment:${NC}"
    echo -e "  source /opt/intel/oneapi/setvars.sh"
    exit 1
fi

# Display SYCL compiler version
echo -e "${GREEN}SYCL Compiler Version:${NC}"
icpx --version | head -n 1
echo ""

# Check for available devices
echo "Checking GPU availability..."
if command -v sycl-ls &> /dev/null; then
    sycl-ls | head -n 5
else
    echo -e "${YELLOW}sycl-ls not found, assuming GPU is available${NC}"
fi
echo ""

# Create build directory if it doesn't exist
echo "Building tests..."
if [ ! -d "$BUILD_DIR" ]; then
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"
    cmake .. -DCMAKE_BUILD_TYPE=Release
else
    echo "Using existing build directory"
fi

cd "$BUILD_DIR"

# Check if CMake configuration succeeded
if [ ! -f "Makefile" ]; then
    echo -e "${RED}CMake configuration not found, running cmake...${NC}"
    cmake .. -DCMAKE_BUILD_TYPE=Release

    if [ $? -ne 0 ]; then
        echo -e "${RED}CMake configuration failed!${NC}"
        exit 1
    fi
fi

# Build the project
echo "Compiling..."
make -j$(nproc)

# Check if build was successful
if [ $? -ne 0 ]; then
    echo -e "${RED}Build failed!${NC}"
    exit 1
fi

echo ""
echo "Running tests..."
echo -e "${GREEN}===================================${NC}"

# Run the tests
cd "$SCRIPT_DIR"
$BUILD_DIR/vectorAdd_sycl_test --gtest_color=yes
TEST_EXIT_CODE=$?

echo ""

# Check test results
if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}===================================${NC}"
    echo -e "${GREEN}✓ All tests PASSED!${NC}"
    echo -e "${GREEN}===================================${NC}"
else
    echo -e "${RED}===================================${NC}"
    echo -e "${RED}✗ Some tests FAILED!${NC}"
    echo -e "${RED}===================================${NC}"
fi

# Output directories
OUTPUT_DIR="${SCRIPT_DIR}/sycl_outputs"
INPUT_DIR="${SCRIPT_DIR}/sycl_inputs"

echo ""
echo -e "${YELLOW}Test Summary:${NC}"
echo -e "------------"
echo -e "Output Directory: ${OUTPUT_DIR}"
echo -e "Input Directory: ${INPUT_DIR}"

# Count generated files
if [ -d "$OUTPUT_DIR" ]; then
    OUTPUT_COUNT=$(find "$OUTPUT_DIR" -type f | wc -l)
    echo -e "Generated ${OUTPUT_COUNT} output files"
else
    echo -e "${YELLOW}No output directory found${NC}"
fi

if [ -d "$INPUT_DIR" ]; then
    INPUT_COUNT=$(find "$INPUT_DIR" -type f | wc -l)
    echo -e "Generated ${INPUT_COUNT} input files"
fi

echo ""
echo "Test Execution Complete!"

exit $TEST_EXIT_CODE
