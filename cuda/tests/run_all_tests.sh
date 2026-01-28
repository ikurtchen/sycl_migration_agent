#!/bin/bash

# CUDA Unit Test Runner
# This script runs all CUDA unit tests and generates benchmark reports

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== CUDA Unit Test Runner ===${NC}"
echo "This script will run all CUDA kernel tests and generate benchmark reports."
echo ""

# Create output directory
mkdir -p cuda_outputs
mkdir -p ../results/cuda_outputs

# Check if CUDA is available
if ! command -v nvcc &> /dev/null; then
    echo -e "${RED}Error: nvcc not found. Please ensure CUDA toolkit is installed and in PATH.${NC}"
    exit 1
fi

# Clean any previous builds
echo -e "${YELLOW}Cleaning previous builds...${NC}"
rm -rf build
mkdir build
cd build

echo -e "${YELLOW}Configuring CMake...${NC}"
cmake .. -DCMAKE_BUILD_TYPE=Release

echo -e "${YELLOW}Building tests...${NC}"
make -j$(nproc)

cd ..

# Function to run a test executable
run_test() {
    local test_name=$1
    local test_executable="build/$test_name"

    if [ ! -f "$test_executable" ]; then
        echo -e "${RED}Test executable $test_executable not found!${NC}"
        return 1
    fi

    echo -e "${YELLOW}Running $test_name tests...${NC}"
    ./$test_executable --gtest_output=xml:cuda_outputs/${test_name}_results.xml

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ $test_name tests passed${NC}"
    else
        echo -e "${RED}✗ $test_name tests failed${NC}"
        return 1
    fi
}

# Function to run benchmarks
run_benchmarks() {
    local test_name=$1
    local test_executable="build/$test_name"

    echo -e "${YELLOW}Running benchmarks for $test_name...${NC}"
    ./$test_executable --gtest_filter="*Benchmark*" --gtest_output=xml:cuda_outputs/${test_name}_benchmark.xml
}

# Create benchmark summary
create_benchmark_summary() {
    echo -e "${YELLOW}Creating benchmark summary...${NC}"

    cat > cuda_outputs/benchmark_summary.md << EOF
# CUDA Kernel Benchmark Summary

Generated on: $(date)

## Test Results Summary

EOF

    # Process each test result
    for file in cuda_outputs/*_benchmark.json; do
        if [ -f "$file" ]; then
            kernel_name=$(basename "$file" _benchmark.json)

            # Extract metrics
            avg_time=$(grep "avg_time_ms" "$file" | cut -d: -f2 | tr -d ', ')
            gflops=$(grep "gflops" "$file" | cut -d: -f2 | tr -d ' ,')

            cat >> cuda_outputs/benchmark_summary.md << EOF
### $kernel_name
- Average Time: ${avg_time} ms
- Throughput: ${gflops} GFLOPS

EOF
        fi
    done

    echo "Benchmark summary saved to cuda_outputs/benchmark_summary.md"
}

# Main execution
echo -e "${GREEN}=== Running All Tests ===${NC}"

# Track overall success
overall_success=0

# Run FP16 tests
if ! run_test "fp16_tests"; then
    overall_success=1
fi

# Run math tests
if ! run_test "math_tests"; then
    overall_success=1
fi

# Run neural network tests
if ! run_test "nn_tests"; then
    overall_success=1
fi

# Run transform tests
if ! run_test "transform_tests"; then
    overall_success=1
fi

# Create benchmark summary
create_benchmark_summary

# Copy outputs to results directory
echo -e "${YELLOW}Copying outputs to results directory...${NC}"
cp -r cuda_outputs/* ../results/cuda_outputs/ 2>/dev/null || true

# Final status
echo ""
if [ $overall_success -eq 0 ]; then
    echo -e "${GREEN}=== All tests completed successfully! ===${NC}"
    echo "Results saved in: cuda_outputs/"
    echo "Benchmark summary: cuda_outputs/benchmark_summary.md"
    exit 0
else
    echo -e "${RED}=== Some tests failed! ===${NC}"
    echo "Please check the test output above for details."
    exit 1
fi