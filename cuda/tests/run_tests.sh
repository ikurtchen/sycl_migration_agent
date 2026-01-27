#!/bin/bash

# VectorAdd Test Execution Script
# This script runs all vectorAdd tests and manages input/output persistence

set -e  # Exit on any error

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Create necessary directories
mkdir -p "$SCRIPT_DIR/cuda_inputs"
mkdir -p "$SCRIPT_DIR/cuda_outputs"
mkdir -p "$SCRIPT_DIR/test_logs"

echo "==================================="
echo "CUDA VectorAdd Test Suite"
echo "==================================="
echo "Build Directory: ${SCRIPT_DIR}"
echo "Time: $(date)"
echo ""

# Check if CUDA is available
echo "Checking CUDA availability..."
if ! command -v nvcc &> /dev/null; then
    echo "ERROR: CUDA compiler (nvcc) not found!"
    exit 1
fi

# Check GPU availability
echo "Checking GPU availability..."
if ! nvidia-smi &> /dev/null; then
    echo "WARNING: nvidia-smi not available. GPU may not be accessible."
fi

# Build the tests
echo "Building tests..."
cd "$SCRIPT_DIR"

# Create build directory
mkdir -p build
cd build

# Configure with CMake
echo "Configuring with CMake..."
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
echo "Compiling..."
make -j$(nproc 2>/dev/null || echo 4)

cd "$SCRIPT_DIR"

# Run tests with output capture
echo ""
echo "Running tests..."
echo "==================================="

TEST_LOG="$SCRIPT_DIR/test_logs/vectorAdd_test_$(date +%Y%m%d_%H%M%S).log"

# Run the test executable with output capture
./build/vectorAdd_test 2>&1 | tee "$TEST_LOG"

# Check if tests passed
if [ $? -eq 0 ]; then
    echo ""
    echo "==================================="
    echo "✓ All tests PASSED!"
    echo "==================================="
else
    echo ""
    echo "==================================="
    echo "✗ Some tests FAILED!"
    echo "==================================="
    echo "Check log file: $TEST_LOG"
    exit 1
fi

# Generate test report
echo ""
echo "Generating test report..."
REPORT_FILE="$SCRIPT_DIR/test_logs/vectorAdd_test_report_$(date +%Y%m%d_%H%M%S).txt"

cat > "$REPORT_FILE" << EOF
CUDA VectorAdd Test Report
=========================
Date: $(date)
Test Log: $TEST_LOG
Build Directory: ${SCRIPT_DIR}/build

Output Files:
--------------
EOF

# List all generated output files
if [ -d "$SCRIPT_DIR/cuda_outputs" ]; then
    echo "CUDA Outputs:" >> "$REPORT_FILE"
    ls -la "$SCRIPT_DIR/cuda_outputs/" >> "$REPORT_FILE"
fi

if [ -d "$SCRIPT_DIR/cuda_inputs" ]; then
    echo "" >> "$REPORT_FILE"
    echo "CUDA Inputs:" >> "$REPORT_FILE"
    ls -la "$SCRIPT_DIR/cuda_inputs/" >> "$REPORT_FILE"
fi

# Extract benchmark results from output files if they exist
echo "" >> "$REPORT_FILE"
echo "Benchmark Results:" >> "$REPORT_FILE"

# Use find to handle the wildcard expansion properly
BenchmarkFiles=$(find "$SCRIPT_DIR/cuda_outputs" -name "*benchmark*.json" 2>/dev/null)
for file in $BenchmarkFiles; do
    if [ -f "$file" ]; then
        echo ""
        cat "$file" >> "$REPORT_FILE"
    fi
done

echo "Test report generated: $REPORT_FILE"

# Display summary
echo ""
echo "Test Summary:"
echo "------------"
echo "Test Log: $TEST_LOG"
echo "Test Report: $REPORT_FILE"
echo "Output Directory: $SCRIPT_DIR/cuda_outputs/"
echo "Input Directory: $SCRIPT_DIR/cuda_inputs/"

# Show output files count
OUTPUT_COUNT=$(find "$SCRIPT_DIR/cuda_outputs" -name "*.bin" 2>/dev/null | wc -l)
echo "Generated $OUTPUT_COUNT output files"

echo ""
echo "Test Execution Complete!"