#!/bin/bash

# CUDA Benchmark Runner

set -e

echo "=== CUDA Benchmark Runner ==="
echo "This script runs all benchmark tests and generates performance reports."
echo ""

mkdir -p cuda_outputs
mkdir -p ../results/cuda_outputs

# Clean build
rm -rf build
mkdir build
cd build

echo "Building all tests..."
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

cd ..

echo -e "\n=== Running Benchmarks ===\n"

# Function to run benchmarks for a specific test
run_benchmark() {
    local test_name=$1
    local test_executable="build/$test_name"

    echo "Running benchmarks for $test_name..."
    ./$test_executable --gtest_filter="*Benchmark*" --gtest_output=xml:cuda_outputs/${test_name}_benchmark.xml

    if [ $? -eq 0 ]; then
        echo "✓ $test_name benchmarks completed"
    else
        echo "✗ $test_name benchmarks failed"
        return 1
    fi
}

# Run benchmarks for all test categories
if [ -f "build/fp16_tests" ]; then
    run_benchmark "fp16_tests"
fi

if [ -f "build/math_tests" ]; then
    run_benchmark "math_tests"
fi

if [ -f "build/nn_tests" ]; then
    run_benchmark "nn_tests"
fi

if [ -f "build/transform_tests" ]; then
    run_benchmark "transform_tests"
fi

# Generate comprehensive benchmark report
echo -e "\nGenerating benchmark report..."

cat > cuda_outputs/benchmark_report.md << EOF
# CUDA Kernel Performance Report

Generated on: $(date)
GPU Device: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
CUDA Version: $(nvcc --version | grep "release" | cut -d, -f2 | cut -d' ' -f2)

## Performance Summary

EOF

# Process all benchmark JSON files
for json_file in cuda_outputs/*_benchmark.json; do
    if [ -f "$json_file" ]; then
        kernel_name=$(basename "$json_file" _benchmark.json)
        echo "Processing $kernel_name..."

        # Extract metrics
        avg_time=$(grep "avg_time_ms" "$json_file" | sed 's/.*: \([0-9.]*\).*/\1/')
        gflops=$(grep "gflops" "$json_file" | sed 's/.*: \([0-9.]*\).*/\1/')
        data_size=$(grep "data_size_bytes" "$json_file" | sed 's/.*: \([0-9]*\).*/\1/')

        # Convert to GB
        data_size_gb=$(echo "scale=2; $data_size / (1024^3)" | bc)

        cat >> cuda_outputs/benchmark_report.md << EOF
### $kernel_name
- **Average Time:** ${avg_time} ms
- **Throughput:** ${gflops} GFLOPS
- **Data Size:** ${data_size_gb} GB

EOF
    fi
done

# Generate CSV report
echo "kernel_name,avg_time_ms,gflops,data_size_gb" > cuda_outputs/benchmark_summary.csv

for json_file in cuda_outputs/*_benchmark.json; do
    if [ -f "$json_file" ]; then
        kernel_name=$(basename "$json_file" _benchmark.json)
        avg_time=$(grep "avg_time_ms" "$json_file" | sed 's/.*: \([0-9.]*\).*/\1/')
        gflops=$(grep "gflops" "$json_file" | sed 's/.*: \([0-9.]*\).*/\1/')
        data_size=$(grep "data_size_bytes" "$json_file" | sed 's/.*: \([0-9]*\).*/\1/')
        data_size_gb=$(echo "scale=6; $data_size / (1024^3)" | bc)

        echo "$kernel_name,$avg_time,$gflops,$data_size_gb" >> cuda_outputs/benchmark_summary.csv
    fi
done

# Copy results to parent directory
cp -r cuda_outputs/* ../results/cuda_outputs/ 2>/dev/null || true

echo -e "\n=== Benchmark Report Generated ==="
echo "Reports saved in:"
echo "  - cuda_outputs/benchmark_report.md"
echo "  - cuda_outputs/benchmark_summary.csv"
echo "  - cuda_outputs/ (individual JSON files)"
echo ""
echo "Results also copied to: ../results/cuda_outputs/"