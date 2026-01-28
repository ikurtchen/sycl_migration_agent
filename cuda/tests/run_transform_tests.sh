#!/bin/bash

# Transform and Reduction Kernel Test Runner

set -e

echo "=== Running Transform & Reduction Kernel Tests ==="
mkdir -p cuda_outputs

# Build and run transform tests
rm -rf build
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make transform_tests
cd ..

echo "Running Transpose tests..."
./build/transform_tests --gtest_filter="TransposeTest.*" --gtest_output=xml:cuda_tests/transpose_results.xml

echo "Running Reshape tests..."
./build/transform_tests --gtest_filter="ReshapeTest.*" --gtest_output=xml:cuda_tests/reshape_results.xml

echo "Running Atomic Sum tests..."
./build/transform_tests --gtest_filter="AtomicSumTest.*" --gtest_output=xml:cuda_tests/atomic_sum_results.xml

echo "Running Transform benchmarks..."
./build/transform_tests --gtest_filter="*Benchmark*" --gtest_output=xml:cuda_tests/transform_benchmark.xml

echo "Transform & Reduction tests completed! Results in cuda_outputs/"