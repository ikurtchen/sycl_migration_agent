#!/bin/bash

# Math Kernel Test Runner

set -e

echo "=== Running Math Kernel Tests ==="
mkdir -p cuda_outputs

# Build and run math tests
rm -rf build
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make math_tests
cd ..

echo "Running Add Vectors tests..."
./build/math_tests --gtest_filter="AddVectorsTest.*" --gtest_output=xml:cuda_tests/add_vectors_results.xml

echo "Running Add Bias tests..."
./build/math_tests --gtest_filter="AddBiasTest.*" --gtest_output=xml:cuda_tests/add_bias_results.xml

echo "Running Batch Norm tests..."
./build/math_tests --gtest_filter="BatchNormTest.*" --gtest_output=xml:cuda_tests/batch_norm_results.xml

echo "Running Softmax tests..."
./build/math_tests --gtest_filter="SoftmaxTest.*" --gtest_output=xml:cuda_tests/softmax_results.xml

echo "Running Reduce Sum tests..."
./build/math_tests --gtest_filter="ReduceSumTest.*" --gtest_output=xml:cuda_tests/reduce_sum_results.xml

echo "Running Math benchmarks..."
./build/math_tests --gtest_filter="*Benchmark*" --gtest_output=xml:cuda_tests/math_benchmark.xml

echo "Math tests completed! Results in cuda_outputs/"