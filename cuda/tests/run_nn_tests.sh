#!/bin/bash

# Neural Network Kernel Test Runner

set -e

echo "=== Running Neural Network Kernel Tests ==="
mkdir -p cuda_outputs

# Build and run NN tests
rm -rf build
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make nn_tests
cd ..

echo "Running Layer Norm tests..."
./build/nn_tests --gtest_filter="Fp16LayerNormTest.*" --gtest_output=xml:cuda_tests/layer_norm_results.xml

echo "Running Policy Head tests..."
./build/nn_tests --gtest_filter="PolicyHeadTest.*" --gtest_output=xml:cuda_tests/policy_head_results.xml

echo "Running Value Head tests..."
./build/nn_tests --gtest_filter="ValueHeadTest.*" --gtest_output=xml:cuda_tests/value_head_results.xml

echo "Running Winograd Transform tests..."
./build/nn_tests --gtest_filter="WinogradTransformTest.*" --gtest_output=xml:cuda_tests/winograd_transform_results.xml

echo "Running NN benchmarks..."
./build/nn_tests --gtest_filter="*Benchmark*" --gtest_output=xml:cuda_tests/nn_benchmark.xml

echo "Neural Network tests completed! Results in cuda_outputs/"