#!/bin/bash

# FP16 Kernel Test Runner

set -e

echo "=== Running FP16 Kernel Tests ==="
mkdir -p cuda_outputs

# Build and run FP16 tests
rm -rf build
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make fp16_tests
cd ..

echo "Running SE Layer NHWC tests..."
./build/fp16_tests --gtest_filter="SELayerNhwcTest.*" --gtest_output=xml:cuda_outputs/se_layer_nhwc_results.xml

echo "Running Output Input Transform tests..."
./build/fp16_tests --gtest_filter="OutputInputTransformTest.*" --gtest_output=xml:cuda_tests/output_input_transform_results.xml

echo "Running FP16 benchmarks..."
./build/fp16_tests --gtest_filter="*Benchmark*" --gtest_output=xml:cuda_tests/fp16_benchmark.xml

echo "FP16 tests completed! Results in cuda_outputs/"