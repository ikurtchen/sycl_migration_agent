#!/bin/bash

# Build and test script for SYCL PolicyMap kernel

set -e

echo "========================================"
echo "  Building SYCL PolicyMap Test"
echo "========================================"

# Create build directory
mkdir -p build
cd build

# Configure with CMake
echo "Configuring with CMake..."
if command -v icx &> /dev/null; then
    # Use Intel oneAPI DPC++ compiler if available
    cmake -DCMAKE_CXX_COMPILER=icx ..
elif command -v clang++ &> /dev/null; then
    # Use clang with SYCL support
    cmake -DCMAKE_CXX_COMPILER=clang++ ..
else
    # Default
    cmake ..
fi

# Build
echo "Building..."
make -j$(nproc)

echo "========================================"
echo "  Running SYCL PolicyMap Test"
echo "========================================"

# Run the test
./test_policy_map

echo "========================================"
echo "  Test Complete"
echo "========================================"