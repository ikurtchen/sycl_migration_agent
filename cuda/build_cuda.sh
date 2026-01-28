#!/usr/bin/env bash

# Build script for LC0 CUDA Kernels
# Usage: ./build_cuda.sh [build_type] [options]

set -e

# Default build type
BUILD_TYPE=${1:-Release}

# Move to script directory
CDPATH= cd -- "$(dirname -- "$0")"

# Create build directory
BUILD_DIR="build-${BUILD_TYPE,,}"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure with CMake
echo "Configuring LC0 CUDA Kernels build (type: $BUILD_TYPE)..."
cmake .. \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DBUILD_TESTS=ON \
    -DBUILD_BENCHMARKS=ON \
    "$@"

# Build
echo "Building LC0 CUDA Kernels..."
cmake --build . --parallel $(nproc)

# Run tests if in debug or relwithdebinfo mode
if [[ "$BUILD_TYPE" == "Debug" || "$BUILD_TYPE" == "RelWithDebInfo" ]]; then
    echo "Running tests..."
    ctest --output-on-failure
fi

echo "Build complete!"
echo "Library location: $(pwd)/liblc0_cuda_kernels.a"
echo "Test executable: $(pwd)/lc0_cuda_tests"