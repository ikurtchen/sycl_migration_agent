#!/bin/bash

# Build and test script for addVectorsHNC_NHC kernel

echo "=========================================="
echo "Testing addVectorsHNC_NHC kernel migration"
echo "=========================================="

# Set the SYCL migration agent directory
DIR="/localdisk/kurt/workspace/code/ai_coding/sycl_migration_agent"
CDIR="$DIR/sycl"

echo "Working directory: $CDIR"

# Create build directory
mkdir -p "$CDIR/build_test_hnc"
cd "$CDIR/build_test_hnc"

# Configure with cmake
echo "Configuring with CMake..."
cp "$CDIR/test_addVectorsHNC_NHC_CMakeLists.txt" ./CMakeLists.txt
cmake . -DCMAKE_BUILD_TYPE=Debug

# Build
echo "Building..."
if make -j$(nproc); then
    echo "Build successful!"
else
    echo "Build failed!"
    exit 1
fi

# Run tests
echo -e "\nRunning test..."
if ./bin/test_addVectorsHNC_NHC; then
    echo -e "\nTest PASSED!"
else
    echo -e "\nTest FAILED!"
    exit 1
fi

echo -e "\n=========================================="
echo "Test completed successfully!"
echo "=========================================="

# Clean up
cd "$DIR"
# rm -rf "$CDIR/build_test_hnc"