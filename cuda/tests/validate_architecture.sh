#!/bin/bash

# Architecture validation script for CUDA tests
# This script validates that the CUDA kernel architecture is properly separated

echo "=== CUDA Test Architecture Validation ==="
echo

# Check thatCUDA kernel file exists
echo "1. Checking CUDA kernel file..."
if [ -f "vectorAdd_kernel.cu" ]; then
    echo "   ✓ vectorAdd_kernel.cu exists"
else
    echo "   ✗ vectorAdd_kernel.cu missing"
    exit 1
fi

# Check that header file exists
echo "2. Checking header file..."
if [ -f "vectorAdd_kernel.h" ]; then
    echo "   ✓ vectorAdd_kernel.h exists"
else
    echo "   ✗ vectorAdd_kernel.h missing"
    exit 1
fi

# Check that test file exists
echo "3. Checking test file..."
if [ -f "test_vectorAdd.cpp" ]; then
    echo "   ✓ test_vectorAdd.cpp exists"
else
    echo "   ✗ test_vectorAdd.cpp missing"
    exit 1
fi

# Check that kernel file contains __global__ function
echo "4. Validating CUDA kernel..."
if grep -q "__global__ void vectorAdd" vectorAdd_kernel.cu; then
    echo "   ✓ CUDA kernel function found"
else
    echo "   ✗ CUDA kernel function missing"
    exit 1
fi

# Check that kernel file contains launch function
echo "5. Validating host launch function..."
if grep -q "cudaError_t launchVectorAdd" vectorAdd_kernel.cu; then
    echo "   ✓ Host launch function found"
else
    echo "   ✗ Host launch function missing"
    exit 1
fi

# Check that header declares both functions
echo "6. Validating header declarations..."
if grep -q "__global__ void vectorAdd" vectorAdd_kernel.h && grep -q "cudaError_t launchVectorAdd" vectorAdd_kernel.h; then
    echo "   ✓ Both functions declared in header"
else
    echo "   ✗ Missing function declarations in header"
    exit 1
fi

# Check that test file includes header
echo "7. Validating test file includes..."
if grep -q '#include "vectorAdd_kernel.h"' test_vectorAdd.cpp; then
    echo "   ✓ Header properly included in test file"
else
    echo "   ✗ Header not included in test file"
    exit 1
fi

# Check that test file calls launch function (not kernel directly)
echo "8. Validating function calls..."
if grep -q "launchVectorAdd" test_vectorAdd.cpp && ! grep -q "vectorAdd<<<" test_vectorAdd.cpp; then
    echo "   ✓ Test calls launch function (not kernel directly)"
else
    echo "   ✗ Test incorrectly calls kernel directly or missing launch calls"
    exit 1
fi

# Check that CMakeLists.txt includes both files
echo "9. Validating build configuration..."
if grep -q "test_vectorAdd.cpp" CMakeLists.txt && grep -q "vectorAdd_kernel.cu" CMakeLists.txt; then
    echo "   ✓ Both source files in CMakeLists.txt"
else
    echo "   ✗ Missing source files in CMakeLists.txt"
    exit 1
fi

echo
echo "=== All Architecture Checks Passed! ==="
echo
echo "Architecture Summary:"
echo "- CUDA kernel (__global__ vectorAdd) lives in .cu file"
echo "- Host launch function (launchVectorAdd) bridges C++ and CUDA"
echo "- Test code (.cpp) calls host function, not kernel directly"
echo "- Header file provides C++/CUDA interop declarations"
echo "- Build system properly compiles both .cu and .cpp files"
echo
echo "This architecture ensures:"
echo "- Clean separation of CUDA and C++ code"
echo "- Proper compilation with nvcc"
echo "- Maintenable code structure"
echo "- Compatibility with Google Test framework"
echo
echo "Ready to build and test!"