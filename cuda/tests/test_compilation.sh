#!/bin/bash

# Quick compilation test script for CUDA tests
# This script tests if the architecture is set up correctly for compilation

echo "=== CUDA Test Compilation Test ==="
echo

# Check that all required files exist
echo "1. Checking required files..."
required_files=("test_vectorAdd.cpp" "vectorAdd_kernel.cu" "vectorAdd_kernel.h" "CMakeLists.txt")
for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo "   ✓ $file found"
    else
        echo "   ✗ $file missing"
        exit 1
    fi
done

echo

# Check that nvcc is available
echo "2. Checking CUDA compiler..."
if command -v nvcc &> /dev/null; then
    echo "   ✓ nvcc found: $(nvcc --version | head -1)"
else
    echo "   ✗ nvcc not found - CUDA toolkit not installed or not in PATH"
    exit 1
fi

echo

# Check syntax of key files without full compilation
echo "3. Performing syntax checks..."

# Check C++ test file syntax
echo "   Checking test_vectorAdd.cpp syntax..."
if g++ -std=c++14 -fsyntax-only -I/usr/include/gtest -I../src test_vectorAdd.cpp 2>/dev/null; then
    echo "   ✓ C++ syntax OK"
else
    echo "   ⚠ C++ syntax issues found (may be due to CUDA-specific code)"
fi

# Check CUDA file syntax
echo "   Checking vectorAdd_kernel.cu syntax..."
if nvcc -std=c++14 --expt-relaxed-constexpr -c vectorAdd_kernel.cu -o /dev/null 2>/dev/null; then
    echo "   ✓ CUDA syntax OK"
else
    echo "   ⚠ CUDA syntax issues found"
fi

echo

# Check key architectural patterns
echo "4. Checking architectural patterns..."

# Verify CUDA kernel is in .cu file
if grep -q "__global__ void vectorAdd" vectorAdd_kernel.cu; then
    echo "   ✓ CUDA kernel found in .cu file"
else
    echo "   ✗ CUDA kernel not found in .cu file"
fi

# Verify launch function exists
if grep -q "cudaError_t launchVectorAdd" vectorAdd_kernel.cu; then
    echo "   ✓ Launch function found in .cu file"
else
    echo "   ✗ Launch function not found in .cu file"
fi

# Verify header declarations
if grep -q "__global__ void vectorAdd" vectorAdd_kernel.h && grep -q "cudaError_t launchVectorAdd" vectorAdd_kernel.h; then
    echo "   ✓ Header has both function declarations"
else
    echo "   ✗ Header incomplete"
fi

# Verify test file includes header
if grep -q '#include "vectorAdd_kernel.h"' test_vectorAdd.cpp; then
    echo "   ✓ Test file includes header"
else
    echo "   ✗ Test file missing header include"
fi

# Verify test calls launch function
if grep -q "launchVectorAdd" test_vectorAdd.cpp && ! grep -q "vectorAdd<<<" test_vectorAdd.cpp; then
    echo "   ✓ Test calls launch function (not kernel directly)"
else
    echo "   ✗ Test incorrectly calls kernel directly"
fi

# Verify no direct kernel calls from C++
if ! grep -q "<<<" test_vectorAdd.cpp; then
    echo "   ✓ No direct kernel launches from C++ code"
else
    echo "   ✗ Direct kernel launches found in C++ code"
fi

echo

# Check for common compilation fixes
echo "5. Checking for common compilation fixes..."

# Check if ASSERT statements are properly handled in return functions
if grep -A5 -B5 "std::pair.*benchmarkVectorAdd" test_vectorAdd.cpp | grep -q "EXPECT"; then
    echo "   ✓ RETURNING function uses EXPECT instead of ASSERT"
else
    echo "   ⚠ RETURNING function may still use ASSERT (compilation risk)"
fi

# Check for proper includes
if grep -q "#include <string>" test_vectorAdd.cpp; then
    echo "   ✓ String header included"
else
    echo "   ⚠ String header may be missing"
fi

echo

echo "=== Compilation Test Complete ==="
echo
echo "Next steps:"
echo "1. Run: mkdir -p build && cd build"
echo "2. Run: cmake .. && make"
echo "3. Run: ./vectorAdd_test"
echo
echo "If compilation fails, verify:"
echo "- CUDA toolkit is properly installed"
echo "- GTest is installed (libgtest-dev)"
echo "- Helper_cuda.h is accessible in ../src/"