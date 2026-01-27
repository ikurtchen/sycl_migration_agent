# SYCL VectorAdd Build Instructions

This directory contains the SYCL/DPC++ implementation of the vector addition kernel, translated from CUDA for Intel GPU execution.

## Prerequisites

- Intel oneAPI Base Toolkit with DPC++/C++ Compiler
- OR any SYCL-compliant compiler (with appropriate modifications)
- CMake 3.20 or higher

## Build Instructions

### Standard Build (for Intel GPU)

```bash
# Create build directory
mkdir build
cd build

# Configure with CMake
cmake ..

# Build the project
make

# Run the executable
./vectorAdd_sycl
```

### Build Options

#### 1. Enable Intel GPU B60 Optimizations
```bash
cmake .. -DENABLE_B60_OPTIMIZATIONS=ON
```
This enables:
- Device-specific optimizations for Intel GPU B60 (device ID: 8800)
- Optimized work-group size (256) and vectorization (4)

#### 2. Enable CPU Fallback
```bash
cmake .. -DENABLE_CPU_FALLBACK=ON  # Default: ON
```
Allows the application to fall back to CPU execution if GPU is not available.

#### 3. Enable Performance Profiling
```bash
cmake .. -DENABLE_PROFILING=ON
```
Enables verbose SYCL output and profiling instrumentation.

#### 4. Custom Device ID
```bash
cmake .. -DDEVICE_ID=8800  # For Intel GPU B60
# Use other device IDs for different Intel GPUs
```

#### 5. Debug Build
```bash
cmake .. -DCMAKE_BUILD_TYPE=Debug
```

### Different SYCL Targets

```bash
# For different Intel GPU architectures
cmake .. -DSYCL_TARGET=spir64_gen  # Intel GPUs (default)

# For CPU execution
cmake .. -DSYCL_TARGET=spir64 -DENABLE_CPU_FALLBACK=OFF

# For other SYCL implementations, you may need to adjust the compiler
export CXX=clang  # If using clang with SYCL support
cmake .. -DCMAKE_CXX_COMPILER=clang
```

## Build Targets

```bash
# Build the main executable
make vectorAdd_sycl

# Build tests (if Google Test is available and BUILD_TESTS=ON)
make vectorAdd_sycl_tests

# Run tests
ctest

# Clean all build artifacts
make clean-all

# Run the executable directly
make run

# Run as benchmark
make benchmark
```

## Install

```bash
# Install to system (requires sudo if installing to system directories)
sudo make install

# Install to custom prefix
cmake .. -DCMAKE_INSTALL_PREFIX=/path/to/install
make install
```

## Expected Output

When run successfully, the program should output:
```
[Vector addition of 50000 elements]
Running on: [Your GPU Name]
Platform: [Intel(R) oneAPI DPC++/C++ Build ID ...]
Copy input data from the host memory to the SYCL device
SYCL kernel launch with global range 50176 and work-group size 256
Copy output data from the SYCL device to the host memory
Test PASSED
Kernel execution time: X.XXXXXX seconds
Achieved memory bandwidth: X.XX GB/s
Done
```

## Troubleshooting

### Compiler Not Found
If you see errors about Intel DPC++ compiler not being found:
1. Install Intel oneAPI Base Toolkit
2. Source the environment: `source /opt/intel/oneapi/setvars.sh`
3. Or ensure the compiler is in your PATH

### SYCL Runtime Errors
- Ensure Intel GPU drivers are installed and up-to-date
- Check that the GPU is available: `sycl-ls` (from oneAPI)
- Try CPU fallback: `cmake .. -DENABLE_CPU_FALLBACK=ON`

### Build Errors
- Ensure CMake version is 3.20 or higher
- Check that all required compilers are available
- Verify SYCL support in your compiler

## Performance Note

The kernel includes optimizations for Intel GPU B60 including:
- Subgroup operations (size 16)
- Vectorized memory access (float4)
- Optimal work-group size (256 for B60)
- Memory bandwidth optimization

For maximum performance, use `-DENABLE_B60_OPTIMIZATIONS=ON` when building for Intel GPU B60.