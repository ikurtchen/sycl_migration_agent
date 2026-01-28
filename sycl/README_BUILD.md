# SYCL Kernel Build System

This directory contains the CMake build system for the SYCL kernels migrated from CUDA.

## Build System Components

### Files
- **CMakeLists.txt** - Main CMake configuration file
- **build_sycl.sh** - Build script for easy compilation
- **run_sycl_tests.sh** - Test execution script
- **intel_gpu_toolchain.cmake** - Toolchain file for Intel GPU targeting
- **README_BUILD.md** - This documentation

## Prerequisites

### Intel oneAPI Base Toolkit
Download and install from: https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html

### Required Components
- Intel DPC++/C++ Compiler
- SYCL runtime
- OpenCL runtime (for GPU execution)

## Building the Project

### Option 1: Using the Build Script (Recommended)

```bash
# Basic build
./build_sycl.sh

# Debug build
./build_sycl.sh --debug

# Clean build
./build_sycl.sh --clean

# Build and run tests
./build_sycl.sh --run-tests

# Specify build directory
./build_sycl.sh --build-dir my_build
```

### Option 2: Using CMake Directly

```bash
# Create build directory
mkdir build && cd build

# Setup Intel oneAPI environment (if needed)
source /opt/intel/oneapi/setvars.sh

# Configure
cmake -DCMAKE_BUILD_TYPE=Release \
      -DSYCL_ARCH_INTEL_GPU=ON \
      -DSYCL_DEBUG=OFF \
      ..

# Build
make -j$(nproc)
```

### Option 3: Using Toolchain File

```bash
mkdir build && cd build

cmake -DCMAKE_TOOLCHAIN_FILE=../intel_gpu_toolchain.cmake \
      -DCMAKE_BUILD_TYPE=Release \
      ..

make -j$(nproc)
```

## Build Targets

The CMake system creates the following targets:

### Library
- `sycl_kernels` - Static library containing all kernel implementations

### Test Executables
- `test_addVectors` - Tests for vector addition kernels
- `test_policy_map` - Tests for policy mapping kernels
- `test_se_layer_nhwc` - Tests for Squeeze-and-Excitation layer kernels
- `test_globalAvgPool` - Tests for global average pooling kernels
- `test_softmax` - Tests for softmax kernels

### Custom Targets
- `run_addVectors` - Run addVectors tests
- `run_policy_map` - Run policy map tests
- `run_se_layer_nhwc` - Run SE Layer tests
- `run_globalAvgPool` - Run global average pool tests
- `run_softmax` - Run softmax tests
- `run_all_tests` - Run all tests via CTest

## Running Tests

### Option 1: Using Test Script
```bash
./run_sycl_tests.sh
```

### Option 2: Using CTest
```bash
cd build
ctest --verbose
```

### Option 3: Running Individual Tests
```bash
cd build
./test_addVectors
./test_policy_map
# etc.
```

## Build Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `CMAKE_BUILD_TYPE` | Release | Debug or Release build |
| `SYCL_ARCH_INTEL_GPU` | ON | Enable Intel GPU compilation |
| `SYCL_DEBUG` | OFF | Enable debug mode |

## Intel GPU Targets

The build system supports various Intel GPU architectures:

- **Generic**: `spir64_gen` (default)
- **Data Center GPU Max**: `spir64_gen-unknown-linux`
- **Intel Arc GPUs**: `spir64_gen-unknown-linux-sycldevice`

To specify a target:
```bash
cmake -DINTEL_GPU_TARGET=spir64_gen-unknown-linux ..
```

## Environment Variables

- `SYCL_COMPILER` - Path to SYCL compiler (overrides auto-detection)
- `ONEAPI_ROOT` - Path to Intel oneAPI installation
- `DPCPP_ROOT` - Path to DPC++ installation

## Example Build Commands

### Production Build
```bash
./build_sycl.sh --clean
```

### Debug Build with Full Info
```bash
./build_sycl.sh --debug --clean --run-tests
```

### Build for Specific GPU Architecture
```bash
mkdir build && cd build
cmake -DINTEL_GPU_TARGET=spir64_gen-unknown-linux \
      -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

## Troubleshooting

### SYCL Compiler Not Found
- Ensure Intel oneAPI Base Toolkit is installed
- Run `source /opt/intel/oneapi/setvars.sh` before building
- Or set `SYCL_COMPILER` environment variable

### GPU Execution Issues
- Verify GPU drivers are installed
- Check OpenCL runtime is available
- Use `sycl-ls` to list available devices

 Linker Errors
- Ensure Intel oneAPI environment is set up
- Check that SYCL flags are properly applied
- Verify architecture targeting is correct

## Performance Notes

- Use Release builds (`-O3`) for performance testing
- Intel GPU targeting (`-fsycl-targets=spir64_gen`) enables GPU execution
- Consider using `-fsycl-targets=spir64_gen-unknown-linux` for specific GPU models

## Installation

The build system supports installation with:
```bash
make install
```

This installs:
- Library to `<prefix>/lib`
- Headers to `<prefix>/include/sycl/neural/backends/sycl`

Default prefix is `/usr/local`, can be changed with:
```bash
cmake -DCMAKE_INSTALL_PREFIX=/custom/path ..
```