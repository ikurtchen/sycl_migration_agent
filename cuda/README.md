# LC0 CUDA Kernels Build System

This directory contains the CMake build configuration for compiling the CUDA kernels from the Leela Chess Zero project.

## Prerequisites

- CUDA Toolkit 11.0 or higher
- CMake 3.18 or higher
- C++17 compatible compiler (GCC 9+, Clang 10+, or MSVC 2019+)
- NVIDIA GPU with Compute Capability 7.0 or higher

## Dependencies

### Required
- **CUDA Toolkit**: Provides nvcc compiler and CUDA runtime libraries
- **cuBLAS**: NVIDIA's BLAS library for GPU-accelerated linear algebra

### Optional
- **cuDNN**: NVIDIA's deep learning library (automatically detected if available)
- **CUTLASS**: Template library for high-performance CUDA kernels (requires Compute Capability 8.0+)

## Build Instructions

### Quick Start

```bash
# Build release version
./build_cuda.sh Release

# Build debug version with tests
./build_cuda.sh Debug

# Custom CMake options
./build_cuda.sh Release -DCMAKE_CUDA_ARCHITECTURES="80;86"
```

### Manual CMake Build

```bash
# Create build directory
mkdir build && cd build

# Configure
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTS=ON \
    -DBUILD_BENCHMARKS=ON

# Build
cmake --build . --parallel $(nproc)

# Run tests
ctest --output-on-failure
```

## Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `CMAKE_BUILD_TYPE` | Release | Debug, Release, RelWithDebInfo, MinSizeRel |
| `CMAKE_CUDA_ARCHITECTURES` | 70;75;80;86 | Target GPU architectures |
| `BUILD_TESTS` | ON | Build unit tests with Google Test |
| `BUILD_BENCHMARKS` | OFF | Build performance benchmarks |
| `USE_CUTLASS` | ON | Enable CUTLASS kernels (if supported) |

## CUDA Architecture Support

The build system automatically targets multiple GPU architectures for compatibility:
- **70**: V100, T4
- **75**: RTX 20xx series
- **80**: A100, RTX 30xx series
- **86**: RTX 30xx series (enhanced)

To target specific architectures only:
```bash
cmake .. -DCMAKE_CUDA_ARCHITECTURES="80;86"
```

## Testing

### Running All Tests
```bash
# If built with tests
./build-release/lc0_cuda_tests

# Or via CTest
cd build-release
ctest --output-on-failure
```

### Test Coverage
The test suite includes:
- Kernel correctness validation
- Numerical precision verification
- Memory access pattern testing
- Performance regression tests

## Output Files

After building, you'll find:

| File | Location | Description |
|------|----------|-------------|
| `liblc0_cuda_kernels.a` | `build-*/` | Static library with all CUDA kernels |
| `lc0_cuda_tests` | `build-*/` | Test executable |
| `lc0_cuda_benchmarks` | `build-*/` | Performance benchmarks (if enabled) |

## Installation

```bash
# Install to system (requires sudo)
sudo cmake --install build-release

# Or install to custom prefix
cmake --install build-release --prefix /opt/lc0-cuda
```

## Integration with Other Projects

The installed package provides CMake config files:

```cmake
find_package(lc0_cuda_kernels REQUIRED)
target_link_libraries(my_target PRIVATE lc0_cuda::lc0_cuda_kernels)
```

## Performance Tips

1. **Native Architecture**: Build with `-DCMAKE_CUDA_ARCHITECTURES=native` for optimal performance on the build machine
2. **Optimization Flags**: Release build uses `--use_fast_math` and other optimizations
3. **Memory Coalescing**: Kernels are written to ensure coalesced memory access patterns
4. **Occupancy**: Launch bounds are set to optimize GPU occupancy

## Troubleshooting

### Common Issues

1. **CUDA not found**: Ensure CUDA Toolkit is installed and `nvcc` is in PATH
2. **Compute capability error**: Check your GPU supports the targeted architecture
3. **cuBLAS/cuDNN not found**: Install the CUDA libraries or update library paths

### Debug Information

Enable verbose build output:
```bash
cmake .. -DCMAKE_VERBOSE_MAKEFILE=ON
```

Check detected GPU capabilities:
```bash
nvidia-smi --query-gpu=compute_cap --format=csv
```

## License

This build configuration is part of the Leela Chess Zero project, licensed under GPL v3.0.