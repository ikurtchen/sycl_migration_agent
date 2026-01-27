# CUDA VectorAdd Test Suite

Comprehensive Google Test unit tests for the CUDA vectorAdd kernel.

## Overview

This test suite provides thorough testing for the `vectorAdd` CUDA kernel with:
- Multiple vector sizes and edge cases
- Benchmark performance testing
- Input/output serialization for SYCL comparison
- Automated test execution and analysis

## Files

### Test Files
- `test_vectorAdd.cpp` - Main test suite with Google Test cases
- `vectorAdd_kernel.cu` - CUDA kernel implementation
- `vectorAdd_kernel.h` - Header file with kernel declaration

### Build Files
- `CMakeLists.txt` - CMake build configuration
- `Makefile` - Alternative build system

### Scripts
- `run_tests.sh` - Automated test execution script
- `analyze_results.py` - Test results analysis tool

## Tests

### Correctness Tests
1. **SmallSize_1000** - Tests with 1,000 elements
2. **MediumSize_50000** - Tests with 50,000 elements (original default)
3. **LargeSize_1000000** - Tests with 1 million elements
4. **VeryLargeSize_10000000** - Tests with 10 million elements
5. **SingleElement** - Edge case with single element
6. **PowerOf2Boundary** - Tests at block size boundary (256 elements)
7. **NegativeValues** - Tests with negative input values
8. **ZeroValues** - Tests with zero inputs
9. **MaximumValues** - Tests with maximum float values

### Benchmark Tests
Performance testing for different vector sizes:
- `Benchmark_Small` - 1,000 elements
- `Benchmark_Medium` - 50,000 elements
- `Benchmark_Large` - 1,000,000 elements
- `Benchmark_VeryLarge` - 10,000,000 elements

## Building and Running

### Prerequisites
- CUDA toolkit
- Google Test framework
- CMake 3.10+ (or use Makefile)

### Using CMake
```bash
mkdir build
cd build
cmake ..
make
./vectorAdd_test
```

### Using the test script
```bash
./run_tests.sh
```

### Using Make
```bash
make test
```

### Installing Dependencies
```bash
make install-deps
```

## Output Files

### Test Outputs
- `cuda_outputs/vectorAdd_{test}_output.bin` - Binary output data
- `cuda_outputs/vectorAdd_benchmark_{size}.json` - Benchmark metrics

### Test Inputs
- `cuda_inputs/vectorAdd_{test}_input_A.bin` - Input vector A
- `cuda_inputs/vectorAdd_{test}_input_B.bin` - Input vector B

### Logs
- `test_logs/vectorAdd_test_{timestamp}.log` - Test execution log
- `test_logs/vectorAdd_test_report_{timestamp}.txt` - Test summary report

## Analysis

### Run Analysis
```bash
python3 analyze_results.py --summary
python3 analyze_results.py -o report.json
```

### Benchmark Metrics
- **Execution Time** - Average kernel execution time in milliseconds
- **GFLOPS** - Performance in Giga Floating Point Operations per Second
- **Throughput** - Elements processed per second

## Test Features

### Input/Output Serialization
- Binary format for data persistence
- Random seed controlled for reproducible tests
- Separate files for each test case

### Performance Characteristics
- Warmup runs to eliminate initialization overhead
- Multiple iterations for accurate timing
- Automatic GFLOPS calculation (2 FLOPs per element)

### Error Handling
- CUDA error checking for all operations
- Google Test assertions with detailed error messages
- Memory leak prevention with proper cleanup

## Usage for SYCL Comparison

The test suite generates:
1. Input data files for SYCL implementation
2. Output reference files for numerical comparison
3. Benchmark metrics for performance comparison

Use the serialized data in `cuda_inputs/` to feed SYCL tests and compare outputs against reference files in `cuda_outputs/`.

## Kernel Configuration

- **Thread Block Size**: 256 threads
- **Grid Size**: Calculated based on vector size
- **CUDA Compute Capability**: 6.0+ (adjust in CMakeLists.txt if needed)

## Troubleshooting

### Common Issues
1. **CUDA not found** - Ensure CUDA toolkit is installed and in PATH
2. **Google Test not found** - Install libgtest-dev or build from source
3. **GPU not available** - Check nvidia-smi and GPU drivers
4. **Permission issues** - Ensure write permissions for output directories

### Debug Mode
Build with debug symbols:
```bash
cmake .. -DCMAKE_BUILD_TYPE=Debug
make
```

### Verbose Output
Run with environment variable:
```bash
CUDA_LAUNCH_BLOCKING=1 ./vectorAdd_test
```