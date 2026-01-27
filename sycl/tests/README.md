# SYCL VectorAdd Test Suite

Comprehensive test suite for the SYCL vector addition kernel on Intel GPUs.

## Setup

### Prerequisites

1. Intel oneAPI DPC++/C++ Compiler
2. Google Test (gtest) framework
3. Intel GPU with Level-Zero support

### Environment Setup

```bash
# Source oneAPI environment
source /opt/intel/oneapi/setvars.sh

# Verify installation
icpx --version
sycl-ls
```

## Building and Running

### Quick Start

```bash
cd tests
./run_tests.sh
```

### Manual Build

```bash
cd tests
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
./vectorAdd_sycl_test
```

## Test Cases

The test suite includes the following test cases:

1. **SmallSize_1000** - Tests with 1,000 elements
2. **MediumSize_50000** - Tests with 50,000 elements
3. **LargeSize_1000000** - Tests with 1,000,000 elements
4. **VeryLargeSize_10000000** - Tests with 10,000,000 elements
5. **SingleElement** - Edge case: single element
6. **PowerOf2Boundary** - Edge case: power of 2 (256 elements)
7. **NegativeValues** - Tests with negative floating-point values
8. **ZeroValues** - Tests with zero values
9. **MaximumValues** - Tests with FLT_MAX (infinity handling)
10. **Benchmark_Small/Medium/Large/VeryLarge** - Performance benchmarks

## Output

### Generated Files

- **sycl_outputs/** - Binary output files from test runs
- **sycl_inputs/** - Binary input files (for reproducibility)
- **test_logs/** (optional) - Test execution logs

### Test Results

All tests verify numerical correctness with tolerance of 1e-5.

Benchmark results are saved as JSON files:
- `vectorAdd_benchmark_small.json`
- `vectorAdd_benchmark_medium.json`
- `vectorAdd_benchmark_large.json`
- `vectorAdd_benchmark_verylarge.json`

## Integration with Remote Execution

To run tests on remote GPU servers:

```bash
# Copy test files
rsync -avz tests/ user@gpu-server:/path/to/tests/

# SSH to server
ssh user@gpu-server

# Run tests
cd /path/to/tests
source /opt/intel/oneapi/setvars.sh
./run_tests.sh
```

## Troubleshooting

### "ocloc: Device name missing"

This error occurs when targeting Intel GPUs without proper device specification.
Solution: Use `spir64` target instead of `spir64_gen`.

### "No SYCL device found"

Ensure oneAPI environment is sourced and GPU drivers are installed:
```bash
source /opt/intel/oneapi/setvars.sh
sycl-ls
```

### Permission Denied

Make sure run_tests.sh is executable:
```bash
chmod +x run_tests.sh
```

## Performance Notes

- Benchmark iterations: 100
- Random seed: 42 (for reproducibility)
- Work-group size: 256 (optimal for Intel GPUs)
- Sub-group sizes: 16, 32 (Intel GPU architecture)

## License

Copyright (c) 2024. All rights reserved.