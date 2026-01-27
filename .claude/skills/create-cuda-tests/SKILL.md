---
name: create-cuda-tests
description: "generate comprehensive Google Test suites for CUDA kernels with input/output serialization and benchmarking"
---

# create-cuda-tests

Generates comprehensive Google Test suites for CUDA kernels with input/output serialization and benchmarking.

## Description

This skill creates production-ready unit tests for CUDA kernels that validate correctness, save outputs for SYCL comparison, and measure performance metrics. Tests follow Google Test best practices and include proper resource management.

## Usage

```bash
python create_cuda_tests.py <cuda_file> <kernel_name> [options]
```

### Arguments

- `cuda_file`: Path to CUDA source file containing the kernel
- `kernel_name`: Name of the kernel to test
- `--output-dir`: Directory for generated test files (default: ./tests)
- `--input-sizes`: Comma-separated list of test sizes (e.g., "256,1024,4096")
- `--save-inputs`: Save input data for SYCL comparison
- `--benchmark`: Include benchmark tests
- `--verify-correctness`: Include CPU reference implementation

### Examples

```bash
# Basic test generation
python create_cuda_tests.py matmul.cu matrixMul

# With multiple input sizes
python create_cuda_tests.py matmul.cu matrixMul --input-sizes "256,1024,4096"

# Full test suite with benchmarks
python create_cuda_tests.py matmul.cu matrixMul \
    --save-inputs \
    --benchmark \
    --verify-correctness \
    --output-dir ./cuda/tests
```

## Generated Test Structure

### Test File Layout

```cpp
// test_matrixMul.cpp
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <fstream>
#include <vector>
#include <chrono>
#include <random>
#include <cmath>

// Test fixture class
class MatrixMulTest : public ::testing::Test {
protected:
    // Setup: Allocate and initialize data
    void SetUp() override;

    // Cleanup: Free resources
    void TearDown() override;

    // Helper methods
    void initializeInputs();
    void saveInputs(const std::string& prefix);
    void saveOutputs(const std::string& filename);
    void cpuReference();
    bool verifyResults(float tolerance = 1e-5);

    // Test parameters
    int N;
    size_t size_bytes;

    // Host arrays
    std::vector<float> h_A, h_B, h_C;
    std::vector<float> h_C_ref;  // CPU reference

    // Device arrays
    float *d_A, *d_B, *d_C;
};

// Test fixture implementation
void MatrixMulTest::SetUp() {
    N = 1024;  // Default size
    size_bytes = N * N * sizeof(float);

    // Allocate host memory
    h_A.resize(N * N);
    h_B.resize(N * N);
    h_C.resize(N * N);
    h_C_ref.resize(N * N);

    // Initialize with random data
    initializeInputs();

    // Allocate device memory
    cudaMalloc(&d_A, size_bytes);
    cudaMalloc(&d_B, size_bytes);
    cudaMalloc(&d_C, size_bytes);

    // Copy inputs to device
    cudaMemcpy(d_A, h_A.data(), size_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), size_bytes, cudaMemcpyHostToDevice);
}

void MatrixMulTest::TearDown() {
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

void MatrixMulTest::initializeInputs() {
    std::random_device rd;
    std::mt19937 gen(42);  // Fixed seed for reproducibility
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (int i = 0; i < N * N; i++) {
        h_A[i] = dist(gen);
        h_B[i] = dist(gen);
    }
}

void MatrixMulTest::saveInputs(const std::string& prefix) {
    std::ofstream file_A(prefix + "_input_A.bin", std::ios::binary);
    file_A.write(reinterpret_cast<const char*>(h_A.data()), size_bytes);
    file_A.close();

    std::ofstream file_B(prefix + "_input_B.bin", std::ios::binary);
    file_B.write(reinterpret_cast<const char*>(h_B.data()), size_bytes);
    file_B.close();
}

void MatrixMulTest::saveOutputs(const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    file.write(reinterpret_cast<const char*>(h_C.data()), size_bytes);
    file.close();
}

void MatrixMulTest::cpuReference() {
    // Simple CPU matrix multiplication for verification
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += h_A[i * N + k] * h_B[k * N + j];
            }
            h_C_ref[i * N + j] = sum;
        }
    }
}

bool MatrixMulTest::verifyResults(float tolerance) {
    int mismatches = 0;
    float max_error = 0.0f;

    for (size_t i = 0; i < h_C.size(); i++) {
        float error = std::abs(h_C[i] - h_C_ref[i]);
        max_error = std::max(max_error, error);

        if (error > tolerance) {
            mismatches++;
            if (mismatches <= 10) {  // Report first 10 mismatches
                std::cout << "Mismatch at index " << i 
                         << ": GPU=" << h_C[i]
                         << ", CPU=" << h_C_ref[i]
                         << ", error=" << error << std::endl;
            }
        }
    }

    std::cout << "Max error: " << max_error << std::endl;
    std::cout << "Mismatches: " << mismatches << " / " << h_C.size() << std::endl;

    return mismatches == 0;
}

// ============================================================================
// Test Cases
// ============================================================================

TEST_F(MatrixMulTest, BasicExecution) {
    // Test that kernel executes without errors
    launchMatrixMul(d_C, d_A, d_B, N);

    cudaError_t err = cudaGetLastError();
    ASSERT_EQ(err, cudaSuccess) << "Kernel launch failed: " 
                                << cudaGetErrorString(err);

    err = cudaDeviceSynchronize();
    ASSERT_EQ(err, cudaSuccess) << "Kernel execution failed: "
                                << cudaGetErrorString(err);
}

TEST_F(MatrixMulTest, CorrectnessSmallSize) {
    // Test with small size for quick verification
    N = 256;
    TearDown();
    SetUp();

    // Run kernel
    launchMatrixMul(d_C, d_A, d_B, N);
    cudaDeviceSynchronize();

    // Copy result
    cudaMemcpy(h_C.data(), d_C, size_bytes, cudaMemcpyDeviceToHost);

    // Compute CPU reference
    cpuReference();

    // Verify
    EXPECT_TRUE(verifyResults(1e-4f));
}

TEST_F(MatrixMulTest, SaveForComparison) {
    // Save inputs
    saveInputs("cuda_outputs/matrixMul");

    // Run kernel
    launchMatrixMul(d_C, d_A, d_B, N);
    cudaDeviceSynchronize();

    // Copy and save outputs
    cudaMemcpy(h_C.data(), d_C, size_bytes, cudaMemcpyDeviceToHost);
    saveOutputs("cuda_outputs/matrixMul_output.bin");

    // Save metadata
    std::ofstream meta("cuda_outputs/matrixMul_meta.json");
    meta << "{\"N\":" << N 
         << ",\"dtype\":\"float32\""
         << ",\"shape\":[" << N << "," << N << "]"
         << "}" << std::endl;
    meta.close();

    SUCCEED();
}

TEST_F(MatrixMulTest, Benchmark) {
    // Warmup runs
    for (int i = 0; i < 5; i++) {
        launchMatrixMul(d_C, d_A, d_B, N);
    }
    cudaDeviceSynchronize();

    // Benchmark runs
    const int num_iterations = 100;
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_iterations; i++) {
        launchMatrixMul(d_C, d_A, d_B, N);
    }
    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // Calculate metrics
    double avg_time_ms = duration.count() / (1000.0 * num_iterations);
    double flops = 2.0 * N * N * N;  // Matrix multiply FLOPs
    double gflops = flops / (avg_time_ms * 1e6);

    // Memory bandwidth
    double bytes = 3.0 * N * N * sizeof(float);  // Read A, B; Write C
    double bandwidth_gbps = bytes / (avg_time_ms * 1e6);

    std::cout << "======================================" << std::endl;
    std::cout << "Benchmark Results (N=" << N << ")" << std::endl;
    std::cout << "======================================" << std::endl;
    std::cout << "Average time: " << avg_time_ms << " ms" << std::endl;
    std::cout << "Throughput: " << gflops << " GFLOPS" << std::endl;
    std::cout << "Memory bandwidth: " << bandwidth_gbps << " GB/s" << std::endl;

    // Save benchmark results
    std::ofstream bench("cuda_outputs/matrixMul_benchmark.json");
    bench << "{"
          << "\"kernel\":\"matrixMul\","
          << "\"N\":" << N << ","
          << "\"time_ms\":" << avg_time_ms << ","
          << "\"gflops\":" << gflops << ","
          << "\"bandwidth_gbps\":" << bandwidth_gbps << ","
          << "\"grid\":[" << grid.x << "," << grid.y << "," << grid.z << "],"
          << "\"block\":[" << block.x << "," << block.y << "," << block.z << "]"
          << "}" << std::endl;
    bench.close();

    // Optional: Set performance expectations
    // EXPECT_GT(gflops, 1000.0) << "Performance below 1 TFLOPS";
}

// Parameterized test for multiple sizes
class MatrixMulParameterizedTest : public ::testing::TestWithParam<int> {
protected:
    void SetUp() override {
        N = GetParam();
        // ... similar setup
    }

    int N;
    // ... same members as MatrixMulTest
};

TEST_P(MatrixMulParameterizedTest, VariousSizes) {
    launchMatrixMul(d_C, d_A, d_B, N);
    cudaDeviceSynchronize();

    SUCCEED() << "Completed test for N=" << N;
}

INSTANTIATE_TEST_SUITE_P(
    MultipleInputSizes,
    MatrixMulParameterizedTest,
    ::testing::Values(256, 512, 1024, 2048, 4096)
);

// Main function
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    // Create output directory
    system("mkdir -p cuda_outputs");

    return RUN_ALL_TESTS();
}
```

## Features

### 1. Resource Management
- RAII pattern with SetUp/TearDown
- Automatic cleanup on test failure
- Memory leak detection

### 2. Input/Output Serialization
- Binary format for efficient storage
- Reproducible random initialization
- Metadata JSON for validation tools

### 3. CPU Reference Implementation
- Optional correctness verification
- Configurable tolerance
- Detailed mismatch reporting

### 4. Benchmarking
- Warmup iterations
- Multiple measurements for accuracy
- Performance metrics calculation
- JSON output for analysis tools

### 5. Parameterized Testing
- Test multiple input sizes
- Grid/block configuration variations
- Data type testing (float, double)

## Advanced Test Patterns

### Memory Pattern Testing

```cpp
TEST_F(MatrixMulTest, CoalescedAccess) {
    // Verify memory access patterns are coalesced
    // Use CUDA profiler API if available
}

TEST_F(MatrixMulTest, BankConflictFree) {
    // Check for shared memory bank conflicts
}
```

### Error Injection Testing

```cpp
TEST_F(MatrixMulTest, InvalidInputSizes) {
    N = 0;

    launchMatrixMul(d_C, d_A, d_B, N);

    // Should handle gracefully or return error
}
```

### Stress Testing

```cpp
TEST_F(MatrixMulTest, LargeInput) {
    N = 8192;  // Large size
    TearDown();
    SetUp();

    // Test memory allocation and execution
}
```

## Integration with CI/CD

```bash
# Run tests with output
./cuda_tests --gtest_output=json:test_results.json

# Run only benchmark tests
./cuda_tests --gtest_filter="*Benchmark*"

# Run with specific device
CUDA_VISIBLE_DEVICES=0 ./cuda_tests
```

## Output Files Generated

```
cuda_outputs/
├── matrixMul_input_A.bin          # Input matrix A
├── matrixMul_input_B.bin          # Input matrix B
├── matrixMul_output.bin           # Kernel output
├── matrixMul_meta.json            # Test metadata
└── matrixMul_benchmark.json       # Performance metrics
```

## Customization Options

### Test Configuration File

```json
{
  "kernel": "matrixMul",
  "test_sizes": [256, 512, 1024, 2048, 4096],
  "verify_correctness": true,
  "tolerance": 1e-5,
  "benchmark_iterations": 100,
  "warmup_iterations": 5,
  "save_inputs": true,
  "save_outputs": true,
  "grid_configs": [
    {"block": [16, 16], "name": "default"},
    {"block": [32, 32], "name": "large_block"}
  ]
}
```

## Related Skills

- `scan-cuda-repo`: Identifies kernels to test
- `generate-cmake`: Creates build system for tests
- `compare-numerical-results`: Uses saved outputs for validation
- `translate-cuda-to-sycl`: SYCL tests mirror CUDA tests

## Skill Metadata

```yaml
name: create-cuda-tests
version: 1.0.0
category: testing
dependencies:
  - python >= 3.8
  - CUDA Toolkit >= 11.0
  - Google Test >= 1.12
inputs:
  - cuda_file: string
  - kernel_name: string
  - options: dict (optional)
outputs:
  - test_file: C++ source file
  - status: success|error
execution_time: 1-3 seconds
```
