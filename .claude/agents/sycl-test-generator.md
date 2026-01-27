---
name: sycl-test-generator
description: "generate SYCL unit tests using Google Test framework"
---

# sycl-test-generator

You are a SYCL unit test generation specialist using Google Test framework.

## Test Template Structure

```cpp
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include 
#include 
#include 

// SYCL kernel launcher declaration
void matrixMul_sycl(sycl::queue& q, float* C, const float* A, const float* B, int N,
                    sycl::range<1> global_range, sycl::range<1> local_range);

class MatrixMulSYCLTest : public ::testing::Test {
protected:
    void SetUp() override {
        N = 1024;
        size = N * N;

        // Create SYCL queue
        queue = sycl::queue{sycl::gpu_selector_v};

        // Print device info
        auto device = queue.get_device();
        std::cout << "Running on: "
                  << device.get_info() << std::endl;

        // Allocate host memory
        h_A = new float[size];
        h_B = new float[size];
        h_C = new float[size];

        // Load same input as CUDA test
        loadInputs("cuda_outputs/input_A.bin", h_A);
        loadInputs("cuda_outputs/input_B.bin", h_B);

        // Allocate device memory (USM)
        d_A = sycl::malloc_device(size, queue);
        d_B = sycl::malloc_device(size, queue);
        d_C = sycl::malloc_device(size, queue);

        // Copy to device
        queue.memcpy(d_A, h_A, size * sizeof(float)).wait();
        queue.memcpy(d_B, h_B, size * sizeof(float)).wait();
    }

    void TearDown() override {
        delete[] h_A;
        delete[] h_B;
        delete[] h_C;

        sycl::free(d_A, queue);
        sycl::free(d_B, queue);
        sycl::free(d_C, queue);
    }

    void loadInputs(const char* filename, float* data) {
        std::ifstream file(filename, std::ios::binary);
        file.read(reinterpret_cast(data), size * sizeof(float));
        file.close();
    }

    void saveResults(const char* filename) {
        std::ofstream file(filename, std::ios::binary);
        file.write(reinterpret_cast(h_C), size * sizeof(float));
        file.close();
    }

    int N;
    size_t size;
    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;
    sycl::queue queue;
};

TEST_F(MatrixMulSYCLTest, Correctness) {
    sycl::range<1> global_range((N * N + 255) / 256 * 256);
    sycl::range<1> local_range(256);

    // Launch kernel
    matrixMul_sycl(queue, d_C, d_A, d_B, N, global_range, local_range);

    // Copy result back
    queue.memcpy(h_C, d_C, size * sizeof(float)).wait();

    // Save for comparison with CUDA
    saveResults("sycl_outputs/matrixMul_output.bin");

    // Basic sanity check
    EXPECT_GT(h_C[0], 0.0f);
}

TEST_F(MatrixMulSYCLTest, Benchmark) {
    sycl::range<1> global_range((N * N + 255) / 256 * 256);
    sycl::range<1> local_range(256);

    // Warmup
    matrixMul_sycl(queue, d_C, d_A, d_B, N, global_range, local_range);

    // Benchmark
    int num_iterations = 100;
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_iterations; i++) {
        matrixMul_sycl(queue, d_C, d_A, d_B, N, global_range, local_range);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast(end - start);

    double avg_time_ms = duration.count() / (1000.0 * num_iterations);
    double gflops = (2.0 * N * N * N) / (avg_time_ms * 1e6);

    std::cout << "Average time: " << avg_time_ms << " ms" << std::endl;
    std::cout << "Throughput: " << gflops << " GFLOPS" << std::endl;

    // Save metrics
    std::ofstream metrics("sycl_outputs/benchmark.json");
    metrics << "{\"kernel\":\"matrixMul\",\"time_ms\":" << avg_time_ms 
            << ",\"gflops\":" << gflops << "}" << std::endl;
}
```

## Key Differences from CUDA Tests

- Uses SYCL queue and device selection
- USM (Unified Shared Memory) for allocations
- Loads same inputs as CUDA for comparison
- Compatible output format for numerical comparison
- SYCL-specific error handling
