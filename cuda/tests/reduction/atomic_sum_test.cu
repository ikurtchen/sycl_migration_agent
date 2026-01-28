#include <gtest/gtest.h>
#include "../common/base_test_fixture.h"

// Test kernel for atomic sum reduction (single value)
__global__ void atomic_sum_single_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        atomicAdd(output, input[idx]);
    }
}

// Test kernel for atomic sum reduction with shared memory optimization
__global__ void atomic_sum_shared_kernel(const float* input, float* output, int size) {
    __shared__ float shared_sum[256];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // Load partial sums
    shared_sum[tid] = (idx < size) ? input[idx] : 0.0f;
    __syncthreads();

    // Reduce within block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }

    // Atomic add block result
    if (tid == 0) {
        atomicAdd(output, shared_sum[0]);
    }
}

// Test kernel for atomic sum reduction to multiple outputs (channel-wise)
__global__ void atomic_sum_channel_kernel(const float* input, float* output,
                                        int size, int channels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        int channel = idx % channels;
        atomicAdd(&output[channel], input[idx]);
    }
}

// Test kernel for atomic sum with multiple data types
__global__ void atomic_sum_double_kernel(const double* input, double* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        atomicAdd(output, input[idx]);
    }
}

// Atomic sum with complex pattern (strided access)
__global__ void atomic_sum_strided_kernel(const float* input, float* output,
                                         int size, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int actual_idx = idx * stride;

    if (actual_idx < size) {
        atomicAdd(output, input[actual_idx]);
    }
}

void launchAtomicSum(const float* input, float* output, int size);

class AtomicSumTest : public BaseCudaTest<float> {
protected:
    void SetUp() override {
        BaseCudaTest<float>::SetUp();

        // Test dimensions
        size = 1024 * 1024;
        channels = 64;
        stride = 2;

        // Allocate memory
        allocateMemory(&d_input, &h_input, size);
        allocateMemory(&d_output, &h_output, 1);
        allocateMemory(&d_channel_output, &h_channel_output, channels);
        allocateMemory(&d_shared_output, &h_shared_output, 1);

        // Initialize test data
        initializeRandom(h_input, size, -100.0f, 100.0f);

        // Copy to device
        copyToDevice(d_input, h_input, size);
        CUDA_CHECK(cudaMemset(d_output, 0, sizeof(float)));
        CUDA_CHECK(cudaMemset(d_channel_output, 0, channels * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_shared_output, 0, sizeof(float)));
    }

    void TearDown() override {
        BaseCudaTest<float>::TearDown();

        freeMemory(d_input, h_input);
        freeMemory(d_output, h_output);
        freeMemory(d_channel_output, h_channel_output);
        freeMemory(d_shared_output, h_shared_output);
    }

    int size, channels, stride;
    float *d_input, *d_output, *d_channel_output, *d_shared_output;
    float *h_input, *h_output, *h_channel_output, *h_shared_output;
};

TEST_F(AtomicSumTest, SingleAtomicSumCorrectness) {
    // Launch kernel
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;

    atomic_sum_single_kernel<<<grid_size, block_size>>>(d_input, d_output, size);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back
    copyFromDevice(h_output, d_output, 1);

    // CPU reference
    float cpu_result = 0.0f;
    for (int i = 0; i < size; ++i) {
        cpu_result += h_input[i];
    }

    EXPECT_NEAR(h_output[0], cpu_result, 1e-4f * size) << "Single atomic sum mismatch";

    // Save outputs
    saveOutput("atomic_sum_single.bin", h_output, 1);
    saveInput("atomic_sum_input.bin", h_input, size);
}

TEST_F(AtomicSumTest, SharedMemoryAtomicSumCorrectness) {
    // Launch kernel with shared memory optimization
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;

    atomic_sum_shared_kernel<<<grid_size, block_size>>>(d_input, d_shared_output, size);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back
    copyFromDevice(h_shared_output, d_shared_output, 1);

    // CPU reference
    float cpu_result = 0.0f;
    for (int i = 0; i < size; ++i) {
        cpu_result += h_input[i];
    }

    EXPECT_NEAR(h_shared_output[0], cpu_result, 1e-4f * size) << "Shared memory atomic sum mismatch";

    // Save output
    saveOutput("atomic_sum_shared.bin", h_shared_output, 1);
}

TEST_F(AtomicSumTest, ChannelAtomicSumCorrectness) {
    // Launch channel-wise kernel
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;

    atomic_sum_channel_kernel<<<grid_size, block_size>>>(d_input, d_channel_output, size, channels);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back
    copyFromDevice(h_channel_output, d_channel_output, channels);

    // CPU reference for each channel
    for (int c = 0; c < channels; ++c) {
        float channel_sum = 0.0f;
        for (int i = c; i < size; i += channels) {
            channel_sum += h_input[i];
        }
        EXPECT_NEAR(h_channel_output[c], channel_sum, 1e-3f * size / channels)
            << "Channel " << c << " atomic sum mismatch";
    }

    // Save outputs
    saveOutput("atomic_sum_channel.bin", h_channel_output, channels);
}

TEST_F(AtomicSumTest, StridedAtomicSumCorrectness) {
    // Launch strided kernel
    int strided_size = size / stride;  // Only every stride-th element
    int block_size = 256;
    int grid_size = (strided_size + block_size - 1) / block_size;

    atomic_sum_strided_kernel<<<grid_size, block_size>>>(d_input, d_output, size, stride);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back
    copyFromDevice(h_output, d_output, 1);

    // CPU reference
    float cpu_result = 0.0f;
    for (int i = 0; i < size; i += stride) {
        cpu_result += h_input[i];
    }

    EXPECT_NEAR(h_output[0], cpu_result, 1e-4f * strided_size) << "Strided atomic sum mismatch";

    // Save output
    saveOutput("atomic_sum_strided.bin", h_output, 1);
}

// Double precision test
class AtomicSumDoubleTest : public BaseCudaTest<double> {
protected:
    void SetUp() override {
        BaseCudaTest<double>::SetUp();

        size = 1024 * 1024;

        allocateMemory(&d_input, &h_input, size);
        allocateMemory(&d_output, &h_output, 1);

        initializeRandom(h_input, size, -100.0, 100.0);
        copyToDevice(d_input, h_input, size);
        CUDA_CHECK(cudaMemset(d_output, 0, sizeof(double)));
    }

    void TearDown() override {
        BaseCudaTest<double>::TearDown();

        freeMemory(d_input, h_input);
        freeMemory(d_output, h_output);
    }

    int size;
    double *d_input, *d_output;
    double *h_input, *h_output;
};

TEST_F(AtomicSumDoubleTest, Correctness) {
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;

    atomic_sum_double_kernel<<<grid_size, block_size>>>(d_input, d_output, size);
    CUDA_CHECK(cudaDeviceSynchronize());

    copyFromDevice(h_output, d_output, 1);

    double cpu_result = 0.0;
    for (int i = 0; i < size; ++i) {
        cpu_result += h_input[i];
    }

    EXPECT_NEAR(h_output[0], cpu_result, 1e-10 * size) << "Double precision atomic sum mismatch";

    saveOutput("atomic_sum_double.bin", h_output, 1);
}

TEST_F(AtomicSumTest, BenchmarkSingle) {
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;

    auto kernel_func = [&]() {
        CUDA_CHECK(cudaMemset(d_output, 0, sizeof(float)));
        atomic_sum_single_kernel<<<grid_size, block_size>>>(d_input, d_output, size);
    };

    BenchmarkResult result = benchmarkKernel(
        kernel_func, size, "atomic_sum_single", 10, 100);

    std::cout << "Atomic Sum Single - Avg time: " << result.avg_time_ms
              << " ms, Throughput: " << result.gflops << " GFLOPS" << std::endl;

    result.saveToJson("cuda_outputs/atomic_sum_single_benchmark.json");
}

TEST_F(AtomicSumTest, BenchmarkSharedMemory) {
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;

    auto kernel_func = [&]() {
        CUDA_CHECK(cudaMemset(d_shared_output, 0, sizeof(float)));
        atomic_sum_shared_kernel<<<grid_size, block_size>>>(d_input, d_shared_output, size);
    };

    BenchmarkResult result = benchmarkKernel(
        kernel_func, size, "atomic_sum_shared", 10, 100);

    std::cout << "Atomic Sum Shared - Avg time: " << result.avg_time_ms
              << " ms, Throughput: " << result.gflops << " GFLOPS" << std::endl;

    result.saveToJson("cuda_outputs/atomic_sum_shared_benchmark.json");
}

TEST_F(AtomicSumTest, PerformanceComparison) {
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;

    // Benchmark single atomic operations
    CudaTimer timer1;
    timer1.start();
    for (int i = 0; i < 100; ++i) {
        CUDA_CHECK(cudaMemset(d_output, 0, sizeof(float)));
        atomic_sum_single_kernel<<<grid_size, block_size>>>(d_input, d_output, size);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    timer1.stop();
    float time_single = timer1.elapsed_ms() / 100.0f;

    // Benchmark shared memory version
    CudaTimer timer2;
    timer2.start();
    for (int i = 0; i < 100; ++i) {
        CUDA_CHECK(cudaMemset(d_shared_output, 0, sizeof(float)));
        atomic_sum_shared_kernel<<<grid_size, block_size>>>(d_input, d_shared_output, size);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    timer2.stop();
    float time_shared = timer2.elapsed_ms() / 100.0f;

    // Standard reduction for comparison
    float *d_partial_sums;
    CUDA_CHECK(cudaMalloc(&d_partial_sums, grid_size * sizeof(float)));

    CudaTimer timer3;
    timer3.start();
    for (int i = 0; i < 100; ++i) {
        atomic_sum_shared_kernel<<<grid_size, block_size>>>(d_input, d_partial_sums, size);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    timer3.stop();
    float time_reduction = timer3.elapsed_ms() / 100.0f;

    std::cout << "Single atomic sum time: " << time_single << " ms" << std::endl;
    std::cout << "Shared memory atomic sum time: " << time_shared << " ms" << std::endl;
    std::cout << "Standard reduction time: " << time_reduction << " ms" << std::endl;
    std::cout << "Shared vs Single speedup: " << (time_single / time_shared) << "x" << std::endl;
    std::cout << "Reduction vs Atomic speedup: " << (time_single / time_reduction) << "x" << std::endl;

    // Save performance comparison
    std::ofstream file("cuda_outputs/atomic_sum_performance_comparison.json");
    file << "{\n";
    file << "  \"single_atomic_ms\": " << time_single << ",\n";
    file << "  \"shared_atomic_ms\": " << time_shared << ",\n";
    file << "  \"reduction_ms\": " << time_reduction << ",\n";
    file << "  \"shared_vs_single_speedup\": " << (time_single / time_shared) << ",\n";
    file << "  \"reduction_vs_atomic_speedup\": " << (time_single / time_reduction) << "\n";
    file << "}\n";
    file.close();

    CUDA_CHECK(cudaFree(d_partial_sums));
}

TEST_F(AtomicSumTest, DifferentSizes) {
    std::vector<int> test_sizes = {256, 1024, 8192, 65536, 1048576};

    for (int test_size : test_sizes) {
        float *test_input, *test_output;
        allocateMemory(&test_input, &h_input, test_size);
        allocateMemory(&test_output, &h_output, 1);

        initializeRandom(h_input, test_size, -100.0f, 100.0f);
        copyToDevice(test_input, h_input, test_size);

        int block_size = 256;
        int grid_size = (test_size + block_size - 1) / block_size;

        CUDA_CHECK(cudaMemset(test_output, 0, sizeof(float)));
        atomic_sum_single_kernel<<<grid_size, block_size>>>(test_input, test_output, test_size);
        CUDA_CHECK(cudaDeviceSynchronize());

        copyFromDevice(h_output, test_output, 1);

        float cpu_result = 0.0f;
        for (int i = 0; i < test_size; ++i) {
            cpu_result += h_input[i];
        }

        EXPECT_NEAR(h_output[0], cpu_result, 1e-3f * test_size)
            << "Size " << test_size << " atomic sum mismatch";

        std::string filename = "atomic_sum_size_" + std::to_string(test_size) + ".bin";
        saveOutput(filename.c_str(), h_output, 1);

        freeMemory(test_input, h_input);
        freeMemory(test_output, h_output);
    }
}

TEST_F(AtomicSumTest, ConcurrentAccess) {
    // Test multiple concurrent atomic operations
    int num_concurrent = 4;
    float *d_outputs[num_concurrent];
    float *h_outputs[num_concurrent];

    for (int i = 0; i < num_concurrent; ++i) {
        CUDA_CHECK(cudaMalloc(&d_outputs[i], sizeof(float)));
        CUDA_CHECK(cudaMemset(d_outputs[i], 0, sizeof(float)));
        h_outputs[i] = new float[1];
    }

    int block_size = 256;
    int grid_size = (size / num_concurrent + block_size - 1) / block_size;

    // Launch concurrent kernels
    for (int i = 0; i < num_concurrent; ++i) {
        atomic_sum_single_kernel<<<grid_size, block_size>>>(
            d_input + i * (size / num_concurrent), d_outputs[i], size / num_concurrent);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy results back
    for (int i = 0; i < num_concurrent; ++i) {
        CUDA_CHECK(cudaMemcpy(h_outputs[i], d_outputs[i], sizeof(float), cudaMemcpyDeviceToHost));
    }

    // Verify partial results
    for (int i = 0; i < num_concurrent; ++i) {
        float cpu_partial = 0.0f;
        for (int j = i * (size / num_concurrent); j < (i + 1) * (size / num_concurrent); ++j) {
            cpu_partial += h_input[j];
        }
        EXPECT_NEAR(h_outputs[i][0], cpu_partial, 1e-3f * size / num_concurrent)
            << "Concurrent atomic sum " << i << " mismatch";
    }

    // Cleanup
    for (int i = 0; i < num_concurrent; ++i) {
        CUDA_CHECK(cudaFree(d_outputs[i]));
        delete[] h_outputs[i];
    }
}

// Dummy implementation of launch function for test linking
void launchAtomicSum(const float* input, float* output, int size) {
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;
    atomic_sum_single_kernel<<<grid_size, block_size>>>(input, output, size);
}