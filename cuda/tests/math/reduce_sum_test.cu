#include <gtest/gtest.h>
#include "../common/base_test_fixture.h"

// Test kernel for sum reduce
__global__ void reduce_sum_kernel(const float* input, float* output, int size) {
    __shared__ float shared_data[256];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // Load data to shared memory
    shared_data[tid] = (idx < size) ? input[idx] : 0.0f;
    __syncthreads();

    // Reduce within block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();
    }

    // Write block result
    if (tid == 0) {
        output[blockIdx.x] = shared_data[0];
    }
}

// Test kernel for atomic sum reduce
__global__ void atomic_reduce_sum_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        atomicAdd(output, input[idx]);
    }
}

// Test kernel for channel-wise reduction (2D)
__global__ void reduce_sum_2d_kernel(const float* input, float* output,
                                    int batch_size, int channels, int spatial_size) {
    extern __shared__ float shared_mem[];

    int channel_idx = blockIdx.x;
    int tid = threadIdx.x;
    int total_per_channel = batch_size * spatial_size;

    if (channel_idx >= channels) return;

    // Reduce across batch and spatial dimensions for this channel
    float sum = 0.0f;
    for (int i = tid; i < total_per_channel; i += blockDim.x) {
        int idx = i / spatial_size * channels * spatial_size + channel_idx * spatial_size + (i % spatial_size);
        sum += input[idx];
    }

    shared_mem[tid] = sum;
    __syncthreads();

    // Reduce within block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_mem[tid] += shared_mem[tid + stride];
        }
        __syncthreads();
    }

    // Write result for this channel
    if (tid == 0) {
        output[channel_idx] = shared_mem[0];
    }
}

void launchReduceSum(const float* input, float* output, int size);

class ReduceSumTest : public BaseCudaTest<float> {
protected:
    void SetUp() override {
        BaseCudaTest<float>::SetUp();

        // Test dimensions
        size = 1024 * 1024;
        batch_size = 8;
        channels = 64;
        spatial_size = 32 * 32;
        total_2d = batch_size * channels * spatial_size;

        // Allocate memory
        allocateMemory(&d_input, &h_input, size);
        allocateMemory(&d_output, &h_output, 256);  // For partial reductions
        allocateMemory(&d_input_2d, &h_input_2d, total_2d);
        allocateMemory(&d_output_2d, &h_output_2d, channels);
        allocateMemory(&d_atomic_output, &h_atomic_output, 1);

        // Initialize test data
        initializeRandom(h_input, size, -100.0f, 100.0f);
        initializeRandom(h_input_2d, total_2d, -100.0f, 100.0f);

        // Copy to device
        copyToDevice(d_input, h_input, size);
        copyToDevice(d_input_2d, h_input_2d, total_2d);
        CUDA_CHECK(cudaMemset(d_atomic_output, 0, sizeof(float)));
    }

    void TearDown() override {
        BaseCudaTest<float>::TearDown();

        freeMemory(d_input, h_input);
        freeMemory(d_output, h_output);
        freeMemory(d_input_2d, h_input_2d);
        freeMemory(d_output_2d, h_output_2d);
        freeMemory(d_atomic_output, h_atomic_output);
    }

    int size, batch_size, channels, spatial_size, total_2d;
    float *d_input, *d_output, *d_input_2d, *d_output_2d, *d_atomic_output;
    float *h_input, *h_output, *h_input_2d, *h_output_2d, *h_atomic_output;
};

TEST_F(ReduceSumTest, ReduceSumCorrectness) {
    // Launch kernel
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;

    reduce_sum_kernel<<<grid_size, block_size>>>(d_input, d_output, size);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy partial results back
    copyFromDevice(h_output, d_output, grid_size);

    // Final reduction on CPU
    float gpu_result = 0.0f;
    for (int i = 0; i < grid_size; ++i) {
        gpu_result += h_output[i];
    }

    // CPU reference
    float cpu_result = 0.0f;
    for (int i = 0; i < size; ++i) {
        cpu_result += h_input[i];
    }

    EXPECT_NEAR(gpu_result, cpu_result, 1e-4f * size) << "Reduce sum mismatch";

    // Save outputs
    saveOutput("reduce_sum_output.bin", &gpu_result, 1);
    saveInput("reduce_sum_input.bin", h_input, size);
}

TEST_F(ReduceSumTest, AtomicReduceSumCorrectness) {
    // Launch kernel
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;

    atomic_reduce_sum_kernel<<<grid_size, block_size>>>(d_input, d_atomic_output, size);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back
    copyFromDevice(h_atomic_output, d_atomic_output, 1);

    // CPU reference
    float cpu_result = 0.0f;
    for (int i = 0; i < size; ++i) {
        cpu_result += h_input[i];
    }

    EXPECT_NEAR(h_atomic_output[0], cpu_result, 1e-4f * size) << "Atomic reduce sum mismatch";

    // Save output
    saveOutput("atomic_reduce_sum_output.bin", h_atomic_output, 1);
}

TEST_F(ReduceSumTest, ReduceSum2DCorrectness) {
    // Launch kernel
    int block_size = 256;
    int shared_mem_size = block_size * sizeof(float);

    reduce_sum_2d_kernel<<<channels, block_size, shared_mem_size>>>(
        d_input_2d, d_output_2d, batch_size, channels, spatial_size);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back
    copyFromDevice(h_output_2d, d_output_2d, channels);

    // CPU reference for each channel
    for (int c = 0; c < channels; ++c) {
        float cpu_result = 0.0f;
        for (int b = 0; b < batch_size; ++b) {
            for (int s = 0; s < spatial_size; ++s) {
                int idx = b * channels * spatial_size + c * spatial_size + s;
                cpu_result += h_input_2d[idx];
            }
        }
        EXPECT_NEAR(h_output_2d[c], cpu_result, 1e-4f * batch_size * spatial_size)
            << "2D reduce sum mismatch for channel " << c;
    }

    // Save outputs
    saveOutput("reduce_sum_2d_output.bin", h_output_2d, channels);
}

TEST_F(ReduceSumTest, BenchmarkReduceSum) {
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;

    auto kernel_func = [&]() {
        reduce_sum_kernel<<<grid_size, block_size>>>(d_input, d_output, size);
    };

    BenchmarkResult result = benchmarkKernel(
        kernel_func, size, "reduce_sum", 10, 100);

    std::cout << "Reduce Sum - Avg time: " << result.avg_time_ms
              << " ms, Throughput: " << result.gflops << " GFLOPS" << std::endl;

    result.saveToJson("cuda_outputs/reduce_sum_benchmark.json");
}

TEST_F(ReduceSumTest, BenchmarkAtomicReduceSum) {
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;

    auto kernel_func = [&]() {
        cudaMemset(d_atomic_output, 0, sizeof(float));
        atomic_reduce_sum_kernel<<<grid_size, block_size>>>(d_input, d_atomic_output, size);
    };

    BenchmarkResult result = benchmarkKernel(
        kernel_func, size, "atomic_reduce_sum", 10, 100);

    std::cout << "Atomic Reduce Sum - Avg time: " << result.avg_time_ms
              << " ms, Throughput: " << result.gflops << " GFLOPS" << std::endl;

    result.saveToJson("cuda_outputs/atomic_reduce_sum_benchmark.json");
}

TEST_F(ReduceSumTest, DifferentSizes) {
    std::vector<int> test_sizes = {256, 1024, 8192, 65536, 1048576};

    for (int test_size : test_sizes) {
        float *test_input, *test_output;
        allocateMemory(&test_input, &h_input, test_size);
        allocateMemory(&test_output, &h_output, 256);

        initializeRandom(h_input, test_size, -100.0f, 100.0f);
        copyToDevice(test_input, h_input, test_size);

        int block_size = 256;
        int test_grid_size = (test_size + block_size - 1) / block_size;

        reduce_sum_kernel<<<test_grid_size, block_size>>>(test_input, test_output, test_size);
        CUDA_CHECK(cudaDeviceSynchronize());

        copyFromDevice(h_output, test_output, test_grid_size);

        // Final reduction
        float gpu_result = 0.0f;
        for (int i = 0; i < test_grid_size; ++i) {
            gpu_result += h_output[i];
        }

        // CPU reference
        float cpu_result = 0.0f;
        for (int i = 0; i < test_size; ++i) {
            cpu_result += h_input[i];
        }

        EXPECT_NEAR(gpu_result, cpu_result, 1e-4f * test_size)
            << "Size " << test_size << " reduce sum mismatch";

        std::string filename = "reduce_sum_size_" + std::to_string(test_size) + ".bin";
        saveOutput(filename.c_str(), &gpu_result, 1);

        freeMemory(test_input, h_input);
        freeMemory(test_output, h_output);
    }
}

TEST_F(ReduceSumTest, PerformanceComparison) {
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;

    // Test both methods
    float* device_temp_output;
    CUDA_CHECK(cudaMalloc((void**)&device_temp_output, 256 * sizeof(float)));

    // Method 1: Shared memory reduction
    CudaTimer timer1;
    timer1.start();
    for (int i = 0; i < 100; ++i) {
        reduce_sum_kernel<<<grid_size, block_size>>>(d_input, d_output, size);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    timer1.stop();
    float time1 = timer1.elapsed_ms() / 100.0f;

    // Method 2: Atomic reduction
    CudaTimer timer2;
    timer2.start();
    for (int i = 0; i < 100; ++i) {
        cudaMemset(d_atomic_output, 0, sizeof(float));
        atomic_reduce_sum_kernel<<<grid_size, block_size>>>(d_input, d_atomic_output, size);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    timer2.stop();
    float time2 = timer2.elapsed_ms() / 100.0f;

    std::cout << "Shared memory reduction time: " << time1 << " ms" << std::endl;
    std::cout << "Atomic reduction time: " << time2 << " ms" << std::endl;
    std::cout << "Speedup: " << (time2 / time1) << "x" << std::endl;

    // Save performance comparison
    std::ofstream file("cuda_outputs/reduce_sum_performance_comparison.json");
    file << "{\n";
    file << "  \"shared_memory_reduction_ms\": " << time1 << ",\n";
    file << "  \"atomic_reduction_ms\": " << time2 << ",\n";
    file << "  \"speedup\": " << (time2 / time1) << "\n";
    file << "}\n";
    file.close();

    cudaFree(device_temp_output);
}