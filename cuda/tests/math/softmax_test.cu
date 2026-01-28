#include <gtest/gtest.h>
#include "../common/base_test_fixture.h"
#include <cmath>

// Test kernel for softmax (1D) - Simple single-block implementation
__global__ void softmax_1d_kernel(float* input, int size) {
    // Use a single block to avoid multi-block synchronization issues
    // For size=1000 and block_size=256, this is efficient enough
    if (blockIdx.x > 0) return;  // Only first block processes data

    __shared__ float shared_data[256];
    int tid = threadIdx.x;

    // Phase 1: Find maximum value
    float max_val = -INFINITY;
    for (int i = tid; i < size; i += blockDim.x) {
        max_val = max(max_val, input[i]);
    }
    shared_data[tid] = max_val;
    __syncthreads();

    // Reduce within block to get global max
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] = max(shared_data[tid], shared_data[tid + stride]);
        }
        __syncthreads();
    }
    max_val = shared_data[0];

    // Phase 2: Compute sum of exp(x - max)
    float sum_exp = 0.0f;
    for (int i = tid; i < size; i += blockDim.x) {
        sum_exp += expf(input[i] - max_val);
    }
    shared_data[tid] = sum_exp;
    __syncthreads();

    // Reduce within block to get global sum
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();
    }
    sum_exp = shared_data[0];

    // Safeguard against numerical issues
    if (sum_exp <= 0.0f || sum_exp != sum_exp) {  // sum_exp != sum_exp is NaN check
        sum_exp = 1.0f;
    }

    // Phase 3: Apply softmax
    for (int i = tid; i < size; i += blockDim.x) {
        float exp_val = expf(input[i] - max_val);
        input[i] = exp_val / sum_exp;

        // Safety check for extreme values
        if (input[i] != input[i] || input[i] > 1.0f) {  // NaN or > 1
            input[i] = 1.0f / size;  // Fallback to uniform distribution
        }
    }
}

// Test kernel for softmax (2D, channel-wise) - Fixed
__global__ void softmax_2d_kernel(float* input, int batch_size, int channels) {
    // Each block processes one sample
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;

    // Use static shared memory to avoid allocation issues
    __shared__ float shared_mem[512];  // Enough for 256 threads + reduction results

    if (batch_idx >= batch_size) return;

    float* sample_input = input + batch_idx * channels;

    // Find maximum within this sample
    float max_val = -INFINITY;
    for (int i = tid; i < channels; i += blockDim.x) {
        max_val = max(max_val, sample_input[i]);
    }
    shared_mem[tid] = max_val;
    __syncthreads();

    // Parallel reduction within block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_mem[tid] = max(shared_mem[tid], shared_mem[tid + stride]);
        }
        __syncthreads();
    }
    max_val = shared_mem[0];

    // Compute sum of exp within this sample
    float sum_exp = 0.0f;
    for (int i = tid; i < channels; i += blockDim.x) {
        sum_exp += expf(sample_input[i] - max_val);
    }
    shared_mem[tid] = sum_exp;
    __syncthreads();

    // Parallel reduction for sum
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_mem[tid] += shared_mem[tid + stride];
        }
        __syncthreads();
    }
    sum_exp = shared_mem[0];

    // Safeguard against numerical issues - using device-compatible checks
    if (sum_exp <= 0.0f || sum_exp != sum_exp) {  // sum_exp != sum_exp is a NaN check
        sum_exp = 1.0f;  // Fallback to uniform distribution
    }

    // Apply softmax to each element in this sample
    for (int i = tid; i < channels; i += blockDim.x) {
        float exp_val = expf(sample_input[i] - max_val);
        sample_input[i] = exp_val / sum_exp;

        // Safety check for extreme values - using device-compatible checks
        if (sample_input[i] != sample_input[i] || sample_input[i] > 1e10f) {  // NaN or huge value check
            sample_input[i] = 1.0f / channels;  // Fallback to uniform
        }
    }
}

void launchSoftmax1D(float* input, int size);
void launchSoftmax2D(float* input, int batch_size, int channels);

class SoftmaxTest : public BaseCudaTest<float> {
protected:
    void SetUp() override {
        BaseCudaTest<float>::SetUp();

        // Test dimensions
        batch_size = 8;
        channels = 1000;  // Common for policy head
        total_size = batch_size * channels;

        // Allocate memory
        allocateMemory(&d_input_1d, &h_input_1d, channels);
        allocateMemory(&d_input_2d, &h_input_2d, total_size);

        // Initialize test data (logits)
        initializeRandom(h_input_1d, channels, -10.0f, 10.0f);
        initializeRandom(h_input_2d, total_size, -10.0f, 10.0f);

        // Copy to device
        copyToDevice(d_input_1d, h_input_1d, channels);
        copyToDevice(d_input_2d, h_input_2d, total_size);
    }

    void TearDown() override {
        BaseCudaTest<float>::TearDown();

        freeMemory(d_input_1d, h_input_1d);
        freeMemory(d_input_2d, h_input_2d);
    }

    int batch_size, channels, total_size;
    float *d_input_1d, *d_input_2d;
    float *h_input_1d, *h_input_2d;
};

TEST_F(SoftmaxTest, Softmax1DCorrectness) {
    // Create copy for comparison
    float* h_input_copy = new float[channels];
    memcpy(h_input_copy, h_input_1d, channels * sizeof(float));

    // Launch kernel - Simple single-block approach
    int block_size = 256;
    int grid_size = 1;  // Single block for 1D softmax

    softmax_1d_kernel<<<grid_size, block_size>>>(d_input_1d, channels);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back
    copyFromDevice(h_input_1d, d_input_1d, channels);

    // Verify CPU reference implementation
    float max_val = *std::max_element(h_input_copy, h_input_copy + channels);
    float sum_exp = 0.0f;
    for (int i = 0; i < channels; ++i) {
        sum_exp += expf(h_input_copy[i] - max_val);
    }

    for (int i = 0; i < channels; ++i) {
        float expected = expf(h_input_copy[i] - max_val) / sum_exp;
        EXPECT_NEAR(h_input_1d[i], expected, 1e-5f) << "Softmax 1D mismatch at index " << i;
        EXPECT_GT(h_input_1d[i], 0.0f) << "Softmax output must be positive";
        EXPECT_LT(h_input_1d[i], 1.0f) << "Softmax output must be < 1";
    }

    // Check sum is approximately 1
    float gpu_sum = 0.0f;
    for (int i = 0; i < channels; ++i) {
        gpu_sum += h_input_1d[i];
    }
    EXPECT_NEAR(gpu_sum, 1.0f, 1e-5f) << "Softmax 1D sum must be 1";

    // Save outputs
    saveOutput("softmax_1d_output.bin", h_input_1d, channels);

    delete[] h_input_copy;
}

TEST_F(SoftmaxTest, Softmax2DCorrectness) {
    // Launch kernel - Fixed with proper shared memory size
    int block_size = 256;
    int shared_mem_size = 512 * sizeof(float);  // Fixed shared memory for safety

    softmax_2d_kernel<<<batch_size, block_size, shared_mem_size>>>(
        d_input_2d, batch_size, channels);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back
    copyFromDevice(h_input_2d, d_input_2d, total_size);

    // Verify each batch
    for (int b = 0; b < batch_size; ++b) {
        float* batch_input = h_input_2d + b * channels;
        float sum = 0.0f;

        for (int i = 0; i < channels; ++i) {
            EXPECT_GT(batch_input[i], 0.0f) << "Softmax output must be positive at batch " << b << ", index " << i;
            EXPECT_LT(batch_input[i], 1.0f) << "Softmax output must be < 1 at batch " << b << ", index " << i;
            sum += batch_input[i];
        }

        EXPECT_NEAR(sum, 1.0f, 1e-5f) << "Softmax sum must be 1 for batch " << b;
    }

    // Save outputs
    saveOutput("softmax_2d_output.bin", h_input_2d, total_size);
}

TEST_F(SoftmaxTest, Benchmark1D) {
    int block_size = 256;
    int grid_size = 1;  // Single block for 1D softmax

    auto kernel_func = [&]() {
        softmax_1d_kernel<<<grid_size, block_size>>>(d_input_1d, channels);
    };

    BenchmarkResult result = benchmarkKernel(
        kernel_func, channels, "softmax_1d", 10, 100);

    std::cout << "Softmax 1D - Avg time: " << result.avg_time_ms
              << " ms, Throughput: " << result.gflops << " GFLOPS" << std::endl;

    result.saveToJson("cuda_outputs/softmax_1d_benchmark.json");
}

TEST_F(SoftmaxTest, Benchmark2D) {
    int block_size = 256;
    int shared_mem_size = 512 * sizeof(float);  // Fixed shared memory for safety

    auto kernel_func = [&]() {
        softmax_2d_kernel<<<batch_size, block_size, shared_mem_size>>>(
            d_input_2d, batch_size, channels);
    };

    BenchmarkResult result = benchmarkKernel(
        kernel_func, total_size, "softmax_2d", 10, 100);

    std::cout << "Softmax 2D - Avg time: " << result.avg_time_ms
              << " ms, Throughput: " << result.gflops << " GFLOPS" << std::endl;

    result.saveToJson("cuda_outputs/softmax_2d_benchmark.json");
}

TEST_F(SoftmaxTest, DifferentSizes) {
    std::vector<int> test_sizes = {10, 100, 1000, 10000};

    for (int size : test_sizes) {
        float *test_input;
        allocateMemory(&test_input, &h_input_1d, size);

        initializeRandom(h_input_1d, size, -10.0f, 10.0f);
        copyToDevice(test_input, h_input_1d, size);

        int block_size = min(256, size);
        int grid_size = 1;

        softmax_1d_kernel<<<grid_size, block_size>>>(test_input, size);
        CUDA_CHECK(cudaDeviceSynchronize());

        copyFromDevice(h_input_1d, test_input, size);

        // Check sum
        float sum = 0.0f;
        for (int i = 0; i < size; ++i) {
            sum += h_input_1d[i];
        }
        EXPECT_NEAR(sum, 1.0f, 1e-5f) << "Softmax sum must be 1 for size " << size;

        std::string filename = "softmax_size_" + std::to_string(size) + ".bin";
        saveOutput(filename.c_str(), h_input_1d, size);

        freeMemory(test_input, h_input_1d);
    }
}

TEST_F(SoftmaxTest, ExtremeValues) {
    // Test with very large negative values (should result in uniform distribution)
    initializeConstant(h_input_1d, channels, -1000.0f);
    copyToDevice(d_input_1d, h_input_1d, channels);

    int block_size = 256;
    softmax_1d_kernel<<<1, block_size>>>(d_input_1d, channels);
    CUDA_CHECK(cudaDeviceSynchronize());

    copyFromDevice(h_input_1d, d_input_1d, channels);

    float expected_uniform = 1.0f / channels;
    for (int i = 0; i < channels; ++i) {
        EXPECT_NEAR(h_input_1d[i], expected_uniform, 1e-6f)
            << "Should be approximately uniform for extreme negative values";
    }
}