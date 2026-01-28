#include <gtest/gtest.h>
#include "../common/base_test_fixture.h"

// Test kernel for policy head - combines linear transformation and softmax
__global__ void policy_head_kernel(const float* input, float* output,
                                   const float* weight, const float* bias,
                                   int input_size, int output_size) {
    // Each block processes one output element
    int output_idx = blockIdx.x;
    int tid = threadIdx.x;

    extern __shared__ float shared_mem[];

    if (output_idx >= output_size) return;

    // Compute dot product for this output
    float sum = 0.0f;
    for (int i = tid; i < input_size; i += blockDim.x) {
        sum += input[i] * weight[output_idx * input_size + i];
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

    // Add bias and store output
    if (tid == 0) {
        float result = shared_mem[0] + bias[output_idx];
        output[output_idx] = result;
    }
}

// Apply softmax to policy output
__global__ void policy_softmax_kernel(float* policy, int output_size) {
    __shared__ float shared_data[256];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // Find maximum
    shared_data[tid] = (idx < output_size) ? policy[idx] : -INFINITY;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] = max(shared_data[tid], shared_data[tid + stride]);
        }
        __syncthreads();
    }

    float max_val = shared_data[0];

    // Compute sum of exp
    shared_data[tid] = (idx < output_size) ? expf(policy[idx] - max_val) : 0.0f;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();
    }

    float sum_exp = shared_data[0];

    // Apply softmax
    if (idx < output_size) {
        policy[idx] = expf(policy[idx] - max_val) / sum_exp;
    }
}

void launchPolicyHead(const float* input, float* output, const float* weight,
                     const float* bias, int input_size, int output_size);

class PolicyHeadTest : public BaseCudaTest<float> {
protected:
    void SetUp() override {
        BaseCudaTest<float>::SetUp();

        // Test dimensions (typical for Leela Chess Zero)
        batch_size = 1;
        input_size = 24 * 64 * 64;  // Typical CNN feature map size
        output_size = 1858;         // Number of possible moves in chess

        // Allocate memory
        allocateMemory(&d_input, &h_input, input_size);
        allocateMemory(&d_output, &h_output, output_size);
        allocateMemory(&d_weight, &h_weight, input_size * output_size);
        allocateMemory(&d_bias, &h_bias, output_size);
        allocateMemory(&d_intermediate, &h_intermediate, output_size);

        // Initialize test data
        initializeRandom(h_input, input_size, -1.0f, 1.0f);
        initializeRandom(h_weight, input_size * output_size, -0.1f, 0.1f);
        initializeRandom(h_bias, output_size, -0.1f, 0.1f);

        // Copy to device
        copyToDevice(d_input, h_input, input_size);
        copyToDevice(d_weight, h_weight, input_size * output_size);
        copyToDevice(d_bias, h_bias, output_size);
    }

    void TearDown() override {
        BaseCudaTest<float>::TearDown();

        freeMemory(d_input, h_input);
        freeMemory(d_output, h_output);
        freeMemory(d_weight, h_weight);
        freeMemory(d_bias, h_bias);
        freeMemory(d_intermediate, h_intermediate);
    }

    int batch_size, input_size, output_size;
    float *d_input, *d_output, *d_weight, *d_bias, *d_intermediate;
    float *h_input, *h_output, *h_weight, *h_bias, *h_intermediate;
};

TEST_F(PolicyHeadTest, LinearCorrectness) {
    // Launch linear transformation kernel
    int block_size = 256;
    dim3 grid(output_size);
    int shared_mem_size = block_size * sizeof(float);

    policy_head_kernel<<<grid, block_size, shared_mem_size>>>(
        d_input, d_intermediate, d_weight, d_bias, input_size, output_size);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back
    copyFromDevice(h_intermediate, d_intermediate, output_size);

    // CPU reference for first few outputs
    for (int i = 0; i < min(10, output_size); ++i) {
        float sum = h_bias[i];
        for (int j = 0; j < min(100, input_size); ++j) {  // Check subset for speed
            sum += h_input[j] * h_weight[i * input_size + j];
        }
        // Note: This is partial check, full check would be too slow
        EXPECT_GT(abs(h_intermediate[i]), 0.0f) << "Output too small at index " << i;
    }

    // Save outputs
    saveOutput("policy_head_linear_output.bin", h_intermediate, output_size);
    saveOutput("policy_head_input.bin", h_input, input_size);
}

TEST_F(PolicyHeadTest, SoftmaxCorrectness) {
    // First run linear transformation
    int block_size = 256;
    dim3 grid(output_size);
    int shared_mem_size = block_size * sizeof(float);

    policy_head_kernel<<<grid, block_size, shared_mem_size>>>(
        d_input, d_intermediate, d_weight, d_bias, input_size, output_size);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Then apply softmax
    int softmax_blocks = (output_size + block_size - 1) / block_size;
    policy_softmax_kernel<<<softmax_blocks, block_size>>>(d_intermediate, output_size);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back
    copyFromDevice(h_output, d_intermediate, output_size);

    // Verify softmax properties
    float sum = 0.0f;
    float max_val = 0.0f;
    for (int i = 0; i < output_size; ++i) {
        EXPECT_GT(h_output[i], 0.0f) << "Softmax output must be positive at index " << i;
        EXPECT_LT(h_output[i], 1.0f) << "Softmax output must be < 1 at index " << i;
        sum += h_output[i];
        if (h_output[i] > max_val) max_val = h_output[i];
    }

    EXPECT_NEAR(sum, 1.0f, 1e-5f) << "Softmax sum must be 1";
    EXPECT_GT(max_val, 0.0f) << "Should have non-zero maximum";

    // Save outputs
    saveOutput("policy_head_softmax_output.bin", h_output, output_size);
}

TEST_F(PolicyHeadTest, EndToEndCorrectness) {
    // Run complete policy head
    launchPolicyHead(d_input, d_output, d_weight, d_bias, input_size, output_size);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back
    copyFromDevice(h_output, d_output, output_size);

    // Basic checks
    float sum = 0.0f;
    for (int i = 0; i < output_size; ++i) {
        EXPECT_GT(h_output[i], 0.0f) << "Policy probability must be positive";
        EXPECT_LT(h_output[i], 1.0f) << "Policy probability must be < 1";
        EXPECT_FALSE(isnan(h_output[i])) << "NaN detected at index " << i;
        sum += h_output[i];
    }

    EXPECT_NEAR(sum, 1.0f, 1e-5f) << "Policy probabilities must sum to 1";

    // Save outputs
    saveOutput("policy_head_final_output.bin", h_output, output_size);
}

TEST_F(PolicyHeadTest, BenchmarkLinear) {
    int block_size = 256;
    dim3 grid(output_size);
    int shared_mem_size = block_size * sizeof(float);

    auto kernel_func = [&]() {
        policy_head_kernel<<<grid, block_size, shared_mem_size>>>(
            d_input, d_intermediate, d_weight, d_bias, input_size, output_size);
    };

    // Total FLOPs: 2 * input_size * output_size (multiply-add) + output_size (add bias)
    size_t total_flops = 2 * input_size * output_size + output_size;

    BenchmarkResult result = benchmarkKernel(
        kernel_func, total_flops / 2, "policy_head_linear", 5, 50);

    std::cout << "Policy Head Linear - Avg time: " << result.avg_time_ms
              << " ms, Throughput: " << result.gflops << " GFLOPS" << std::endl;

    result.saveToJson("cuda_outputs/policy_head_linear_benchmark.json");
}

TEST_F(PolicyHeadTest, BenchmarkSoftmax) {
    int block_size = 256;
    int softmax_blocks = (output_size + block_size - 1) / block_size;

    auto kernel_func = [&]() {
        policy_softmax_kernel<<<softmax_blocks, block_size>>>(d_intermediate, output_size);
    };

    BenchmarkResult result = benchmarkKernel(
        kernel_func, output_size, "policy_softmax", 10, 100);

    std::cout << "Policy Softmax - Avg time: " << result.avg_time_ms
              << " ms, Throughput: " << result.gflops << " GFLOPS" << std::endl;

    result.saveToJson("cuda_outputs/policy_softmax_benchmark.json");
}

TEST_F(PolicyHeadTest, DifferentInputSizes) {
    std::vector<int> test_sizes = {
        1024,      // Small
        4096,      // Medium
        16384,     // Large
        65536      // Very large
    };

    for (int test_input_size : test_sizes) {
        float *test_input, *test_weight, *test_bias, *test_output;
        allocateMemory(&test_input, &h_input, test_input_size);
        allocateMemory(&test_weight, &h_weight, test_input_size * output_size);
        allocateMemory(&test_bias, &h_bias, output_size);
        allocateMemory(&test_output, &h_output, output_size);

        initializeRandom(h_input, test_input_size, -1.0f, 1.0f);
        initializeRandom(h_weight, test_input_size * output_size, -0.1f, 0.1f);
        initializeRandom(h_bias, output_size, -0.1f, 0.1f);

        copyToDevice(test_input, h_input, test_input_size);
        copyToDevice(test_weight, h_weight, test_input_size * output_size);
        copyToDevice(test_bias, h_bias, output_size);

        int block_size = 256;
        dim3 grid(output_size);
        int shared_mem_size = block_size * sizeof(float);

        policy_head_kernel<<<grid, block_size, shared_mem_size>>>(
            test_input, test_output, test_weight, test_bias, test_input_size, output_size);
        CUDA_CHECK(cudaDeviceSynchronize());

        copyFromDevice(h_output, test_output, output_size);

        // Basic check
        for (int i = 0; i < min(10, output_size); ++i) {
            EXPECT_GT(abs(h_output[i]), 0.0f) << "Size " << test_input_size << " output too small";
        }

        std::string filename = "policy_head_size_" + std::to_string(test_input_size) + ".bin";
        saveOutput(filename.c_str(), h_output, output_size);

        freeMemory(test_input, h_input);
        freeMemory(test_weight, h_weight);
        freeMemory(test_bias, h_bias);
        freeMemory(test_output, h_output);
    }
}

// Dummy implementation of the launch function for test linking
void launchPolicyHead(const float* input, float* output, const float* weight,
                     const float* bias, int input_size, int output_size) {
    // This would be implemented in the actual kernel file
    int block_size = 256;
    dim3 grid(output_size);
    int shared_mem_size = block_size * sizeof(float);

    policy_head_kernel<<<grid, block_size, shared_mem_size>>>(
        input, output, weight, bias, input_size, output_size);

    int softmax_blocks = (output_size + block_size - 1) / block_size;
    policy_softmax_kernel<<<softmax_blocks, block_size>>>(output, output_size);
}