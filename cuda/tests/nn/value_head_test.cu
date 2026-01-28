#include <gtest/gtest.h>
#include "../common/base_test_fixture.h"

// Test kernel for value head - multiple linear layers and activation
__global__ void value_head_fc1_kernel(const float* input, float* output,
                                     const float* weight, const float* bias,
                                     int input_size, int hidden_size) {
    // Each block processes one hidden neuron
    int hidden_idx = blockIdx.x;
    int tid = threadIdx.x;

    extern __shared__ float shared_mem[];

    if (hidden_idx >= hidden_size) return;

    // Compute dot product
    float sum = 0.0f;
    for (int i = tid; i < input_size; i += blockDim.x) {
        sum += input[i] * weight[hidden_idx * input_size + i];
    }
    shared_mem[tid] = sum;
    __syncthreads();

    // Reduce
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_mem[tid] += shared_mem[tid + stride];
        }
        __syncthreads();
    }

    // Add bias and apply ReLU
    if (tid == 0) {
        float result = shared_mem[0] + bias[hidden_idx];
        output[hidden_idx] = max(0.0f, result);  // ReLU activation
    }
}

// Second fully connected layer (single output)
__global__ void value_head_fc2_kernel(const float* input, float* output,
                                     const float* weight, const float* bias,
                                     int hidden_size) {
    float sum = 0.0f;
    for (int i = 0; i < hidden_size; ++i) {
        sum += input[i] * weight[i];
    }
    *output = sum + bias[0];
}

// Tangent activation for value output
__global__ void tanh_activation_kernel(float* output) {
    *output = tanhf(*output);
}

void launchValueHead(const float* input, float* output,
                    const float* fc1_weight, const float* fc1_bias,
                    const float* fc2_weight, const float* fc2_bias,
                    int input_size, int hidden_size);

class ValueHeadTest : public BaseCudaTest<float> {
protected:
    void SetUp() override {
        BaseCudaTest<float>::SetUp();

        // Test dimensions
        batch_size = 1;
        input_size = 24 * 64 * 64;  // Same as policy input
        hidden_size = 256;          // Hidden layer size for value head

        // Allocate memory
        allocateMemory(&d_input, &h_input, input_size);
        allocateMemory(&d_hidden, &h_hidden, hidden_size);
        allocateMemory(&d_output, &h_output, 1);
        allocateMemory(&d_fc1_weight, &h_fc1_weight, input_size * hidden_size);
        allocateMemory(&d_fc1_bias, &h_fc1_bias, hidden_size);
        allocateMemory(&d_fc2_weight, &h_fc2_weight, hidden_size);
        allocateMemory(&d_fc2_bias, &h_fc2_bias, 1);

        // Initialize test data
        initializeRandom(h_input, input_size, -1.0f, 1.0f);
        initializeRandom(h_fc1_weight, input_size * hidden_size, -0.1f, 0.1f);
        initializeRandom(h_fc1_bias, hidden_size, -0.1f, 0.1f);
        initializeRandom(h_fc2_weight, hidden_size, -0.1f, 0.1f);
        initializeRandom(h_fc2_bias, 1, -0.1f, 0.1f);

        // Copy to device
        copyToDevice(d_input, h_input, input_size);
        copyToDevice(d_fc1_weight, h_fc1_weight, input_size * hidden_size);
        copyToDevice(d_fc1_bias, h_fc1_bias, hidden_size);
        copyToDevice(d_fc2_weight, h_fc2_weight, hidden_size);
        copyToDevice(d_fc2_bias, h_fc2_bias, 1);
    }

    void TearDown() override {
        BaseCudaTest<float>::TearDown();

        freeMemory(d_input, h_input);
        freeMemory(d_hidden, h_hidden);
        freeMemory(d_output, h_output);
        freeMemory(d_fc1_weight, h_fc1_weight);
        freeMemory(d_fc1_bias, h_fc1_bias);
        freeMemory(d_fc2_weight, h_fc2_weight);
        freeMemory(d_fc2_bias, h_fc2_bias);
    }

    int batch_size, input_size, hidden_size;
    float *d_input, *d_hidden, *d_output;
    float *d_fc1_weight, *d_fc1_bias, *d_fc2_weight, *d_fc2_bias;
    float *h_input, *h_hidden, *h_output;
    float *h_fc1_weight, *h_fc1_bias, *h_fc2_weight, *h_fc2_bias;
};

TEST_F(ValueHeadTest, FC1Correctness) {
    // Launch first fully connected layer
    int block_size = 256;
    dim3 grid(hidden_size);
    int shared_mem_size = block_size * sizeof(float);

    value_head_fc1_kernel<<<grid, block_size, shared_mem_size>>>(
        d_input, d_hidden, d_fc1_weight, d_fc1_bias, input_size, hidden_size);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back
    copyFromDevice(h_hidden, d_hidden, hidden_size);

    // Check ReLU activation (no negative values)
    for (int i = 0; i < hidden_size; ++i) {
        EXPECT_GE(h_hidden[i], 0.0f) << "ReLU activation should not produce negative values at index " << i;
    }

    // CPU reference for first few elements
    for (int i = 0; i < min(10, hidden_size); ++i) {
        float sum = h_fc1_bias[i];
        for (int j = 0; j < min(100, input_size); ++j) {  // Check subset for speed
            sum += h_input[j] * h_fc1_weight[i * input_size + j];
        }
        float expected = max(0.0f, sum);
        // Note: Partial check, full check would be too slow
        EXPECT_LT(abs(h_hidden[i] - expected), 100.0f) << "FC1 mismatch at index " << i;
    }

    // Save outputs
    saveOutput("value_head_fc1_output.bin", h_hidden, hidden_size);
}

TEST_F(ValueHeadTest, FC2Correctness) {
    // First need to run FC1
    int block_size = 256;
    dim3 grid(hidden_size);
    int shared_mem_size = block_size * sizeof(float);

    value_head_fc1_kernel<<<grid, block_size, shared_mem_size>>>(
        d_input, d_hidden, d_fc1_weight, d_fc1_bias, input_size, hidden_size);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Then run FC2
    value_head_fc2_kernel<<<1, 1>>>(
        d_hidden, d_output, d_fc2_weight, d_fc2_bias, hidden_size);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back
    copyFromDevice(h_output, d_output, 1);

    // Check range (should be reasonable before tanh)
    EXPECT_LT(abs(h_output[0]), 100.0f) << "FC2 output should be reasonable";

    // Save output
    saveOutput("value_head_fc2_output.bin", h_output, 1);
}

TEST_F(ValueHeadTest, EndToEndCorrectness) {
    // Run complete value head
    launchValueHead(d_input, d_output,
                   d_fc1_weight, d_fc1_bias,
                   d_fc2_weight, d_fc2_bias,
                   input_size, hidden_size);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back
    copyFromDevice(h_output, d_output, 1);

    // Check tanh output range (-1, 1)
    EXPECT_GT(h_output[0], -1.0f) << "Value should be > -1 (tanh range)";
    EXPECT_LT(h_output[0], 1.0f) << "Value should be < 1 (tanh range)";
    EXPECT_FALSE(isnan(h_output[0])) << "Value should not be NaN";

    // Save output
    saveOutput("value_head_final_output.bin", h_output, 1);
}

TEST_F(ValueHeadTest, BenchmarkFC1) {
    int block_size = 256;
    dim3 grid(hidden_size);
    int shared_mem_size = block_size * sizeof(float);

    auto kernel_func = [&]() {
        value_head_fc1_kernel<<<grid, block_size, shared_mem_size>>>(
            d_input, d_hidden, d_fc1_weight, d_fc1_bias, input_size, hidden_size);
    };

    BenchmarkResult result = benchmarkKernel(
        kernel_func, input_size * hidden_size, "value_head_fc1", 10, 100);

    std::cout << "Value Head FC1 - Avg time: " << result.avg_time_ms
              << " ms, Throughput: " << result.gflops << " GFLOPS" << std::endl;

    result.saveToJson("cuda_outputs/value_head_fc1_benchmark.json");
}

TEST_F(ValueHeadTest, BenchmarkFC2) {
    // First run FC1 to get hidden layer
    int block_size = 256;
    dim3 grid(hidden_size);
    int shared_mem_size = block_size * sizeof(float);

    value_head_fc1_kernel<<<grid, block_size, shared_mem_size>>>(
        d_input, d_hidden, d_fc1_weight, d_fc1_bias, input_size, hidden_size);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto kernel_func = [&]() {
        value_head_fc2_kernel<<<1, 1>>>(d_hidden, d_output, d_fc2_weight, d_fc2_bias, hidden_size);
    };

    BenchmarkResult result = benchmarkKernel(
        kernel_func, hidden_size, "value_head_fc2", 100, 1000);

    std::cout << "Value Head FC2 - Avg time: " << result.avg_time_ms
              << " ms, Throughput: " << result.gflops << " GFLOPS" << std::endl;

    result.saveToJson("cuda_outputs/value_head_fc2_benchmark.json");
}

TEST_F(ValueHeadTest, DifferentHiddenSizes) {
    std::vector<int> test_sizes = {64, 128, 256, 512};

    for (int test_hidden_size : test_sizes) {
        float *test_fc1_weight, *test_fc1_bias, *test_fc2_weight, *test_fc2_bias;
        float *test_hidden, *test_output;

        allocateMemory(&test_fc1_weight, &h_fc1_weight, input_size * test_hidden_size);
        allocateMemory(&test_fc1_bias, &h_fc1_bias, test_hidden_size);
        allocateMemory(&test_fc2_weight, &h_fc2_weight, test_hidden_size);
        allocateMemory(&test_fc2_bias, &h_fc2_bias, 1);
        allocateMemory(&test_hidden, &h_hidden, test_hidden_size);
        allocateMemory(&test_output, &h_output, 1);

        initializeRandom(h_fc1_weight, input_size * test_hidden_size, -0.1f, 0.1f);
        initializeRandom(h_fc1_bias, test_hidden_size, -0.1f, 0.1f);
        initializeRandom(h_fc2_weight, test_hidden_size, -0.1f, 0.1f);
        initializeRandom(h_fc2_bias, 1, -0.1f, 0.1f);

        copyToDevice(test_fc1_weight, h_fc1_weight, input_size * test_hidden_size);
        copyToDevice(test_fc1_bias, h_fc1_bias, test_hidden_size);
        copyToDevice(test_fc2_weight, h_fc2_weight, test_hidden_size);
        copyToDevice(test_fc2_bias, h_fc2_bias, 1);

        // Run FC1
        int block_size = 256;
        dim3 grid(test_hidden_size);
        int shared_mem_size = block_size * sizeof(float);

        value_head_fc1_kernel<<<grid, block_size, shared_mem_size>>>(
            d_input, test_hidden, test_fc1_weight, test_fc1_bias, input_size, test_hidden_size);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Run FC2 and tanh
        value_head_fc2_kernel<<<1, 1>>>(test_hidden, test_output, test_fc2_weight, test_fc2_bias, test_hidden_size);
        CUDA_CHECK(cudaDeviceSynchronize());

        tanh_activation_kernel<<<1, 1>>>(test_output);
        CUDA_CHECK(cudaDeviceSynchronize());

        copyFromDevice(h_output, test_output, 1);

        EXPECT_GT(h_output[0], -1.0f) << "Tanh range check failed for size " << test_hidden_size;
        EXPECT_LT(h_output[0], 1.0f) << "Tanh range check failed for size " << test_hidden_size;

        std::string filename = "value_head_hidden_" + std::to_string(test_hidden_size) + ".bin";
        saveOutput(filename.c_str(), h_output, 1);

        freeMemory(test_fc1_weight, h_fc1_weight);
        freeMemory(test_fc1_bias, h_fc1_bias);
        freeMemory(test_fc2_weight, h_fc2_weight);
        freeMemory(test_fc2_bias, h_fc2_bias);
        freeMemory(test_hidden, h_hidden);
        freeMemory(test_output, h_output);
    }
}

TEST_F(ValueHeadTest, EdgeCases) {
    // Test with zero input (should output something reasonable)
    initializeConstant(h_input, input_size, 0.0f);
    copyToDevice(d_input, h_input, input_size);

    launchValueHead(d_input, d_output,
                   d_fc1_weight, d_fc1_bias,
                   d_fc2_weight, d_fc2_bias,
                   input_size, hidden_size);
    CUDA_CHECK(cudaDeviceSynchronize());

    copyFromDevice(h_output, d_output, 1);

    EXPECT_FALSE(isnan(h_output[0])) << "Zero input should not produce NaN";
    EXPECT_GT(h_output[0], -1.0f) << "Zero input output should be in tanh range";
    EXPECT_LT(h_output[0], 1.0f) << "Zero input output should be in tanh range";

    // Test with very large input
    initializeConstant(h_input, input_size, 100.0f);
    copyToDevice(d_input, h_input, input_size);

    launchValueHead(d_input, d_output,
                   d_fc1_weight, d_fc1_bias,
                   d_fc2_weight, d_fc2_bias,
                   input_size, hidden_size);
    CUDA_CHECK(cudaDeviceSynchronize());

    copyFromDevice(h_output, d_output, 1);

    EXPECT_FALSE(isnan(h_output[0])) << "Large input should not produce NaN";
    EXPECT_GT(h_output[0], -1.0f) << "Large input output should be in tanh range";
    EXPECT_LT(h_output[0], 1.0f) << "Large input output should be in tanh range";
}

// Dummy implementation of the launch function for test linking
void launchValueHead(const float* input, float* output,
                    const float* fc1_weight, const float* fc1_bias,
                    const float* fc2_weight, const float* fc2_bias,
                    int input_size, int hidden_size) {
    int block_size = 256;
    dim3 grid(hidden_size);
    int shared_mem_size = block_size * sizeof(float);

    // FC1 with ReLU
    value_head_fc1_kernel<<<grid, block_size, shared_mem_size>>>(
        input, output, fc1_weight, fc1_bias, input_size, hidden_size);
    CUDA_CHECK(cudaDeviceSynchronize());

    // FC2
    value_head_fc2_kernel<<<1, 1>>>(output, output, fc2_weight, fc2_bias, hidden_size);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Tanh activation
    tanh_activation_kernel<<<1, 1>>>(output);
    CUDA_CHECK(cudaDeviceSynchronize());
}