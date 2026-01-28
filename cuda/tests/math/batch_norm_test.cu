#include <gtest/gtest.h>
#include "../common/base_test_fixture.h"

// Test kernel for batch normalization
__global__ void batch_norm_kernel(float* input, const float* gamma, const float* beta,
                                 const float* running_mean, const float* running_var,
                                 float epsilon, int batch_size, int channels,
                                 int spatial_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = batch_size * channels * spatial_size;

    if (idx < total_size) {
        int c = (idx / spatial_size) % channels;

        float x = input[idx];
        float mean = running_mean[c];
        float var = running_var[c];
        float std_dev = sqrtf(var + epsilon);

        float normalized = (x - mean) / std_dev;
        input[idx] = gamma[c] * normalized + beta[c];
    }
}

// Simplified version for inference
__global__ void batch_norm_inference_kernel(const float* input, float* output,
                                           const float* gamma, const float* beta,
                                           const float* mean, const float* var,
                                           float epsilon, int batch_size, int channels,
                                           int spatial_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = batch_size * channels * spatial_size;

    if (idx < total_size) {
        int c = (idx / spatial_size) % channels;

        float x = input[idx];
        float std_dev = sqrtf(var[c] + epsilon);

        float normalized = (x - mean[c]) / std_dev;
        output[idx] = gamma[c] * normalized + beta[c];
    }
}

void launchBatchNorm(float* input, const float* gamma, const float* beta,
                    const float* running_mean, const float* running_var,
                    float epsilon, int batch_size, int channels, int spatial_size);

class BatchNormTest : public BaseCudaTest<float> {
protected:
    void SetUp() override {
        BaseCudaTest<float>::SetUp();

        // Test dimensions
        batch_size = 8;
        channels = 64;
        height = 32;
        width = 32;
        spatial_size = height * width;
        total_size = batch_size * channels * spatial_size;
        epsilon = 1e-5f;

        // Allocate memory
        allocateMemory(&d_input, &h_input, total_size);
        allocateMemory(&d_output, &h_output, total_size);
        allocateMemory(&d_gamma, &h_gamma, channels);
        allocateMemory(&d_beta, &h_beta, channels);
        allocateMemory(&d_mean, &h_mean, channels);
        allocateMemory(&d_var, &h_var, channels);

        // Initialize test data
        initializeRandom(h_input, total_size, -5.0f, 5.0f);
        initializeRandom(h_gamma, channels, 0.5f, 2.0f);
        initializeRandom(h_beta, channels, -1.0f, 1.0f);
        initializeRandom(h_mean, channels, -1.0f, 1.0f);
        initializeRandom(h_var, channels, 0.1f, 2.0f);

        // Copy to device
        copyToDevice(d_input, h_input, total_size);
        copyToDevice(d_gamma, h_gamma, channels);
        copyToDevice(d_beta, h_beta, channels);
        copyToDevice(d_mean, h_mean, channels);
        copyToDevice(d_var, h_var, channels);
    }

    void TearDown() override {
        BaseCudaTest<float>::TearDown();

        freeMemory(d_input, h_input);
        freeMemory(d_output, h_output);
        freeMemory(d_gamma, h_gamma);
        freeMemory(d_beta, h_beta);
        freeMemory(d_mean, h_mean);
        freeMemory(d_var, h_var);
    }

    int batch_size, channels, height, width, spatial_size, total_size;
    float epsilon;
    float *d_input, *d_output, *d_gamma, *d_beta, *d_mean, *d_var;
    float *h_input, *h_output, *h_gamma, *h_beta, *h_mean, *h_var;
};

TEST_F(BatchNormTest, InferenceCorrectness) {
    // Launch kernel
    int block_size = 256;
    int grid_size = (total_size + block_size - 1) / block_size;

    batch_norm_inference_kernel<<<grid_size, block_size>>>(
        d_input, d_output, d_gamma, d_beta, d_mean, d_var,
        epsilon, batch_size, channels, spatial_size);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back
    copyFromDevice(h_output, d_output, total_size);

    // Verify results with CPU reference
    for (int i = 0; i < total_size; ++i) {
        int c = (i / spatial_size) % channels;
        float x = h_input[i];
        float std_dev = sqrtf(h_var[c] + epsilon);
        float normalized = (x - h_mean[c]) / std_dev;
        float expected = h_gamma[c] * normalized + h_beta[c];

        EXPECT_NEAR(h_output[i], expected, 1e-5f) << "BatchNorm mismatch at index " << i;
    }

    // Save outputs
    saveOutput("batch_norm_output.bin", h_output, total_size);
    saveOutput("batch_norm_input.bin", h_input, total_size);
    saveOutput("batch_norm_params.bin", h_gamma, channels * 4); // gamma, beta, mean, var
}

TEST_F(BatchNormTest, Benchmark) {
    int block_size = 256;
    int grid_size = (total_size + block_size - 1) / block_size;

    auto kernel_func = [&]() {
        batch_norm_inference_kernel<<<grid_size, block_size>>>(
            d_input, d_output, d_gamma, d_beta, d_mean, d_var,
            epsilon, batch_size, channels, spatial_size);
    };

    BenchmarkResult result = benchmarkKernel(
        kernel_func, total_size, "batch_norm", 10, 100);

    std::cout << "Batch Norm - Avg time: " << result.avg_time_ms
              << " ms, Throughput: " << result.gflops << " GFLOPS" << std::endl;

    result.saveToJson("cuda_outputs/batch_norm_benchmark.json");
}

TEST_F(BatchNormTest, DifferentConfigurations) {
    struct Config {
        int batch, channels, height, width;
    };

    std::vector<Config> configs = {
        {1, 32, 16, 16},
        {4, 64, 32, 32},
        {8, 128, 64, 64}
    };

    for (auto config : configs) {
        int test_spatial = config.height * config.width;
        int test_total = config.batch * config.channels * test_spatial;

        float *test_input, *test_output, *test_gamma, *test_beta, *test_mean, *test_var;
        allocateMemory(&test_input, &h_input, test_total);
        allocateMemory(&test_output, &h_output, test_total);
        allocateMemory(&test_gamma, &h_gamma, config.channels);
        allocateMemory(&test_beta, &h_beta, config.channels);
        allocateMemory(&test_mean, &h_mean, config.channels);
        allocateMemory(&test_var, &h_var, config.channels);

        initializeRandom(h_input, test_total, -5.0f, 5.0f);
        initializeRandom(h_gamma, config.channels, 0.5f, 2.0f);
        initializeRandom(h_beta, config.channels, -1.0f, 1.0f);
        initializeRandom(h_mean, config.channels, -1.0f, 1.0f);
        initializeRandom(h_var, config.channels, 0.1f, 2.0f);

        copyToDevice(test_input, h_input, test_total);
        copyToDevice(test_gamma, h_gamma, config.channels);
        copyToDevice(test_beta, h_beta, config.channels);
        copyToDevice(test_mean, h_mean, config.channels);
        copyToDevice(test_var, h_var, config.channels);

        int block_size = 256;
        int grid_size = (test_total + block_size - 1) / block_size;

        batch_norm_inference_kernel<<<grid_size, block_size>>>(
            test_input, test_output, test_gamma, test_beta, test_mean, test_var,
            epsilon, config.batch, config.channels, test_spatial);
        CUDA_CHECK(cudaDeviceSynchronize());

        copyFromDevice(h_output, test_output, test_total);

        // Verify a few random elements
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, test_total - 1);
        for (int i = 0; i < 10; ++i) {
            int idx = dis(gen);
            int c = (idx / test_spatial) % config.channels;
            float x = h_input[idx];
            float std_dev = sqrtf(h_var[c] + epsilon);
            float normalized = (x - h_mean[c]) / std_dev;
            float expected = h_gamma[c] * normalized + h_beta[c];
            EXPECT_NEAR(h_output[idx], expected, 1e-4f)
                << "Config " << config.batch << "x" << config.channels << "x"
                << config.height << "x" << config.width << " mismatch at index " << idx;
        }

        std::string filename = "batch_norm_config_" + std::to_string(config.batch) + "_" +
                               std::to_string(config.channels) + "_" +
                               std::to_string(config.height) + "_" +
                               std::to_string(config.width) + ".bin";
        saveOutput(filename.c_str(), h_output, test_total);

        freeMemory(test_input, h_input);
        freeMemory(test_output, h_output);
        freeMemory(test_gamma, h_gamma);
        freeMemory(test_beta, h_beta);
        freeMemory(test_mean, h_mean);
        freeMemory(test_var, h_var);
    }
}

TEST_F(BatchNormTest, EdgeCases) {
    // Test with near-zero variance
    for (int i = 0; i < channels; ++i) {
        h_var[i] = 1e-7f;
    }
    copyToDevice(d_var, h_var, channels);

    int block_size = 256;
    int grid_size = (total_size + block_size - 1) / block_size;

    batch_norm_inference_kernel<<<grid_size, block_size>>>(
        d_input, d_output, d_gamma, d_beta, d_mean, d_var,
        epsilon, batch_size, channels, spatial_size);
    CUDA_CHECK(cudaDeviceSynchronize());

    copyFromDevice(h_output, d_output, total_size);

    // Check for numerical stability
    for (int i = 0; i < total_size; ++i) {
        EXPECT_FALSE(isnan(h_output[i])) << "NaN detected at index " << i;
        EXPECT_FALSE(isinf(h_output[i])) << "Inf detected at index " << i;
        EXPECT_LT(abs(h_output[i]), 1000.0f) << "Output too large at index " << i;
    }
}