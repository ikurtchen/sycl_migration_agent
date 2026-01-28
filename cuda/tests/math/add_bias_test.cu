#include <gtest/gtest.h>
#include "../common/base_test_fixture.h"

// Test kernel for add bias (broadcasting)
__global__ void add_bias_1d_kernel(float* input, const float* bias, int channels, int spatial_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = channels * spatial_size;

    if (idx < total_size) {
        int c = idx % channels;
        input[idx] += bias[c];
    }
}

__global__ void add_bias_2d_kernel(float* input, const float* bias, int batch_size, int channels, int spatial_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = batch_size * channels * spatial_size;

    if (idx < total_size) {
        int c = (idx / spatial_size) % channels;
        input[idx] += bias[c];
    }
}

void launchAddBias1D(float* input, const float* bias, int channels, int spatial_size);
void launchAddBias2D(float* input, const float* bias, int batch_size, int channels, int spatial_size);

class AddBiasTest : public BaseCudaTest<float> {
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

        // Allocate memory for 2D case
        allocateMemory(&d_input, &h_input, total_size);
        allocateMemory(&d_bias, &h_bias, channels);
        allocateMemory(&d_output, &h_output, total_size);

        // Initialize test data
        initializeRandom(h_input, total_size, -10.0f, 10.0f);
        initializeRandom(h_bias, channels, -5.0f, 5.0f);

        // Copy to device
        copyToDevice(d_input, h_input, total_size);
        copyToDevice(d_bias, h_bias, channels);
    }

    void TearDown() override {
        BaseCudaTest<float>::TearDown();

        freeMemory(d_input, h_input);
        freeMemory(d_bias, h_bias);
        freeMemory(d_output, h_output);
    }

    int batch_size, channels, height, width, spatial_size, total_size;
    float *d_input, *d_bias, *d_output;
    float *h_input, *h_bias, *h_output;
};

TEST_F(AddBiasTest, AddBias1D) {
    // Copy input to output for in-place operation
    copyToDevice(d_output, h_input, total_size);

    // Launch kernel for single sample
    int block_size = 256;
    int grid_size = (channels * spatial_size + block_size - 1) / block_size;

    add_bias_1d_kernel<<<grid_size, block_size>>>(d_output, d_bias, channels, spatial_size);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back
    copyFromDevice(h_output, d_output, channels * spatial_size);

    // Verify results
    for (int i = 0; i < channels * spatial_size; ++i) {
        int c = i % channels;
        float expected = h_input[i] + h_bias[c];
        EXPECT_NEAR(h_output[i], expected, 1e-5f) << "1D bias mismatch at index " << i;
    }

    // Save outputs
    saveOutput("add_bias_1d_output.bin", h_output, channels * spatial_size);
    saveOutput("add_bias_1d_bias.bin", h_bias, channels);
}

TEST_F(AddBiasTest, AddBias2D) {
    // Launch kernel for batch
    int block_size = 256;
    int grid_size = (total_size + block_size - 1) / block_size;

    add_bias_2d_kernel<<<grid_size, block_size>>>(d_input, d_bias, batch_size, channels, spatial_size);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back
    copyFromDevice(h_output, d_input, total_size);

    // Verify results
    for (int i = 0; i < total_size; ++i) {
        int c = (i / spatial_size) % channels;
        float expected = h_input[i] + h_bias[c];
        EXPECT_NEAR(h_output[i], expected, 1e-5f) << "2D bias mismatch at index " << i;
    }

    // Save outputs
    saveOutput("add_bias_2d_output.bin", h_output, total_size);
    saveOutput("add_bias_2d_input.bin", h_input, total_size);
}

TEST_F(AddBiasTest, Benchmark1D) {
    int block_size = 256;
    int grid_size = (channels * spatial_size + block_size - 1) / block_size;

    auto kernel_func = [&]() {
        add_bias_1d_kernel<<<grid_size, block_size>>>(d_input, d_bias, channels, spatial_size);
    };

    BenchmarkResult result = benchmarkKernel(
        kernel_func, channels * spatial_size, "add_bias_1d", 10, 100);

    std::cout << "Add Bias 1D - Avg time: " << result.avg_time_ms
              << " ms, Throughput: " << result.gflops << " GFLOPS" << std::endl;

    result.saveToJson("cuda_outputs/add_bias_1d_benchmark.json");
}

TEST_F(AddBiasTest, Benchmark2D) {
    int block_size = 256;
    int grid_size = (total_size + block_size - 1) / block_size;

    auto kernel_func = [&]() {
        add_bias_2d_kernel<<<grid_size, block_size>>>(d_input, d_bias, batch_size, channels, spatial_size);
    };

    BenchmarkResult result = benchmarkKernel(
        kernel_func, total_size, "add_bias_2d", 10, 100);

    std::cout << "Add Bias 2D - Avg time: " << result.avg_time_ms
              << " ms, Throughput: " << result.gflops << " GFLOPS" << std::endl;

    result.saveToJson("cuda_outputs/add_bias_2d_benchmark.json");
}

TEST_F(AddBiasTest, DifferentConfigurations) {
    struct Config {
        int batch, channels, height, width;
    };

    std::vector<Config> configs = {
        {1, 32, 16, 16},
        {4, 64, 32, 32},
        {8, 128, 64, 64},
        {16, 256, 128, 128}
    };

    for (auto config : configs) {
        int test_spatial = config.height * config.width;
        int test_total = config.batch * config.channels * test_spatial;

        float *test_input, *test_bias;
        allocateMemory(&test_input, &h_input, test_total);
        allocateMemory(&test_bias, &h_bias, config.channels);

        initializeRandom(h_input, test_total, -10.0f, 10.0f);
        initializeRandom(h_bias, config.channels, -5.0f, 5.0f);
        copyToDevice(test_input, h_input, test_total);
        copyToDevice(test_bias, h_bias, config.channels);

        int block_size = 256;
        int grid_size = (test_total + block_size - 1) / block_size;

        add_bias_2d_kernel<<<grid_size, block_size>>>(
            test_input, test_bias, config.batch, config.channels, test_spatial);
        CUDA_CHECK(cudaDeviceSynchronize());

        copyFromDevice(h_output, test_input, test_total);

        // Verify a few random elements
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, test_total - 1);
        for (int i = 0; i < 20; ++i) {
            int idx = dis(gen);
            int c = (idx / test_spatial) % config.channels;
            float expected = h_input[idx] + h_bias[c];
            EXPECT_NEAR(h_output[idx], expected, 1e-4f)
                << "Config " << config.batch << "x" << config.channels << "x"
                << config.height << "x" << config.width << " mismatch at index " << idx;
        }

        std::string filename = "add_bias_config_" + std::to_string(config.batch) + "_" +
                               std::to_string(config.channels) + "_" +
                               std::to_string(config.height) + "_" +
                               std::to_string(config.width) + ".bin";
        saveOutput(filename.c_str(), h_output, test_total);

        freeMemory(test_input, h_input);
        freeMemory(test_bias, h_bias);
    }
}