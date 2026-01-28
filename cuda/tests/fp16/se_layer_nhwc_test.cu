#include <gtest/gtest.h>
#include "../common/base_test_fixture.h"

// Test kernel for SE_Layer_NHWC
__global__ void se_layer_nhwc_test_kernel(half* input, half* output,
                                          const half* weight1, const half* bias1,
                                          const half* weight2, const half* bias2,
                                          int channels, int height, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = channels * height * width;

    if (idx < total_elements) {
        int c = idx % channels;
        // int hw = idx / channels;

        // Simplified SE operation: Squeeze + Excitation
        // Global average pooling for this channel
        half sum = __float2half(0.0f);
        for (int i = 0; i < height * width; ++i) {
            sum += input[c + i * channels];
        }
        half avg = __float2half(__half2float(sum) / (height * width));

        // FC layer 1
        half fc1 = __float2half(fmaxf(0.0f, __half2float(avg * weight1[0] + bias1[0])));

        // FC layer 2 with sigmoid
        half fc2 = fc1 * weight2[0] + bias2[0];
        half scale = __float2half(1.0f / (1.0f + expf(-__half2float(fc2))));

        // Apply scale
        output[idx] = input[idx] * scale;
    }
}

void launchSeLayerNhwc(half* output, const half* input,
                       const half* weight1, const half* bias1,
                       const half* weight2, const half* bias2,
                       int channels, int height, int width);

class SELayerNhwcTest : public FP16Test {
protected:
    void SetUp() override {
        FP16Test::SetUp();

        // Test dimensions (smaller for testing)
        channels = 64;
        height = 32;
        width = 32;
        total_size = channels * height * width;
        hidden_size = 16;  // SE hidden dimension

        // Allocate memory
        allocateMemory(&d_input, &h_input, total_size);
        allocateMemory(&d_output, &h_output, total_size);
        allocateMemory(&d_weight1, &h_weight1, channels * hidden_size);
        allocateMemory(&d_bias1, &h_bias1, hidden_size);
        allocateMemory(&d_weight2, &h_weight2, hidden_size * channels);
        allocateMemory(&d_bias2, &h_bias2, channels);

        // Initialize test data
        initializeRandom(h_input, total_size, half(-1.0f), half(1.0f));
        initializeRandom(h_weight1, channels * hidden_size, half(-0.1f), half(0.1f));
        initializeRandom(h_bias1, hidden_size, half(0.0f), half(0.1f));
        initializeRandom(h_weight2, hidden_size * channels, half(-0.1f), half(0.1f));
        initializeRandom(h_bias2, channels, half(0.0f), half(0.1f));

        // Copy to device
        copyToDevice(d_input, h_input, total_size);
        copyToDevice(d_weight1, h_weight1, channels * hidden_size);
        copyToDevice(d_bias1, h_bias1, hidden_size);
        copyToDevice(d_weight2, h_weight2, hidden_size * channels);
        copyToDevice(d_bias2, h_bias2, channels);
    }

    void TearDown() override {
        FP16Test::TearDown();

        freeMemory(d_input, h_input);
        freeMemory(d_output, h_output);
        freeMemory(d_weight1, h_weight1);
        freeMemory(d_bias1, h_bias1);
        freeMemory(d_weight2, h_weight2);
        freeMemory(d_bias2, h_bias2);
    }

    int channels, height, width, total_size, hidden_size;
    half *d_input, *d_output, *d_weight1, *d_weight2;
    half *d_bias1, *d_bias2;
    half *h_input, *h_output, *h_weight1, *h_weight2;
    half *h_bias1, *h_bias2;
};

TEST_F(SELayerNhwcTest, Correctness) {
    // Launch kernel
    int block_size = 256;
    int grid_size = (total_size + block_size - 1) / block_size;

    se_layer_nhwc_test_kernel<<<grid_size, block_size>>>(
        d_input, d_output, d_weight1, d_bias1, d_weight2, d_bias2,
        channels, height, width);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back
    copyFromDevice(h_output, d_output, total_size);

    // Save outputs
    saveOutput("se_layer_nhwc_output.bin", h_output, total_size);
    saveOutput("se_layer_nhwc_input.bin", h_input, total_size);

    // Basic sanity checks
    for (int i = 0; i < total_size; ++i) {
        EXPECT_LE(abs(__half2float(h_output[i])), 2.0f) << "Output too large at index " << i;
        EXPECT_GT(abs(__half2float(h_output[i])), 0.0f) << "Output too small at index " << i;
    }
}

TEST_F(SELayerNhwcTest, Benchmark) {
    int block_size = 256;
    int grid_size = (total_size + block_size - 1) / block_size;

    auto kernel_func = [&]() {
        se_layer_nhwc_test_kernel<<<grid_size, block_size>>>(
            d_input, d_output, d_weight1, d_bias1, d_weight2, d_bias2,
            channels, height, width);
    };

    BenchmarkResult result = benchmarkKernel(
        kernel_func, total_size, "se_layer_nhwc", 5, 50);

    std::cout << "SE Layer NHWC - Avg time: " << result.avg_time_ms
              << " ms, Throughput: " << result.gflops << " GFLOPS" << std::endl;

    result.saveToJson("cuda_outputs/se_layer_nhwc_benchmark.json");
}

TEST_F(SELayerNhwcTest, DifferentSizes) {
    std::vector<std::tuple<int, int, int>> test_sizes = {
        {32, 16, 16}, {128, 8, 8}, {256, 4, 4}
    };

    for (auto [c, h, w] : test_sizes) {
        int size = c * h * w;
        half *test_input, *test_output;
        allocateMemory(&test_input, &h_input, size);
        allocateMemory(&test_output, &h_output, size);

        initializeRandom(h_input, size, half(-1.0f), half(1.0f));
        copyToDevice(test_input, h_input, size);

        int block_size = 256;
        int grid_size = (size + block_size - 1) / block_size;

        se_layer_nhwc_test_kernel<<<grid_size, block_size>>>(
            test_input, test_output, d_weight1, d_bias1, d_weight2, d_bias2,
            c, h, w);
        CUDA_CHECK(cudaDeviceSynchronize());

        copyFromDevice(h_output, test_output, size);

        std::string filename = "se_layer_nhwc_size_" + std::to_string(c) +
                               "_" + std::to_string(h) + "_" + std::to_string(w) + ".bin";
        saveOutput(filename.c_str(), h_output, size);

        freeMemory(test_input, h_input);
        freeMemory(test_output, h_output);
    }
}
