#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <fstream>
#include <cmath>

#include "neural/backends/sycl/common_kernels.h"

using namespace lczero::sycl_backend;
using namespace std::chrono;

class GlobalAvgPoolTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create SYCL queue
        queue = sycl::queue{sycl::gpu_selector_v};

        // Print device info
        auto device = queue.get_device();
        std::cout << "Running on: " << device.get_info<sycl::info::device::name>() << std::endl;

        // Initialize random number generator
        rng.seed(42);

        // Create output directory
        system("mkdir -p sycl_outputs");
    }

    void TearDown() override {
        // Cleanup
    }

    // CPU reference implementation for global average pooling
    void globalAvgPoolCPU(float* output, const float* input, const float* prevLayerBias,
                        int N, int C, bool nhwc = false) {
        const int kPlaneSize = 64; // 8x8 board

        if (nhwc) {
            // NHWC layout (used with fp16)
            for (int n = 0; n < N; n++) {
                for (int c = 0; c < C; c++) {
                    float sum = 0.0f;
                    for (int i = 0; i < kPlaneSize; i++) {
                        int inputIndex = n * C * kPlaneSize + c + i * C;
                        sum += input[inputIndex];
                    }
                    float avg = sum / kPlaneSize;
                    if (prevLayerBias) {
                        avg += prevLayerBias[c];
                    }
                    output[n * C + c] = avg;
                }
            }
        } else {
            // NCHW layout (used with fp32)
            for (int n = 0; n < N; n++) {
                for (int c = 0; c < C; c++) {
                    float sum = 0.0f;
                    int planeStart = n * C * kPlaneSize + c * kPlaneSize;
                    for (int i = 0; i < kPlaneSize; i++) {
                        sum += input[planeStart + i];
                    }
                    float avg = sum / kPlaneSize;
                    if (prevLayerBias) {
                        avg += prevLayerBias[c];
                    }
                    output[n * C + c] = avg;
                }
            }
        }
    }

    // Save data to binary file
    void saveToBinary(const std::string& filename, const void* data, size_t size) {
        std::ofstream file(filename, std::ios::binary);
        file.write(reinterpret_cast<const char*>(data), size);
        file.close();
    }

    // Compare two arrays with tolerance
    bool compareArrays(const float* a, const float* b, size_t size, float tolerance = 1e-5f) {
        for (size_t i = 0; i < size; i++) {
            if (std::abs(a[i] - b[i]) > tolerance) {
                std::cout << "Mismatch at index " << i << ": expected " << a[i]
                         << ", got " << b[i] << ", diff: " << std::abs(a[i] - b[i]) << std::endl;
                return false;
            }
        }
        return true;
    }

    void generateRandomData(float* data, size_t size, float min = -5.0f, float max = 5.0f) {
        std::uniform_real_distribution<float> dist(min, max);
        for (size_t i = 0; i < size; i++) {
            data[i] = dist(rng);
        }
    }

    sycl::queue queue;
    std::mt19937 rng;
};

TEST_F(GlobalAvgPoolTest, BasicFunctionality_NCHW) {
    const int N = 2;
    const int C = 64;
    const int kPlaneSize = 64; // 8x8 board
    const int inputSize = N * C * kPlaneSize;
    const int outputSize = N * C;

    // Allocate host memory
    std::vector<float> h_input(inputSize);
    std::vector<float> h_bias(C);
    std::vector<float> h_sycl(outputSize);
    std::vector<float> h_cpu(outputSize);

    // Generate random input data
    generateRandomData(h_input.data(), inputSize);
    generateRandomData(h_bias.data(), C, -1.0f, 1.0f);

    // Allocate device memory (USM)
    float* d_input = sycl::malloc_device<float>(inputSize, queue);
    float* d_bias = sycl::malloc_device<float>(C, queue);
    float* d_output = sycl::malloc_device<float>(outputSize, queue);

    // Copy inputs to device
    queue.memcpy(d_input, h_input.data(), inputSize * sizeof(float)).wait();
    queue.memcpy(d_bias, h_bias.data(), C * sizeof(float)).wait();

    // Execute SYCL kernel
    globalAvgPool(queue, N, C, d_output, d_input, d_bias, false); // NCHW layout

    // Copy result back
    queue.memcpy(h_sycl.data(), d_output, outputSize * sizeof(float)).wait();

    // CPU reference
    globalAvgPoolCPU(h_cpu.data(), h_input.data(), h_bias.data(), N, C, false);

    // Save results
    saveToBinary("sycl_outputs/global_avg_pool_input.bin", h_input.data(), inputSize * sizeof(float));
    saveToBinary("sycl_outputs/global_avg_pool_bias.bin", h_bias.data(), C * sizeof(float));
    saveToBinary("sycl_outputs/global_avg_pool_output.bin", h_sycl.data(), outputSize * sizeof(float));

    // Verify correctness
    EXPECT_TRUE(compareArrays(h_cpu.data(), h_sycl.data(), outputSize));

    // Cleanup
    sycl::free(d_input, queue);
    sycl::free(d_bias, queue);
    sycl::free(d_output, queue);
}

TEST_F(GlobalAvgPoolTest, BasicFunctionality_NHWC_FP16) {
    const int N = 2;
    const int C = 64;
    const int kPlaneSize = 64; // 8x8 board
    const int inputSize = N * C * kPlaneSize;
    const int outputSize = N * C;

    // Allocate host memory
    std::vector<float> h_input_float(inputSize);
    std::vector<float> h_bias_float(C);
    std::vector<sycl::half> h_input(inputSize);
    std::vector<sycl::half> h_bias(C);
    std::vector<sycl::half> h_sycl(outputSize);
    std::vector<float> h_cpu(outputSize);

    // Generate random input data (in range suitable for fp16)
    generateRandomData(h_input_float.data(), inputSize, -5.0f, 5.0f);
    generateRandomData(h_bias_float.data(), C, -1.0f, 1.0f);

    // Convert to half
    for (int i = 0; i < inputSize; i++) h_input[i] = static_cast<sycl::half>(h_input_float[i]);
    for (int i = 0; i < C; i++) h_bias[i] = static_cast<sycl::half>(h_bias_float[i]);

    // Allocate device memory (USM)
    sycl::half* d_input = sycl::malloc_device<sycl::half>(inputSize, queue);
    sycl::half* d_bias = sycl::malloc_device<sycl::half>(C, queue);
    sycl::half* d_output = sycl::malloc_device<sycl::half>(outputSize, queue);

    // Copy inputs to device
    queue.memcpy(d_input, h_input.data(), inputSize * sizeof(sycl::half)).wait();
    queue.memcpy(d_bias, h_bias.data(), C * sizeof(sycl::half)).wait();

    // Execute SYCL kernel with NHWC layout
    globalAvgPool(queue, N, C, d_output, d_input, d_bias, true); // NHWC layout

    // Copy result back
    queue.memcpy(h_sycl.data(), d_output, outputSize * sizeof(sycl::half)).wait();

    // CPU reference with NHWC layout
    globalAvgPoolCPU(h_cpu.data(), h_input_float.data(), h_bias_float.data(), N, C, true);

    // Compare with tolerance for fp16
    for (int i = 0; i < outputSize; i++) {
        EXPECT_NEAR(float(h_sycl[i]), h_cpu[i], 1e-3f) << "FP16 mismatch at index " << i;
    }

    // Save FP16 results for comparison
    std::vector<float> h_sycl_float(outputSize);
    for (int i = 0; i < outputSize; i++) h_sycl_float[i] = float(h_sycl[i]);
    saveToBinary("sycl_outputs/global_avg_pool_fp16_output.bin", h_sycl_float.data(), outputSize * sizeof(float));

    // Cleanup
    sycl::free(d_input, queue);
    sycl::free(d_bias, queue);
    sycl::free(d_output, queue);
}

TEST_F(GlobalAvgPoolTest, WithoutBias) {
    const int N = 3;
    const int C = 128;
    const int kPlaneSize = 64; // 8x8 board
    const int inputSize = N * C * kPlaneSize;
    const int outputSize = N * C;

    // Allocate host memory
    std::vector<float> h_input(inputSize);
    std::vector<float> h_sycl(outputSize);
    std::vector<float> h_cpu(outputSize);

    // Generate random input data
    generateRandomData(h_input.data(), inputSize);

    // Allocate device memory (USM)
    float* d_input = sycl::malloc_device<float>(inputSize, queue);
    float* d_output = sycl::malloc_device<float>(outputSize, queue);

    // Copy input to device
    queue.memcpy(d_input, h_input.data(), inputSize * sizeof(float)).wait();

    // Execute SYCL kernel without bias
    globalAvgPool(queue, N, C, d_output, d_input, nullptr, false); // NCHW layout, no bias

    // Copy result back
    queue.memcpy(h_sycl.data(), d_output, outputSize * sizeof(float)).wait();

    // CPU reference without bias
    globalAvgPoolCPU(h_cpu.data(), h_input.data(), nullptr, N, C, false);

    // Verify correctness
    EXPECT_TRUE(compareArrays(h_cpu.data(), h_sycl.data(), outputSize));

    // Cleanup
    sycl::free(d_input, queue);
    sycl::free(d_output, queue);
}

TEST_F(GlobalAvgPoolTest, DifferentDimensions) {
    std::vector<std::tuple<int, int>> test_cases = {
        {1, 32},   // Single batch, 32 channels
        {4, 64},   // 4 batches, 64 channels
        {2, 128},  // 2 batches, 128 channels
        {8, 256},  // 8 batches, 256 channels
        {1, 384}   // Single batch, 384 channels
    };

    for (auto [N, C] : test_cases) {
        const int kPlaneSize = 64; // 8x8 board
        const int inputSize = N * C * kPlaneSize;
        const int outputSize = N * C;

        // Allocate host memory
        std::vector<float> h_input(inputSize);
        std::vector<float> h_bias(C);
        std::vector<float> h_sycl(outputSize);
        std::vector<float> h_cpu(outputSize);

        // Generate random input data
        generateRandomData(h_input.data(), inputSize);
        generateRandomData(h_bias.data(), C);

        // Allocate device memory (USM)
        float* d_input = sycl::malloc_device<float>(inputSize, queue);
        float* d_bias = sycl::malloc_device<float>(C, queue);
        float* d_output = sycl::malloc_device<float>(outputSize, queue);

        // Copy inputs to device
        queue.memcpy(d_input, h_input.data(), inputSize * sizeof(float)).wait();
        queue.memcpy(d_bias, h_bias.data(), C * sizeof(float)).wait();

        // Execute SYCL kernel
        globalAvgPool(queue, N, C, d_output, d_input, d_bias, false); // NCHW layout

        // Copy result back
        queue.memcpy(h_sycl.data(), d_output, outputSize * sizeof(float)).wait();

        // CPU reference
        globalAvgPoolCPU(h_cpu.data(), h_input.data(), h_bias.data(), N, C, false);

        // Verify correctness
        EXPECT_TRUE(compareArrays(h_cpu.data(), h_sycl.data(), outputSize))
            << "Failed for N=" << N << ", C=" << C;

        // Cleanup
        sycl::free(d_input, queue);
        sycl::free(d_bias, queue);
        sycl::free(d_output, queue);
    }
}

TEST_F(GlobalAvgPoolTest, PerformanceBenchmark) {
    const int N = 16;
    const int C = 384;
    const int kPlaneSize = 64; // 8x8 board
    const int inputSize = N * C * kPlaneSize;
    const int outputSize = N * C;

    // Allocate host memory
    std::vector<float> h_input(inputSize);
    std::vector<float> h_bias(C);
    std::vector<float> h_sycl(outputSize);

    // Generate random input data
    generateRandomData(h_input.data(), inputSize);
    generateRandomData(h_bias.data(), C);

    // Allocate device memory (USM)
    float* d_input = sycl::malloc_device<float>(inputSize, queue);
    float* d_bias = sycl::malloc_device<float>(C, queue);
    float* d_output = sycl::malloc_device<float>(outputSize, queue);

    // Copy inputs to device
    queue.memcpy(d_input, h_input.data(), inputSize * sizeof(float)).wait();
    queue.memcpy(d_bias, h_bias.data(), C * sizeof(float)).wait();

    // Warmup
    globalAvgPool(queue, N, C, d_output, d_input, d_bias, false);

    // Benchmark
    const int num_iterations = 100;
    auto start = high_resolution_clock::now();

    for (int i = 0; i < num_iterations; i++) {
        globalAvgPool(queue, N, C, d_output, d_input, d_bias, false);
    }

    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start);

    double avg_time_ms = duration.count() / (1000.0 * num_iterations);
    double bandwidth_gbps = (inputSize * sizeof(float) + outputSize * sizeof(float)) / (avg_time_ms * 1e6);
    double reduction_efficiency = (N * C * kPlaneSize) / (avg_time_ms * 1e6); // Reduction ops per second

    std::cout << "Average time: " << avg_time_ms << " ms" << std::endl;
    std::cout << "Effective bandwidth: " << bandwidth_gbps << " GB/s" << std::endl;
    std::cout << "Reduction efficiency: " << reduction_efficiency << " M elements/sec" << std::endl;

    // Save metrics
    std::ofstream metrics("sycl_outputs/benchmark_global_avg_pool.json");
    metrics << "{\"kernel\":\"globalAvgPool\",\"time_ms\":" << avg_time_ms
            << ",\"bandwidth_gbps\":" << bandwidth_gbps
            << ",\"reduction_ops_per_sec\":" << reduction_efficiency << "}" << std::endl;
    metrics.close();

    // Cleanup
    sycl::free(d_input, queue);
    sycl::free(d_bias, queue);
    sycl::free(d_output, queue);

    // Performance should be reasonable
    EXPECT_LT(avg_time_ms, 5.0) << "Kernel takes too long";
}

TEST_F(GlobalAvgPoolTest, EdgeCases) {
    // Test single element
    {
        const int N = 1;
        const int C = 1;
        const int kPlaneSize = 64;
        const int inputSize = N * C * kPlaneSize;
        const int outputSize = N * C;

        std::vector<float> h_input(inputSize, 3.14f); // All same value
        std::vector<float> h_sycl(outputSize);
        std::vector<float> h_cpu(outputSize);

        float* d_input = sycl::malloc_device<float>(inputSize, queue);
        float* d_output = sycl::malloc_device<float>(outputSize, queue);

        queue.memcpy(d_input, h_input.data(), inputSize * sizeof(float)).wait();

        globalAvgPool(queue, N, C, d_output, d_input, nullptr, false);
        queue.memcpy(h_sycl.data(), d_output, outputSize * sizeof(float)).wait();

        globalAvgPoolCPU(h_cpu.data(), h_input.data(), nullptr, N, C, false);

        // Expected result is the same as input since all values are identical
        EXPECT_NEAR(h_sycl[0], 3.14f, 1e-5f) << "Average should preserve constant values";
        EXPECT_NEAR(h_sycl[0], h_cpu[0], 1e-5f) << "CPU and SYCL results should match";

        sycl::free(d_input, queue);
        sycl::free(d_output, queue);
    }

    // Test with zero values
    {
        const int N = 2;
        const int C = 32;
        const int kPlaneSize = 64;
        const int inputSize = N * C * kPlaneSize;
        const int outputSize = N * C;

        std::vector<float> h_input(inputSize, 0.0f); // All zeros
        std::vector<float> h_bias(C, 1.0f); // Bias of 1
        std::vector<float> h_sycl(outputSize);
        std::vector<float> h_cpu(outputSize);

        float* d_input = sycl::malloc_device<float>(inputSize, queue);
        float* d_bias = sycl::malloc_device<float>(C, queue);
        float* d_output = sycl::malloc_device<float>(outputSize, queue);

        queue.memcpy(d_input, h_input.data(), inputSize * sizeof(float)).wait();
        queue.memcpy(d_bias, h_bias.data(), C * sizeof(float)).wait();

        globalAvgPool(queue, N, C, d_output, d_input, d_bias, false);
        queue.memcpy(h_sycl.data(), d_output, outputSize * sizeof(float)).wait();

        globalAvgPoolCPU(h_cpu.data(), h_input.data(), h_bias.data(), N, C, false);

        // Expected result is just the bias (1.0)
        for (int i = 0; i < outputSize; i++) {
            EXPECT_NEAR(h_sycl[i], 1.0f, 1e-5f) << "Zero input should yield bias output";
            EXPECT_NEAR(h_sycl[i], h_cpu[i], 1e-5f) << "CPU and SYCL results should match";
        }

        sycl::free(d_input, queue);
        sycl::free(d_bias, queue);
        sycl::free(d_output, queue);
    }

    // Test alternating pattern
    {
        const int N = 1;
        const int C = 8;
        const int kPlaneSize = 64;
        const int inputSize = N * C * kPlaneSize;
        const int outputSize = N * C;

        std::vector<float> h_input(inputSize);
        std::vector<float> h_sycl(outputSize);
        std::vector<float> h_cpu(outputSize);

        // Create alternating pattern: 0, 1, 0, 1, ...
        for (int i = 0; i < inputSize; i++) {
            h_input[i] = (i % 2) ? 1.0f : 0.0f;
        }

        float* d_input = sycl::malloc_device<float>(inputSize, queue);
        float* d_output = sycl::malloc_device<float>(outputSize, queue);

        queue.memcpy(d_input, h_input.data(), inputSize * sizeof(float)).wait();

        globalAvgPool(queue, N, C, d_output, d_input, nullptr, false);
        queue.memcpy(h_sycl.data(), d_output, outputSize * sizeof(float)).wait();

        globalAvgPoolCPU(h_cpu.data(), h_input.data(), nullptr, N, C, false);

        // Expected average is 0.5 for each plane (half zeros, half ones)
        for (int i = 0; i < outputSize; i++) {
            EXPECT_NEAR(h_sycl[i], 0.5f, 1e-5f) << "Alternating pattern should average to 0.5";
            EXPECT_NEAR(h_sycl[i], h_cpu[i], 1e-5f) << "CPU and SYCL results should match";
        }

        sycl::free(d_input, queue);
        sycl::free(d_output, queue);
    }
}