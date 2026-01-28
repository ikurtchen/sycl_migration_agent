#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <fstream>
#include <cmath>

#include "neural/backends/sycl/common_kernels.h"
#include "neural/tables/activation_function.h"

using namespace lczero::sycl_backend;
using namespace std::chrono;

class BatchNormTest : public ::testing::Test {
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

    // CPU reference implementation for batch normalization
    void batchNormCPU(float* output, const float* input, const float* skipInput,
                     int N, int C, int H, int W, const float* means,
                     const float* varMultipliers, ActivationFunction activation,
                     bool nhwc = false) {
        const int total_elements = N * C * H * W;

        for (int idx = 0; idx < total_elements; idx++) {
            int wIndex;
            if (nhwc) {
                wIndex = idx % C;  // NHWC layout
            } else {
                wIndex = (idx / (H * W)) % C;  // NCHW layout
            }

            float el = input[idx];
            float mean = means[wIndex];
            float varMulti = varMultipliers[wIndex];

            // Batch normalization: (x - mean) * variance_multiplier
            el -= mean;
            el *= varMulti;

            // Add skip connection if present
            if (skipInput) {
                el += skipInput[idx];
            }

            // Apply activation function
            switch (activation) {
                case ActivationFunction::ACTIVATION_NONE:
                    break;
                case ActivationFunction::ACTIVATION_RELU:
                    el = std::max(0.0f, el);
                    break;
                case ActivationFunction::ACTIVATION_RELU_2:
                    el = el > 0 ? el * el : 0.0f;
                    break;
                case ActivationFunction::ACTIVATION_SIGMOID:
                    el = 1.0f / (1.0f + std::exp(-el));
                    break;
                default:
                    break;
            }

            output[idx] = el;
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

    void generateRandomData(float* data, size_t size, float min = -2.0f, float max = 2.0f) {
        std::uniform_real_distribution<float> dist(min, max);
        for (size_t i = 0; i < size; i++) {
            data[i] = dist(rng);
        }
    }

    void generateNormalizationParams(float* means, float* varMultis, int C, float mean_range = 1.0f) {
        std::uniform_real_distribution<float> mean_dist(-mean_range, mean_range);
        std::uniform_real_distribution<float> var_dist(0.5f, 2.0f);
        for (int i = 0; i < C; i++) {
            means[i] = mean_dist(rng);
            varMultis[i] = var_dist(rng);
        }
    }

    sycl::queue queue;
    std::mt19937 rng;
};

TEST_F(BatchNormTest, BasicFunctionality_NCHW) {
    const int N = 4;
    const int C = 64;
    const int H = 8;
    const int W = 8;
    const int totalSize = N * C * H * W;

    // Allocate host memory
    std::vector<float> h_input(totalSize);
    std::vector<float> h_skip(totalSize);
    std::vector<float> h_means(C);
    std::vector<float> h_varMultis(C);
    std::vector<float> h_sycl(totalSize);
    std::vector<float> h_cpu(totalSize);

    // Generate random input data
    generateRandomData(h_input.data(), totalSize);
    generateRandomData(h_skip.data(), totalSize, -0.5f, 0.5f);
    generateNormalizationParams(h_means.data(), h_varMultis.data(), C);

    // Allocate device memory (USM)
    float* d_input = sycl::malloc_device<float>(totalSize, queue);
    float* d_skip = sycl::malloc_device<float>(totalSize, queue);
    float* d_means = sycl::malloc_device<float>(C, queue);
    float* d_varMultis = sycl::malloc_device<float>(C, queue);
    float* d_output = sycl::malloc_device<float>(totalSize, queue);

    // Copy inputs to device
    queue.memcpy(d_input, h_input.data(), totalSize * sizeof(float)).wait();
    queue.memcpy(d_skip, h_skip.data(), totalSize * sizeof(float)).wait();
    queue.memcpy(d_means, h_means.data(), C * sizeof(float)).wait();
    queue.memcpy(d_varMultis, h_varMultis.data(), C * sizeof(float)).wait();

    // Execute SYCL kernel
    batchNorm(queue, d_output, d_input, d_skip, N, C, H, W, d_means, d_varMultis,
              ActivationFunction::ACTIVATION_NONE);

    // Copy result back
    queue.memcpy(h_sycl.data(), d_output, totalSize * sizeof(float)).wait();

    // CPU reference
    batchNormCPU(h_cpu.data(), h_input.data(), h_skip.data(), N, C, H, W,
                 h_means.data(), h_varMultis.data(), ActivationFunction::ACTIVATION_NONE);

    // Save results
    saveToBinary("sycl_outputs/batch_norm_input.bin", h_input.data(), totalSize * sizeof(float));
    saveToBinary("sycl_outputs/batch_norm_skip.bin", h_skip.data(), totalSize * sizeof(float));
    saveToBinary("sycl_outputs/batch_norm_means.bin", h_means.data(), C * sizeof(float));
    saveToBinary("sycl_outputs/batch_norm_var_multis.bin", h_varMultis.data(), C * sizeof(float));
    saveToBinary("sycl_outputs/batch_norm_output.bin", h_sycl.data(), totalSize * sizeof(float));

    // Verify correctness
    EXPECT_TRUE(compareArrays(h_cpu.data(), h_sycl.data(), totalSize));

    // Cleanup
    sycl::free(d_input, queue);
    sycl::free(d_skip, queue);
    sycl::free(d_means, queue);
    sycl::free(d_varMultis, queue);
    sycl::free(d_output, queue);
}

TEST_F(BatchNormTest, WithActivation_NCHW) {
    const int N = 2;
    const int C = 32;
    const int H = 8;
    const int W = 8;
    const int totalSize = N * C * H * W;

    std::vector<ActivationFunction> activations = {
        ActivationFunction::ACTIVATION_NONE,
        ActivationFunction::ACTIVATION_RELU,
        ActivationFunction::ACTIVATION_RELU_2,
        ActivationFunction::ACTIVATION_SIGMOID
    };

    for (auto activation : activations) {
        // Allocate host memory
        std::vector<float> h_input(totalSize);
        std::vector<float> h_skip(totalSize);
        std::vector<float> h_means(C);
        std::vector<float> h_varMultis(C);
        std::vector<float> h_sycl(totalSize);
        std::vector<float> h_cpu(totalSize);

        // Generate random input data
        generateRandomData(h_input.data(), totalSize);
        generateRandomData(h_skip.data(), totalSize, -0.5f, 0.5f);
        generateNormalizationParams(h_means.data(), h_varMultis.data(), C);

        // Allocate device memory (USM)
        float* d_input = sycl::malloc_device<float>(totalSize, queue);
        float* d_skip = sycl::malloc_device<float>(totalSize, queue);
        float* d_means = sycl::malloc_device<float>(C, queue);
        float* d_varMultis = sycl::malloc_device<float>(C, queue);
        float* d_output = sycl::malloc_device<float>(totalSize, queue);

        // Copy inputs to device
        queue.memcpy(d_input, h_input.data(), totalSize * sizeof(float)).wait();
        queue.memcpy(d_skip, h_skip.data(), totalSize * sizeof(float)).wait();
        queue.memcpy(d_means, h_means.data(), C * sizeof(float)).wait();
        queue.memcpy(d_varMultis, h_varMultis.data(), C * sizeof(float)).wait();

        // Execute SYCL kernel
        batchNorm(queue, d_output, d_input, d_skip, N, C, H, W, d_means, d_varMultis, activation);

        // Copy result back
        queue.memcpy(h_sycl.data(), d_output, totalSize * sizeof(float)).wait();

        // CPU reference
        batchNormCPU(h_cpu.data(), h_input.data(), h_skip.data(), N, C, H, W,
                     h_means.data(), h_varMultis.data(), activation);

        // Verify correctness
        EXPECT_TRUE(compareArrays(h_cpu.data(), h_sycl.data(), totalSize, 1e-5f))
            << "Failed with activation " << static_cast<int>(activation);

        // Cleanup
        sycl::free(d_input, queue);
        sycl::free(d_skip, queue);
        sycl::free(d_means, queue);
        sycl::free(d_varMultis, queue);
        sycl::free(d_output, queue);
    }
}

TEST_F(BatchNormTest, WithoutSkipConnection) {
    const int N = 2;
    const int C = 64;
    const int H = 8;
    const int W = 8;
    const int totalSize = N * C * H * W;

    // Allocate host memory
    std::vector<float> h_input(totalSize);
    std::vector<float> h_means(C);
    std::vector<float> h_varMultis(C);
    std::vector<float> h_sycl(totalSize);
    std::vector<float> h_cpu(totalSize);

    // Generate random input data
    generateRandomData(h_input.data(), totalSize);
    generateNormalizationParams(h_means.data(), h_varMultis.data(), C);

    // Allocate device memory (USM)
    float* d_input = sycl::malloc_device<float>(totalSize, queue);
    float* d_means = sycl::malloc_device<float>(C, queue);
    float* d_varMultis = sycl::malloc_device<float>(C, queue);
    float* d_output = sycl::malloc_device<float>(totalSize, queue);

    // Copy inputs to device
    queue.memcpy(d_input, h_input.data(), totalSize * sizeof(float)).wait();
    queue.memcpy(d_means, h_means.data(), C * sizeof(float)).wait();
    queue.memcpy(d_varMultis, h_varMultis.data(), C * sizeof(float)).wait();

    // Execute SYCL kernel (skip = nullptr)
    batchNorm(queue, d_output, d_input, nullptr, N, C, H, W, d_means, d_varMultis,
              ActivationFunction::ACTIVATION_RELU);

    // Copy result back
    queue.memcpy(h_sycl.data(), d_output, totalSize * sizeof(float)).wait();

    // CPU reference (skip = nullptr)
    batchNormCPU(h_cpu.data(), h_input.data(), nullptr, N, C, H, W,
                 h_means.data(), h_varMultis.data(), ActivationFunction::ACTIVATION_RELU);

    // Verify correctness
    EXPECT_TRUE(compareArrays(h_cpu.data(), h_sycl.data(), totalSize));

    // Cleanup
    sycl::free(d_input, queue);
    sycl::free(d_means, queue);
    sycl::free(d_varMultis, queue);
    sycl::free(d_output, queue);
}

TEST_F(BatchNormTest, FP16_NHWC) {
    const int N = 2;
    const int C = 64;
    const int H = 8;
    const int W = 8;
    const int totalSize = N * C * H * W;

    // Allocate host memory
    std::vector<float> h_input_float(totalSize);
    std::vector<float> h_skip_float(totalSize);
    std::vector<float> h_means(C);
    std::vector<float> h_varMultis(C);
    std::vector<sycl::half> h_input(totalSize);
    std::vector<sycl::half> h_skip(totalSize);
    std::vector<sycl::half> h_sycl(totalSize);
    std::vector<float> h_cpu(totalSize);

    // Generate random input data (in range suitable for fp16)
    generateRandomData(h_input_float.data(), totalSize, -2.0f, 2.0f);
    generateRandomData(h_skip_float.data(), totalSize, -0.5f, 0.5f);
    generateNormalizationParams(h_means.data(), h_varMultis.data(), C);

    // Convert to half
    for (int i = 0; i < totalSize; i++) {
        h_input[i] = static_cast<sycl::half>(h_input_float[i]);
        h_skip[i] = static_cast<sycl::half>(h_skip_float[i]);
    }

    // Allocate device memory (USM)
    sycl::half* d_input = sycl::malloc_device<sycl::half>(totalSize, queue);
    sycl::half* d_skip = sycl::malloc_device<sycl::half>(totalSize, queue);
    float* d_means = sycl::malloc_device<float>(C, queue);
    float* d_varMultis = sycl::malloc_device<float>(C, queue);
    sycl::half* d_output = sycl::malloc_device<sycl::half>(totalSize, queue);

    // Copy inputs to device
    queue.memcpy(d_input, h_input.data(), totalSize * sizeof(sycl::half)).wait();
    queue.memcpy(d_skip, h_skip.data(), totalSize * sizeof(sycl::half)).wait();
    queue.memcpy(d_means, h_means.data(), C * sizeof(float)).wait();
    queue.memcpy(d_varMultis, h_varMultis.data(), C * sizeof(float)).wait();

    // Execute SYCL kernel with NHWC layout
    batchNorm(queue, d_output, d_input, d_skip, N, C, H, W, d_means, d_varMultis,
              ActivationFunction::ACTIVATION_RELU);

    // Copy result back
    queue.memcpy(h_sycl.data(), d_output, totalSize * sizeof(sycl::half)).wait();

    // CPU reference with NHWC layout
    batchNormCPU(h_cpu.data(), h_input_float.data(), h_skip_float.data(), N, C, H, W,
                 h_means.data(), h_varMultis.data(), ActivationFunction::ACTIVATION_RELU, true);

    // Compare with tolerance for fp16
    for (int i = 0; i < totalSize; i++) {
        EXPECT_NEAR(float(h_sycl[i]), h_cpu[i], 1e-3f) << "FP16 mismatch at index " << i;
    }

    // Save FP16 results for comparison
    std::vector<float> h_sycl_float(totalSize);
    for (int i = 0; i < totalSize; i++) h_sycl_float[i] = float(h_sycl[i]);
    saveToBinary("sycl_outputs/batch_norm_fp16_output.bin", h_sycl_float.data(), totalSize * sizeof(float));

    // Cleanup
    sycl::free(d_input, queue);
    sycl::free(d_skip, queue);
    sycl::free(d_means, queue);
    sycl::free(d_varMultis, queue);
    sycl::free(d_output, queue);
}

TEST_F(BatchNormTest, PerformanceBenchmark) {
    const int N = 8;
    const int C = 256;
    const int H = 8;
    const int W = 8;
    const int totalSize = N * C * H * W;

    // Allocate host memory
    std::vector<float> h_input(totalSize);
    std::vector<float> h_skip(totalSize);
    std::vector<float> h_means(C);
    std::vector<float> h_varMultis(C);
    std::vector<float> h_sycl(totalSize);

    // Generate random input data
    generateRandomData(h_input.data(), totalSize);
    generateRandomData(h_skip.data(), totalSize, -0.5f, 0.5f);
    generateNormalizationParams(h_means.data(), h_varMultis.data(), C);

    // Allocate device memory (USM)
    float* d_input = sycl::malloc_device<float>(totalSize, queue);
    float* d_skip = sycl::malloc_device<float>(totalSize, queue);
    float* d_means = sycl::malloc_device<float>(C, queue);
    float* d_varMultis = sycl::malloc_device<float>(C, queue);
    float* d_output = sycl::malloc_device<float>(totalSize, queue);

    // Copy inputs to device
    queue.memcpy(d_input, h_input.data(), totalSize * sizeof(float)).wait();
    queue.memcpy(d_skip, h_skip.data(), totalSize * sizeof(float)).wait();
    queue.memcpy(d_means, h_means.data(), C * sizeof(float)).wait();
    queue.memcpy(d_varMultis, h_varMultis.data(), C * sizeof(float)).wait();

    // Warmup
    batchNorm(queue, d_output, d_input, d_skip, N, C, H, W, d_means, d_varMultis,
              ActivationFunction::ACTIVATION_NONE);

    // Benchmark
    const int num_iterations = 100;
    auto start = high_resolution_clock::now();

    for (int i = 0; i < num_iterations; i++) {
        batchNorm(queue, d_output, d_input, d_skip, N, C, H, W, d_means, d_varMultis,
                 ActivationFunction::ACTIVATION_RELU);
    }

    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start);

    double avg_time_ms = duration.count() / (1000.0 * num_iterations);
    double bandwidth_gbps = (3.0 * totalSize * sizeof(float) + 2.0 * C * sizeof(float)) / (avg_time_ms * 1e6);

    std::cout << "Average time: " << avg_time_ms << " ms" << std::endl;
    std::cout << "Effective bandwidth: " << bandwidth_gbps << " GB/s" << std::endl;

    // Save metrics
    std::ofstream metrics("sycl_outputs/benchmark_batch_norm.json");
    metrics << "{\"kernel\":\"batchNorm\",\"time_ms\":" << avg_time_ms
            << ",\"bandwidth_gbps\":" << bandwidth_gbps << "}" << std::endl;
    metrics.close();

    // Cleanup
    sycl::free(d_input, queue);
    sycl::free(d_skip, queue);
    sycl::free(d_means, queue);
    sycl::free(d_varMultis, queue);
    sycl::free(d_output, queue);

    // Performance should be reasonable
    EXPECT_LT(avg_time_ms, 15.0) << "Kernel takes too long";
}

TEST_F(BatchNormTest, EdgeCases) {
    // Test single channel
    {
        const int N = 1;
        const int C = 1;
        const int H = 4;
        const int W = 4;
        const int totalSize = N * C * H * W;

        std::vector<float> h_input(totalSize);
        std::vector<float> h_means(C);
        std::vector<float> h_varMultis(C);
        std::vector<float> h_sycl(totalSize);
        std::vector<float> h_cpu(totalSize);

        generateRandomData(h_input.data(), totalSize);
        h_means[0] = 0.5f;
        h_varMultis[0] = 1.5f;

        float* d_input = sycl::malloc_device<float>(totalSize, queue);
        float* d_means = sycl::malloc_device<float>(C, queue);
        float* d_varMultis = sycl::malloc_device<float>(C, queue);
        float* d_output = sycl::malloc_device<float>(totalSize, queue);

        queue.memcpy(d_input, h_input.data(), totalSize * sizeof(float)).wait();
        queue.memcpy(d_means, h_means.data(), C * sizeof(float)).wait();
        queue.memcpy(d_varMultis, h_varMultis.data(), C * sizeof(float)).wait();

        batchNorm(queue, d_output, d_input, nullptr, N, C, H, W, d_means, d_varMultis,
                  ActivationFunction::ACTIVATION_NONE);
        queue.memcpy(h_sycl.data(), d_output, totalSize * sizeof(float)).wait();

        batchNormCPU(h_cpu.data(), h_input.data(), nullptr, N, C, H, W,
                     h_means.data(), h_varMultis.data(), ActivationFunction::ACTIVATION_NONE);

        EXPECT_TRUE(compareArrays(h_cpu.data(), h_sycl.data(), totalSize));

        sycl::free(d_input, queue);
        sycl::free(d_means, queue);
        sycl::free(d_varMultis, queue);
        sycl::free(d_output, queue);
    }

    // Test zero variance multiplier
    {
        const int N = 2;
        const int C = 4;
        const int H = 2;
        const int W = 2;
        const int totalSize = N * C * H * W;

        std::vector<float> h_input(totalSize);
        std::vector<float> h_means(C);
        std::vector<float> h_varMultis(C, 0.0f); // All zeros
        std::vector<float> h_sycl(totalSize);
        std::vector<float> h_cpu(totalSize);

        generateRandomData(h_input.data(), totalSize);
        generateNormalizationParams(h_means.data(), h_varMultis.data(), C);
        for (int i = 0; i < C; i++) h_varMultis[i] = 0.0f; // Override with zeros

        float* d_input = sycl::malloc_device<float>(totalSize, queue);
        float* d_means = sycl::malloc_device<float>(C, queue);
        float* d_varMultis = sycl::malloc_device<float>(C, queue);
        float* d_output = sycl::malloc_device<float>(totalSize, queue);

        queue.memcpy(d_input, h_input.data(), totalSize * sizeof(float)).wait();
        queue.memcpy(d_means, h_means.data(), C * sizeof(float)).wait();
        queue.memcpy(d_varMultis, h_varMultis.data(), C * sizeof(float)).wait();

        batchNorm(queue, d_output, d_input, nullptr, N, C, H, W, d_means, d_varMultis,
                  ActivationFunction::ACTIVATION_NONE);
        queue.memcpy(h_sycl.data(), d_output, totalSize * sizeof(float)).wait();

        batchNormCPU(h_cpu.data(), h_input.data(), nullptr, N, C, H, W,
                     h_means.data(), h_varMultis.data(), ActivationFunction::ACTIVATION_NONE);

        EXPECT_TRUE(compareArrays(h_cpu.data(), h_sycl.data(), totalSize));

        sycl::free(d_input, queue);
        sycl::free(d_means, queue);
        sycl::free(d_varMultis, queue);
        sycl::free(d_output, queue);
    }
}