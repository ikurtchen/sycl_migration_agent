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

class AddBiasBatchedTest : public ::testing::Test {
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

    // CPU reference implementation for add bias batched
    void addBiasBatchedCPU(float* output, const float* input, const float* bias,
                          int Batch, int N, int C, ActivationFunction activation) {
        for (int batch = 0; batch < Batch; batch++) {
            for (int n = 0; n < N; n++) {
                for (int c = 0; c < C; c++) {
                    int inputIndex = batch * N * C + n * C + c;
                    int biasIndex = batch * C + c;

                    float val = input[inputIndex] + bias[biasIndex];

                    // Apply activation function
                    switch (activation) {
                        case ActivationFunction::ACTIVATION_NONE:
                            break;
                        case ActivationFunction::ACTIVATION_RELU:
                            val = std::max(0.0f, val);
                            break;
                        case ActivationFunction::ACTIVATION_RELU_2:
                            val = val > 0 ? val * val : 0.0f;
                            break;
                        case ActivationFunction::ACTIVATION_SELU: {
                            constexpr float alpha = 1.67326324f;
                            constexpr float scale = 1.05070098f;
                            if (val > 0)
                                val = scale * val;
                            else
                                val = scale * alpha * (std::exp(val) - 1.0f);
                            break;
                        }
                        case ActivationFunction::ACTIVATION_MISH: {
                            float e = std::exp(val);
                            float n = e * e + 2.0f * e;
                            float d = val / (n + 2.0f);
                            if (val <= -0.6f) {
                                val = n * d;
                            } else {
                                val = val - 2.0f * d;
                            }
                            break;
                        }
                        case ActivationFunction::ACTIVATION_SWISH:
                            val = val / (1.0f + std::exp(-val));
                            break;
                        default:
                            break;
                    }

                    output[inputIndex] = val;
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

    void generateRandomData(float* data, size_t size, float min = -1.0f, float max = 1.0f) {
        std::uniform_real_distribution<float> dist(min, max);
        for (size_t i = 0; i < size; i++) {
            data[i] = dist(rng);
        }
    }

    sycl::queue queue;
    std::mt19937 rng;
};

TEST_F(AddBiasBatchedTest, BasicFunctionality_None) {
    const int Batch = 3;
    const int N = 8;
    const int C = 64; // Must be multiple of 4 and <= 4096
    const int totalSize = Batch * N * C;

    // Allocate host memory
    std::vector<float> h_input(totalSize);
    std::vector<float> h_bias(Batch * C);
    std::vector<float> h_sycl(totalSize);
    std::vector<float> h_cpu(totalSize);

    // Generate random input data
    generateRandomData(h_input.data(), totalSize, -2.0f, 2.0f);
    generateRandomData(h_bias.data(), Batch * C, -1.0f, 1.0f);

    // Allocate device memory (USM)
    float* d_input = sycl::malloc_device<float>(totalSize, queue);
    float* d_bias = sycl::malloc_device<float>(Batch * C, queue);
    float* d_output = sycl::malloc_device<float>(totalSize, queue);

    // Copy inputs to device
    queue.memcpy(d_input, h_input.data(), totalSize * sizeof(float)).wait();
    queue.memcpy(d_bias, h_bias.data(), (Batch * C) * sizeof(float)).wait();

    // Execute SYCL kernel
    addBiasBatched(d_output, d_input, d_bias, Batch, N, C, ActivationFunction::ACTIVATION_NONE, queue);

    // Copy result back
    queue.memcpy(h_sycl.data(), d_output, totalSize * sizeof(float)).wait();

    // CPU reference
    addBiasBatchedCPU(h_cpu.data(), h_input.data(), h_bias.data(), Batch, N, C,
                     ActivationFunction::ACTIVATION_NONE);

    // Save results
    saveToBinary("sycl_outputs/add_bias_batched_input.bin", h_input.data(), totalSize * sizeof(float));
    saveToBinary("sycl_outputs/add_bias_batched_bias.bin", h_bias.data(), (Batch * C) * sizeof(float));
    saveToBinary("sycl_outputs/add_bias_batched_output.bin", h_sycl.data(), totalSize * sizeof(float));

    // Verify correctness
    EXPECT_TRUE(compareArrays(h_cpu.data(), h_sycl.data(), totalSize));

    // Cleanup
    sycl::free(d_input, queue);
    sycl::free(d_bias, queue);
    sycl::free(d_output, queue);
}

TEST_F(AddBiasBatchedTest, ActivationFunctions) {
    const int Batch = 2;
    const int N = 4;
    const int C = 128; // Must be multiple of 4 and <= 4096
    const int totalSize = Batch * N * C;

    std::vector<ActivationFunction> activations = {
        ActivationFunction::ACTIVATION_NONE,
        ActivationFunction::ACTIVATION_RELU,
        ActivationFunction::ACTIVATION_RELU_2,
        ActivationFunction::ACTIVATION_SELU,
        ActivationFunction::ACTIVATION_SWISH
    };

    for (auto activation : activations) {
        // Allocate host memory
        std::vector<float> h_input(totalSize);
        std::vector<float> h_bias(Batch * C);
        std::vector<float> h_sycl(totalSize);
        std::vector<float> h_cpu(totalSize);

        // Generate random input data
        generateRandomData(h_input.data(), totalSize, -2.0f, 2.0f);
        generateRandomData(h_bias.data(), Batch * C, -1.0f, 1.0f);

        // Allocate device memory (USM)
        float* d_input = sycl::malloc_device<float>(totalSize, queue);
        float* d_bias = sycl::malloc_device<float>(Batch * C, queue);
        float* d_output = sycl::malloc_device<float>(totalSize, queue);

        // Copy inputs to device
        queue.memcpy(d_input, h_input.data(), totalSize * sizeof(float)).wait();
        queue.memcpy(d_bias, h_bias.data(), (Batch * C) * sizeof(float)).wait();

        // Execute SYCL kernel
        addBiasBatched(d_output, d_input, d_bias, Batch, N, C, activation, queue);

        // Copy result back
        queue.memcpy(h_sycl.data(), d_output, totalSize * sizeof(float)).wait();

        // CPU reference
        addBiasBatchedCPU(h_cpu.data(), h_input.data(), h_bias.data(), Batch, N, C, activation);

        // Verify correctness
        EXPECT_TRUE(compareArrays(h_cpu.data(), h_sycl.data(), totalSize, 1e-5f))
            << "Failed with activation " << static_cast<int>(activation);

        // Cleanup
        sycl::free(d_input, queue);
        sycl::free(d_bias, queue);
        sycl::free(d_output, queue);
    }
}

TEST_F(AddBiasBatchedTest, WithStride) {
    const int Batch = 2;
    const int N = 8;
    const int Nstride = 10; // Padded dimension
    const int C = 64; // Must be multiple of 4 and <= 4096
    const int totalSize = Batch * N * C;
    const int inputSize = Batch * Nstride * C;

    // Allocate host memory (with padding)
    std::vector<float> h_input(inputSize, 0.0f); // Initialize to zero
    std::vector<float> h_bias(Batch * C);
    std::vector<float> h_sycl(totalSize);
    std::vector<float> h_cpu(totalSize);

    // Generate random input data (only for valid entries)
    for (int batch = 0; batch < Batch; batch++) {
        for (int n = 0; n < N; n++) {
            for (int c = 0; c < C; c++) {
                int idx = batch * Nstride * C + n * C + c;
                h_input[idx] = static_cast<float>(rng() % 100) / 50.0f - 1.0f;
            }
        }
    }
    generateRandomData(h_bias.data(), Batch * C, -1.0f, 1.0f);

    // Allocate device memory (USM)
    float* d_input = sycl::malloc_device<float>(inputSize, queue);
    float* d_bias = sycl::malloc_device<float>(Batch * C, queue);
    float* d_output = sycl::malloc_device<float>(totalSize, queue);

    // Copy inputs to device
    queue.memcpy(d_input, h_input.data(), inputSize * sizeof(float)).wait();
    queue.memcpy(d_bias, h_bias.data(), (Batch * C) * sizeof(float)).wait();

    // Execute SYCL kernel with stride
    addBiasBatched(d_output, d_input, d_bias, Batch, N, C, Nstride, ActivationFunction::ACTIVATION_RELU, queue);

    // Copy result back
    queue.memcpy(h_sycl.data(), d_output, totalSize * sizeof(float)).wait();

    // CPU reference
    for (int batch = 0; batch < Batch; batch++) {
        for (int n = 0; n < N; n++) {
            for (int c = 0; c < C; c++) {
                int outputIdx = batch * N * C + n * C + c;
                int inputIdx = batch * Nstride * C + n * C + c;
                int biasIdx = batch * C + c;

                float val = h_input[inputIdx] + h_bias[biasIdx];
                val = std::max(0.0f, val); // ReLU
                h_cpu[outputIdx] = val;
            }
        }
    }

    // Verify correctness
    EXPECT_TRUE(compareArrays(h_cpu.data(), h_sycl.data(), totalSize));

    // Cleanup
    sycl::free(d_input, queue);
    sycl::free(d_bias, queue);
    sycl::free(d_output, queue);
}

TEST_F(AddBiasBatchedTest, FP16_Precision) {
    const int Batch = 2;
    const int N = 4;
    const int C = 256; // Must be multiple of 4 and <= 4096
    const int totalSize = Batch * N * C;

    // Allocate host memory
    std::vector<float> h_input_float(totalSize);
    std::vector<float> h_bias_float(Batch * C);
    std::vector<sycl::half> h_input(totalSize);
    std::vector<sycl::half> h_bias(Batch * C);
    std::vector<sycl::half> h_sycl(totalSize);
    std::vector<float> h_cpu(totalSize);

    // Generate random input data (in range suitable for fp16)
    generateRandomData(h_input_float.data(), totalSize, -2.0f, 2.0f);
    generateRandomData(h_bias_float.data(), Batch * C, -1.0f, 1.0f);

    // Convert to half
    for (int i = 0; i < totalSize; i++) h_input[i] = static_cast<sycl::half>(h_input_float[i]);
    for (int i = 0; i < Batch * C; i++) h_bias[i] = static_cast<sycl::half>(h_bias_float[i]);

    // Allocate device memory (USM)
    sycl::half* d_input = sycl::malloc_device<sycl::half>(totalSize, queue);
    sycl::half* d_bias = sycl::malloc_device<sycl::half>(Batch * C, queue);
    sycl::half* d_output = sycl::malloc_device<sycl::half>(totalSize, queue);

    // Copy inputs to device
    queue.memcpy(d_input, h_input.data(), totalSize * sizeof(sycl::half)).wait();
    queue.memcpy(d_bias, h_bias.data(), (Batch * C) * sizeof(sycl::half)).wait();

    // Execute SYCL kernel
    addBiasBatched(d_output, d_input, d_bias, Batch, N, C, ActivationFunction::ACTIVATION_RELU, queue);

    // Copy result back
    queue.memcpy(h_sycl.data(), d_output, totalSize * sizeof(sycl::half)).wait();

    // CPU reference (using float)
    addBiasBatchedCPU(h_cpu.data(), h_input_float.data(), h_bias_float.data(), Batch, N, C,
                     ActivationFunction::ACTIVATION_RELU);

    // Compare with tolerance for fp16
    for (int i = 0; i < totalSize; i++) {
        EXPECT_NEAR(float(h_sycl[i]), h_cpu[i], 1e-3f) << "FP16 mismatch at index " << i;
    }

    // Cleanup
    sycl::free(d_input, queue);
    sycl::free(d_bias, queue);
    sycl::free(d_output, queue);
}

TEST_F(AddBiasBatchedTest, PerformanceBenchmark) {
    const int Batch = 4;
    const int N = 1024;
    const int C = 1024; // Must be multiple of 4 and <= 4096
    const int totalSize = Batch * N * C;

    // Allocate host memory
    std::vector<float> h_input(totalSize);
    std::vector<float> h_bias(Batch * C);
    std::vector<float> h_sycl(totalSize);

    // Generate random input data
    generateRandomData(h_input.data(), totalSize);
    generateRandomData(h_bias.data(), Batch * C);

    // Allocate device memory (USM)
    float* d_input = sycl::malloc_device<float>(totalSize, queue);
    float* d_bias = sycl::malloc_device<float>(Batch * C, queue);
    float* d_output = sycl::malloc_device<float>(totalSize, queue);

    // Copy inputs to device
    queue.memcpy(d_input, h_input.data(), totalSize * sizeof(float)).wait();
    queue.memcpy(d_bias, h_bias.data(), (Batch * C) * sizeof(float)).wait();

    // Warmup
    addBiasBatched(d_output, d_input, d_bias, Batch, N, C, ActivationFunction::ACTIVATION_NONE, queue);

    // Benchmark
    const int num_iterations = 50;
    auto start = high_resolution_clock::now();

    for (int i = 0; i < num_iterations; i++) {
        addBiasBatched(d_output, d_input, d_bias, Batch, N, C, ActivationFunction::ACTIVATION_RELU, queue);
    }

    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start);

    double avg_time_ms = duration.count() / (1000.0 * num_iterations);
    double bandwidth_gbps = (3.0 * totalSize * sizeof(float)) / (avg_time_ms * 1e6); // 3x for read, read, write

    std::cout << "Average time: " << avg_time_ms << " ms" << std::endl;
    std::cout << "Effective bandwidth: " << bandwidth_gbps << " GB/s" << std::endl;

    // Save metrics
    std::ofstream metrics("sycl_outputs/benchmark_add_bias_batched.json");
    metrics << "{\"kernel\":\"addBiasBatched\",\"time_ms\":" << avg_time_ms
            << ",\"bandwidth_gbps\":" << bandwidth_gbps << "}" << std::endl;
    metrics.close();

    // Cleanup
    sycl::free(d_input, queue);
    sycl::free(d_bias, queue);
    sycl::free(d_output, queue);

    // Performance should be reasonable
    EXPECT_LT(avg_time_ms, 20.0) << "Kernel takes too long";
}

TEST_F(AddBiasBatchedTest, EdgeCases) {
    // Test minimum valid C (must be multiple of 4)
    {
        const int Batch = 1;
        const int N = 2;
        const int C = 4;
        const int totalSize = Batch * N * C;

        std::vector<float> h_input(totalSize);
        std::vector<float> h_bias(Batch * C);
        std::vector<float> h_sycl(totalSize);
        std::vector<float> h_cpu(totalSize);

        generateRandomData(h_input.data(), totalSize);
        generateRandomData(h_bias.data(), Batch * C);

        float* d_input = sycl::malloc_device<float>(totalSize, queue);
        float* d_bias = sycl::malloc_device<float>(Batch * C, queue);
        float* d_output = sycl::malloc_device<float>(totalSize, queue);

        queue.memcpy(d_input, h_input.data(), totalSize * sizeof(float)).wait();
        queue.memcpy(d_bias, h_bias.data(), (Batch * C) * sizeof(float)).wait();

        addBiasBatched(d_output, d_input, d_bias, Batch, N, C, ActivationFunction::ACTIVATION_NONE, queue);
        queue.memcpy(h_sycl.data(), d_output, totalSize * sizeof(float)).wait();

        addBiasBatchedCPU(h_cpu.data(), h_input.data(), h_bias.data(), Batch, N, C,
                         ActivationFunction::ACTIVATION_NONE);

        EXPECT_TRUE(compareArrays(h_cpu.data(), h_sycl.data(), totalSize));

        sycl::free(d_input, queue);
        sycl::free(d_bias, queue);
        sycl::free(d_output, queue);
    }

    // Test error case - invalid C size
    {
        const int Batch = 1;
        const int N = 2;
        const int C = 7; // Not multiple of

        EXPECT_THROW({
            addBiasBatched(nullptr, nullptr, nullptr, Batch, N, C, ActivationFunction::ACTIVATION_NONE, queue);
        }, std::runtime_error);
    }
}