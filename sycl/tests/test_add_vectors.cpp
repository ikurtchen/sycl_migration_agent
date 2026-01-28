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

class AddVectorsTest : public ::testing::Test {
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

    // CPU reference implementation for add vectors
    void addVectorsCPU(float* output, const float* a, const float* b, int size,
                      int asize, int bsize, ActivationFunction activation) {
        for (int i = 0; i < size; i++) {
            float aVal = a ? a[i % asize] : 0.0f;
            float bVal = b ? b[i % bsize] : 0.0f;
            float cVal = aVal + bVal;

            // Apply activation function
            switch (activation) {
                case ActivationFunction::ACTIVATION_NONE:
                    break;
                case ActivationFunction::ACTIVATION_RELU:
                    cVal = std::max(0.0f, cVal);
                    break;
                case ActivationFunction::ACTIVATION_RELU_2:
                    cVal = cVal > 0 ? cVal * cVal : 0.0f;
                    break;
                case ActivationFunction::ACTIVATION_TANH:
                    cVal = std::tanh(cVal);
                    break;
                case ActivationFunction::ACTIVATION_SIGMOID:
                    cVal = 1.0f / (1.0f + std::exp(-cVal));
                    break;
                case ActivationFunction::ACTIVATION_SWISH:
                    cVal = cVal / (1.0f + std::exp(-cVal));
                    break;
                default:
                    break;
            }
            output[i] = cVal;
        }
    }

    // Save data to binary file
    void saveToBinary(const std::string& filename, const void* data, size_t size) {
        std::ofstream file(filename, std::ios::binary);
        file.write(reinterpret_cast<const char*>(data), size);
        file.close();
    }

    // Load data from binary file
    void loadFromBinary(const std::string& filename, void* data, size_t size) {
        std::ifstream file(filename, std::ios::binary);
        file.read(reinterpret_cast<char*>(data), size);
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

TEST_F(AddVectorsTest, BasicFunctionality_None) {
    const int size = 1024;
    const int asize = 256;
    const int bsize = 512;

    // Allocate host memory
    std::vector<float> h_a(asize);
    std::vector<float> h_b(bsize);
    std::vector<float> h_sycl(size);
    std::vector<float> h_cpu(size);

    // Generate random input data
    generateRandomData(h_a.data(), asize);
    generateRandomData(h_b.data(), bsize);

    // Allocate device memory (USM)
    float* d_a = sycl::malloc_device<float>(asize, queue);
    float* d_b = sycl::malloc_device<float>(bsize, queue);
    float* d_c = sycl::malloc_device<float>(size, queue);

    // Copy inputs to device
    queue.memcpy(d_a, h_a.data(), asize * sizeof(float)).wait();
    queue.memcpy(d_b, h_b.data(), bsize * sizeof(float)).wait();

    // Execute SYCL kernel
    addVectors(d_c, d_a, d_b, size, asize, bsize, ActivationFunction::ACTIVATION_NONE, queue);

    // Copy result back
    queue.memcpy(h_sycl.data(), d_c, size * sizeof(float)).wait();

    // CPU reference
    addVectorsCPU(h_cpu.data(), h_a.data(), h_b.data(), size, asize, bsize,
                  ActivationFunction::ACTIVATION_NONE);

    // Save results
    saveToBinary("sycl_outputs/add_vectors_input_A.bin", h_a.data(), asize * sizeof(float));
    saveToBinary("sycl_outputs/add_vectors_input_B.bin", h_b.data(), bsize * sizeof(float));
    saveToBinary("sycl_outputs/add_vectors_output.bin", h_sycl.data(), size * sizeof(float));

    // Verify correctness
    EXPECT_TRUE(compareArrays(h_cpu.data(), h_sycl.data(), size));

    // Cleanup
    sycl::free(d_a, queue);
    sycl::free(d_b, queue);
    sycl::free(d_c, queue);
}

TEST_F(AddVectorsTest, BasicFunctionality_RelU) {
    const int size = 1024;
    const int asize = 256;
    const int bsize = 512;

    // Allocate host memory
    std::vector<float> h_a(asize);
    std::vector<float> h_b(bsize);
    std::vector<float> h_sycl(size);
    std::vector<float> h_cpu(size);

    // Generate random input data
    generateRandomData(h_a.data(), asize, -5.0f, 5.0f);
    generateRandomData(h_b.data(), bsize, -5.0f, 5.0f);

    // Allocate device memory (USM)
    float* d_a = sycl::malloc_device<float>(asize, queue);
    float* d_b = sycl::malloc_device<float>(bsize, queue);
    float* d_c = sycl::malloc_device<float>(size, queue);

    // Copy inputs to device
    queue.memcpy(d_a, h_a.data(), asize * sizeof(float)).wait();
    queue.memcpy(d_b, h_b.data(), bsize * sizeof(float)).wait();

    // Execute SYCL kernel
    addVectors(d_c, d_a, d_b, size, asize, bsize, ActivationFunction::ACTIVATION_RELU, queue);

    // Copy result back
    queue.memcpy(h_sycl.data(), d_c, size * sizeof(float)).wait();

    // CPU reference
    addVectorsCPU(h_cpu.data(), h_a.data(), h_b.data(), size, asize, bsize,
                  ActivationFunction::ACTIVATION_RELU);

    // Verify correctness
    EXPECT_TRUE(compareArrays(h_cpu.data(), h_sycl.data(), size));

    // Cleanup
    sycl::free(d_a, queue);
    sycl::free(d_b, queue);
    sycl::free(d_c, queue);
}

TEST_F(AddVectorsTest, BasicFunctionality_FP16) {
    const int size = 1024;
    const int asize = 256;
    const int bsize = 512;

    // Allocate host memory
    std::vector<sycl::half> h_a(asize);
    std::vector<sycl::half> h_b(bsize);
    std::vector<sycl::half> h_sycl(size);
    std::vector<float> h_a_float(asize);
    std::vector<float> h_b_float(bsize);
    std::vector<float> h_cpu(size);

    // Generate random input data
    generateRandomData(h_a_float.data(), asize, -1.0f, 1.0f);
    generateRandomData(h_b_float.data(), bsize, -1.0f, 1.0f);

    // Convert to half
    for (int i = 0; i < asize; i++) h_a[i] = static_cast<sycl::half>(h_a_float[i]);
    for (int i = 0; i < bsize; i++) h_b[i] = static_cast<sycl::half>(h_b_float[i]);

    // Allocate device memory (USM)
    sycl::half* d_a = sycl::malloc_device<sycl::half>(asize, queue);
    sycl::half* d_b = sycl::malloc_device<sycl::half>(bsize, queue);
    sycl::half* d_c = sycl::malloc_device<sycl::half>(size, queue);

    // Copy inputs to device
    queue.memcpy(d_a, h_a.data(), asize * sizeof(sycl::half)).wait();
    queue.memcpy(d_b, h_b.data(), bsize * sizeof(sycl::half)).wait();

    // Execute SYCL kernel
    addVectors(d_c, d_a, d_b, size, asize, bsize, ActivationFunction::ACTIVATION_NONE, queue);

    // Copy result back
    queue.memcpy(h_sycl.data(), d_c, size * sizeof(sycl::half)).wait();

    // CPU reference (using float for comparison)
    addVectorsCPU(h_cpu.data(), h_a_float.data(), h_b_float.data(), size, asize, bsize,
                  ActivationFunction::ACTIVATION_NONE);

    // Compare with tolerance for fp16
    for (int i = 0; i < size; i++) {
        EXPECT_NEAR(float(h_sycl[i]), h_cpu[i], 1e-3f) << "Mismatch at index " << i;
    }

    // Save results for comparison
    std::vector<float> h_sycl_float(size);
    for (int i = 0; i < size; i++) h_sycl_float[i] = float(h_sycl[i]);
    saveToBinary("sycl_outputs/add_vectors_fp16_output.bin", h_sycl_float.data(), size * sizeof(float));

    // Cleanup
    sycl::free(d_a, queue);
    sycl::free(d_b, queue);
    sycl::free(d_c, queue);
}

TEST_F(AddVectorsTest, PerformanceBenchmark) {
    const int size = 4 * 1024 * 1024; // 4M elements
    const int asize = 1024;
    const int bsize = 1024;

    // Allocate host memory
    std::vector<float> h_a(asize);
    std::vector<float> h_b(bsize);
    std::vector<float> h_sycl(size);

    // Generate random input data
    generateRandomData(h_a.data(), asize);
    generateRandomData(h_b.data(), bsize);

    // Allocate device memory (USM)
    float* d_a = sycl::malloc_device<float>(asize, queue);
    float* d_b = sycl::malloc_device<float>(bsize, queue);
    float* d_c = sycl::malloc_device<float>(size, queue);

    // Copy inputs to device
    queue.memcpy(d_a, h_a.data(), asize * sizeof(float)).wait();
    queue.memcpy(d_b, h_b.data(), bsize * sizeof(float)).wait();

    // Warmup
    addVectors(d_c, d_a, d_b, size, asize, bsize, ActivationFunction::ACTIVATION_NONE, queue);

    // Benchmark
    const int num_iterations = 100;
    auto start = high_resolution_clock::now();

    for (int i = 0; i < num_iterations; i++) {
        addVectors(d_c, d_a, d_b, size, asize, bsize, ActivationFunction::ACTIVATION_RELU, queue);
    }

    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start);

    double avg_time_ms = duration.count() / (1000.0 * num_iterations);
    double bandwidth_gbps = (3.0 * size * sizeof(float)) / (avg_time_ms * 1e6); // 3x for read, read, write

    std::cout << "Average time: " << avg_time_ms << " ms" << std::endl;
    std::cout << "Effective bandwidth: " << bandwidth_gbps << " GB/s" << std::endl;

    // Save metrics
    std::ofstream metrics("sycl_outputs/benchmark_add_vectors.json");
    metrics << "{\"kernel\":\"addVectors\",\"time_ms\":" << avg_time_ms
            << ",\"bandwidth_gbps\":" << bandwidth_gbps << "}" << std::endl;
    metrics.close();

    // Cleanup
    sycl::free(d_a, queue);
    sycl::free(d_b, queue);
    sycl::free(d_c, queue);

    // Performance should be reasonable
    EXPECT_LT(avg_time_ms, 10.0) << "Kernel takes too long";
}

TEST_F(AddVectorsTest, EdgeCases) {
    const int size = 8;

    // Test case 1: nullptr inputs
    {
        std::vector<float> h_sycl(size);
        float* d_c = sycl::malloc_device<float>(size, queue);

        // Both nullptr
        addVectors(d_c, nullptr, nullptr, size, 0, 0, ActivationFunction::ACTIVATION_NONE, queue);
        queue.memcpy(h_sycl.data(), d_c, size * sizeof(float)).wait();

        for (int i = 0; i < size; i++) {
            EXPECT_EQ(h_sycl[i], 0.0f) << "Expected all zeros with nullptr inputs";
        }

        sycl::free(d_c, queue);
    }

    // Test case 2: Single element vectors
    {
        std::vector<float> h_a(1);
        std::vector<float> h_b(1);
        std::vector<float> h_sycl(size);
        std::vector<float> h_cpu(size);

        h_a[0] = 3.14f;
        h_b[0] = 2.71f;

        float* d_a = sycl::malloc_device<float>(1, queue);
        float* d_b = sycl::malloc_device<float>(1, queue);
        float* d_c = sycl::malloc_device<float>(size, queue);

        queue.memcpy(d_a, h_a.data(), sizeof(float)).wait();
        queue.memcpy(d_b, h_b.data(), sizeof(float)).wait();

        addVectors(d_c, d_a, d_b, size, 1, 1, ActivationFunction::ACTIVATION_NONE, queue);
        queue.memcpy(h_sycl.data(), d_c, size * sizeof(float)).wait();

        addVectorsCPU(h_cpu.data(), h_a.data(), h_b.data(), size, 1, 1,
                      ActivationFunction::ACTIVATION_NONE);

        EXPECT_TRUE(compareArrays(h_cpu.data(), h_sycl.data(), size));

        sycl::free(d_a, queue);
        sycl::free(d_b, queue);
        sycl::free(d_c, queue);
    }
}