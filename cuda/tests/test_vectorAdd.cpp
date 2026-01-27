#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>

// Kernel declaration
__global__ void vectorAdd(const float *A, const float *B, float *C, int numElements);

class VectorAddTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Set random seed for reproducible tests
        srand(42);
    }

    void TearDown() override {
        // Cleanup handled by individual test cases
    }

    void testVectorAdd(int numElements, const std::string& test_name) {
        size_t size = numElements * sizeof(float);

        // Allocate host memory
        std::vector<float> h_A(numElements);
        std::vector<float> h_B(numElements);
        std::vector<float> h_C(numElements);
        std::vector<float> h_C_ref(numElements);

        // Initialize with test data
        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dis(-1000.0f, 1000.0f);
        for (int i = 0; i < numElements; i++) {
            h_A[i] = dis(gen);
            h_B[i] = dis(gen);
            h_C_ref[i] = h_A[i] + h_B[i] + 0.0f;
        }

        // Allocate device memory
        float *d_A = nullptr;
        float *d_B = nullptr;
        float *d_C = nullptr;

        ASSERT_EQ(cudaSuccess, cudaMalloc(&d_A, size));
        ASSERT_EQ(cudaSuccess, cudaMalloc(&d_B, size));
        ASSERT_EQ(cudaSuccess, cudaMalloc(&d_C, size));

        // Copy to device
        ASSERT_EQ(cudaSuccess, cudaMemcpy(d_A, h_A.data(), size, cudaMemcpyHostToDevice));
        ASSERT_EQ(cudaSuccess, cudaMemcpy(d_B, h_B.data(), size, cudaMemcpyHostToDevice));

        // Launch kernel
        int threadsPerBlock = 256;
        int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

        vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
        ASSERT_EQ(cudaSuccess, cudaGetLastError());

        // Copy result back
        ASSERT_EQ(cudaSuccess, cudaMemcpy(h_C.data(), d_C, size, cudaMemcpyDeviceToHost));

        // Verify results
        const float tolerance = 1e-5f;
        for (int i = 0; i < numElements; i++) {
            EXPECT_NEAR(h_C[i], h_C_ref[i], tolerance)
                << "Mismatch at element " << i << " in test " << test_name
                << " (A=" << h_A[i] << ", B=" << h_B[i]
                << ", Expected=" << h_C_ref[i] << ", Got=" << h_C[i] << ")";
        }

        // Save results for comparison
        saveResults(h_C.data(), size, test_name);

        // Cleanup
        ASSERT_EQ(cudaSuccess, cudaFree(d_A));
        ASSERT_EQ(cudaSuccess, cudaFree(d_B));
        ASSERT_EQ(cudaSuccess, cudaFree(d_C));
    }

    void saveResults(const float* data, size_t size, const std::string& test_name) {
        std::string filename = "cuda_outputs/vectorAdd_" + test_name + "_output.bin";
        std::string path = "/localdisk/kurt/workspace/code/ai_coding/sycl_migration_agent/cuda/tests/" + filename;

        std::ofstream file(path, std::ios::binary);
        if (file.is_open()) {
            file.write(reinterpret_cast<const char*>(data), size);
            file.close();
        } else {
            std::cerr << "Warning: Could not save results to " << filename << std::endl;
        }
    }

    void saveInputData(const std::vector<float>& A, const std::vector<float>& B, const std::string& test_name) {
        std::string filename_A = "cuda_inputs/vectorAdd_" + test_name + "_input_A.bin";
        std::string filename_B = "cuda_inputs/vectorAdd_" + test_name + "_input_B.bin";
        std::string path_A = "/localdisk/kurt/workspace/code/ai_coding/sycl_migration_agent/cuda/tests/" + filename_A;
        std::string path_B = "/localdisk/kurt/workspace/code/ai_coding/sycl_migration_agent/cuda/tests/" + filename_B;

        // Create inputs directory if it doesn't exist
        system("mkdir -p /localdisk/kurt/workspace/code/ai_coding/sycl_migration_agent/cuda/tests/cuda_inputs");

        std::ofstream file_A(path_A, std::ios::binary);
        std::ofstream file_B(path_B, std::ios::binary);

        if (file_A.is_open()) {
            file_A.write(reinterpret_cast<const char*>(A.data()), A.size() * sizeof(float));
            file_A.close();
        }

        if (file_B.is_open()) {
            file_B.write(reinterpret_cast<const char*>(B.data()), B.size() * sizeof(float));
            file_B.close();
        }
    }

    std::pair<double, double> benchmarkVectorAdd(int numElements, int iterations = 100) {
        size_t size = numElements * sizeof(float);

        // Allocate host memory
        std::vector<float> h_A(numElements);
        std::vector<float> h_B(numElements);
        std::vector<float> h_C(numElements);

        // Initialize with test data
        for (int i = 0; i < numElements; i++) {
            h_A[i] = static_cast<float>(rand()) / RAND_MAX;
            h_B[i] = static_cast<float>(rand()) / RAND_MAX;
        }

        // Allocate device memory
        float *d_A = nullptr;
        float *d_B = nullptr;
        float *d_C = nullptr;

        ASSERT_EQ(cudaSuccess, cudaMalloc(&d_A, size));
        ASSERT_EQ(cudaSuccess, cudaMalloc(&d_B, size));
        ASSERT_EQ(cudaSuccess, cudaMalloc(&d_C, size));

        // Copy to device
        ASSERT_EQ(cudaSuccess, cudaMemcpy(d_A, h_A.data(), size, cudaMemcpyHostToDevice));
        ASSERT_EQ(cudaSuccess, cudaMemcpy(d_B, h_B.data(), size, cudaMemcpyHostToDevice));

        // Setup kernel launch parameters
        int threadsPerBlock = 256;
        int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

        // Warmup
        vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
        ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

        // Benchmark
        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < iterations; i++) {
            vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
        }
        ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        double avg_time_us = static_cast<double>(duration.count()) / iterations;
        double avg_time_ms = avg_time_us / 1000.0;

        // Calculate throughput in GFLOPS (2 FLOPs per element: 1 add + 1 copy)
        double gflops = (2.0 * numElements) / (avg_time_us * 1e3);

        // Cleanup
        ASSERT_EQ(cudaSuccess, cudaFree(d_A));
        ASSERT_EQ(cudaSuccess, cudaFree(d_B));
        ASSERT_EQ(cudaSuccess, cudaFree(d_C));

        return std::make_pair(avg_time_ms, gflops);
    }
};

// Test small vector size
TEST_F(VectorAddTest, SmallSize_1000) {
    testVectorAdd(1000, "small_1000");
}

// Test medium vector size
TEST_F(VectorAddTest, MediumSize_50000) {
    testVectorAdd(50000, "medium_50000");
}

// Test large vector size
TEST_F(VectorAddTest, LargeSize_1000000) {
    testVectorAdd(1000000, "large_1000000");
}

// Test very large vector size
TEST_F(VectorAddTest, VeryLargeSize_10000000) {
    testVectorAdd(10000000, "verylarge_10000000");
}

// Test edge case: single element
TEST_F(VectorAddTest, SingleElement) {
    testVectorAdd(1, "single");
}

// Test edge case: power of 2 boundary
TEST_F(VectorAddTest, PowerOf2Boundary) {
    testVectorAdd(256, "powerof2_256");
}

// Test negative values
TEST_F(VectorAddTest, NegativeValues) {
    int numElements = 1000;
    size_t size = numElements * sizeof(float);

    // Allocate host memory
    std::vector<float> h_A(numElements);
    std::vector<float> h_B(numElements);
    std::vector<float> h_C(numElements);
    std::vector<float> h_C_ref(numElements);

    // Initialize with negative values
    for (int i = 0; i < numElements; i++) {
        h_A[i] = -100.0f - static_cast<float>(i);
        h_B[i] = -200.0f - static_cast<float>(i);
        h_C_ref[i] = h_A[i] + h_B[i] + 0.0f;
    }

    // Allocate device memory
    float *d_A = nullptr;
    float *d_B = nullptr;
    float *d_C = nullptr;

    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_A, size));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_B, size));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_C, size));

    // Copy to device
    ASSERT_EQ(cudaSuccess, cudaMemcpy(d_A, h_A.data(), size, cudaMemcpyHostToDevice));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(d_B, h_B.data(), size, cudaMemcpyHostToDevice));

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
    ASSERT_EQ(cudaSuccess, cudaGetLastError());

    // Copy result back
    ASSERT_EQ(cudaSuccess, cudaMemcpy(h_C.data(), d_C, size, cudaMemcpyDeviceToHost));

    // Verify results
    const float tolerance = 1e-5f;
    for (int i = 0; i < numElements; i++) {
        EXPECT_NEAR(h_C[i], h_C_ref[i], tolerance)
            << "Negative value test failed at element " << i;
    }

    // Save results
    saveResults(h_C.data(), size, "negative_values");

    // Cleanup
    ASSERT_EQ(cudaSuccess, cudaFree(d_A));
    ASSERT_EQ(cudaSuccess, cudaFree(d_B));
    ASSERT_EQ(cudaSuccess, cudaFree(d_C));
}

// Test zero values
TEST_F(VectorAddTest, ZeroValues) {
    int numElements = 1000;
    size_t size = numElements * sizeof(float);

    // Allocate host memory
    std::vector<float> h_A(numElements, 0.0f);
    std::vector<float> h_B(numElements, 0.0f);
    std::vector<float> h_C(numElements);
    std::vector<float> h_C_ref(numElements, 0.0f);

    // Allocate device memory
    float *d_A = nullptr;
    float *d_B = nullptr;
    float *d_C = nullptr;

    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_A, size));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_B, size));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_C, size));

    // Copy to device
    ASSERT_EQ(cudaSuccess, cudaMemcpy(d_A, h_A.data(), size, cudaMemcpyHostToDevice));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(d_B, h_B.data(), size, cudaMemcpyHostToDevice));

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
    ASSERT_EQ(cudaSuccess, cudaGetLastError());

    // Copy result back
    ASSERT_EQ(cudaSuccess, cudaMemcpy(h_C.data(), d_C, size, cudaMemcpyDeviceToHost));

    // Verify results
    const float tolerance = 1e-5f;
    for (int i = 0; i < numElements; i++) {
        EXPECT_NEAR(h_C[i], h_C_ref[i], tolerance)
            << "Zero value test failed at element " << i;
    }

    // Save results
    saveResults(h_C.data(), size, "zero_values");

    // Cleanup
    ASSERT_EQ(cudaSuccess, cudaFree(d_A));
    ASSERT_EQ(cudaSuccess, cudaFree(d_B));
    ASSERT_EQ(cudaSuccess, cudaFree(d_C));
}

// Test maximum values
TEST_F(VectorAddTest, MaximumValues) {
    int numElements = 1000;
    size_t size = numElements * sizeof(float);

    // Allocate host memory
    std::vector<float> h_A(numElements, FLT_MAX);
    std::vector<float> h_B(numElements, FLT_MAX);
    std::vector<float> h_C(numElements);
    std::vector<float> h_C_ref(numElements, FLT_MAX + FLT_MAX);  // This will be inf

    // Allocate device memory
    float *d_A = nullptr;
    float *d_B = nullptr;
    float *d_C = nullptr;

    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_A, size));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_B, size));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_C, size));

    // Copy to device
    ASSERT_EQ(cudaSuccess, cudaMemcpy(d_A, h_A.data(), size, cudaMemcpyHostToDevice));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(d_B, h_B.data(), size, cudaMemcpyHostToDevice));

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
    ASSERT_EQ(cudaSuccess, cudaGetLastError());

    // Copy result back
    ASSERT_EQ(cudaSuccess, cudaMemcpy(h_C.data(), d_C, size, cudaMemcpyDeviceToHost));

    // Verify results (should be infinity)
    for (int i = 0; i < numElements; i++) {
        EXPECT_TRUE(std::isinf(h_C[i]))
            << "Maximum value test failed at element " << i << ", got " << h_C[i];
    }

    // Save results
    saveResults(h_C.data(), size, "max_values");

    // Cleanup
    ASSERT_EQ(cudaSuccess, cudaFree(d_A));
    ASSERT_EQ(cudaSuccess, cudaFree(d_B));
    ASSERT_EQ(cudaSuccess, cudaFree(d_C));
}

// Benchmark tests
TEST_F(VectorAddTest, Benchmark_Small) {
    auto result = benchmarkVectorAdd(1000);
    std::cout << "Small (1000 elements): " << result.first << " ms, "
              << result.second << " GFLOPS" << std::endl;

    // Save metrics
    std::ofstream metrics("cuda_outputs/vectorAdd_benchmark_small.json");
    metrics << "{\"kernel\":\"vectorAdd\",\"size\":1000,\"time_ms\":" << result.first
            << ",\"gflops\":" << result.second << "}" << std::endl;
}

TEST_F(VectorAddTest, Benchmark_Medium) {
    auto result = benchmarkVectorAdd(50000);
    std::cout << "Medium (50000 elements): " << result.first << " ms, "
              << result.second << " GFLOPS" << std::endl;

    // Save metrics
    std::ofstream metrics("cuda_outputs/vectorAdd_benchmark_medium.json");
    metrics << "{\"kernel\":\"vectorAdd\",\"size\":50000,\"time_ms\":" << result.first
            << ",\"gflops\":" << result.second << "}" << std::endl;
}

TEST_F(VectorAddTest, Benchmark_Large) {
    auto result = benchmarkVectorAdd(1000000);
    std::cout << "Large (1000000 elements): " << result.first << " ms, "
              << result.second << " GFLOPS" << std::endl;

    // Save metrics
    std::ofstream metrics("cuda_outputs/vectorAdd_benchmark_large.json");
    metrics << "{\"kernel\":\"vectorAdd\",\"size\":1000000,\"time_ms\":" << result.first
            << ",\"gflops\":" << result.second << "}" << std::endl;
}

TEST_F(VectorAddTest, Benchmark_VeryLarge) {
    auto result = benchmarkVectorAdd(10000000);
    std::cout << "Very Large (10000000 elements): " << result.first << " ms, "
              << result.second << " GFLOPS" << std::endl;

    // Save metrics
    std::ofstream metrics("cuda_outputs/vectorAdd_benchmark_verylarge.json");
    metrics << "{\"kernel\":\"vectorAdd\",\"size\":10000000,\"time_ms\":" << result.first
            << ",\"gflops\":" << result.second << "}" << std::endl;
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
