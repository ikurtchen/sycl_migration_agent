#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <string>
#include <filesystem>
#include <cstdlib>
#include <unistd.h>
#include <sys/stat.h>
#include "vectorAdd_kernel.h"

// Helper function to get the executable's directory
std::filesystem::path getCurrentDir() {
    return std::filesystem::current_path();
}

// Helper function to create directory if it doesn't exist
void ensureDirExists(const std::filesystem::path& dir) {
    if (!std::filesystem::exists(dir)) {
        std::filesystem::create_directories(dir);
    }
}

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

        // Launch kernel using the launch function
        cudaError_t cudaStatus = launchVectorAdd(d_A, d_B, d_C, numElements);
        ASSERT_EQ(cudaSuccess, cudaStatus) << "Kernel launch failed";

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

        // Save inputs for comparison
        saveInputData(h_A, h_B, test_name);

        // Save results for comparison
        saveResults(h_C.data(), size, test_name);

        // Cleanup
        ASSERT_EQ(cudaSuccess, cudaFree(d_A));
        ASSERT_EQ(cudaSuccess, cudaFree(d_B));
        ASSERT_EQ(cudaSuccess, cudaFree(d_C));
    }

    void saveResults(const float* data, size_t size, const std::string& test_name) {
        std::string filename = "cuda_outputs/vectorAdd_" + test_name + "_output.bin";
        std::filesystem::path exeDir = getCurrentDir();
        std::filesystem::path outputPath = exeDir / filename;

        ensureDirExists(outputPath.parent_path());

        std::ofstream file(outputPath, std::ios::binary);
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
        std::filesystem::path exeDir = getCurrentDir();
        std::filesystem::path path_A = exeDir / filename_A;
        std::filesystem::path path_B = exeDir / filename_B;

        // Create inputs directory if it doesn't exist
        ensureDirExists(path_A.parent_path());

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

        // Initialize with test data (same seed as other tests for consistency)
        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dis(-1000.0f, 1000.0f);
        for (int i = 0; i < numElements; i++) {
            h_A[i] = dis(gen);
            h_B[i] = dis(gen);
        }

        // Allocate device memory - use EXPECT instead of ASSERT since this function returns a value
        float *d_A = nullptr;
        float *d_B = nullptr;
        float *d_C = nullptr;

        EXPECT_EQ(cudaSuccess, cudaMalloc(&d_A, size)) << "Failed to allocate device memory for A";
        EXPECT_EQ(cudaSuccess, cudaMalloc(&d_B, size)) << "Failed to allocate device memory for B";
        EXPECT_EQ(cudaSuccess, cudaMalloc(&d_C, size)) << "Failed to allocate device memory for C";

        // Check if any allocation failed
        if (d_A == nullptr || d_B == nullptr || d_C == nullptr) {
            return std::make_pair(-1.0, -1.0);  // Return error values
        }

        // Copy to device
        cudaError_t copyStatus = cudaMemcpy(d_A, h_A.data(), size, cudaMemcpyHostToDevice);
        EXPECT_EQ(cudaSuccess, copyStatus) << "Failed to copy A to device";
        copyStatus = cudaMemcpy(d_B, h_B.data(), size, cudaMemcpyHostToDevice);
        EXPECT_EQ(cudaSuccess, copyStatus) << "Failed to copy B to device";

        // Warmup using launch function
        cudaError_t status = launchVectorAdd(d_A, d_B, d_C, numElements);
        EXPECT_EQ(cudaSuccess, status) << "Warmup kernel launch failed";
        if (status != cudaSuccess) {
            // Cleanup before returning error
            cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
            return std::make_pair(-1.0, -1.0);
        }

        // Benchmark
        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < iterations; i++) {
            status = launchVectorAdd(d_A, d_B, d_C, numElements);
            EXPECT_EQ(cudaSuccess, status) << "Benchmark kernel launch failed at iteration " << i;
            if (status != cudaSuccess) {
                // Cleanup before returning error
                cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
                return std::make_pair(-1.0, -1.0);
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        double avg_time_us = static_cast<double>(duration.count()) / iterations;
        double avg_time_ms = avg_time_us / 1000.0;

        // Calculate throughput in GFLOPS (2 FLOPs per element: 1 add + 1 copy)
        double gflops = (2.0 * numElements) / (avg_time_us * 1e3);

        // Cleanup
        cudaError_t freeStatus = cudaFree(d_A);
        EXPECT_EQ(cudaSuccess, freeStatus) << "Failed to free d_A";
        freeStatus = cudaFree(d_B);
        EXPECT_EQ(cudaSuccess, freeStatus) << "Failed to free d_B";
        freeStatus = cudaFree(d_C);
        EXPECT_EQ(cudaSuccess, freeStatus) << "Failed to free d_C";

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
    
    cudaError_t status = launchVectorAdd(d_A, d_B, d_C, numElements);
    ASSERT_EQ(cudaSuccess, status) << "Kernel launch failed";

    // Copy result back
    ASSERT_EQ(cudaSuccess, cudaMemcpy(h_C.data(), d_C, size, cudaMemcpyDeviceToHost));

    // Verify results
    const float tolerance = 1e-5f;
    for (int i = 0; i < numElements; i++) {
        EXPECT_NEAR(h_C[i], h_C_ref[i], tolerance)
            << "Negative value test failed at element " << i;
    }

    // Save inputs for comparison
    saveInputData(h_A, h_B, "negative_values");

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
    
    cudaError_t status = launchVectorAdd(d_A, d_B, d_C, numElements);
    ASSERT_EQ(cudaSuccess, status) << "Kernel launch failed";

    // Copy result back
    ASSERT_EQ(cudaSuccess, cudaMemcpy(h_C.data(), d_C, size, cudaMemcpyDeviceToHost));

    // Verify results
    const float tolerance = 1e-5f;
    for (int i = 0; i < numElements; i++) {
        EXPECT_NEAR(h_C[i], h_C_ref[i], tolerance)
            << "Zero value test failed at element " << i;
    }

    // Save inputs for comparison
    saveInputData(h_A, h_B, "zero_values");

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
    
    cudaError_t status = launchVectorAdd(d_A, d_B, d_C, numElements);
    ASSERT_EQ(cudaSuccess, status) << "Kernel launch failed";

    // Copy result back
    ASSERT_EQ(cudaSuccess, cudaMemcpy(h_C.data(), d_C, size, cudaMemcpyDeviceToHost));

    // Verify results (should be infinity)
    for (int i = 0; i < numElements; i++) {
        EXPECT_TRUE(std::isinf(h_C[i]))
            << "Maximum value test failed at element " << i << ", got " << h_C[i];
    }

    // Save inputs for comparison
    saveInputData(h_A, h_B, "max_values");

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
    std::filesystem::path exeDir = getCurrentDir();
    std::filesystem::path metricsPath = exeDir / "cuda_outputs/vectorAdd_benchmark_small.json";
    ensureDirExists(metricsPath.parent_path());
    std::ofstream metrics(metricsPath);
    metrics << "{\"kernel\":\"vectorAdd\",\"size\":1000,\"time_ms\":" << result.first
            << ",\"gflops\":" << result.second << "}" << std::endl;
}

TEST_F(VectorAddTest, Benchmark_Medium) {
    auto result = benchmarkVectorAdd(50000);
    std::cout << "Medium (50000 elements): " << result.first << " ms, "
              << result.second << " GFLOPS" << std::endl;

    // Save metrics
    std::filesystem::path exeDir = getCurrentDir();
    std::filesystem::path metricsPath = exeDir / "cuda_outputs/vectorAdd_benchmark_medium.json";
    ensureDirExists(metricsPath.parent_path());
    std::ofstream metrics(metricsPath);
    metrics << "{\"kernel\":\"vectorAdd\",\"size\":50000,\"time_ms\":" << result.first
            << ",\"gflops\":" << result.second << "}" << std::endl;
}

TEST_F(VectorAddTest, Benchmark_Large) {
    auto result = benchmarkVectorAdd(1000000);
    std::cout << "Large (1000000 elements): " << result.first << " ms, "
              << result.second << " GFLOPS" << std::endl;

    // Save metrics
    std::filesystem::path exeDir = getCurrentDir();
    std::filesystem::path metricsPath = exeDir / "cuda_outputs/vectorAdd_benchmark_large.json";
    ensureDirExists(metricsPath.parent_path());
    std::ofstream metrics(metricsPath);
    metrics << "{\"kernel\":\"vectorAdd\",\"size\":1000000,\"time_ms\":" << result.first
            << ",\"gflops\":" << result.second << "}" << std::endl;
}

TEST_F(VectorAddTest, Benchmark_VeryLarge) {
    auto result = benchmarkVectorAdd(10000000);
    std::cout << "Very Large (10000000 elements): " << result.first << " ms, "
              << result.second << " GFLOPS" << std::endl;

    // Save metrics
    std::filesystem::path exeDir = getCurrentDir();
    std::filesystem::path metricsPath = exeDir / "cuda_outputs/vectorAdd_benchmark_verylarge.json";
    ensureDirExists(metricsPath.parent_path());
    std::ofstream metrics(metricsPath);
    metrics << "{\"kernel\":\"vectorAdd\",\"size\":10000000,\"time_ms\":" << result.first
            << ",\"gflops\":" << result.second << "}" << std::endl;
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
