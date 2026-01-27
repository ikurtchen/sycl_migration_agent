#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
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
#ifdef __APPLE__
#include <mach-o/dyld.h>
#endif
#include "vectorAdd_kernel_sycl.h"

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

class VectorAddSYCLTest : public ::testing::Test {
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

        // Launch kernel using the launch function
        int status = launchVectorAddSYCL(h_A.data(), h_B.data(), h_C.data(), numElements);
        ASSERT_EQ(0, status) << "Kernel launch failed";

        // Verify results
        const float tolerance = 1e-5f;
        int mismatches = 0;
        for (int i = 0; i < numElements; i++) {
            float diff = std::abs(h_C[i] - h_C_ref[i]);
            if (diff > tolerance) {
                if (mismatches < 5) { // Only print first 5 mismatches
                    std::cout << "Mismatch at element " << i << ": Expected=" << h_C_ref[i]
                              << ", Got=" << h_C[i] << ", Diff=" << diff << std::endl;
                }
                mismatches++;
            }
        }

        EXPECT_EQ(0, mismatches) << "Found " << mismatches << " mismatches in test " << test_name;

        // Save inputs for comparison
        saveInputData(h_A, h_B, test_name);

        // Save results for comparison
        saveResults(h_C.data(), size, test_name);
    }

    void saveResults(const float* data, size_t size, const std::string& test_name) {
        std::string filename = "sycl_outputs/vectorAdd_" + test_name + "_output.bin";
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
        std::string filename_A = "sycl_inputs/vectorAdd_" + test_name + "_input_A.bin";
        std::string filename_B = "sycl_inputs/vectorAdd_" + test_name + "_input_B.bin";
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
        try {
            // Find available GPU devices
            auto gpuDevices = sycl::device::get_devices(sycl::info::device_type::gpu);
            if (gpuDevices.empty()) {
                std::cerr << "No GPU devices found, using CPU" << std::endl;
            }

            sycl::queue q(sycl::gpu_selector{});
            std::cout << "Running on: " << q.get_device().get_info<sycl::info::device::name>() << std::endl;

            // Create buffers
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

            // Create SYCL buffers
            sycl::buffer<float, 1> bufA(h_A);
            sycl::buffer<float, 1> bufB(h_B);
            sycl::buffer<float, 1> bufC(h_C);

            // Warmup run
            q.submit([&](sycl::handler& h) {
                auto A = bufA.get_access<sycl::access::mode::read>(h);
                auto B = bufB.get_access<sycl::access::mode::read>(h);
                auto C = bufC.get_access<sycl::access::mode::write>(h);
                h.parallel_for(sycl::range<1>(numElements), [=](sycl::id<1> idx) {
                    C[idx] = A[idx] + B[idx];
                });
            });
            q.wait();

            // Benchmark
            auto start = std::chrono::high_resolution_clock::now();

            for (int i = 0; i < iterations; i++) {
                q.submit([&](sycl::handler& h) {
                    auto A = bufA.get_access<sycl::access::mode::read>(h);
                    auto B = bufB.get_access<sycl::access::mode::read>(h);
                    auto C = bufC.get_access<sycl::access::mode::write>(h);
                    h.parallel_for(sycl::range<1>(numElements), [=](sycl::id<1> idx) {
                        C[idx] = A[idx] + B[idx];
                    });
                });
                q.wait();
            }

            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

            double avg_time_us = static_cast<double>(duration.count()) / iterations;
            double avg_time_ms = avg_time_us / 1000.0;

            // Calculate throughput in GFLOPS (2 FLOPs per element: 1 add + 1 copy)
            double gflops = (2.0 * numElements) / (avg_time_us * 1e3);

            return std::make_pair(avg_time_ms, gflops);
        } catch (const std::exception& e) {
            std::cerr << "Benchmark error: " << e.what() << std::endl;
            return std::make_pair(-1.0, -1.0);
        }
    }
};

// Test small vector size
TEST_F(VectorAddSYCLTest, SmallSize_1000) {
    testVectorAdd(1000, "small_1000");
}

// Test medium vector size
TEST_F(VectorAddSYCLTest, MediumSize_50000) {
    testVectorAdd(50000, "medium_50000");
}

// Test large vector size
TEST_F(VectorAddSYCLTest, LargeSize_1000000) {
    testVectorAdd(1000000, "large_1000000");
}

// Test very large vector size
TEST_F(VectorAddSYCLTest, VeryLargeSize_10000000) {
    testVectorAdd(10000000, "verylarge_10000000");
}

// Test edge case: single element
TEST_F(VectorAddSYCLTest, SingleElement) {
    testVectorAdd(1, "single");
}

// Test edge case: power of 2 boundary
TEST_F(VectorAddSYCLTest, PowerOf2Boundary) {
    testVectorAdd(256, "powerof2_256");
}

// Test negative values
TEST_F(VectorAddSYCLTest, NegativeValues) {
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

    // Launch kernel
    int status = launchVectorAddSYCL(h_A.data(), h_B.data(), h_C.data(), numElements);
    ASSERT_EQ(0, status) << "Kernel launch failed";

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
}

// Test zero values
TEST_F(VectorAddSYCLTest, ZeroValues) {
    int numElements = 1000;
    size_t size = numElements * sizeof(float);

    // Allocate host memory
    std::vector<float> h_A(numElements, 0.0f);
    std::vector<float> h_B(numElements, 0.0f);
    std::vector<float> h_C(numElements);
    std::vector<float> h_C_ref(numElements, 0.0f);

    // Launch kernel
    int status = launchVectorAddSYCL(h_A.data(), h_B.data(), h_C.data(), numElements);
    ASSERT_EQ(0, status) << "Kernel launch failed";

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
}

// Test maximum values
TEST_F(VectorAddSYCLTest, MaximumValues) {
    int numElements = 1000;
    size_t size = numElements * sizeof(float);

    // Allocate host memory
    std::vector<float> h_A(numElements, FLT_MAX);
    std::vector<float> h_B(numElements, FLT_MAX);
    std::vector<float> h_C(numElements);

    // Launch kernel
    int status = launchVectorAddSYCL(h_A.data(), h_B.data(), h_C.data(), numElements);
    ASSERT_EQ(0, status) << "Kernel launch failed";

    // Verify results (should be infinity)
    for (int i = 0; i < numElements; i++) {
        EXPECT_EQ(h_C[i], std::numeric_limits<float>::infinity())
            << "Maximum value test failed at element " << i << ", got " << h_C[i];
    }

    // Save inputs for comparison
    saveInputData(h_A, h_B, "max_values");

    // Save results
    saveResults(h_C.data(), size, "max_values");
}

// Benchmark tests
TEST_F(VectorAddSYCLTest, Benchmark_Small) {
    auto result = benchmarkVectorAdd(1000);
    std::cout << "Small (1000 elements): " << result.first << " ms, "
              << result.second << " GFLOPS" << std::endl;

    // Save metrics
    std::filesystem::path exeDir = getCurrentDir();
    std::filesystem::path metricsPath = exeDir / "sycl_outputs/vectorAdd_benchmark_small.json";
    ensureDirExists(metricsPath.parent_path());
    std::ofstream metrics(metricsPath);
    if (metrics.is_open()) {
        metrics << "{\"kernel\":\"vectorAdd\",\"size\":1000,\"time_ms\":" << result.first
                << ",\"gflops\":" << result.second << "}" << std::endl;
    }
}

TEST_F(VectorAddSYCLTest, Benchmark_Medium) {
    auto result = benchmarkVectorAdd(50000);
    std::cout << "Medium (50000 elements): " << result.first << " ms, "
              << result.second << " GFLOPS" << std::endl;

    // Save metrics
    std::filesystem::path exeDir = getCurrentDir();
    std::filesystem::path metricsPath = exeDir / "sycl_outputs/vectorAdd_benchmark_medium.json";
    ensureDirExists(metricsPath.parent_path());
    std::ofstream metrics(metricsPath);
    if (metrics.is_open()) {
        metrics << "{\"kernel\":\"vectorAdd\",\"size\":50000,\"time_ms\":" << result.first
                << ",\"gflops\":" << result.second << "}" << std::endl;
    }
}

TEST_F(VectorAddSYCLTest, Benchmark_Large) {
    auto result = benchmarkVectorAdd(1000000);
    std::cout << "Large (1000000 elements): " << result.first << " ms, "
              << result.second << " GFLOPS" << std::endl;

    // Save metrics
    std::filesystem::path exeDir = getCurrentDir();
    std::filesystem::path metricsPath = exeDir / "sycl_outputs/vectorAdd_benchmark_large.json";
    ensureDirExists(metricsPath.parent_path());
    std::ofstream metrics(metricsPath);
    if (metrics.is_open()) {
        metrics << "{\"kernel\":\"vectorAdd\",\"size\":1000000,\"time_ms\":" << result.first
                << ",\"gflops\":" << result.second << "}" << std::endl;
    }
}

TEST_F(VectorAddSYCLTest, Benchmark_VeryLarge) {
    auto result = benchmarkVectorAdd(10000000);
    std::cout << "Very Large (10000000 elements): " << result.first << " ms, "
              << result.second << " GFLOPS" << std::endl;

    // Save metrics
    std::filesystem::path exeDir = getCurrentDir();
    std::filesystem::path metricsPath = exeDir / "sycl_outputs/vectorAdd_benchmark_verylarge.json";
    ensureDirExists(metricsPath.parent_path());
    std::ofstream metrics(metricsPath);
    if (metrics.is_open()) {
        metrics << "{\"kernel\":\"vectorAdd\",\"size\":10000000,\"time_ms\":" << result.first
                << ",\"gflops\":" << result.second << "}" << std::endl;
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
