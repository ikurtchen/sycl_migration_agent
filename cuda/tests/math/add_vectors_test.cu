#include <gtest/gtest.h>
#include <iostream>
#include <iomanip>
#include <cmath>
#include "../common/base_test_fixture.h"

// Test kernel for addVectors
__global__ void add_vectors_kernel(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

__global__ void add_vectors_kernel_double(const double* a, const double* b, double* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

void launchAddVectors(float* c, const float* a, const float* b, int n);
void launchAddVectorsDouble(double* c, const double* a, const double* b, int n);

class AddVectorsTest : public BaseCudaTest<float> {
protected:
    void SetUp() override {
        BaseCudaTest<float>::SetUp();

        // Test size
        n = 1024 * 1024;

        // Allocate memory
        allocateMemory(&d_a, &h_a, n);
        allocateMemory(&d_b, &h_b, n);
        allocateMemory(&d_c, &h_c, n);

        // Initialize test data
        initializeRandom(h_a, n, -100.0f, 100.0f);
        initializeRandom(h_b, n, -100.0f, 100.0f);

        // Copy to device
        copyToDevice(d_a, h_a, n);
        copyToDevice(d_b, h_b, n);
    }

    void TearDown() override {
        BaseCudaTest<float>::TearDown();

        freeMemory(d_a, h_a);
        freeMemory(d_b, h_b);
        freeMemory(d_c, h_c);
    }

    int n;
    float *d_a, *d_b, *d_c;
    float *h_a, *h_b, *h_c;
};

TEST_F(AddVectorsTest, Correctness) {
    // Launch kernel
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;

    add_vectors_kernel<<<grid_size, block_size>>>(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back
    copyFromDevice(h_c, d_c, n);

    // Verify results with detailed mismatch reporting
    int mismatch_count = 0;
    const int max_mismatches_to_print = 5; // Print first 5 mismatches for debugging

    for (int i = 0; i < n; ++i) {
        float expected = h_a[i] + h_b[i];
        float actual = h_c[i];
        float diff = std::abs(actual - expected);

        if (diff > 1e-5f) {
            mismatch_count++;
            if (mismatch_count <= max_mismatches_to_print) {
                std::cout << "=== MISMATCH #" << mismatch_count << " ===" << std::endl;
                std::cout << "Index: " << i << std::endl;
                std::cout << "Input A: " << std::scientific << std::setprecision(10) << h_a[i] << std::endl;
                std::cout << "Input B: " << std::scientific << std::setprecision(10) << h_b[i] << std::endl;
                std::cout << "Expected: " << std::scientific << std::setprecision(10) << expected << std::endl;
                std::cout << "Actual:   " << std::scientific << std::setprecision(10) << actual << std::endl;
                std::cout << "Difference: " << std::scientific << std::setprecision(10) << diff << std::endl;
                std::cout << "Relative Error: " << std::scientific << std::setprecision(10) << (diff / std::abs(expected)) << std::endl;
                std::cout << "----------------------------------------" << std::endl;
            }
            EXPECT_NEAR(h_c[i], expected, 1e-5f) << "Mismatch at index " << i <<
                " | a=" << h_a[i] << " | b=" << h_b[i] << " | expected=" << expected <<
                " | actual=" << h_c[i] << " | diff=" << diff;
        }
    }

    std::cout << "Total mismatches found: " << mismatch_count << " out of " << n << " elements" << std::endl;

    // Save outputs
    saveOutput("add_vectors_output.bin", h_c, n);
    saveOutput("add_vectors_input_a.bin", h_a, n);
    saveOutput("add_vectors_input_b.bin", h_b, n);
}

TEST_F(AddVectorsTest, Benchmark) {
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;

    auto kernel_func = [&]() {
        add_vectors_kernel<<<grid_size, block_size>>>(d_a, d_b, d_c, n);
    };

    BenchmarkResult result = benchmarkKernel(
        kernel_func, n, "add_vectors", 10, 100);

    std::cout << "Add Vectors - Avg time: " << result.avg_time_ms
              << " ms, Throughput: " << result.gflops << " GFLOPS" << std::endl;

    result.saveToJson("cuda_outputs/add_vectors_benchmark.json");
}

TEST_F(AddVectorsTest, DifferentSizes) {
    std::vector<int> test_sizes = {256, 1024, 8192, 65536, 1048576};

    for (int size : test_sizes) {
        float *test_a, *test_b, *test_c;
        allocateMemory(&test_a, &h_a, size);
        allocateMemory(&test_b, &h_b, size);
        allocateMemory(&test_c, &h_c, size);

        initializeRandom(h_a, size, -100.0f, 100.0f);
        initializeRandom(h_b, size, -100.0f, 100.0f);
        copyToDevice(test_a, h_a, size);
        copyToDevice(test_b, h_b, size);

        int block_size = 256;
        int grid_size = (size + block_size - 1) / block_size;

        add_vectors_kernel<<<grid_size, block_size>>>(test_a, test_b, test_c, size);
        CUDA_CHECK(cudaDeviceSynchronize());

        copyFromDevice(h_c, test_c, size);

        // Verify a few random elements with detailed reporting
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, size - 1);

        int mismatch_count = 0;
        for (int i = 0; i < 10; ++i) {
            int idx = dis(gen);
            float expected = h_a[idx] + h_b[idx];
            float actual = h_c[idx];
            float diff = std::abs(actual - expected);

            if (diff > 1e-5f) {
                mismatch_count++;
                std::cout << "=== SIZE " << size << " RANDOM CHECK MISMATCH #" << mismatch_count << " ===" << std::endl;
                std::cout << "Index: " << idx << std::endl;
                std::cout << "Input A: " << std::scientific << std::setprecision(10) << h_a[idx] << std::endl;
                std::cout << "Input B: " << std::scientific << std::setprecision(10) << h_b[idx] << std::endl;
                std::cout << "Expected: " << std::scientific << std::setprecision(10) << expected << std::endl;
                std::cout << "Actual:   " << std::scientific << std::setprecision(10) << actual << std::endl;
                std::cout << "Difference: " << std::scientific << std::setprecision(10) << diff << std::endl;
                std::cout << "----------------------------------------" << std::endl;
            }
            EXPECT_NEAR(actual, expected, 1e-5f) << "Size " << size << " mismatch at random index " << idx <<
                " | a=" << h_a[idx] << " | b=" << h_b[idx] << " | expected=" << expected <<
                " | actual=" << actual << " | diff=" << diff;
        }

        if (mismatch_count > 0) {
            std::cout << "Size " << size << ": " << mismatch_count << " mismatches out of 10 random checks" << std::endl;
        }

        std::string filename = "add_vectors_size_" + std::to_string(size) + ".bin";
        saveOutput(filename.c_str(), h_c, size);

        freeMemory(test_a, h_a);
        freeMemory(test_b, h_b);
        freeMemory(test_c, h_c);
    }
}

// Double precision test
class AddVectorsDoubleTest : public BaseCudaTest<double> {
protected:
    void SetUp() override {
        BaseCudaTest<double>::SetUp();

        n = 1024 * 1024;

        allocateMemory(&d_a, &h_a, n);
        allocateMemory(&d_b, &h_b, n);
        allocateMemory(&d_c, &h_c, n);

        initializeRandom(h_a, n, -100.0, 100.0);
        initializeRandom(h_b, n, -100.0, 100.0);

        copyToDevice(d_a, h_a, n);
        copyToDevice(d_b, h_b, n);
    }

    void TearDown() override {
        BaseCudaTest<double>::TearDown();

        freeMemory(d_a, h_a);
        freeMemory(d_b, h_b);
        freeMemory(d_c, h_c);
    }

    int n;
    double *d_a, *d_b, *d_c;
    double *h_a, *h_b, *h_c;
};

TEST_F(AddVectorsDoubleTest, Correctness) {
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;

    add_vectors_kernel_double<<<grid_size, block_size>>>(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    copyFromDevice(h_c, d_c, n);

    // Verify results with detailed mismatch reporting for double precision
    int mismatch_count = 0;
    const int max_mismatches_to_print = 5; // Print first 5 mismatches for debugging

    for (int i = 0; i < n; ++i) {
        double expected = h_a[i] + h_b[i];
        double actual = h_c[i];
        double diff = std::abs(actual - expected);

        if (diff > 1e-12) {
            mismatch_count++;
            if (mismatch_count <= max_mismatches_to_print) {
                std::cout << "=== DOUBLE PRECISION MISMATCH #" << mismatch_count << " ===" << std::endl;
                std::cout << "Index: " << i << std::endl;
                std::cout << "Input A: " << std::scientific << std::setprecision(17) << h_a[i] << std::endl;
                std::cout << "Input B: " << std::scientific << std::setprecision(17) << h_b[i] << std::endl;
                std::cout << "Expected: " << std::scientific << std::setprecision(17) << expected << std::endl;
                std::cout << "Actual:   " << std::scientific << std::setprecision(17) << actual << std::endl;
                std::cout << "Difference: " << std::scientific << std::setprecision(17) << diff << std::endl;
                std::cout << "Relative Error: " << std::scientific << std::setprecision(17) << (diff / std::abs(expected)) << std::endl;
                std::cout << "----------------------------------------" << std::endl;
            }
            EXPECT_NEAR(h_c[i], expected, 1e-12) << "Double precision mismatch at index " << i <<
                " | a=" << std::scientific << std::setprecision(17) << h_a[i] <<
                " | b=" << std::scientific << std::setprecision(17) << h_b[i] <<
                " | expected=" << std::scientific << std::setprecision(17) << expected <<
                " | actual=" << std::scientific << std::setprecision(17) << h_c[i] <<
                " | diff=" << std::scientific << std::setprecision(17) << diff;
        }
    }

    std::cout << "Total double precision mismatches found: " << mismatch_count << " out of " << n << " elements" << std::endl;

    saveOutput("add_vectors_double_output.bin", h_c, n);
}