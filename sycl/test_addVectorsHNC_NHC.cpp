#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <iomanip>

#include <sycl/sycl.hpp>
#include "src/neural/backends/sycl/kernels.h"

using namespace lczero;
using namespace sycl_backend;

// Function to compare results with tolerance
bool compareVectors(const float* a, const float* b, int size, float tolerance = 1e-5) {
    for (int i = 0; i < size; ++i) {
        if (std::abs(a[i] - b[i]) > tolerance) {
            std::cout << "Mismatch at index " << i << ": " << a[i] << " vs " << b[i] << std::endl;
            return false;
        }
    }
    return true;
}

// CPU reference implementation for HNC_NHC layout addition
void addVectorsHNC_NHC_CPU(float* a, const float* b, int N, int H, int C) {
    std::vector<float> a_copy(N * H * C);
    // Copy a to preserve original values
    for (int i = 0; i < N * H * C; ++i) {
        a_copy[i] = a[i];
    }

    // Perform the addition
    for (int i = 0; i < N * H * C; ++i) {
        int orig_i = i;
        int c = i % C;
        i /= C;
        int n = i % N;
        i /= N;
        int h = i;

        float aVal = a_copy[orig_i];
        float bVal = b[n * H * C + h * C + c];  // NHC index
        a[orig_i] = aVal + bVal;
    }
}

int main() {
    try {
        // Initialize SYCL queue
        sycl::queue q(sycl::default_selector{},
                      [](sycl::exception_list eL) {
                          for (auto e : eL) {
                              std::cerr << "SYCL Exception: " << e.what() << std::endl;
                          }
                      });

        std::cout << "Running on device: " << q.get_device().get_info<sycl::info::device::name>() << std::endl;

        // Test parameters
        const int N = 4;  // Batch size
        const int H = 8;  // Height (e.g., 8x8 board)
        const int C = 32; // Channels
        const int total_size = N * H * C;

        std::vector<float> host_a_hnc(total_size);  // HNC layout
        std::vector<float> host_b_nhc(total_size);  // NHC layout
        std::vector<float> sycl_result(total_size);
        std::vector<float> reference_result(total_size);

        // Initialize data with random values
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-10.0f, 10.0f);

        // Fill input arrays
        for (int i = 0; i < total_size; ++i) {
            host_a_hnc[i] = dis(gen);
            host_b_nhc[i] = dis(gen);
        }

        // Allocate device memory
        float* device_a = sycl::malloc_device<float>(total_size, q);
        float* device_b = sycl::malloc_device<float>(total_size, q);

        // Copy a to two copies (one for SYCL, one for reference)
        q.memcpy(device_a, host_a_hnc.data(), total_size * sizeof(float));
        q.memcpy(device_b, host_b_nhc.data(), total_size * sizeof(float));
        q.wait();

        // Run SYCL kernel
        std::cout << "\nRunning addVectorsHNC_NHC SYCL kernel..." << std::endl;
        addVectorsHNC_NHC<float>(device_a, device_b, N, H, C, q);
        q.wait();

        // Copy result back
        q.memcpy(sycl_result.data(), device_a, total_size * sizeof(float));
        q.wait();

        // CPU reference computation
        std::cout << "Computing CPU reference result..." << std::endl;
        reference_result = host_a_hnc;  // Start with original a values
        addVectorsHNC_NHC_CPU(reference_result.data(), host_b_nhc.data(), N, H, C);

        // Compare results
        bool passed = compareVectors(sycl_result.data(), reference_result.data(), total_size);

        std::cout << "\nTest " << (passed ? "PASSED" : "FAILED") << std::endl;

        if (!passed) {
            std::cout << "\nDetailed comparison (first 10 elements):" << std::endl;
            std::cout << "Index\tSYCL\t\tCPU\t\tDifference" << std::endl;
            std::cout << "-----\t----\t\t---\t\t----------" << std::endl;
            for (int i = 0; i < 10; ++i) {
                float diff = std::abs(sycl_result[i] - reference_result[i]);
                std::cout << i << "\t"
                         << std::fixed << std::setprecision(6) << sycl_result[i] << "\t"
                         << std::fixed << std::setprecision(6) << reference_result[i] << "\t"
                         << std::scientific << diff << std::endl;
            }

            // Verify layout conversion logic with a small example
            std::cout << "\nVerifying layout conversion for N=" << N << ", H=" << H << ", C=" << C << std::endl;
            for (int idx = 0; idx < std::min(5, total_size); ++idx) {
                int i = idx;
                int c = i % C;
                i /= C;
                int n = i % N;
                i /= N;
                int h = i;

                int hnc_index = h * N * C + n * C + c;
                int nhc_index = n * H * C + h * C + c;

                std::cout << "Linear index " << idx << ": HNC(" << h << "," << n << "," << c << ")="
                         << hnc_index << ", NHC(" << n << "," << h << "," << c << ")="
                         << nhc_index << std::endl;
            }
        }

        // Test with different dimensions
        std::vector<std::tuple<int, int, int>> test_dims = {
            {1, 64, 16},   // Single batch
            {2, 64, 32},   // Small batch, more channels
            {8, 64, 64},   // Larger batch and channels
            {1, 8, 128}    // Small height, many channels
        };

        for (const auto& dims : test_dims) {
            int n = std::get<0>(dims);
            int h = std::get<1>(dims);
            int c = std::get<2>(dims);
            int size = n * h * c;

            std::cout << "\nTesting with dimensions: N=" << n << ", H=" << h << ", C=" << c << std::endl;

            // Reallocate for new size if needed
            sycl::free(device_a, q);
            sycl::free(device_b, q);
            device_a = sycl::malloc_device<float>(size, q);
            device_b = sycl::malloc_device<float>(size, q);

            std::vector<float> test_a(size), test_b(size), test_result(size), cpu_ref(size);

            // Fill with test data
            for (int i = 0; i < size; ++i) {
                test_a[i] = dis(gen);
                test_b[i] = dis(gen);
            }

            q.memcpy(device_a, test_a.data(), size * sizeof(float));
            q.memcpy(device_b, test_b.data(), size * sizeof(float));
            q.wait();

            // Run SYCL kernel
            addVectorsHNC_NHC<float>(device_a, device_b, n, h, c, q);
            q.wait();

            // Copy back
            q.memcpy(test_result.data(), device_a, size * sizeof(float));
            q.wait();

            // CPU reference
            cpu_ref = test_a;
            addVectorsHNC_NHC_CPU(cpu_ref.data(), test_b.data(), n, h, c);

            bool dim_passed = compareVectors(test_result.data(), cpu_ref.data(), size);
            std::cout << "Dimensions N=" << n << ", H=" << h << ", C=" << c << ": "
                     << (dim_passed ? "PASSED" : "FAILED") << std::endl;
        }

        // Clean up
        sycl::free(device_a, q);
        sycl::free(device_b, q);

        std::cout << "\nAll tests completed!" << std::endl;
        return 0;

    } catch (std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}