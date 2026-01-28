/*
  Test program for the policyMap kernel migration from CUDA to SYCL.
  This test verifies that the SYCL implementation produces identical results
  to the CUDA version for the same inputs.
*/

#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <cmath>
#include <sycl/sycl.hpp>

#include "neural/backends/sycl/common_kernels.h"
#include "neural/tables/activation_function.h"

using namespace lczero::sycl_backend;

// Simple test to verify policyMap kernel
bool test_policyMap_simple() {
    std::cout << "Testing PolicyMap kernel (simple case)..." << std::endl;

    try {
        // Create SYCL queue
        sycl::queue q(sycl::default_selector_v);

        // Test parameters
        const int N = 2;           // Batch size
        const int inputSize = 10;  // Input size
        const int usedSize = 8;    // Used size (valid indices)
        const int outputSize = 12; // Output size

        // Create index mapping array (maps from input indices to output indices)
        std::vector<short> h_indices = {0, 2, 4, 6, 8, 10, -1, -1};

        // Create input data
        std::vector<float> h_input(N * inputSize);
        for (int i = 0; i < N * inputSize; i++) {
            h_input[i] = static_cast<float>(i + 1);
        }

        // Create expected output
        std::vector<float> h_expected(outputSize, 0.0f);
        for (int n = 0; n < N; n++) {
            for (int i = 0; i < usedSize; i++) {
                int j = h_indices[i];
                if (j >= 0) {
                    h_expected[n * outputSize + j] = h_input[n * inputSize + i];
                }
            }
        }

        // Allocate device memory
        float* d_input = sycl::malloc_device<float>(N * inputSize, q);
        float* d_output = sycl::malloc_device<float>(N * outputSize, q);
        short* d_indices = sycl::malloc_device<short>(usedSize, q);

        // Copy data to device
        q.memcpy(d_input, h_input.data(), N * inputSize * sizeof(float)).wait();
        q.memcpy(d_indices, h_indices.data(), usedSize * sizeof(short)).wait();
        q.memset(d_output, 0, N * outputSize * sizeof(float)).wait();

        // Call the kernel
        PolicyMap<float>(N, d_output, d_input, d_indices, inputSize, usedSize, outputSize, q);

        // Copy results back
        std::vector<float> h_result(N * outputSize);
        q.memcpy(h_result.data(), d_output, N * outputSize * sizeof(float)).wait();

        // Verify results
        bool success = true;
        float tolerance = 1e-5f;

        for (int i = 0; i < N * outputSize; i++) {
            if (std::abs(h_result[i] - h_expected[i]) > tolerance) {
                std::cout << "  Mismatch at index " << i
                         << ": expected " << h_expected[i]
                         << ", got " << h_result[i] << std::endl;
                success = false;
            }
        }

        // Print results for verification
        std::cout << "  Input data:" << std::endl;
        for (int n = 0; n < N; n++) {
            std::cout << "    Batch " << n << ": ";
            for (int i = 0; i < inputSize; i++) {
                std::cout << std::setw(4) << h_input[n * inputSize + i] << " ";
            }
            std::cout << std::endl;
        }

        std::cout << "  Indices mapping: ";
        for (int i = 0; i < usedSize; i++) {
            std::cout << h_indices[i] << " ";
        }
        std::cout << std::endl;

        std::cout << "  Expected output:" << std::endl;
        for (int n = 0; n < N; n++) {
            std::cout << "    Batch " << n << ": ";
            for (int i = 0; i < outputSize; i++) {
                std::cout << std::setw(4) << h_expected[n * outputSize + i] << " ";
            }
            std::cout << std::endl;
        }

        std::cout << "  Actual output:" << std::endl;
        for (int n = 0; n < N; n++) {
            std::cout << "    Batch " << n << ": ";
            for (int i = 0; i < outputSize; i++) {
                std::cout << std::setw(4) << h_result[n * outputSize + i] << " ";
            }
            std::cout << std::endl;
        }

        // Clean up
        sycl::free(d_input, q);
        sycl::free(d_output, q);
        sycl::free(d_indices, q);

        std::cout << "  Test " << (success ? "PASSED" : "FAILED") << std::endl;
        return success;

    } catch (sycl::exception const& e) {
        std::cerr << "SYCL exception caught: " << e.what() << std::endl;
        return false;
    }
}

// Test with chess-specific parameters (similar to LCZero usage)
bool test_policyMap_chess() {
    std::cout << "Testing PolicyMap kernel (chess-specific case)..." << std::endl;

    try {
        // Create SYCL queue
        sycl::queue q(sycl::default_selector_v);

        // Typical chess policy mapping parameters
        const int N = 4;               // Batch size (multiple positions)
        const int inputSize = 1858;    // Total possible moves
        const int usedSize = 1858;     // Used size (all moves for this test)
        const int outputSize = 1858;   // Output size

        // Create index mapping array (identity mapping for this test)
        std::vector<short> h_indices(usedSize);
        for (int i = 0; i < usedSize; i++) {
            h_indices[i] = i;  // Identity mapping
        }

        // Create input data with some pattern
        std::vector<float> h_input(N * inputSize);
        std::mt19937 gen(42);  // Fixed seed for reproducibility
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

        for (int i = 0; i < N * inputSize; i++) {
            h_input[i] = dist(gen);
        }

        // Expected output should be identical to input for identity mapping
        std::vector<float> h_expected = h_input;

        // Allocate device memory
        float* d_input = sycl::malloc_device<float>(N * inputSize, q);
        float* d_output = sycl::malloc_device<float>(N * outputSize, q);
        short* d_indices = sycl::malloc_device<short>(usedSize, q);

        // Copy data to device
        q.memcpy(d_input, h_input.data(), N * inputSize * sizeof(float)).wait();
        q.memcpy(d_indices, h_indices.data(), usedSize * sizeof(short)).wait();

        // Call the kernel
        PolicyMap<float>(N, d_output, d_input, d_indices, inputSize, usedSize, outputSize, q);

        // Copy results back
        std::vector<float> h_result(N * outputSize);
        q.memcpy(h_result.data(), d_output, N * outputSize * sizeof(float)).wait();

        // Verify results
        bool success = true;
        float tolerance = 1e-5f;

        for (int i = 0; i < N * outputSize; i++) {
            if (std::abs(h_result[i] - h_expected[i]) > tolerance) {
                std::cout << "  Mismatch at index " << i
                         << ": expected " << h_expected[i]
                         << ", got " << h_result[i] << std::endl;
                success = false;
            }
        }

        // Clean up
        sycl::free(d_input, q);
        sycl::free(d_output, q);
        sycl::free(d_indices, q);

        std::cout << "  Test " << (success ? "PASSED" : "FAILED") << std::endl;
        return success;

    } catch (sycl::exception const& e) {
        std::cerr << "SYCL exception caught: " << e.what() << std::endl;
        return false;
    }
}

// Test with half precision (sycl::half)
bool test_policyMap_half() {
    std::cout << "Testing PolicyMap kernel (half precision)..." << std::endl;

    try {
        // Create SYCL queue
        sycl::queue q(sycl::default_selector_v);

        // Test parameters
        const int N = 2;
        const int inputSize = 8;
        const int usedSize = 6;
        const int outputSize = 10;

        // Create index mapping
        std::vector<short> h_indices = {1, 3, 5, 7, 9, -1};

        // Create input data
        std::vector<sycl::half> h_input(N * inputSize);
        for (int i = 0; i < N * inputSize; i++) {
            h_input[i] = sycl::half(i * 0.5f);
        }

        // Calculate expected output
        std::vector<sycl::half> h_expected(outputSize, sycl::half(0.0f));
        for (int n = 0; n < N; n++) {
            for (int i = 0; i < usedSize; i++) {
                int j = h_indices[i];
                if (j >= 0) {
                    h_expected[n * outputSize + j] = h_input[n * inputSize + i];
                }
            }
        }

        // Allocate device memory
        sycl::half* d_input = sycl::malloc_device<sycl::half>(N * inputSize, q);
        sycl::half* d_output = sycl::malloc_device<sycl::half>(N * outputSize, q);
        short* d_indices = sycl::malloc_device<short>(usedSize, q);

        // Copy data to device
        q.memcpy(d_input, h_input.data(), N * inputSize * sizeof(sycl::half)).wait();
        q.memcpy(d_indices, h_indices.data(), usedSize * sizeof(short)).wait();
        q.memset(d_output, 0, N * outputSize * sizeof(sycl::half)).wait();

        // Call the kernel
        PolicyMap<sycl::half>(N, d_output, d_input, d_indices, inputSize, usedSize, outputSize, q);

        // Copy results back
        std::vector<sycl::half> h_result(N * outputSize);
        q.memcpy(h_result.data(), d_output, N * outputSize * sizeof(sycl::half)).wait();

        // Verify results
        bool success = true;
        float tolerance = 1e-3f;  // Larger tolerance for half precision

        for (int i = 0; i < N * outputSize; i++) {
            float diff = std::abs(static_cast<float>(h_result[i]) - static_cast<float>(h_expected[i]));
            if (diff > tolerance) {
                std::cout << "  Mismatch at index " << i
                         << ": expected " << static_cast<float>(h_expected[i])
                         << ", got " << static_cast<float>(h_result[i]) << std::endl;
                success = false;
            }
        }

        // Clean up
        sycl::free(d_input, q);
        sycl::free(d_output, q);
        sycl::free(d_indices, q);

        std::cout << "  Test " << (success ? "PASSED" : "FAILED") << std::endl;
        return success;

    } catch (sycl::exception const& e) {
        std::cerr << "SYCL exception caught: " << e.what() << std::endl;
        return false;
    }
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  SYCL PolicyMap Kernel Migration Test" << std::endl;
    std::cout << "========================================" << std::endl;

    bool all_passed = true;

    // Run tests
    all_passed &= test_policyMap_simple();
    std::cout << std::endl;

    all_passed &= test_policyMap_chess();
    std::cout << std::endl;

    all_passed &= test_policyMap_half();
    std::cout << std::endl;

    // Final result
    std::cout << "========================================" << std::endl;
    std::cout << "  Overall result: " << (all_passed ? "ALL TESTS PASSED" : "SOME TESTS FAILED") << std::endl;
    std::cout << "========================================" << std::endl;

    return all_passed ? 0 : 1;
}