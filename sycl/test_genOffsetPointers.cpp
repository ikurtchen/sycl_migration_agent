/*
  Test for genOffsetPointers kernel migration from CUDA to SYCL
  This test verifies that the SYCL version produces identical pointers
  to the CUDA version.
*/

#include <iostream>
#include <vector>
#include <cassert>
#include <sycl/sycl.hpp>

// Include the proper header
#include "src/neural/backends/sycl/kernels.h"

// Simple test to verify genOffsetPointers produces correct pointer offsets
int main() {
    try {
        sycl::queue q(sycl::default_selector_v);

        // Test parameters (small values for testing)
        const int heads = 4;
        const int max_batch = 2;
        const int depth = 64;
        const int d_model = 128;
        const int block_size = heads * max_batch;

        // Allocate device memory
        const size_t total_mem = 64 * d_model * max_batch + heads * depth + 5 * block_size * 64 * 64;

        float* d_k = sycl::malloc_device<float>(total_mem, q);
        float* d_q = sycl::malloc_device<float>(total_mem, q);
        float* d_v = sycl::malloc_device<float>(total_mem, q);
        float* d_b1 = sycl::malloc_device<float>(total_mem, q);
        float* d_b2 = sycl::malloc_device<float>(total_mem, q);

        // Allocate array for pointers on device
        float** d_offsets = sycl::malloc_device<float*>(5 * block_size, q);

        if (!d_k || !d_q || !d_v || !d_b1 || !d_b2 || !d_offsets) {
            std::cerr << "Memory allocation failed" << std::endl;
            return 1;
        }

        // Call the SYCL kernel
        lczero::sycl_backend::genOffsetPointers<float>(
            q, d_offsets, heads, max_batch, depth, d_model,
            d_k, d_q, d_b1, d_v, d_b2);

        // Copy pointers back to host for verification
        std::vector<float*> h_offsets(5 * block_size);
        q.memcpy(h_offsets.data(), d_offsets, sizeof(float*) * 5 * block_size).wait();

        // Verify pointer calculations
        bool test_passed = true;
        for (int n = 0; n < max_batch; ++n) {
            for (int h = 0; h < heads; ++h) {
                int i = n * heads + h;

                // Check k pointer
                float* expected_k = d_k + h * depth + 64 * d_model * n;
                if (h_offsets[i] != expected_k) {
                    std::cerr << "K pointer mismatch at n=" << n << ", h=" << h
                             << ": expected=" << expected_k << ", got=" << h_offsets[i] << std::endl;
                    test_passed = false;
                }

                // Check q pointer
                float* expected_q = d_q + h * depth + 64 * d_model * n;
                if (h_offsets[i + block_size] != expected_q) {
                    std::cerr << "Q pointer mismatch at n=" << n << ", h=" << h
                             << ": expected=" << expected_q << ", got=" << h_offsets[i + block_size] << std::endl;
                    test_passed = false;
                }

                // Check v pointer
                float* expected_v = d_v + h * depth + 64 * d_model * n;
                if (h_offsets[i + 3 * block_size] != expected_v) {
                    std::cerr << "V pointer mismatch at n=" << n << ", h=" << h
                             << ": expected=" << expected_v << ", got=" << h_offsets[i + 3 * block_size] << std::endl;
                    test_passed = false;
                }

                // Check b2 pointer
                float* expected_b2 = d_b2 + h * depth + 64 * d_model * n;
                if (h_offsets[i + 4 * block_size] != expected_b2) {
                    std::cerr << "B2 pointer mismatch at n=" << n << ", h=" << h
                             << ": expected=" << expected_b2 << ", got=" << h_offsets[i + 4 * block_size] << std::endl;
                    test_passed = false;
                }

                // Check b1 pointer (special indexing)
                float* expected_b1 = d_b1 + i * 64 * 64;
                if (h_offsets[i + 2 * block_size] != expected_b1) {
                    std::cerr << "B1 pointer mismatch at n=" << n << ", h=" << h
                             << ": expected=" << expected_b1 << ", got=" << h_offsets[i + 2 * block_size] << std::endl;
                    test_passed = false;
                }
            }
        }

        // Cleanup
        sycl::free(d_offsets, q);
        sycl::free(d_k, q);
        sycl::free(d_q, q);
        sycl::free(d_v, q);
        sycl::free(d_b1, q);
        sycl::free(d_b2, q);

        if (test_passed) {
            std::cout << "SYCL genOffsetPointers test PASSED!" << std::endl;
            return 0;
        } else {
            std::cout << "SYCL genOffsetPointers test FAILED!" << std::endl;
            return 1;
        }

    } catch (sycl::exception const& e) {
        std::cerr << "SYCL exception caught: " << e.what() << std::endl;
        return 1;
    } catch (std::exception const& e) {
        std::cerr << "Standard exception caught: " << e.what() << std::endl;
        return 1;
    }
}