/*
  Test for input_gating kernel migration from CUDA to SYCL
*/

#include <iostream>
#include <vector>
#include <cmath>
#include <sycl/sycl.hpp>
#include "neural/backends/sycl/kernels.h"

using namespace lczero;
using namespace sycl_backend;

void test_input_gating() {
    // Test parameters
    const int N = 2;    // batch size
    const int HW = 64;  // flattened height*width (8x8)
    const int C = 32;   // channels

    // Create SYCL queue
    sycl::queue queue{sycl::cpu_selector{}};

    // Allocate host memory
    std::vector<float> h_input(N * HW * C);
    std::vector<float> h_mult(C * HW);     // transposed weights
    std::vector<float> h_add(C * HW);      // transposed bias
    std::vector<float> h_output(N * HW * C, 0.0f);

    // Initialize test data
    for (int i = 0; i < N * HW * C; ++i) {
        h_input[i] = static_cast<float>(i % 100) / 10.0f;
    }

    for (int i = 0; i < C * HW; ++i) {
        h_mult[i] = static_cast<float>(i % 10) / 5.0f;
        h_add[i] = static_cast<float>((i * 3) % 20) / 10.0f;
    }

    // Allocate device memory using USM
    float* d_input = sycl::malloc_device<float>(N * HW * C, queue);
    float* d_mult = sycl::malloc_device<float>(C * HW, queue);
    float* d_add = sycl::malloc_device<float>(C * HW, queue);
    float* d_output = sycl::malloc_device<float>(N * HW * C, queue);

    // Copy data to device
    queue.memcpy(d_input, h_input.data(), N * HW * C * sizeof(float)).wait();
    queue.memcpy(d_mult, h_mult.data(), C * HW * sizeof(float)).wait();
    queue.memcpy(d_add, h_add.data(), C * HW * sizeof(float)).wait();

    // Apply input gating
    applyInputGating<float>(queue, d_output, d_input, d_mult, d_add, N, HW, C);

    // Copy results back
    queue.memcpy(h_output.data(), d_output, N * HW * C * sizeof(float)).wait();

    // Verify results
    bool correct = true;
    for (int n = 0; n < N; ++n) {
        for (int hw = 0; hw < HW; ++hw) {
            for (int c = 0; c < C; ++c) {
                int input_idx = n * HW * C + hw * C + c;
                int weight_idx = c * HW + hw;  // transposed indexing

                float expected = h_input[input_idx] * h_mult[weight_idx] + h_add[weight_idx];
                float actual = h_output[input_idx];

                if (std::abs(expected - actual) > 1e-5f) {
                    std::cout << "Mismatch at [" << n << "," << hw << "," << c << "]: "
                              << "expected=" << expected << ", actual=" << actual << std::endl;
                    correct = false;
                }
            }
        }
    }

    if (correct) {
        std::cout << "input_gating kernel test PASSED!" << std::endl;
    } else {
        std::cout << "input_gating kernel test FAILED!" << std::endl;
    }

    // Cleanup
    sycl::free(d_input, queue);
    sycl::free(d_mult, queue);
    sycl::free(d_add, queue);
    sycl::free(d_output, queue);
}

int main() {
    try {
        test_input_gating();
        std::cout << "Test completed successfully!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}