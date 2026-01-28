/*
  Test file for promotion_logits_kernel SYCL implementation
*/

#include <iostream>
#include <vector>
#include <cmath>
#include <sycl/sycl.hpp>

#include "sycl_common.h"

using namespace lczero::sycl_backend;

template <typename T>
void test_promotion_logits() {
    // Test parameters
    const int N = 2;      // Batch size
    const int C = 64;     // Channels
    constexpr int output_stride = 64 * 64 + 8 * 24;

    // Create SYCL queue
    sycl::queue q(sycl::default_selector_v);

    // Allocate and initialize host memory
    std::vector<T> h_keys(N * 64 * C, static_cast<T>(1.0f));
    std::vector<T> h_ppo(4 * C, static_cast<T>(0.5f));
    std::vector<T> h_policy_attn_logits(N * 64 * 64, static_cast<T>(0.1f));
    std::vector<T> h_output(N * 64 * 64 + N * 8 * 24, static_cast<T>(0.0f));

    // Initialize with distinct patterns
    for (int i = 0; i < N * 64 * C; i++) {
        h_keys[i] = static_cast<T>(0.1f + (i % 10) * 0.01f);
    }
    for (int i = 0; i < 4 * C; i++) {
        h_ppo[i] = static_cast<T>(0.2f + (i % 5) * 0.02f);
    }
    for (int i = 0; i < N * 64 * 64; i++) {
        h_policy_attn_logits[i] = static_cast<T>(0.3f + (i % 8) * 0.03f);
    }

    // Allocate device memory
    T* d_keys = sycl::malloc_device<T>(N * 64 * C, q);
    T* d_ppo = sycl::malloc_device<T>(4 * C, q);
    T* d_policy_attn_logits = sycl::malloc_device<T>(N * 64 * 64, q);
    T* d_output = sycl::malloc_device<T>(N * 64 * 64 + N * 8 * 24, q);

    // Copy data to device
    q.memcpy(d_keys, h_keys.data(), N * 64 * C * sizeof(T));
    q.memcpy(d_ppo, h_ppo.data(), 4 * C * sizeof(T));
    q.memcpy(d_policy_attn_logits, h_policy_attn_logits.data(), N * 64 * 64 * sizeof(T));
    q.memcpy(d_output, h_output.data(), (N * 64 * 64 + N * 8 * 24) * sizeof(T));

    // Execute the kernel
    ComputePromotionLogits<T>(q, N, C, d_output, d_keys, d_ppo, d_policy_attn_logits);

    // Copy results back
    q.memcpy(h_output.data(), d_output, (N * 64 * 64 + N * 8 * 24) * sizeof(T));
    q.wait();

    // Validate results - check that promotion logits section is non-zero
    bool success = true;
    for (int n = 0; n < N; n++) {
        for (int i = 0; i < 8 * 24; i++) {
            T val = h_output[n * output_stride + i];
            if (val == static_cast<T>(0.0f)) {
                std::cout << "ERROR: Promotion logit [" << n << "][" << i << "] is zero!" << std::endl;
                success = false;
            }
        }
    }

    if (success) {
        std::cout << "SUCCESS: promotion_logits_kernel test passed for ";
        if (std::is_same<T, float>::value) {
            std::cout << "float" << std::endl;
        } else {
            std::cout << "half" << std::endl;
        }

        // Print some sample outputs
        std::cout << "Sample promotion logits:" << std::endl;
        for (int n = 0; n < std::min(N, 1); n++) {
            for (int i = 0; i < std::min(8 * 24, 10); i++) {
                std::cout << "  [" << n << "][" << i << "] = " << static_cast<float>(h_output[n * output_stride + i]) << std::endl;
            }
        }
    }

    // Clean up
    sycl::free(d_keys, q);
    sycl::free(d_ppo, q);
    sycl::free(d_policy_attn_logits, q);
    sycl::free(d_output, q);
}

int main() {
    try {
        std::cout << "Testing promotion_logits_kernel..." << std::endl;

        // Test with float
        test_promotion_logits<float>();

        // Test with half
        test_promotion_logits<sycl::half>();

        std::cout << "All tests completed successfully!" << std::endl;
        return 0;
    } catch (sycl::exception const& e) {
        std::cerr << "SYCL exception caught: " << e.what() << std::endl;
        return 1;
    } catch (std::exception const& e) {
        std::cerr << "Standard exception caught: " << e.what() << std::endl;
        return 1;
    }
}