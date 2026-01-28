/*
  Test for SYCL layer normalization kernel
  Compares results with expected values computed on CPU
*/

#include <gtest/gtest.h>
#include <random>
#include <vector>
#include <cmath>
#include <iomanip>
#include <fstream>

#include "sycl_common.h"
#include "../src/neural/backends/sycl/common_kernels.cpp"  // Include kernel implementation

using namespace lczero::sycl_backend;

class LayerNormTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize SYCL queue
        try {
            q = sycl::queue{sycl::default_selector{}};
        } catch (const sycl::exception& e) {
            GTEST_SKIP() << "SYCL device not available: " << e.what();
        }

        // Set random seed for reproducible tests
        rng.seed(42);
    }

    sycl::queue q;
    std::mt19937 rng;
};

// CPU reference implementation of layer normalization
template <typename T>
void layer_norm_cpu_reference(int N, int C, std::vector<T>& output,
                              const std::vector<T>& input,
                              const std::vector<T>& bias,
                              const std::vector<T>* skip,
                              const std::vector<T>& gammas,
                              const std::vector<T>& betas,
                              float epsilon, float alpha,
                              ActivationFunction activation) {
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            int idx = n * C + c;

            // Add bias and optional skip, apply activation and scaling
            float val = static_cast<float>(input[idx]) + static_cast<float>(bias[c]);

            if (skip != nullptr) {
                val = activate_device(val, activation) * alpha + static_cast<float>((*skip)[idx]);
            } else {
                val = activate_device(val, activation) * alpha;
            }

            output[idx] = static_cast<T>(val);
        }

        // Compute mean across C dimension
        float mean = 0.0f;
        for (int c = 0; c < C; ++c) {
            mean += static_cast<float>(output[n * C + c]);
        }
        mean /= C;

        // Compute variance across C dimension
        float var = 0.0f;
        for (int c = 0; c < C; ++c) {
            float diff = static_cast<float>(output[n * C + c]) - mean;
            var += diff * diff;
        }
        var /= C;

        // Normalize: (x - mean) / sqrt(var + epsilon) * gamma + beta
        for (int c = 0; c < C; ++c) {
            int idx = n * C + c;
            float x = static_cast<float>(output[idx]);
            float normalized = (x - mean) / std::sqrt(var + epsilon);
            float result = normalized * static_cast<float>(gammas[c]) + static_cast<float>(betas[c]);
            output[idx] = static_cast<T>(result);
        }
    }
}

// Test layer normalization with float32
TEST_F(LayerNormTest, TestFloat32) {
    const int N = 4;    // Batch size
    const int C = 64;   // Channels (must be multiple of 16)
    const float epsilon = 1e-5f;
    const float alpha = 1.0f;
    const ActivationFunction activation = ActivationFunction::ACTIVATION_RELU;

    // Allocate host memory
    std::vector<float> input(N * C);
    std::vector<float> bias(C);
    std::vector<float> skip(N * C);
    std::vector<float> gammas(C);
    std::vector<float> betas(C);
    std::vector<float> output_sycl(N * C);
    std::vector<float> output_cpu(N * C);

    // Generate random test data
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& x : input) x = dist(rng);
    for (auto& x : bias) x = dist(rng) * 0.1f;
    for (auto& x : skip) x = dist(rng) * 0.5f;
    for (auto& x : gammas) x = 0.5f + dist(rng) * 0.5f; // Positive values for gamma
    for (auto& x : betas) x = dist(rng) * 0.1f;

    // Run CPU reference implementation
    layer_norm_cpu_reference(N, C, output_cpu, input, bias, &skip, gammas, betas, epsilon, alpha, activation);

    // Allocate device memory
    float* d_input = sycl::malloc_device<float>(N * C, q);
    float* d_bias = sycl::malloc_device<float>(C, q);
    float* d_skip = sycl::malloc_device<float>(N * C, q);
    float* d_gammas = sycl::malloc_device<float>(C, q);
    float* d_betas = sycl::malloc_device<float>(C, q);
    float* d_output = sycl::malloc_device<float>(N * C, q);

    // Copy data to device
    q.memcpy(d_input, input.data(), N * C * sizeof(float));
    q.memcpy(d_bias, bias.data(), C * sizeof(float));
    q.memcpy(d_skip, skip.data(), N * C * sizeof(float));
    q.memcpy(d_gammas, gammas.data(), C * sizeof(float));
    q.memcpy(d_betas, betas.data(), C * sizeof(float));

    // Run SYCL kernel
    LayerNorm(q, N, C, d_output, d_input, d_bias, d_skip, d_gammas, d_betas, epsilon, alpha, activation);

    // Copy results back
    q.memcpy(output_sycl.data(), d_output, N * C * sizeof(float));
    q.wait();

    // Compare results
    const float tolerance = 1e-4f;
    for (int i = 0; i < N * C; ++i) {
        EXPECT_NEAR(output_sycl[i], output_cpu[i], tolerance)
            << "Mismatch at index " << i << ": SYCL=" << output_sycl[i] << ", CPU=" << output_cpu[i];
    }

    // Cleanup
    sycl::free(d_input, q);
    sycl::free(d_bias, q);
    sycl::free(d_skip, q);
    sycl::free(d_gammas, q);
    sycl::free(d_betas, q);
    sycl::free(d_output, q);
}

// Test layer normalization with skip = nullptr
TEST_F(LayerNormTest, TestNoSkip) {
    const int N = 2;
    const int C = 32;   // Must be multiple of 16
    const float epsilon = 1e-6f;
    const float alpha = 0.5f;
    const ActivationFunction activation = ActivationFunction::ACTIVATION_NONE;

    std::vector<float> input(N * C);
    std::vector<float> bias(C);
    std::vector<float> gammas(C);
    std::vector<float> betas(C);
    std::vector<float> output_sycl(N * C);
    std::vector<float> output_cpu(N * C);

    // Generate test data
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    for (auto& x : input) x = dist(rng);
    for (auto& x : bias) x = dist(rng) * 0.2f;
    for (auto& x : gammas) x = 1.0f + dist(rng) * 0.3f; // Around 1.0
    for (auto& x : betas) x = dist(rng) * 0.2f;

    // CPU reference
    layer_norm_cpu_reference(N, C, output_cpu, input, bias, nullptr, gammas, betas, epsilon, alpha, activation);

    // Allocate and copy to device
    float* d_input = sycl::malloc_device<float>(N * C, q);
    float* d_bias = sycl::malloc_device<float>(C, q);
    float* d_gammas = sycl::malloc_device<float>(C, q);
    float* d_betas = sycl::malloc_device<float>(C, q);
    float* d_output = sycl::malloc_device<float>(N * C, q);

    q.memcpy(d_input, input.data(), N * C * sizeof(float));
    q.memcpy(d_bias, bias.data(), C * sizeof(float));
    q.memcpy(d_gammas, gammas.data(), C * sizeof(float));
    q.memcpy(d_betas, betas.data(), C * sizeof(float));

    // Run SYCL kernel with skip = nullptr
    LayerNorm(q, N, C, d_output, d_input, d_bias, nullptr, d_gammas, d_betas, epsilon, alpha, activation);

    q.memcpy(output_sycl.data(), d_output, N * C * sizeof(float));
    q.wait();

    // Compare results
    const float tolerance = 1e-5f;
    for (int i = 0; i < N * C; ++i) {
        EXPECT_NEAR(output_sycl[i], output_cpu[i], tolerance)
            << "Mismatch at index " << i << ": SYCL=" << output_sycl[i] << ", CPU=" << output_cpu[i];
    }

    // Cleanup
    sycl::free(d_input, q);
    sycl::free(d_bias, q);
    sycl::free(d_gammas, q);
    sycl::free(d_betas, q);
    sycl::free(d_output, q);
}

// Test with half precision (if supported by device)
TEST_F(LayerNormTest, TestHalfPrecision) {
    // Check if device supports half precision
    if (!q.get_device().has(sycl::aspect::fp16)) {
        GTEST_SKIP() << "Device does not support half precision";
    }

    const int N = 2;
    const int C = 64;  // Must be multiple of 16
    const float epsilon = 1e-5f;
    const float alpha = 1.0f;
    const ActivationFunction activation = ActivationFunction::ACTIVATION_SWISH;

    std::vector<sycl::half> input(N * C);
    std::vector<sycl::half> bias(C);
    std::vector<sycl::half> skip(N * C);
    std::vector<sycl::half> gammas(C);
    std::vector<sycl::half> betas(C);
    std::vector<sycl::half> output_sycl(N * C);

    // Generate test data with smaller range to avoid overflow
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& x : input) x = static_cast<sycl::half>(dist(rng));
    for (auto& x : bias) x = static_cast<sycl::half>(dist(rng) * 0.1f);
    for (auto& x : skip) x = static_cast<sycl::half>(dist(rng) * 0.5f);
    for (auto& x : gammas) x = static_cast<sycl::half>(0.5f + dist(rng) * 0.5f);
    for (auto& x : betas) x = static_cast<sycl::half>(dist(rng) * 0.1f);

    // Allocate device memory
    sycl::half* d_input = sycl::malloc_device<sycl::half>(N * C, q);
    sycl::half* d_bias = sycl::malloc_device<sycl::half>(C, q);
    sycl::half* d_skip = sycl::malloc_device<sycl::half>(N * C, q);
    sycl::half* d_gammas = sycl::malloc_device<sycl::half>(C, q);
    sycl::half* d_betas = sycl::malloc_device<sycl::half>(C, q);
    sycl::half* d_output = sycl::malloc_device<sycl::half>(N * C, q);

    // Copy to device
    q.memcpy(d_input, input.data(), N * C * sizeof(sycl::half));
    q.memcpy(d_bias, bias.data(), C * sizeof(sycl::half));
    q.memcpy(d_skip, skip.data(), N * C * sizeof(sycl::half));
    q.memcpy(d_gammas, gammas.data(), C * sizeof(sycl::half));
    q.memcpy(d_betas, betas.data(), C * sizeof(sycl::half));

    // Run SYCL kernel
    LayerNorm(q, N, C, d_output, d_input, d_bias, d_skip, d_gammas, d_betas, epsilon, alpha, activation);

    // Copy back and validate basic properties
    q.memcpy(output_sycl.data(), d_output, N * C * sizeof(sycl::half));
    q.wait();

    // Basic sanity checks
    for (int n = 0; n < N; ++n) {
        // Check that outputs are finite
        for (int c = 0; c < C; ++c) {
            int idx = n * C + c;
            float val = static_cast<float>(output_sycl[idx]);
            EXPECT_TRUE(std::isfinite(val)) << "Non-finite value at index " << idx << ": " << val;
        }
    }

    // Cleanup
    sycl::free(d_input, q);
    sycl::free(d_bias, q);
    sycl::free(d_skip, q);
    sycl::free(d_gammas, q);
    sycl::free(d_betas, q);
    sycl::free(d_output, q);
}

// Test edge cases
TEST_F(LayerNormTest, TestEdgeCases) {
    const int N = 1;
    const int C = 16;  // Minimum supported size (multiple of 16)
    const float epsilon = 1e-6f;

    // Test with all zeros
    std::vector<float> input_zeros(N * C, 0.0f);
    std::vector<float> bias(C, 0.1f);
    std::vector<float> gammas(C, 1.0f);
    std::vector<float> betas(C, 0.0f);
    std::vector<float> output(N * C);

    float* d_input = sycl::malloc_device<float>(N * C, q);
    float* d_bias = sycl::malloc_device<float>(C, q);
    float* d_gammas = sycl::malloc_device<float>(C, q);
    float* d_betas = sycl::malloc_device<float>(C, q);
    float* d_output = sycl::malloc_device<float>(N * C, q);

    q.memcpy(d_input, input_zeros.data(), N * C * sizeof(float));
    q.memcpy(d_bias, bias.data(), C * sizeof(float));
    q.memcpy(d_gammas, gammas.data(), C * sizeof(float));
    q.memcpy(d_betas, betas.data(), C * sizeof(float));

    LayerNorm(q, N, C, d_output, d_input, d_bias, nullptr, d_gammas, d_betas, epsilon, 1.0f, ActivationFunction::ACTIVATION_NONE);

    q.memcpy(output.data(), d_output, N * C * sizeof(float));
    q.wait();

    // Check that all outputs are betas (since input is zero)
    for (int i = 0; i < N * C; ++i) {
        EXPECT_NEAR(output[i], 0.0f, 1e-6f) << "Zero input test failed at index " << i;
    }

    // Cleanup
    sycl::free(d_input, q);
    sycl::free(d_bias, q);
    sycl::free(d_gammas, q);
    sycl::free(d_betas, q);
    sycl::free(d_output, q);
}