/*
  Test for preprocess_for_attention_body kernel
*/

#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <sycl/sycl.hpp>
#include "neural/backends/sycl/kernels.h"
#include "neural/backends/sycl/sycl_common.h"

using namespace lczero;
using namespace sycl_backend;

// Helper function to compare arrays with tolerance
bool compare_arrays(const float* a, const float* b, size_t size, float tolerance = 1e-5f) {
    for (size_t i = 0; i < size; ++i) {
        if (std::abs(a[i] - b[i]) > tolerance) {
            std::cout << "Mismatch at index " << i << ": " << a[i] << " vs " << b[i] << std::endl;
            return false;
        }
    }
    return true;
}

// CPU reference implementation
void preprocess_for_attention_body_cpu_ref(
    float* output, const float* input, const float* encoding,
    int N, int input_size, int encoding_size, bool is_pe_dense_embedding) {

    for (int n = 0; n < N; ++n) {
        for (int hw = 0; hw < 64; ++hw) {
            for (int c = 0; c < input_size + encoding_size; ++c) {
                float op;
                if (c >= input_size) {
                    // concatenate from position encoding array
                    if (is_pe_dense_embedding) {
                        op = encoding[n * 64 * encoding_size + hw * encoding_size + (c - input_size)];
                    } else {
                        op = encoding[64 * hw + (c - input_size)];
                    }
                } else {
                    // Input is in NCHW format
                    op = input[n * input_size * 64 + c * 64 + hw];
                }

                // Output is in NHWC format
                output[n * 64 * (input_size + encoding_size) + hw * (input_size + encoding_size) + c] = op;
            }
        }
    }
}

int main() {
    try {
        // Create SYCL queue
        sycl::queue q(sycl::cpu_selector_v);
        std::cout << "Running on CPU for testing" << std::endl;

        // Test parameters
        const int N = 2;         // Batch size
        const int input_size = 112;  // Standard input planes for Leela Chess
        const int encoding_size = 64; // Position encoding size
        const bool is_pe_dense_embedding = false;

        // Calculate total sizes
        const int input_total = N * input_size * 64;
        const int encoding_total = is_pe_dense_embedding ? N * 64 * encoding_size : 64 * encoding_size;
        const int output_total = N * 64 * (input_size + encoding_size);

        // Allocate host memory
        std::vector<float> h_input(input_total);
        std::vector<float> h_encoding(encoding_total);
        std::vector<float> h_output_gpu(output_total);
        std::vector<float> h_output_cpu(output_total);

        // Initialize input and encoding with random data
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

        for (int i = 0; i < input_total; ++i) {
            h_input[i] = dis(gen);
        }
        for (int i = 0; i < encoding_total; ++i) {
            h_encoding[i] = dis(gen);
        }

        // Allocate device memory
        float* d_input = sycl::malloc_device<float>(input_total, q);
        float* d_encoding = sycl::malloc_device<float>(encoding_total, q);
        float* d_output = sycl::malloc_device<float>(output_total, q);

        if (!d_input || !d_encoding || !d_output) {
            std::cerr << "Failed to allocate device memory" << std::endl;
            return 1;
        }

        // Copy data to device
        q.memcpy(d_input, h_input.data(), input_total * sizeof(float));
        q.memcpy(d_encoding, h_encoding.data(), encoding_total * sizeof(float));
        q.wait();

        // Run SYCL kernel
        std::cout << "Running SYCL kernel..." << std::endl;
        inputPreprocessForAttentionBody<float>(q, d_output, d_input, d_encoding,
                                               N, input_size, encoding_size,
                                               is_pe_dense_embedding);

        // Copy results back
        q.memcpy(h_output_gpu.data(), d_output, output_total * sizeof(float));
        q.wait();

        // Run CPU reference implementation
        std::cout << "Running CPU reference..." << std::endl;
        preprocess_for_attention_body_cpu_ref(
            h_output_cpu.data(), h_input.data(), h_encoding.data(),
            N, input_size, encoding_size, is_pe_dense_embedding);

        // Compare results
        std::cout << "Comparing results..." << std::endl;
        if (compare_arrays(h_output_gpu.data(), h_output_cpu.data(), output_total)) {
            std::cout << "✅ Test PASSED! SYCL results match CPU reference" << std::endl;

            // Print a few sample values for verification
            std::cout << "\nSample output values:" << std::endl;
            std::cout << "Batch 0, Position 0, Channels 0-3:" << std::endl;
            for (int c = 0; c < 4; ++c) {
                int idx = 0 * 64 * (input_size + encoding_size) + 0 * (input_size + encoding_size) + c;
                std::cout << "  Channel " << c << ": " << std::fixed << std::setprecision(6) << h_output_gpu[idx] << std::endl;
            }
        } else {
            std::cerr << "❌ Test FAILED! SYCL results don't match CPU reference" << std::endl;
            return 1;
        }

        // Clean up
        sycl::free(d_input, q);
        sycl::free(d_encoding, q);
        sycl::free(d_output, q);

    } catch (sycl::exception const& e) {
        std::cerr << "SYCL exception caught: " << e.what() << std::endl;
        return 1;
    } catch (std::exception const& e) {
        std::cerr << "Standard exception caught: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}