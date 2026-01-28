/*
  Test for SE_Layer_NHWC kernel migration
*/

#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <vector>
#include <random>
#include <chrono>
#include <fstream>
#include <iomanip>
#include "neural/backends/sycl/fp16_kernels.h"
#include "neural/tables/activation_function.h"
#include "utils/float16.h"
#include "test_utils.h"

using namespace lczero;
using namespace sycl_backend;
using sycl::half;

class SELayerNHWCTest : public ::testing::Test {
protected:
    void SetUp() override {
        q = sycl::queue(sycl::default_selector_v);

        // Test parameters
        N = 2;
        C = 64;
        numFc1Out = 32;
        elementsPerThread = 64; // 8x8 board

        // Allocate host memory
        h_input.resize(N * C * elementsPerThread);
        h_skip.resize(N * C * elementsPerThread);
        h_output_cuda.resize(N * C * elementsPerThread);
        h_output_sycl.resize(N * C * elementsPerThread);
        h_w1.resize(C * numFc1Out);
        h_b1.resize(numFc1Out);
        h_w2.resize(numFc1Out * 2 * C);
        h_b2.resize(2 * C);
        h_bPrev.resize(C);

        // Generate test data
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

        // Initialize input and skip
        for (size_t i = 0; i < h_input.size(); ++i) {
            h_input[i] = Float16_float2half(dis(gen));
            h_skip[i] = Float16_float2half(dis(gen));
        }

        // Initialize weights and biases
        for (size_t i = 0; i < h_w1.size(); ++i) {
            h_w1[i] = Float16_float2half(dis(gen) * 0.1f);
        }
        for (size_t i = 0; i < h_b1.size(); ++i) {
            h_b1[i] = Float16_float2half(dis(gen) * 0.1f);
        }
        for (size_t i = 0; i < h_w2.size(); ++i) {
            h_w2[i] = Float16_float2half(dis(gen) * 0.1f);
        }
        for (size_t i = 0; i < h_b2.size(); ++i) {
            h_b2[i] = Float16_float2half(dis(gen) * 0.1f);
        }
        for (size_t i = 0; i < h_bPrev.size(); ++i) {
            h_bPrev[i] = Float16_float2half(dis(gen) * 0.05f);
        }

        // Allocate device memory
        d_input = sycl::malloc_device<half>(h_input.size(), q);
        d_skip = sycl::malloc_device<half>(h_skip.size(), q);
        d_output = sycl::malloc_device<half>(h_output_sycl.size(), q);
        d_w1 = sycl::malloc_device<half>(h_w1.size(), q);
        d_b1 = sycl::malloc_device<half>(h_b1.size(), q);
        d_w2 = sycl::malloc_device<half>(h_w2.size(), q);
        d_b2 = sycl::malloc_device<half>(h_b2.size(), q);
        d_bPrev = sycl::malloc_device<half>(h_bPrev.size(), q);

        // Copy to device
        q.memcpy(d_input, h_input.data(), h_input.size() * sizeof(half));
        q.memcpy(d_skip, h_skip.data(), h_skip.size() * sizeof(half));
        q.memcpy(d_w1, h_w1.data(), h_w1.size() * sizeof(half));
        q.memcpy(d_b1, h_b1.data(), h_b1.size() * sizeof(half));
        q.memcpy(d_w2, h_w2.data(), h_w2.size() * sizeof(half));
        q.memcpy(d_b2, h_b2.data(), h_b2.size() * sizeof(half));
        q.memcpy(d_bPrev, h_bPrev.data(), h_bPrev.size() * sizeof(half));
        q.wait();
    }

    void TearDown() override {
        sycl::free(d_input, q);
        sycl::free(d_skip, q);
        sycl::free(d_output, q);
        sycl::free(d_w1, q);
        sycl::free(d_b1, q);
        sycl::free(d_w2, q);
        sycl::free(d_b2, q);
        sycl::free(d_bPrev, q);
    }

    void CompareResults(const std::vector<half>& ref, const std::vector<half>& test,
                      float tolerance = 1e-3f) {
        ASSERT_EQ(ref.size(), test.size());

        int mismatch_count = 0;
        float max_diff = 0.0f;

        for (size_t i = 0; i < ref.size(); ++i) {
            float ref_val = Float16_half2float(ref[i]);
            float test_val = Float16_half2float(test[i]);
            float diff = std::abs(ref_val - test_val);

            if (diff > tolerance) {
                mismatch_count++;
                max_diff = std::max(max_diff, diff);

                if (mismatch_count <= 10) {
                    std::cout << "Mismatch at index " << i
                              << ": ref=" << ref_val
                              << ", test=" << test_val
                              << ", diff=" << diff << std::endl;
                }
            }
        }

        std::cout << "Total mismatches: " << mismatch_count << "/" << ref.size() << std::endl;
        std::cout << "Maximum difference: " << max_diff << std::endl;

        EXPECT_LT(mismatch_count, ref.size() * 0.001f); // Allow 0.1% mismatches
        EXPECT_LT(max_diff, tolerance * 10); // Allow some tolerance for half precision
    }

    sycl::queue q;
    int N, C, numFc1Out, elementsPerThread;

    std::vector<half> h_input, h_skip, h_output_cuda, h_output_sycl;
    std::vector<half> h_w1, h_b1, h_w2, h_b2, h_bPrev;

    half* d_input;
    half* d_skip;
    half* d_output;
    half* d_w1;
    half* d_b1;
    half* d_w2;
    half* d_b2;
    half* d_bPrev;
};

TEST_F(SELayerNHWCTest, BasicFunctionality) {
    ActivationFunction activation = ActivationFunction::ACTIVATION_RELU;

    // Run SYCL kernel
    bool success = Se_Fp16_NHWC(q, N, C, numFc1Out, d_output, d_skip, d_input,
                                d_w1, d_b1, d_w2, d_b2, d_bPrev, activation);
    ASSERT_TRUE(success);

    // Copy results back
    q.memcpy(h_output_sycl.data(), d_output, h_output_sycl.size() * sizeof(half));
    q.wait();

    // For this test, we'll create a reference implementation on CPU
    // to verify the SYCL results
    std::vector<half> h_reference(N * C * elementsPerThread);

    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            // 1. Compute global average for this channel
            float sum = 0.0f;
            for (int i = 0; i < elementsPerThread; ++i) {
                int idx = n * C * elementsPerThread + i * C + c;
                float input_val = Float16_half2float(h_input[idx]) + Float16_half2float(h_bPrev[c]);
                sum += input_val;
            }
            float avg = sum / elementsPerThread;

            // 2. First fully connected layer
            if (c < numFc1Out) {
                sum = 0.0f;
                for (int i = 0; i < C; ++i) {
                    // Get avg for channel i (computed above)
                    // For simplicity, recompute it here
                    float ch_avg = 0.0f;
                    for (int j = 0; j < elementsPerThread; ++j) {
                        int idx = n * C * elementsPerThread + j * C + i;
                        ch_avg += Float16_half2float(h_input[idx]) + Float16_half2float(h_bPrev[i]);
                    }
                    ch_avg /= elementsPerThread;
                    sum += ch_avg * Float16_half2float(h_w1[i * numFc1Out + c]);
                }
                sum += Float16_half2float(h_b1[c]);
                sum = activate(sum, activation);
            }

            // 3. Second fully connected layer (only for c < C)
            float S = 0.0f, B = 0.0f;
            for (int i = 0; i < numFc1Out; ++i) {
                // Get FC1 output for i (computed above)
                float fc1_out = 0.0f;
                for (int j = 0; j < C; ++j) {
                    float ch_avg = 0.0f;
                    for (int k = 0; k < elementsPerThread; ++k) {
                        int idx = n * C * elementsPerThread + k * C + j;
                        ch_avg += Float16_half2float(h_input[idx]) + Float16_half2float(h_bPrev[j]);
                    }
                    ch_avg /= elementsPerThread;
                    fc1_out += ch_avg * Float16_half2float(h_w1[j * numFc1Out + i]);
                }
                fc1_out += Float16_half2float(h_b1[i]);
                fc1_out = activate(fc1_out, activation);

                S += fc1_out * Float16_half2float(h_w2[i * 2 * C + c]);
                B += fc1_out * Float16_half2float(h_w2[i * 2 * C + (c + C)]);
            }
            S += Float16_half2float(h_b2[c]);
            B += Float16_half2float(h_b2[c + C]);

            // Sigmoid
            S = 1.0f / (1.0f + exp(-S));

            // 4. Apply to all elements
            for (int i = 0; i < elementsPerThread; ++i) {
                int idx = n * C * elementsPerThread + i * C + c;
                float input_val = Float16_half2float(h_input[idx]) + Float16_half2float(h_bPrev[c]);
                float skip_val = Float16_half2float(h_skip[idx]);
                float val = skip_val + input_val * S + B;
                val = activate(val, activation);
                h_reference[idx] = Float16_float2half(val);
            }
        }
    }

    CompareResults(h_reference, h_output_sycl);
}

TEST_F(SELayerNHWCTest, DifferentActivations) {
    std::vector<ActivationFunction> activations = {
        ActivationFunction::ACTIVATION_NONE,
        ActivationFunction::ACTIVATION_RELU,
        ActivationFunction::ACTIVATION_MISH,
        ActivationFunction::ACTIVATION_SIGMOID
    };

    for (auto activation : activations) {
        std::cout << "Testing activation: " << static_cast<int>(activation) << std::endl;

        bool success = Se_Fp16_NHWC(q, N, C, numFc1Out, d_output, d_skip, d_input,
                                    d_w1, d_b1, d_w2, d_b2, d_bPrev, activation);
        ASSERT_TRUE(success);

        q.memcpy(h_output_sycl.data(), d_output, h_output_sycl.size() * sizeof(half));
        q.wait();

        // Verify outputs are reasonable (no NaNs or infinities)
        for (size_t i = 0; i < h_output_sycl.size(); ++i) {
            float val = Float16_half2float(h_output_sycl[i]);
            EXPECT_FALSE(std::isnan(val));
            EXPECT_FALSE(std::isinf(val));

            // RELU outputs should be non-negative
            if (activation == ActivationFunction::ACTIVATION_RELU) {
                EXPECT_GE(val, 0.0f);
            }
        }
    }
}

TEST_F(SELayerNHWCTest,Performance) {
    const int warmup_runs = 5;
    const int timed_runs = 20;

    ActivationFunction activation = ActivationFunction::ACTIVATION_RELU;

    // Warmup
    for (int i = 0; i < warmup_runs; ++i) {
        Se_Fp16_NHWC(q, N, C, numFc1Out, d_output, d_skip, d_input,
                    d_w1, d_b1, d_w2, d_b2, d_bPrev, activation);
    }
    q.wait();

    // Timed runs
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < timed_runs; ++i) {
        Se_Fp16_NHWC(q, N, C, numFc1Out, d_output, d_skip, d_input,
                    d_w1, d_b1, d_w2, d_b2, d_bPrev, activation);
    }
    q.wait();
    auto end = std::chrono::high_resolution_clock::now();

    double elapsed_ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
    double avg_time_ms = elapsed_ms / timed_runs;

    // Total operations: N * C * elementsPerThread (for averaging) +
    //                 C * numFc1Out (first FC) +
    //                 C * numFc1Out (second FC) +
    //                 N * C * elementsPerThread (final apply)
    size_t total_ops = N * C * elementsPerThread * 2 + C * numFc1Out * 2;
    double gflops = (total_ops / 1e9) / (avg_time_ms / 1000.0);

    std::cout << "Average execution time: " << avg_time_ms << " ms" << std::endl;
    std::cout << "Performance: " << gflops << " GFLOPS" << std::endl;

    // Save performance data
    std::ofstream perf_file("results_vector_add/se_layer_nhwc_performance.csv", std::ios::app);
    if (perf_file.is_open()) {
        perf_file << N << "," << C << "," << numFc1Out << "," << avg_time_ms << "," << gflops << std::endl;
        perf_file.close();
    }

    // Reasonable performance expectation (this will vary greatly by hardware)
    EXPECT_LT(avg_time_ms, 10.0); // Should complete in less than 10ms
}