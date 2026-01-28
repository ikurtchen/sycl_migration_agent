/*
  Test for SYCL Multi-Head Attention kernel
*/

#include <gtest/gtest.h>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <sycl/sycl.hpp>
#include "neural/backends/sycl/kernels.h"
#include "neural/backends/sycl/sycl_common.h"

using data_type = sycl::half;

// Reference implementation of attention (simplified for testing)
void referenceAttention(data_type* output, const data_type* query, const data_type* key,
                      const data_type* value, const data_type* bias,
                      int batch_size, int num_heads, int depth, bool use_bias) {
  const int kNumQueries = 64;
  const int kNumKeys = 64;
  const float scale = 1.0f / sqrtf(depth);

  for (int b = 0; b < batch_size; ++b) {
    for (int h = 0; h < num_heads; ++h) {
      for (int q_idx = 0; q_idx < kNumQueries; ++q_idx) {
        // Compute attention scores
        std::vector<float> attn_scores(kNumKeys);
        for (int k_idx = 0; k_idx < kNumKeys; ++k_idx) {
          float score = 0.0f;
          for (int d = 0; d < depth; ++d) {
            float q_val = static_cast<float>(query[b * depth * num_heads * kNumQueries + h * depth * kNumQueries + q_idx * depth + d]);
            float k_val = static_cast<float>(key[b * depth * num_heads * kNumKeys + h * depth * kNumKeys + k_idx * depth + d]);
            score += q_val * k_val;
          }
          score *= scale;

          if (use_bias) {
            score += static_cast<float>(bias[b * num_heads * kNumQueries * kNumKeys + h * kNumQueries * kNumKeys + q_idx * kNumKeys + k_idx]);
          }

          attn_scores[k_idx] = score;
        }

        // Softmax
        float max_score = *std::max_element(attn_scores.begin(), attn_scores.end());
        float sum = 0.0f;
        for (int i = 0; i < kNumKeys; ++i) {
          attn_scores[i] = expf(attn_scores[i] - max_score);
          sum += attn_scores[i];
        }
        for (int i = 0; i < kNumKeys; ++i) {
          attn_scores[i] /= sum;
        }

        // Compute weighted sum of values
        for (int d = 0; d < depth; ++d) {
          float weighted_sum = 0.0f;
          for (int k_idx = 0; k_idx < kNumKeys; ++k_idx) {
            float v_val = static_cast<float>(value[b * depth * num_heads * kNumKeys + h * depth * kNumKeys + k_idx * depth + d]);
            weighted_sum += attn_scores[k_idx] * v_val;
          }
          output[b * depth * num_heads * kNumQueries + h * depth * kNumQueries + q_idx * depth + d] = static_cast<data_type>(weighted_sum);
        }
      }
    }
  }
}

class FusedMHATest : public ::testing::Test {
protected:
  void SetUp() override {
    queue = sycl::queue{sycl::default_selector()};

    batch_size = 2;
    num_heads = 4;
    depth = 32;
    kNumQueries = 64;
    kNumKeys = 64;

    // Calculate total sizes
    int query_size = batch_size * num_heads * kNumQueries * depth;
    int key_size = batch_size * num_heads * kNumKeys * depth;
    int value_size = batch_size * num_heads * kNumKeys * depth;
    int bias_size = batch_size * num_heads * kNumQueries * kNumKeys;
    int output_size = batch_size * num_heads * kNumQueries * depth;

    // Allocate host memory
    host_query.resize(query_size);
    host_key.resize(key_size);
    host_value.resize(value_size);
    host_bias.resize(bias_size);
    host_output_sycl.resize(output_size);
    host_output_ref.resize(output_size);

    // Generate random data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    for (auto& val : host_query) {
      val = static_cast<data_type>(dis(gen));
    }
    for (auto& val : host_key) {
      val = static_cast<data_type>(dis(gen));
    }
    for (auto& val : host_value) {
      val = static_cast<data_type>(dis(gen));
    }
    for (auto& val : host_bias) {
      val = static_cast<data_type>(dis(gen) * 0.1f);  // Smaller bias values
    }

    // Allocate device memory
    device_query = sycl::malloc_device<data_type>(query_size, queue);
    device_key = sycl::malloc_device<data_type>(key_size, queue);
    device_value = sycl::malloc_device<data_type>(value_size, queue);
    device_bias = sycl::malloc_device<data_type>(bias_size, queue);
    device_output = sycl::malloc_device<data_type>(output_size, queue);

    // Copy data to device
    queue.memcpy(device_query, host_query.data(), query_size * sizeof(data_type));
    queue.memcpy(device_key, host_key.data(), key_size * sizeof(data_type));
    queue.memcpy(device_value, host_value.data(), value_size * sizeof(data_type));
    queue.memcpy(device_bias, host_bias.data(), bias_size * sizeof(data_type));
    queue.wait();
  }

  void TearDown() override {
    sycl::free(device_query, queue);
    sycl::free(device_key, queue);
    sycl::free(device_value, queue);
    sycl::free(device_bias, queue);
    sycl::free(device_output, queue);
  }

  sycl::queue queue;
  int batch_size;
  int num_heads;
  int depth;
  int kNumQueries;
  int kNumKeys;

  std::vector<data_type> host_query, host_key, host_value, host_bias;
  std::vector<data_type> host_output_sycl, host_output_ref;

  data_type* device_query;
  data_type* device_key;
  data_type* device_value;
  data_type* device_bias;
  data_type* device_output;
};

TEST_F(FusedMHATest, TestWithoutBias) {
  // Run SYCL version without bias
  lczero::sycl_backend::fusedMHA(device_output, device_query, device_key, device_value,
                                nullptr, batch_size, num_heads, depth, queue);

  // Copy result back
  queue.memcpy(host_output_sycl.data(), device_output, host_output_sycl.size() * sizeof(data_type));
  queue.wait();

  // Compute reference
  referenceAttention(host_output_ref.data(), host_query.data(), host_key.data(),
                   host_value.data(), nullptr, batch_size, num_heads, depth, false);

  // Compare results with tolerance
  const float tolerance = 1e-3f;
  for (size_t i = 0; i < host_output_sycl.size(); ++i) {
    float sycl_val = static_cast<float>(host_output_sycl[i]);
    float ref_val = static_cast<float>(host_output_ref[i]);
    float error = std::abs(sycl_val - ref_val);
    EXPECT_LT(error, tolerance) << "Mismatch at index " << i
                               << ": SYCL=" << sycl_val << ", REF=" << ref_val
                               << ", ERROR=" << error;
  }
}

TEST_F(FusedMHATest, TestWithBias) {
  // Run SYCL version with bias
  lczero::sycl_backend::fusedMHA(device_output, device_query, device_key, device_value,
                                 device_bias, batch_size, num_heads, depth, queue);

  // Copy result back
  queue.memcpy(host_output_sycl.data(), device_output, host_output_sycl.size() * sizeof(data_type));
  queue.wait();

  // Compute reference
  referenceAttention(host_output_ref.data(), host_query.data(), host_key.data(),
                   host_value.data(), host_bias.data(), batch_size, num_heads, depth, true);

  // Compare results with tolerance
  const float tolerance = 1e-3f;
  for (size_t i = 0; i < host_output_sycl.size(); ++i) {
    float sycl_val = static_cast<float>(host_output_sycl[i]);
    float ref_val = static_cast<float>(host_output_ref[i]);
    float error = std::abs(sycl_val - ref_val);
    EXPECT_LT(error, tolerance) << "Mismatch at index " << i
                               << ": SYCL=" << sycl_val << ", REF=" << ref_val
                               << ", ERROR=" << error;
  }
}

TEST_F(FusedMHATest, TestInvalidParameters) {
  // Test with null pointers
  EXPECT_THROW(
    lczero::sycl_backend::fusedMHA(nullptr, device_query, device_key, device_value,
                                  nullptr, batch_size, num_heads, depth, queue),
    std::invalid_argument
  );

  // Test with invalid dimensions
  EXPECT_THROW(
    lczero::sycl_backend::fusedMHA(device_output, device_query, device_key, device_value,
                                  nullptr, -1, num_heads, depth, queue),
    std::invalid_argument
  );
}