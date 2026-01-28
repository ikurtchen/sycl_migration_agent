/*
  Test suite for SYCL softmax kernels
  Tests both the optimized C=64 version and general version
*/

#include <gtest/gtest.h>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <iostream>

// SYCL includes
#include <sycl/sycl.hpp>

// Include our kernels
#include "sycl_common.h"
#include "common_kernels.cpp"  // Including the implementation for testing

using namespace lczero::sycl_backend;

// Helper function to compute softmax on CPU for verification
std::vector<float> computeSoftmaxCPU(const std::vector<float>& input, int N, int C, const std::vector<float>* input2 = nullptr) {
  std::vector<float> output(N * C);

  for (int n = 0; n < N; ++n) {
    // Find max for numerical stability
    float max_val = -std::numeric_limits<float>::max();
    for (int c = 0; c < C; ++c) {
      int idx = n * C + c;
      float val = input[idx];
      if (input2) val += (*input2)[idx];
      max_val = std::max(max_val, val);
    }

    // Compute exp and sum
    float sum = 0.0f;
    std::vector<float> exp_vals(C);
    for (int c = 0; c < C; ++c) {
      int idx = n * C + c;
      float val = input[idx];
      if (input2) val += (*input2)[idx];
      exp_vals[c] = std::exp(val - max_val);
      sum += exp_vals[c];
    }

    // Normalize
    for (int c = 0; c < C; ++c) {
      int idx = n * C + c;
      output[idx] = exp_vals[c] / sum;
    }
  }

  return output;
}

// Test the general softmax kernel
TEST(SoftmaxTest, GeneralKernelFloat) {
  // Set up SYCL queue
  sycl::queue q;

  const int N = 8;
  const int C = 128;  // Not 64, so it will use the general kernel

  // Generate random input data
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(-5.0f, 5.0f);

  std::vector<float> input_host(N * C);
  for (auto& val : input_host) {
    val = dis(gen);
  }

  // Allocate device memory
  float* input_device = sycl::malloc_device<float>(N * C, q);
  float* output_device = sycl::malloc_device<float>(N * C, q);

  // Copy input to device
  q.memcpy(input_device, input_host.data(), N * C * sizeof(float)).wait();

  // Run the kernel
  ASSERT_NO_THROW(Softmax<float>(q, N, C, output_device, input_device, nullptr));

  // Copy result back
  std::vector<float> output_host(N * C);
  q.memcpy(output_host.data(), output_device, N * C * sizeof(float)).wait();

  // Compare with CPU implementation
  std::vector<float> expected = computeSoftmaxCPU(input_host, N, C);

  // Check results with tolerance
  const float tolerance = 1e-6f;
  for (int i = 0; i < N * C; ++i) {
    EXPECT_NEAR(output_host[i], expected[i], tolerance)
      << "Mismatch at index " << i << ": got " << output_host[i] << ", expected " << expected[i];
  }

  // Check that each row sums to 1
  for (int n = 0; n < N; ++n) {
    float row_sum = 0.0f;
    for (int c = 0; c < C; ++c) {
      row_sum += output_host[n * C + c];
    }
    EXPECT_NEAR(row_sum, 1.0f, 1e-6f) << "Row " << n << " doesn't sum to 1: sum = " << row_sum;
  }

  // Cleanup
  sycl::free(input_device, q);
  sycl::free(output_device, q);
}

// Test the optimized C=64 kernel
TEST(SoftmaxTest, OptimizedKernelFloat) {
  // Set up SYCL queue
  sycl::queue q;

  const int N = 5;
  const int C = 64;  // This will use the optimized kernel

  // Generate random input data
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(-5.0f, 5.0f);

  std::vector<float> input_host(N * C);
  for (auto& val : input_host) {
    val = dis(gen);
  }

  // Allocate device memory
  float* input_device = sycl::malloc_device<float>(N * C, q);
  float* output_device = sycl::malloc_device<float>(N * C, q);

  // Copy input to device
  q.memcpy(input_device, input_host.data(), N * C * sizeof(float)).wait();

  // Run the kernel
  ASSERT_NO_THROW(Softmax<float>(q, N, C, output_device, input_device, nullptr));

  // Copy result back
  std::vector<float> output_host(N * C);
  q.memcpy(output_host.data(), output_device, N * C * sizeof(float)).wait();

  // Compare with CPU implementation
  std::vector<float> expected = computeSoftmaxCPU(input_host, N, C);

  // Check results with tolerance
  const float tolerance = 1e-6f;
  for (int i = 0; i < N * C; ++i) {
    EXPECT_NEAR(output_host[i], expected[i], tolerance)
      << "Mismatch at index " << i << ": got " << output_host[i] << ", expected " << expected[i];
  }

  // Check that each row sums to 1
  for (int n = 0; n < N; ++n) {
    float row_sum = 0.0f;
    for (int c = 0; c < C; ++c) {
      row_sum += output_host[n * C + c];
    }
    EXPECT_NEAR(row_sum, 1.0f, 1e-6f) << "Row " << n << " doesn't sum to 1: sum = " << row_sum;
  }

  // Cleanup
  sycl::free(input_device, q);
  sycl::free(output_device, q);
}

// Test with input2 (addition before softmax)
TEST(SoftmaxTest, WithInput2) {
  // Set up SYCL queue
  sycl::queue q;

  const int N = 4;
  const int C = 64;

  // Generate random input data
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(-5.0f, 5.0f);

  std::vector<float> input_host(N * C);
  std::vector<float> input2_host(N * C);
  for (auto& val : input_host) val = dis(gen);
  for (auto& val : input2_host) val = dis(gen);

  // Allocate device memory
  float* input_device = sycl::malloc_device<float>(N * C, q);
  float* input2_device = sycl::malloc_device<float>(N * C, q);
  float* output_device = sycl::malloc_device<float>(N * C, q);

  // Copy inputs to device
  q.memcpy(input_device, input_host.data(), N * C * sizeof(float)).wait();
  q.memcpy(input2_device, input2_host.data(), N * C * sizeof(float)).wait();

  // Run the kernel
  ASSERT_NO_THROW(Softmax<float>(q, N, C, output_device, input_device, input2_device));

  // Copy result back
  std::vector<float> output_host(N * C);
  q.memcpy(output_host.data(), output_device, N * C * sizeof(float)).wait();

  // Compare with CPU implementation
  std::vector<float> expected = computeSoftmaxCPU(input_host, N, C, &input2_host);

  // Check results with tolerance
  const float tolerance = 1e-6f;
  for (int i = 0; i < N * C; ++i) {
    EXPECT_NEAR(output_host[i], expected[i], tolerance)
      << "Mismatch at index " << i << ": got " << output_host[i] << ", expected " << expected[i];
  }

  // Check that each row sums to 1
  for (int n = 0; n < N; ++n) {
    float row_sum = 0.0f;
    for (int c = 0; c < C; ++c) {
      row_sum += output_host[n * C + c];
    }
    EXPECT_NEAR(row_sum, 1.0f, 1e-6f) << "Row " << n << " doesn't sum to 1: sum = " << row_sum;
  }

  // Cleanup
  sycl::free(input_device, q);
  sycl::free(input2_device, q);
  sycl::free(output_device, q);
}

// Test edge cases
TEST(SoftmaxTest, EdgeCases) {
  // Set up SYCL queue
  sycl::queue q;

  const int N = 3;
  const int C = 64;

  // Test case 1: All zeros
  std::vector<float> zeros(N * C, 0.0f);
  float* input_device = sycl::malloc_device<float>(N * C, q);
  float* output_device = sycl::malloc_device<float>(N * C, q);
  q.memcpy(input_device, zeros.data(), N * C * sizeof(float)).wait();

  Softmax<float>(q, N, C, output_device, input_device, nullptr);

  std::vector<float> output_host(N * C);
  q.memcpy(output_host.data(), output_device, N * C * sizeof(float)).wait();

  // All outputs should be equal (1/C)
  const float expected = 1.0f / C;
  const float tolerance = 1e-6f;
  for (int i = 0; i < N * C; ++i) {
    EXPECT_NEAR(output_host[i], expected, tolerance);
  }

  // Test case 2: Large values (test clamping)
  std::vector<float> large_values(N * C);
  for (int i = 0; i < N * C; ++i) {
    large_values[i] = (i % 2 == 0) ? 1000.0f : -1000.0f;
  }

  q.memcpy(input_device, large_values.data(), N * C * sizeof(float)).wait();
  Softmax<float>(q, N, C, output_device, input_device, nullptr);
  q.memcpy(output_host.data(), output_device, N * C * sizeof(float)).wait();

  // Check that rows sum to 1
  for (int n = 0; n < N; ++n) {
    float row_sum = 0.0f;
    for (int c = 0; c < C; ++c) {
      row_sum += output_host[n * C + c];
    }
    EXPECT_NEAR(row_sum, 1.0f, 1e-5f) << "Row " << n << " doesn't sum to 1 with large values";
  }

  // Cleanup
  sycl::free(input_device, q);
  sycl::free(output_device, q);
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}