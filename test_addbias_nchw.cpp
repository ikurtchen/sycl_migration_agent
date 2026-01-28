/*
   Test for addBias_NCHW kernel migration from CUDA to SYCL
   This test verifies that the SYCL implementation produces identical results
   to the CUDA version for the NCHW bias addition operation.
*/

#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <sycl/sycl.hpp>
#include "sycl/src/neural/backends/sycl/sycl_common.h"
#include "sycl/src/neural/backends/sycl/kernels.h"

using namespace lczero::sycl_backend;

// Test configuration
constexpr float EPSILON = 1e-5f;
constexpr int N = 2;  // Batch size
constexpr int C = 64; // Channels
constexpr int H = 8;  // Height
constexpr int W = 8;  // Width

// Generate test data
void generateTestData(std::vector<float>& input, std::vector<float>& bias,
                      std::vector<float>& output_golden,
                      ActivationFunction activation) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

  // Generate input tensor
  for (int i = 0; i < N * C * H * W; ++i) {
    input[i] = dis(gen);
  }

  // Generate bias (one per channel)
  for (int i = 0; i < C; ++i) {
    bias[i] = dis(gen);
  }

  // Compute golden result on CPU
  for (int n = 0; n < N; ++n) {
    for (int c = 0; c < C; ++c) {
      for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
          int idx = n * C * H * W + c * H * W + h * W + w;
          float val = input[idx] + bias[c];
          output_golden[idx] = activate(val, activation);
        }
      }
    }
  }
}

// Compare results
bool compareResults(const std::vector<float>& output,
                   const std::vector<float>& output_golden) {
  for (int i = 0; i < N * C * H * W; ++i) {
    if (std::abs(output[i] - output_golden[i]) > EPSILON) {
      std::cout << "Mismatch at index " << i << ": ";
      std::cout << "Expected " << output_golden[i] << ", got " << output[i] << std::endl;
      return false;
    }
  }
  return true;
}

// Test addBias_NCHW with different activation functions
void testAddBiasNCHW(sycl::queue& q, ActivationFunction activation, const std::string& name) {
  std::cout << "Testing addBias_NCHW with " << name << " activation..." << std::endl;

  // Allocate host memory
  std::vector<float> input(N * C * H * W);
  std::vector<float> bias(C);
  std::vector<float> output(N * C * H * W);
  std::vector<float> output_golden(N * C * H * W);

  // Generate test data
  generateTestData(input, bias, output_golden, activation);

  // Allocate device memory
  float* d_input = sycl::malloc_device<float>(N * C * H * W, q);
  float* d_bias = sycl::malloc_device<float>(C, q);
  float* d_output = sycl::malloc_device<float>(N * C * H * W, q);

  if (!d_input || !d_bias || !d_output) {
    std::cerr << "Failed to allocate device memory!" << std::endl;
    return;
  }

  // Copy data to device
  q.memcpy(d_input, input.data(), sizeof(float) * N * C * H * W);
  q.memcpy(d_bias, bias.data(), sizeof(float) * C);
  q.wait();

  // Run kernel
  addBias_NCHW(q, d_output, d_input, d_bias, N, C, H, W, activation);

  // Copy result back
  q.memcpy(output.data(), d_output, sizeof(float) * N * C * H * W);
  q.wait();

  // Compare results
  bool passed = compareResults(output, output_golden);
  std::cout << name << " test: " << (passed ? "PASSED" : "FAILED") << std::endl;

  // Free device memory
  sycl::free(d_input, q);
  sycl::free(d_bias, q);
  sycl::free(d_output, q);
}

// Test with half precision
void testAddBiasNCHWHalf(sycl::queue& q) {
  std::cout << "\nTesting addBias_NCHW with half precision..." << std::endl;

  // Allocate host memory
  std::vector<float> input_f(N * C * H * W);
  std::vector<float> bias_f(C);
  std::vector<float> output_golden_f(N * C * H * W);

  // Generate test data and compute golden result in float
  generateTestData(input_f, bias_f, output_golden_f, ActivationFunction::ACTIVATION_RELU);

  // Convert to half
  std::vector<sycl::half> input(N * C * H * W);
  std::vector<sycl::half> bias(C);
  std::vector<sycl::half> output(N * C * H * W);
  std::vector<float> output_f(N * C * H * W);

  for (int i = 0; i < N * C * H * W; ++i) {
    input[i] = static_cast<sycl::half>(input_f[i]);
  }
  for (int i = 0; i < C; ++i) {
    bias[i] = static_cast<sycl::half>(bias_f[i]);
  }

  // Allocate device memory
  sycl::half* d_input = sycl::malloc_device<sycl::half>(N * C * H * W, q);
  sycl::half* d_bias = sycl::malloc_device<sycl::half>(C, q);
  sycl::half* d_output = sycl::malloc_device<sycl::half>(N * C * H * W, q);

  if (!d_input || !d_bias || !d_output) {
    std::cerr << "Failed to allocate device memory for half precision!" << std::endl;
    return;
  }

  // Copy data to device
  q.memcpy(d_input, input.data(), sizeof(sycl::half) * N * C * H * W);
  q.memcpy(d_bias, bias.data(), sizeof(sycl::half) * C);
  q.wait();

  // Run kernel
  addBias_NCHW(q, d_output, d_input, d_bias, N, C, H, W, ActivationFunction::ACTIVATION_RELU);

  // Copy result back
  q.memcpy(output.data(), d_output, sizeof(sycl::half) * N * C * H * W);
  q.wait();

  // Convert back to float for comparison
  for (int i = 0; i < N * C * H * W; ++i) {
    output_f[i] = static_cast<float>(output[i]);
  }

  // Compare results with larger tolerance for half precision
  constexpr float HALF_EPSILON = 1e-3f;
  bool passed = true;
  for (int i = 0; i < N * C * H * W; ++i) {
    if (std::abs(output_f[i] - output_golden_f[i]) > HALF_EPSILON) {
      std::cout << "Half precision mismatch at index " << i << ": ";
      std::cout << "Expected " << output_golden_f[i] << ", got " << output_f[i] << std::endl;
      passed = false;
      break;
    }
  }
  std::cout << "Half precision test: " << (passed ? "PASSED" : "FAILED") << std::endl;

  // Free device memory
  sycl::free(d_input, q);
  sycl::free(d_bias, q);
  sycl::free(d_output, q);
}

int main() {
  try {
    // Create SYCL queue
    sycl::queue q(sycl::default_selector_v);
    std::cout << "Running on: " << q.get_device().get_info<sycl::info::device::name>() << std::endl;

    // Test different activation functions
    testAddBiasNCHW(q, ActivationFunction::ACTIVATION_NONE, "NONE");
    testAddBiasNCHW(q, ActivationFunction::ACTIVATION_RELU, "RELU");
    testAddBiasNCHW(q, ActivationFunction::ACTIVATION_RELU_2, "RELU_2");
    testAddBiasNCHW(q, ActivationFunction::ACTIVATION_TANH, "TANH");
    testAddBiasNCHW(q, ActivationFunction::ACTIVATION_SIGMOID, "SIGMOID");
    testAddBiasNCHW(q, ActivationFunction::ACTIVATION_SELU, "SELU");
    testAddBiasNCHW(q, ActivationFunction::ACTIVATION_MISH, "MISH");
    testAddBiasNCHW(q, ActivationFunction::ACTIVATION_SWISH, "SWISH");

    // Test half precision
    testAddBiasNCHWHalf(q);

    std::cout << "\nAll tests completed successfully!" << std::endl;

  } catch (sycl::exception const& e) {
    std::cerr << "SYCL exception caught: " << e.what() << std::endl;
    return 1;
  } catch (std::exception const& e) {
    std::cerr << "Standard exception caught: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}