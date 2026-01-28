/*
  Test program for globalAvgPool kernel migration validation
  Compares CUDA and SYCL implementations for numerical correctness
*/

#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <iomanip>
#include <sycl/sycl.hpp>

// Include our SYCL implementation
#include "../sycl/src/neural/backends/sycl/sycl_common.h"

// Test configuration
constexpr float TOLERANCE = 1e-6f;  // Numerical tolerance for fp32
constexpr float TOLERANCE_FP16 = 1e-3f;  // Larger tolerance for fp32

// Simple activation function for testing
float test_activate(float val, int activation_type) {
  switch (activation_type) {
    case 0:  // NONE
      return val;
    case 1:  // RELU
      return std::max(0.0f, val);
    default:
      return val;
  }
}

// CPU reference implementation of global average pooling
template <typename T>
void cpu_globalAvgPool_ref(int N, int C, T* output, const T* input,
                          const T* prevLayerBias, bool nhwc) {
  const int kPlaneSize = 64;  // 8x8 board

  if (nhwc) {
    // NHWC layout: N x C x 64, where 64 = 8x8 spatial
    for (int n = 0; n < N; n++) {
      for (int c = 0; c < C; c++) {
        float sum = 0.0f;
        // Sum all 64 spatial elements for this channel
        for (int i = 0; i < kPlaneSize; i++) {
          int index = n * C * kPlaneSize + c + i * C;  // NHWC indexing
          sum += static_cast<float>(input[index]);
        }
        float avg = sum / kPlaneSize;
        if (prevLayerBias) {
          avg += static_cast<float>(prevLayerBias[c]);
        }
        output[n * C + c] = static_cast<T>(avg);
      }
    }
  } else {
    // NCHW layout: N x C x 64, where 64 = 8x8 spatial
    for (int n = 0; n < N; n++) {
      for (int c = 0; c < C; c++) {
        float sum = 0.0f;
        // Sum all 64 spatial elements for this channel
        for (int i = 0; i < kPlaneSize; i++) {
          int index = n * C * kPlaneSize + c * kPlaneSize + i;  // NCHW indexing
          sum += static_cast<float>(input[index]);
        }
        float avg = sum / kPlaneSize;
        if (prevLayerBias) {
          avg += static_cast<float>(prevLayerBias[n * C + c]);
        }
        output[n * C + c] = static_cast<T>(avg);
      }
    }
  }
}

// Generate random test data
template <typename T>
void generateTestData(std::vector<T>& data, float min_val = -1.0f, float max_val = 1.0f) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(min_val, max_val);

  for (auto& val : data) {
    val = static_cast<T>(dis(gen));
  }
}

// Test the globalAvgPool implementation
template <typename T>
bool testGlobalAvgPool(sycl::queue& q, int N, int C, bool nhwc, bool with_bias) {
  const int kPlaneSize = 64;
  const int inputSize = N * C * kPlaneSize;
  const int outputSize = N * C;

  std::cout << "\nTesting globalAvgPool with N=" << N << ", C=" << C
            << ", nhwc=" << (nhwc ? "true" : "false")
            << ", with_bias=" << (with_bias ? "true" : "false") << std::endl;
  std::cout << "Data type: " << (std::is_same_v<T, float> ? "float" : "half") << std::endl;

  // Allocate test data
  std::vector<T> input(inputSize);
  std::vector<T> prevLayerBias(with_bias ? C : 0);
  std::vector<T> sycl_output(outputSize);
  std::vector<T> cpu_output(outputSize);

  // Generate random input data
  generateTestData(input, -2.0f, 2.0f);
  if (with_bias) {
    generateTestData(prevLayerBias, -0.5f, 0.5f);
  }

  // Run CPU reference implementation
  cpu_globalAvgPool_ref(N, C, cpu_output.data(), input.data(),
                       with_bias ? prevLayerBias.data() : nullptr, nhwc);

 // Run SYCL implementation
  try {
    // Copy input data to device
    T* d_input = sycl::malloc_device<T>(inputSize, q);
    T* d_output = sycl::malloc_device<T>(outputSize, q);
    T* d_bias = nullptr;

    if (with_bias) {
      d_bias = sycl::malloc_device<T>(prevLayerBias.size(), q);
    }

    // Copy data to device
    q.memcpy(d_input, input.data(), inputSize * sizeof(T)).wait();
    if (with_bias) {
      q.memcpy(d_bias, prevLayerBias.data(), prevLayerBias.size() * sizeof(T)).wait();
    }

    // Run SYCL kernel
    lczero::sycl_backend::globalAvgPool(q, N, C, d_output, d_input, d_bias, nhwc);

    // Copy results back
    q.memcpy(sycl_output.data(), d_output, outputSize * sizeof(T)).wait();

    // Free device memory
    sycl::free(d_input, q);
    sycl::free(d_output, q);
    if (d_bias) {
      sycl::free(d_bias, q);
    }

  } catch (sycl::exception const& e) {
    std::cerr << "SYCL error: " << e.what() << std::endl;
    return false;
  }

  // Compare results
  float max_diff = 0.0f;
  int error_count = 0;
  float tolerance = std::is_same_v<T, float> ? TOLERANCE : TOLERANCE_FP16;

  for (int i = 0; i < outputSize; i++) {
    float sycl_val = static_cast<float>(sycl_output[i]);
    float cpu_val = static_cast<float>(cpu_output[i]);
    float diff = std::abs(sycl_val - cpu_val);

    if (diff > tolerance) {
      error_count++;
      if (error_count <= 5) {  // Show first 5 errors
        std::cout << "Error at index " << i << ": SYCL=" << sycl_val
                  << ", CPU=" << cpu_val << ", diff=" << diff << std::endl;
      }
    }
    max_diff = std::max(max_diff, diff);
  }

  std::cout << "Max difference: " << max_diff << std::endl;
  std::cout << "Errors: " << error_count << " / " << outputSize << std::endl;

  bool passed = error_count == 0;
  std::cout << "Test " << (passed ? "PASSED" : "FAILED") << std::endl;

  return passed;
}

// Test with specific configuration
bool runTestSuite(sycl::queue& q) {
  std::cout << "=== Global Average Pooling Test Suite ===" << std::endl;
  std::cout << "Device: " << q.get_device().get_info<sycl::info::device::name>() << std::endl;

  bool all_passed = true;

  // Test configurations
  struct TestConfig {
    int N, C;
    bool nhwc, with_bias;
  };

  std::vector<TestConfig> configs = {
    {1, 64, false, false},   // Small NCHW, no bias
    {1, 64, false, true},    // Small NCHW, with bias
    {2, 128, false, false},  // Medium NCHW, no bias
    {2, 128, false, true},   // Medium NCHW, with bias
    {1, 64, true, false},    // Small NHWC, no bias
    {1, 64, true, true},     // Small NHWC, with bias
    {2, 128, true, false},   // Medium NHWC, no bias
    {2, 128, true, true},    // Medium NHWC, with bias
  };

  for (const auto& config : configs) {
    // Test with float (NCHW only for float)
    if (!config.nhwc) {
      bool result = testGlobalAvgPool<float>(q, config.N, config.C,
                                           config.nhwc, config.with_bias);
      all_passed = all_passed && result;
    }

    // Test with half (both NCHW and NHWC)
    bool result = testGlobalAvgPool<sycl::half>(q, config.N, config.C,
                                                config.nhwc, config.with_bias);
    all_passed = all_passed && result;
  }

  return all_passed;
}

int main() {
  try {
    // Create SYCL queue
    sycl::queue q(sycl::default_selector_v);

    std::cout << "Global Average Pooling Kernel Validation" << std::endl;
    std::cout << "========================================" << std::endl;

    bool all_tests_passed = runTestSuite(q);

    std::cout << "\n=== Test Summary ===" << std::endl;
    if (all_tests_passed) {
      std::cout << "✓ ALL TESTS PASSED!" << std::endl;
      std::cout << "SYCL implementation matches CPU reference within tolerance." << std::endl;
      return 0;
    } else {
      std::cout << "✗ SOME TESTS FAILED!" << std::endl;
      std::cout << "Check the output above for details." << std::endl;
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