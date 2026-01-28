/*
  Test file for globalScale kernel SYCL translation
  This tests both fp32 (NCHW) and fp16 (NHWC) variants of the globalScale kernel
*/

#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <vector>
#include <random>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <cstring>
#include "neural/backends/sycl/sycl_common.h"

// Function declarations for the SYCL kernels
namespace lczero {
namespace sycl_backend {
  template <typename T>
  void globalScale(int N, int C, T* output, const T* input, const T* scaleBias,
                   const T* prevLayerBias, bool nhwc,
                   ActivationFunction activation, sycl::queue& q);
}
}

using namespace lczero::sycl_backend;

template <typename T>
class GlobalScaleTest : public ::testing::Test {
protected:
  void SetUp() override {
    q = sycl::queue{sycl::default_selector{}};

    // Test parameters
    N = 4;
    C = 32;
    H = 8;
    W = 8;
    kPlaneSize = H * W;
    inputSize = N * C * kPlaneSize;

    // Initialize random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    // Allocate and initialize test data
    inputHost.resize(inputSize);
    outputHost.resize(inputSize);
    scaleBiasHost.resize(N * 2 * C);
    prevLayerBiasHost.resize(C);

    if constexpr (std::is_same_v<T, float>) {
      // For fp32 NCHW layout
      outputInit.resize(N * C * kPlaneSize);

      for (int i = 0; i < inputSize; ++i) {
        inputHost[i] = dis(gen);
        outputHost[i] = dis(gen);
        outputInit[i] = outputHost[i];  // Save initial output for comparison
      }

      for (int i = 0; i < N * 2 * C; ++i) {
        scaleBiasHost[i] = dis(gen);
      }

      for (int i = 0; i < C; ++i) {
        prevLayerBiasHost[i] = dis(gen) * 0.5f;  // Smaller bias values
      }

    } else {
      // For fp16 NHWC layout
      outputInit.resize(N * C * kPlaneSize);

      for (int i = 0; i < inputSize; ++i) {
        inputHost[i] = sycl::half(dis(gen));
        outputHost[i] = sycl::half(dis(gen));
        outputInit[i] = outputHost[i];
      }

      for (int i = 0; i < N * 2 * C; ++i) {
        scaleBiasHost[i] = sycl::half(dis(gen));
      }

      for (int i = 0; i < C; ++i) {
        prevLayerBiasHost[i] = sycl::half(dis(gen) * 0.5f);
      }
    }

    // Allocate device memory
    inputDevice = sycl::malloc_device<T>(inputSize, q);
    outputDevice = sycl::malloc_device<T>(inputSize, q);
    outputInitDevice = sycl::malloc_device<T>(inputSize, q);
    scaleBiasDevice = sycl::malloc_device<T>(N * 2 * C, q);
    prevLayerBiasDevice = sycl::malloc_device<T>(C, q);

    // Copy host data to device
    q.memcpy(inputDevice, inputHost.data(), inputSize * sizeof(T));
    q.memcpy(outputDevice, outputHost.data(), inputSize * sizeof(T));
    q.memcpy(outputInitDevice, outputInit.data(), inputSize * sizeof(T));
    q.memcpy(scaleBiasDevice, scaleBiasHost.data(), N * 2 * C * sizeof(T));
    q.memcpy(prevLayerBiasDevice, prevLayerBiasHost.data(), C * sizeof(T));
    q.wait();
  }

  void TearDown() override {
    // Free device memory
    sycl::free(inputDevice, q);
    sycl::free(outputDevice, q);
    sycl::free(outputInitDevice, q);
    sycl::free(scaleBiasDevice, q);
    sycl::free(prevLayerBiasDevice, q);
  }

  // CPU reference implementation for globalScale (NCHW layout)
  void cpuReferenceGlobalScale_NCHW(std::vector<T>& output, const std::vector<T>& input,
                                    const std::vector<T>& scaleBias, const std::vector<T>& prevLayerBias,
                                    ActivationFunction activation) {
    const int kPlaneSize = H * W;

    for (int n = 0; n < N; ++n) {
      for (int c = 0; c < C; ++c) {
        float scaleValue = static_cast<float>(scaleBias[n * 2 * C + c]);
        float biasValue = static_cast<float>(scaleBias[n * 2 * C + C + c]);

        // Apply sigmoid to scale
        float scaleFactor = 1.0f / (1.0f + std::exp(-scaleValue));

        float prevBias = prevLayerBias.empty() ? 0.0f : static_cast<float>(prevLayerBias[c]);

        for (int i = 0; i < kPlaneSize; ++i) {
          int idx = n * C * kPlaneSize + c * kPlaneSize + i;

          float inputVal = static_cast<float>(input[idx]);
          float outputVal = static_cast<float>(output[idx]);

          float scaled = (inputVal + prevBias) * scaleFactor;
          float result = scaled + outputVal + biasValue;

          // Apply activation
          result = activate(result, activation);

          output[idx] = static_cast<T>(result);
        }
      }
    }
  }

  // CPU reference implementation for globalScale (NHWC layout)
  void cpuReferenceGlobalScale_NHWC(std::vector<T>& output, const std::vector<T>& input,
                                    const std::vector<T>& scaleBias, const std::vector<T>& prevLayerBias,
                                    ActivationFunction activation) {
    int HWC = H * W * C;

    for (int tid = 0; tid < inputSize; ++tid) {
      // Calculate n and c for NHWC layout
      int c = tid % C;
      int n = tid / HWC;

      float scaleValue = static_cast<float>(scaleBias[n * 2 * C + c]);
      float biasValue = static_cast<float>(scaleBias[n * 2 * C + C + c]);

      // Apply sigmoid to scale
      float scaleFactor = 1.0f / (1.0f + std::exp(-scaleValue));

      float prevBias = prevLayerBias.empty() ? 0.0f : static_cast<float>(prevLayerBias[c]);

      float inputVal = static_cast<float>(input[tid]);
      float outputVal = static_cast<float>(output[tid]);

      float scaled = (inputVal + prevBias) * scaleFactor;
      float result = scaled + outputVal + biasValue;

      // Apply activation
      result = activate(result, activation);

      output[tid] = static_cast<T>(result);
    }
  }

  void compareResults(const std::vector<T>& syclResult, const std::vector<T>& cpuResult,
                     float tolerance = 1e-5f) {
    ASSERT_EQ(syclResult.size(), cpuResult.size()) << "Result sizes don't match";

    int mismatchCount = 0;
    float maxDiff = 0.0f;

    for (size_t i = 0; i < syclResult.size(); ++i) {
      float diff = std::abs(static_cast<float>(syclResult[i]) - static_cast<float>(cpuResult[i]));
      maxDiff = std::max(maxDiff, diff);

      if (diff > tolerance) {
        mismatchCount++;
        if (mismatchCount <= 10) {  // Print first 10 mismatches
          std::cout << "Mismatch at index " << i << ": SYCL=" << static_cast<float>(syclResult[i])
                    << ", CPU=" << static_cast<float>(cpuResult[i]) << ", diff=" << diff << std::endl;
        }
      }
    }

    EXPECT_LE(mismatchCount, 0) << "Found " << mismatchCount << " mismatches, max difference: " << maxDiff;
    if (mismatchCount == 0) {
      std::cout << "All results match within tolerance " << tolerance << " (max diff: " << maxDiff << ")" << std::endl;
    }
  }

  sycl::queue q;
  int N, C, H, W, kPlaneSize, inputSize;
  std::vector<T> inputHost, outputHost, outputInit, scaleBiasHost, prevLayerBiasHost;
  T *inputDevice, *outputDevice, *outputInitDevice, *scaleBiasDevice, *prevLayerBiasDevice;
};

// Test instantiation for float (NCHW)
class GlobalScaleFloatTest : public GlobalScaleTest<float> {};

// Test instantiation for sycl::half (NHWC)
class GlobalScaleHalfTest : public GlobalScaleTest<sycl::half> {};

TEST_F(GlobalScaleFloatTest, GlobalScale_NCHW_None) {
  // Test with ACTIVATION_NONE (NCHW layout for fp32)
  ActivationFunction activation = ActivationFunction::ACTIVATION_NONE;

  // Run SYCL version
  std::vector<float> syclOutput = outputInit;
  q.memcpy(outputDevice, syclOutput.data(), inputSize * sizeof(float));
  globalScale(N, C, outputDevice, inputDevice, scaleBiasDevice, prevLayerBiasDevice, false, activation, q);
  q.memcpy(syclOutput.data(), outputDevice, inputSize * sizeof(float));
  q.wait();

  // Run CPU reference
  std::vector<float> cpuOutput = outputInit;
  cpuReferenceGlobalScale_NCHW(cpuOutput, inputHost, scaleBiasHost, prevLayerBiasHost, activation);

  // Compare results
  compareResults(syclOutput, cpuOutput, 1e-5f);
}

TEST_F(GlobalScaleFloatTest, GlobalScale_NCHW_Relu) {
  // Test with ACTIVATION_RELU (NCHW layout for fp32)
  ActivationFunction activation = ActivationFunction::ACTIVATION_RELU;

  // Run SYCL version
  std::vector<float> syclOutput = outputInit;
  q.memcpy(outputDevice, syclOutput.data(), inputSize * sizeof(float));
  globalScale(N, C, outputDevice, inputDevice, scaleBiasDevice, prevLayerBiasDevice, false, activation, q);
  q.memcpy(syclOutput.data(), outputDevice, inputSize * sizeof(float));
  q.wait();

  // Run CPU reference
  std::vector<float> cpuOutput = outputInit;
  cpuReferenceGlobalScale_NCHW(cpuOutput, inputHost, scaleBiasHost, prevLayerBiasHost, activation);

  // Compare results
  compareResults(syclOutput, cpuOutput, 1e-5f);
}

TEST_F(GlobalScaleHalfTest, GlobalScale_NHWC_None) {
  // Test with ACTIVATION_NONE (NHWC layout for fp16)
  ActivationFunction activation = ActivationFunction::ACTIVATION_NONE;

  // Run SYCL version
  std::vector<sycl::half> syclOutput = outputInit;
  q.memcpy(outputDevice, syclOutput.data(), inputSize * sizeof(sycl::half));
  globalScale(N, C, outputDevice, inputDevice, scaleBiasDevice, prevLayerBiasDevice, true, activation, q);
  q.memcpy(syclOutput.data(), outputDevice, inputSize * sizeof(sycl::half));
  q.wait();

  // Run CPU reference
  std::vector<sycl::half> cpuOutput = outputInit;
  cpuReferenceGlobalScale_NHWC(cpuOutput, inputHost, scaleBiasHost, prevLayerBiasHost, activation);

  // Compare results
  compareResults(syclOutput, cpuOutput, 1e-3f);  // Higher tolerance for fp16
}

TEST_F(GlobalScaleHalfTest, GlobalScale_NHWC_Relu) {
  // Test with ACTIVATION_RELU (NHWC layout for fp16)
  ActivationFunction activation = ActivationFunction::ACTIVATION_RELU;

  // Run SYCL version
  std::vector<sycl::half> syclOutput = outputInit;
  q.memcpy(outputDevice, syclOutput.data(), inputSize * sizeof(sycl::half));
  globalScale(N, C, outputDevice, inputDevice, scaleBiasDevice, prevLayerBiasDevice, true, activation, q);
  q.memcpy(syclOutput.data(), outputDevice, inputSize * sizeof(sycl::half));
  q.wait();

  // Run CPU reference
  std::vector<sycl::half> cpuOutput = outputInit;
  cpuReferenceGlobalScale_NHWC(cpuOutput, inputHost, scaleBiasHost, prevLayerBiasHost, activation);

  // Compare results
  compareResults(syclOutput, cpuOutput, 1e-3f);  // Higher tolerance for fp16
}

TEST_F(GlobalScaleFloatTest, GlobalScale_NCHW_NoPrevBias) {
  // Test without prevLayerBias (NCHW layout for fp32)
  ActivationFunction activation = ActivationFunction::ACTIVATION_NONE;

  // Run SYCL version with null prevLayerBias
  std::vector<float> syclOutput = outputInit;
  q.memcpy(outputDevice, syclOutput.data(), inputSize * sizeof(float));
  globalScale(N, C, outputDevice, inputDevice, scaleBiasDevice, nullptr, false, activation, q);
  q.memcpy(syclOutput.data(), outputDevice, inputSize * sizeof(float));
  q.wait();

  // Run CPU reference with empty prevLayerBias
  std::vector<float> cpuOutput = outputInit;
  std::vector<float> emptyPrevBias;
  cpuReferenceGlobalScale_NCHW(cpuOutput, inputHost, scaleBiasHost, emptyPrevBias, activation);

  // Compare results
  compareResults(syclOutput, cpuOutput, 1e-5f);
}

// Performance test to verify SYCL kernel is working correctly
TEST_F(GlobalScaleFloatTest, GlobalScale_Performance) {
  ActivationFunction activation = ActivationFunction::ACTIVATION_RELU;

  // Larger test for performance
  const int largeN = 32, largeC = 256;
  const int largeInputSize = largeN * largeC * 64;

  // Allocate larger buffers
  float *largeInput = sycl::malloc_device<float>(largeInputSize, q);
  float *largeOutput = sycl::malloc_device<float>(largeInputSize, q);
  float *largeScaleBias = sycl::malloc_device<float>(largeN * 2 * largeC, q);
  float *largePrevBias = sycl::malloc_device<float>(largeC, q);

  // Initialize with some data
  std::vector<float> hostInput(largeInputSize, 1.0f);
  std::vector<float> hostOutput(largeInputSize, 0.5f);
  std::vector<float> hostScaleBias(largeN * 2 * largeC, 0.1f);
  std::vector<float> hostPrevBias(largeC, 0.01f);

  q.memcpy(largeInput, hostInput.data(), largeInputSize * sizeof(float));
  q.memcpy(largeOutput, hostOutput.data(), largeInputSize * sizeof(float));
  q.memcpy(largeScaleBias, hostScaleBias.data(), largeN * 2 * largeC * sizeof(float));
  q.memcpy(largePrevBias, hostPrevBias.data(), largeC * sizeof(float));

  // Measure SYCL execution time
  auto start = std::chrono::high_resolution_clock::now();

  globalScale(largeN, largeC, largeOutput, largeInput, largeScaleBias, largePrevBias,
              false, activation, q);

  auto stop = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

  std::cout << "SYCL globalScale execution time for " << largeN << "x" << largeC << " tensors: "
            << duration.count() << " microseconds" << std::endl;

  // Clean up
  sycl::free(largeInput, q);
  sycl::free(largeOutput, q);
  sycl::free(largeScaleBias, q);
  sycl::free(largePrevBias, q);

  // Just ensure it didn't crash - the actual timing is for information
  SUCCEED();
}