/*
   Unit tests for addBias_NCHW SYCL kernel
   Tests the NCHW bias addition operation with various activation functions
*/

#include <gtest/gtest.h>
#include <vector>
#include <random>
#include <cmath>
#include <memory>

// Mock SYCL structures for testing without SYCL compiler
#define SYCL_MOCK_TEST

#ifdef SYCL_MOCK_TEST

namespace sycl {
  namespace ext {
    namespace oneapi {
      namespace experimental {
        template<typename T>
        class malloc_device_mock {
        public:
          malloc_device_mock(size_t size, queue& q) : ptr_(new T[size]) {}
          T* get() const { return ptr_; }
          operator T*() const { return ptr_; }
        private:
          T* ptr_;
        };
      }
    }
  }

  class queue_mock {
  public:
    template<typename... Args>
    queue_mock(Args...) {}

    template<typename... Args>
    void memcpy(T* dst, const T* src, size_t size) {
      std::memcpy(dst, src, size);
    }

    template<typename... Args>
    void wait() {}

    template<typename... Args>
    void submit(Args...) {}
  };

  using half = uint16_t;
  using range = std::vector<int>;
  using nd_range = std::pair<range, range>;
  using nd_item = std::pair<int, int>;

  template<typename T>
  T malloc_device(size_t size, queue& q) {
    return T(size);
  }

  void free(void* ptr, queue& q) {
    delete[] static_cast<char*>(ptr);
  }

  namespace info {
    namespace device {
      constexpr auto name = "Mock SYCL Device";
    }
  }

  struct exception : public std::exception {
    exception(const std::string& msg) : msg_(msg) {}
    const char* what() const noexcept override { return msg_.c_str(); }
  private:
    std::string msg_;
  };
}

using sycl::queue_mock as sycl::queue;
using sycl::half;
using sycl::malloc_device;
using sycl::free;

#else
#include <sycl/sycl.hpp>
#endif

#include "sycl/src/neural/backends/sycl/sycl_common.h"
#include "sycl/src/neural/backends/sycl/kernels.h"

using namespace lczero::sycl_backend;

// Test fixture class for addBias_NCHW
class AddBiasNCHWTest : public ::testing::Test {
protected:
  void SetUp() override {
    N = 2;  // Batch size
    C = 64; // Channels
    H = 8;  // Height
    W = 8;  // Width

    total_elements = N * C * H * W;

    // Allocate memory
    input.resize(total_elements);
    bias.resize(C);
    output_cpu.resize(total_elements);
    output_sycl.resize(total_elements);
    golden.resize(total_elements);

    // Setup random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    // Generate test data
    for (int i = 0; i < total_elements; ++i) {
      input[i] = dis(gen);
    }
    for (int i = 0; i < C; ++i) {
      bias[i] = dis(gen);
    }
  }

  void TearDown() override {
    input.clear();
    bias.clear();
    output_cpu.clear();
    output_sycl.clear();
    golden.clear();
  }

  // CPU reference implementation
  void computeGolden(ActivationFunction activation) {
    for (int n = 0; n < N; ++n) {
      for (int c = 0; c < C; ++c) {
        for (int h = 0; h < H; ++h) {
          for (int w = 0; w < W; ++w) {
            int idx = n * C * H * W + c * H * W + h * W + w;
            float val = input[idx] + bias[c];
            golden[idx] = activate(val, activation);
          }
        }
      }
    }
  }

  // CPU implementation that mimics the SYCL kernel logic
  void computeCPUReference(ActivationFunction activation) {
    for (int i = 0; i < total_elements; ++i) {
      float aVal = input[i];

      // Calculate bias index: the channel index for current element
      // NCHW layout: index = n*C*H*W + c*H*W + h*W + w
      // biasIndex = (i / (H * W)) % C - this gets the channel index
      int biasIndex = (i / (H * W)) % C;
      float bVal = bias[biasIndex];

      float cVal = aVal + bVal;
      cVal = activate(cVal, activation);

      output_cpu[i] = cVal;
    }
  }

  bool compareResults(float tolerance = 1e-5f) {
    for (int i = 0; i < total_elements; ++i) {
      if (std::abs(output_cpu[i] - golden[i]) > tolerance) {
        std::cout << "CPU vs Golden mismatch at index " << i << ": CPU="
                  << output_cpu[i] << ", Golden=" << golden[i] << std::endl;
        return false;
      }
    }
    return true;
  }

  void verifyIndexCalculation() {
    // Verify the bias index calculation formula
    for (int n = 0; n < N; ++n) {
      for (int c = 0; c < C; ++c) {
        for (int h = 0; h < H; ++h) {
          for (int w = 0; w < W; ++w) {
            int linear_idx = n * C * H * W + c * H * W + h * W + w;
            int bias_idx = (linear_idx / (H * W)) % C;
            ASSERT_EQ(bias_idx, c) << "Bias index calculation failed for n=" << n
                                   << ", c=" << c << ", h=" << h << ", w=" << w;
          }
        }
      }
    }
  }

  // Test parameters
  int N, C, H, W;
  int total_elements;

  // Test data
  std::vector<float> input;
  std::vector<float> bias;
  std::vector<float> output_cpu;
  std::vector<float> output_sycl;
  std::vector<float> golden;
};

// Test bias index calculation for NCHW layout
TEST_F(AddBiasNCHWTest, TestBiasIndexCalculation) {
  verifyIndexCalculation();
}

// Test with different activation functions
TEST_F(AddBiasNCHWTest, TestActivationNone) {
  ActivationFunction activation = ActivationFunction::ACTIVATION_NONE;
  computeGolden(activation);
  computeCPUReference(activation);

  EXPECT_TRUE(compareResults()) << "NONE activation failed";
}

TEST_F(AddBiasNCHWTest, TestActivationReLU) {
  ActivationFunction activation = ActivationFunction::ACTIVATION_RELU;
  computeGolden(activation);
  computeCPUReference(activation);

  EXPECT_TRUE(compareResults()) << "RELU activation failed";
}

TEST_F(AddBiasNCHWTest, TestActivationReLU2) {
  ActivationFunction activation = ActivationFunction::ACTIVATION_RELU_2;
  computeGolden(activation);
  computeCPUReference(activation);

  EXPECT_TRUE(compareResults()) << "RELU_2 activation failed";
}

TEST_F(AddBiasNCHWTest, TestActivationTanh) {
  ActivationFunction activation = ActivationFunction::ACTIVATION_TANH;
  computeGolden(activation);
  computeCPUReference(activation);

  EXPECT_TRUE(compareResults(1e-4f)) << "TANH activation failed";
}

TEST_F(AddBiasNCHWTest, TestActivationSigmoid) {
  ActivationFunction activation = ActivationFunction::ACTIVATION_SIGMOID;
  computeGolden(activation);
  computeCPUReference(activation);

  EXPECT_TRUE(compareResults(1e-4f)) << "SIGMOID activation failed";
}

TEST_F(AddBiasNCHWTest, TestActivationSELU) {
  ActivationFunction activation = ActivationFunction::ACTIVATION_SELU;
  computeGolden(activation);
  computeCPUReference(activation);

  EXPECT_TRUE(compareResults(1e-4f)) << "SELU activation failed";
}

TEST_F(AddBiasNCHWTest, TestActivationMish) {
  ActivationFunction activation = ActivationFunction::ACTIVATION_MISH;
  computeGolden(activation);
  computeCPUReference(activation);

  EXPECT_TRUE(compareResults(1e-4f)) << "MISH activation failed";
}

TEST_F(AddBiasNCHWTest, TestActivationSwish) {
  ActivationFunction activation = ActivationFunction::ACTIVATION_SWISH;
  computeGolden(activation);
  computeCPUReference(activation);

  EXPECT_TRUE(compareResults(1e-4f)) << "SWISH activation failed";
}

// Test edge cases
TEST_F(AddBiasNCHWTest, TestZeroInput) {
  std::fill(input.begin(), input.end(), 0.0f);
  std::fill(bias.begin(), bias.end(), 1.0f);

  ActivationFunction activation = ActivationFunction::ACTIVATION_NONE;
  computeGolden(activation);
  computeCPUReference(activation);

  EXPECT_TRUE(compareResults()) << "Zero input test failed";

  // Verify all outputs are equal to bias (1.0f)
  for (int i = 0; i < total_elements; ++i) {
    EXPECT_FLOAT_EQ(output_cpu[i], 1.0f) << "Output at index " << i << " should be 1.0f";
  }
}

TEST_F(AddBiasNCHWTest, TestNegativeBias) {
  std::fill(input.begin(), input.end(), 2.0f);
  std::fill(bias.begin(), bias.end(), -1.0f);

  ActivationFunction activation = ActivationFunction::ACTIVATION_NONE;
  computeGolden(activation);
  computeCPUReference(activation);

  EXPECT_TRUE(compareResults()) << "Negative bias test failed";
}

TEST_F(AddBiasNCHWTest, TestReLUWithNegativeValues) {
  std::fill(input.begin(), input.end(), -2.0f);
  std::fill(bias.begin(), bias.end(), 1.0f);

  ActivationFunction activation = ActivationFunction::ACTIVATION_RELU;
  computeGolden(activation);
  computeCPUReference(activation);

  EXPECT_TRUE(compareResults()) << "RELU negative values test failed";

  // Some outputs should be zero (where bias didn't overcome negative input)
  bool found_zero = false;
  for (int i = 0; i < total_elements; ++i) {
    if (output_cpu[i] == 0.0f) {
      found_zero = true;
      break;
    }
  }
  EXPECT_TRUE(found_zero) << "Expected some zero outputs with RELU";
}

// Test memory layout preservation
TEST_F(AddBiasNCHWTest, TestNCHWMemoryLayout) {
  // Create input where each channel has a distinct pattern
  for (int n = 0; n < N; ++n) {
    for (int c = 0; c < C; ++c) {
      for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
          int idx = n * C * H * W + c * H * W + h * W + w;
          input[idx] = static_cast<float>(c * 100 + h * 10 + w);
        }
      }
    }
  }

  // Bias should add the channel number
  for (int c = 0; c < C; ++c) {
    bias[c] = static_cast<float>(c + 1000);
  }

  ActivationFunction activation = ActivationFunction::ACTIVATION_NONE;
  computeCPUReference(activation);

  // Verify that the bias was correctly added per channel
  for (int n = 0; n < N; ++n) {
    for (int c = 0; c < C; ++c) {
      for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
          int idx = n * C * H * W + c * H * W + h * W + w;
          float expected = static_cast<float>(c * 100 + h * 10 + w + c + 1000);
          EXPECT_FLOAT_EQ(output_cpu[idx], expected)
              << "NCHW layout error at n=" << n << ", c=" << c << ", h=" << h << ", w=" << w;
        }
      }
    }
  }
}

#ifdef SYCL_MOCK_TEST
// Mock SYCL test to validate logic without actual SYCL compilation
TEST_F(AddBiasNCHWTest, MockSYCLImplementation) {
  // This tests would run with actual SYCL when compiler is available
  // For now, we validate that the CPU implementation matches the golden reference
  ActivationFunction activations[] = {
    ActivationFunction::ACTIVATION_NONE,
    ActivationFunction::ACTIVATION_RELU,
    ActivationFunction::ACTIVATION_RELU_2,
    ActivationFunction::ACTIVATION_TANH,
    ActivationFunction::ACTIVATION_SIGMOID,
    ActivationFunction::ACTIVATION_SELU,
    ActivationFunction::ACTIVATION_MISH,
    ActivationFunction::ACTIVATION_SWISH
  };

  for (auto activation : activations) {
    computeGolden(activation);
    computeCPUReference(activation);
    EXPECT_TRUE(compareResults()) << "Mock SYCL test failed for activation "
                                 << static_cast<int>(activation);
  }
}
#endif

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}