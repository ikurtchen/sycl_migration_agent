/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018-2019 The LCZero Authors

  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.

  Additional permission under GNU GPL version 3 section 7

  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/

#pragma once

#include <sycl/sycl.hpp>
#include <cmath>
#include <iostream>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace lczero {
namespace sycl_backend {

// Use sycl::half directly if available
using half = sycl::half;

// Activation function enum matching CUDA version
enum class ActivationFunction {
  ACTIVATION_NONE,
  ACTIVATION_RELU,
  ACTIVATION_RELU_2,
  ACTIVATION_TANH,
  ACTIVATION_SIGMOID,
  ACTIVATION_SELU,
  ACTIVATION_MISH,
  ACTIVATION_SWISH,
  ACTIVATION_DEFAULT,
  ACTIVATION_SOFTMAX
};

// Error handling
inline void ReportSYCLErrors(sycl::exception const& e, const char* file, const int& line) {
  std::cerr << "SYCL error at " << file << ":" << line << " - " << e.what() << std::endl;
}

#define ReportSYCLErrors(e) ReportSYCLErrors(e, __FILE__, __LINE__)

// Utility functions
inline int DivUp(int a, int b) { return (a + b - 1) / b; }

// SYCL activation function
inline float activate(float cVal, ActivationFunction activation) {
  switch (activation) {
    case ActivationFunction::ACTIVATION_RELU:
      if (cVal < 0) cVal = 0;
      break;
    case ActivationFunction::ACTIVATION_RELU_2:
      if (cVal < 0) cVal = 0;
      cVal *= cVal;
      break;
    case ActivationFunction::ACTIVATION_TANH:
      cVal = std::tanh(cVal);
      break;
    case ActivationFunction::ACTIVATION_SIGMOID:
      cVal = 1.0f / (1.0f + std::exp(-cVal));
      break;
    case ActivationFunction::ACTIVATION_SELU: {
      float alpha = 1.67326324f, scale = 1.05070098f;
      if (cVal > 0)
        cVal = scale * cVal;
      else
        cVal = scale * alpha * (std::exp(cVal) - 1.0f);
      break;
    }
    case ActivationFunction::ACTIVATION_MISH: {
      auto e = std::exp(cVal);
      auto n = e * e + 2.0f * e;
      auto d = cVal / (n + 2.0f);
      if (cVal <= -0.6f) {
        cVal = n * d;
      } else {
        cVal = cVal - 2.0f * d;
      }
      break;
    }
    case ActivationFunction::ACTIVATION_SWISH:
      cVal /= (1.0f + std::exp(-cVal));
      break;
    case ActivationFunction::ACTIVATION_NONE:
      break;
    default:
      // Trigger an error if we ever get here.
      std::terminate();
  }
  return cVal;
}

// Helper function to do vector loads/stores
template <typename T>
inline void copyAs(void* dst, const void* src) {
  *((T*)(dst)) = *((const T*)(src));
}

} // namespace sycl_backend
} // namespace lczero