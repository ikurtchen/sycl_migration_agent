/*
  Minimal activation function definitions for testing
  This is a simplified version for the migration test
*/

#pragma once

#include <cmath>

namespace lczero {

// Simple activation function enum for testing
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

// Device-side activation function implementation
inline float activate(float val, ActivationFunction activation) {
  switch (activation) {
    case ActivationFunction::ACTIVATION_RELU:
      return (val < 0.0f) ? 0.0f : val;
    case ActivationFunction::ACTIVATION_RELU_2:
      return (val < 0.0f) ? 0.0f : (val * val);
    case ActivationFunction::ACTIVATION_TANH:
      return std::tanh(val);
    case ActivationFunction::ACTIVATION_SIGMOID:
      return 1.0f / (1.0f + std::exp(-val));
    case ActivationFunction::ACTIVATION_SELU: {
      constexpr float alpha = 1.67326324f;
      constexpr float scale = 1.05070098f;
      return (val > 0.0f) ? (scale * val) : (scale * alpha * (std::exp(val) - 1.0f));
    }
    case ActivationFunction::ACTIVATION_MISH: {
      auto e = std::exp(val);
      auto n = e * e + 2.0f * e;
      auto d = val / (n + 2.0f);
      return (val <= -0.6f) ? (n * d) : (val - 2.0f * d);
    }
    case ActivationFunction::ACTIVATION_SWISH:
      return val / (1.0f + std::exp(-val));
    case ActivationFunction::ACTIVATION_NONE:
    default:
      return val;
  }
}

inline float mishActivate(float el) {
  auto e = std::exp(el);
  auto n = e * e + 2.0f * e;
  auto d = el / (n + 2.0f);
  if (el <= -0.6f) {
    return n * d;
  } else {
    return el - 2.0f * d;
  }
}

} // namespace lczero