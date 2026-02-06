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
  combining it with Intel Corporation's oneAPI DPC++ libraries,
  containing parts covered by the terms of the respective license
  agreement, the licensors of this Program grant you additional
  permission to convey the resulting work.
*/

#pragma once

#include <sycl/sycl.hpp>
#include <iostream>
#include <stdexcept>
#include <cmath>

#include "neural/tables/activation_function.h"

#if defined(__INTEL_LLVM_COMPILER) || defined(__LIBSYCL_MAJOR_VERSION)
// Intel-specific headers for FP16 and extended features
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/ext/oneapi/bfloat16.hpp>
#endif

namespace lczero {
namespace sycl_backend {

// Maximum supported filter count for fast path
// TODO: extend it to cover bigger networks!
// (We are limited by no of registers per thread)
static constexpr int kMaxResBlockFusingChannels = 384;  // limit on num_filters
static constexpr int kMaxResBlockFusingSeK = 128;  // limit on (num_filters / se_ratio)

// Error handling for SYCL
inline void SycleError(sycl::exception const& e, const char* file, const int& line) {
  std::cerr << "SYCL error at " << file << ":" << line << std::endl;
  std::cerr << "  " << e.what() << std::endl;
  std::exit(EXIT_FAILURE);
}

#define ReportSYCLErrors(e) try { \
  e; \
} catch (sycl::exception const& err) { \
  SycleError(err, __FILE__, __LINE__); \
}

inline int DivUp(int a, int b) { return (a + b - 1) / b; }

// SYCL-specific constants for Intel optimization
static constexpr int kMaxWorkGroupSize = 256;   // Typical for Intel GPUs
static constexpr int kSubGroupSize = 16;       // Common for Intel GPUs
static constexpr int kOptimalWorkGroupSize = 128; // Balance occupancy and performance

// Helper function to get optimal work-group size based on device
inline int getOptimalWorkGroupSize(const sycl::queue& queue) {
  auto device = queue.get_device();
  auto max_wg_size = device.get_info<sycl::info::device::max_work_group_size>();
  return static_cast<int>(std::min(max_wg_size, static_cast<size_t>(kOptimalWorkGroupSize)));
}

// Helper function to get subgroup size
inline int getSubGroupSize(const sycl::queue& queue) {
  auto device = queue.get_device();
  // Modern SYCL uses sub_group() function directly
  try {
    return device.get_info<sycl::info::device::sub_group_sizes>().back();
  } catch (...) {
    // Fallback for devices that don't support sub-group size query
    return kSubGroupSize;  // Default for Intel GPUs
  }
}

// Create a SYCL queue with appropriate selector
inline sycl::queue createOptimizedQueue() {
  // Try to select Intel GPU first, fall back to other GPUs or CPU
  sycl::device target_device;

  try {
    // Try Intel GPU
    target_device = sycl::device(sycl::gpu_selector_v);
    // Check if it's an Intel GPU
    if (target_device.get_info<sycl::info::device::vendor>().find("Intel") == std::string::npos) {
      throw std::runtime_error("Not Intel GPU");
    }
  } catch (...) {
    try {
      // Fall back to any GPU
      target_device = sycl::device(sycl::gpu_selector_v);
    } catch (...) {
      // Fall back to CPU
      target_device = sycl::device(sycl::cpu_selector_v);
    }
  }

  sycl::property_list props{
    sycl::property::queue::enable_profiling{},
    sycl::property::queue::in_order{}
  };

  return sycl::queue(target_device, props);
}

// Convenience functions for SYCL queue management
inline sycl::queue& sycl_default_queue() {
  static sycl::queue queue = createOptimizedQueue();
  return queue;
}

// LC0 network constants
static constexpr int kInputPlanes = 112;
static constexpr int kNumOutputPolicy = 1858;

// Activation function helpers
inline float apply_activation(float x, int activation) {
  switch(activation) {
    case ACTIVATION_RELU:
      return x > 0.0f ? x : 0.0f;
    case ACTIVATION_TANH:
      return std::tanh(x);
    case ACTIVATION_SIGMOID:
      return 1.0f / (1.0f + std::exp(-x));
    case ACTIVATION_SWISH:
      return x / (1.0f + std::exp(-x)); // Re-approximation
    default:
      return x; // ACTIVATION_NONE or ACTIVATION_DEFAULT
  }
}

}  // namespace sycl_backend
}  // namespace lczero