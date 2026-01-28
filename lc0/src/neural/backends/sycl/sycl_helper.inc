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

#include "sycl_common.h"
#include "neural/tables/activation_function.h"

namespace lczero {
namespace sycl_backend {

// MISH activation function (same as CUDA version)
inline float mishActivate(float x) {
  float n = x + 1.0f;
  float d = 1.0f + exp(-x);
  float sp = sycl::log(d);
  float el = sp / d;
  if (x <= 0.0f)
    return x * sycl::tanh(sp);
  n = x + 1.0f;
  if (n >= d)
    return x * sycl::tanh(sp);
  else
    return x * el;
}

// Device activation function for SYCL kernels (same logic as CUDA)
inline float activate(float cVal, ActivationFunction activation) {
  switch (activation) {
    case ACTIVATION_RELU:
      if (cVal < 0) cVal = 0;
      break;
    case ACTIVATION_RELU_2:
      if (cVal < 0) cVal = 0;
      cVal *= cVal;
      break;
    case ACTIVATION_TANH:
      cVal = sycl::tanh(cVal);
      break;
    case ACTIVATION_SIGMOID:
      cVal = 1.0f / (1.0f + sycl::exp(-cVal));
      break;
    case ACTIVATION_SELU: {
      float alpha = 1.67326324f, scale = 1.05070098f;
      if (cVal > 0)
        cVal = scale * cVal;
      else
        cVal = scale * alpha * (sycl::exp(cVal) - 1.0f);
      break;
    }
    case ACTIVATION_MISH:
      cVal = mishActivate(cVal);
      break;
    case ACTIVATION_SWISH:
      cVal /= (1.0f + sycl::exp(-cVal));
      break;
    case ACTIVATION_NONE:
    default:
      // No activation
      break;
  }
  return cVal;
}

// SYCL subgroup reduction functions (replacing CUDA warp primitives)
template <typename Group>
inline float subgroup_reduce(Group sg, float x) {
  // Using SYCL's built-in reduce operations for subgroups
  return sycl::reduce_over_group(sg, x, sycl::plus<float>());
}

template <typename Group>
inline float subgroup_max(Group sg, float x) {
  // Using SYCL's built-in max operations for subgroups
  return sycl::reduce_over_group(sg, x, sycl::maximum<float>());
}

// Atomic operations for SYCL (where needed)
template <typename Acc>
inline void atomic_max_float(Acc acc, float val) {
  // SYCL doesn't have direct atomic max for float, need to use integer operations
  // This mirrors the CUDA implementation
  using int_type = int;
  sycl::atomic<int_type> atomic_int(*reinterpret_cast<int_type*>(acc.get_pointer()));

  if (!sycl::signbit(val)) {
    // For positive values
    int_type int_val = *reinterpret_cast<const int_type*>(&val);
    sycl::atomic<bool> atomic_float(*reinterpret_cast<bool*>(acc.get_pointer()));
    sycl::atomic_fetch_max(atomic_int, int_val);
  } else {
    // For negative values (need min operation)
    unsigned int int_val = *reinterpret_cast<const unsigned int*>(&val);
    sycl::atomic<unsigned int> atomic_uint(*reinterpret_cast<unsigned int*>(acc.get_pointer()));
    sycl::atomic_fetch_min(atomic_uint, int_val);
  }
}

// Memory copy utilities for different data types
template <typename T>
inline T load_value(const T* ptr, size_t index) {
  return ptr[index];
}

// Specialization for FP16 vectorized loads
inline sycl::half2 load_half2(const sycl::half* ptr, size_t index) {
  // SYCL half2 equivalent
  auto val = sycl::vec<sycl::half, 2>::load(0, ptr + index);
  return val;
}

inline void store_half2(sycl::half* ptr, size_t index, const sycl::half2& val) {
  // SYCL half2 store
  sycl::vec<sycl::half, 2> hvec = val;
  hvec.store(0, ptr + index);
}

// Helper functions for type conversion
template <typename T>
inline T to_device_type(float val) {
  return static_cast<T>(val);
}

template <>
inline sycl::half to_device_type<sycl::half>(float val) {
  return sycl::half(val);
}

#if defined(__INTEL_LLVM_COMPILER) || defined(__LIBSYCL_MAJOR_VERSION)
template <>
inline sycl::ext::oneapi::bfloat16 to_device_type<sycl::ext::oneapi::bfloat16>(float val) {
  return sycl::ext::oneapi::bfloat16(val);
}
#endif

// Clamping function for FP16 overflow protection (same as CUDA)
inline constexpr float kTwiceHalfMax = 65504.0f;  // max FP16 value

inline float clamp_for_fp16(float x) {
  return sycl::clamp(x, -kTwiceHalfMax, kTwiceHalfMax);
}

// Memory layout conversion helpers
template <typename DstType, typename SrcType>
inline DstType readNCHW(const SrcType* input_tensor, int n, int c, int h, int w,
                        int Nin, int Cin, int H, int W) {
  if (n >= Nin || c >= Cin) return 0;

  int index;
  index = n;
  index *= Cin;
  index += c;
  index *= H;
  index += h;
  index *= W;
  index += w;

  return static_cast<DstType>(input_tensor[index]);
}

// Subgroup shuffle operations (replacing CUDA __shfl_sync)
template <typename Group, typename T>
inline T subgroup_shuffle(Group sg, T val, int target_local_id) {
  return sycl::group_broadcast(sg, val, target_local_id);
}

template <typename Group, typename T>
inline T subgroup_shuffle_down(Group sg, T val, int delta) {
  // SYCL doesn't have direct shuffle_down, need to implement with permute
  int target = sg.get_local_id() + delta;
  if (target < sg.get_max_local_range()[0]) {
    return subgroup_shuffle(sg, val, target);
  } else {
    return val;
  }
}

template <typename Group, typename T>
inline T subgroup_shuffle_xor(Group sg, T val, int mask) {
  // SYCL doesn't have direct shuffle_xor, need to implement with permute_xors
  return sycl::select_from_group(sg, val, sg.get_local_id() ^ mask);
}

// Intel-specific optimizations
#if defined(__INTEL_LLVM_COMPILER) || defined(__LIBSYCL_MAJOR_VERSION)
// Intel Xe Matrix Extensions (XMX) hint for better performance on Intel GPUs
template <typename T>
[[intel::max_global_work_dim(1)]] // Typically 1D work-items for best vectorization
[[intel::num_simd_work_items(16)]] // Request SIMD width of 16 for Intel GPUs
[[intel::reqd_sub_group_size(16)]] // Force subgroup size of 16
inline void intel_optimized_mem_rd(const T* src, const size_t offset, T* dst) {
  dst[0] = src[offset];
}
#endif

}  // namespace sycl_backend
}  // namespace lczero