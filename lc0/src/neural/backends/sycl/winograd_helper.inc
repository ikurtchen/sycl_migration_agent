/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2020 The LCZero Authors

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

#include "sycl_common.h"
#include "sycl_helper.inc"
#include "neural/tables/activation_function.h"

namespace lczero {
namespace sycl_backend {

// MISH activation function (SYCL-compatible)
inline float mishActivate(float el) {
  auto e = sycl::exp(el);
  auto n = e * e + 2.0f * e;
  auto d = el / (n + 2.0f);
  if (el <= -0.6f) {
    return n * d;
  } else {
    return el - 2.0f * d;
  }
}

// Device activation function for SYCL kernels (same as CUDA, SYCL-compatible)
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
      break;
    case ACTIVATION_DEFAULT:
    case ACTIVATION_SOFTMAX:
      // Trigger an error if we ever get here.
      assert(false && "Invalid activation function");
      break;
  }
  return cVal;
}

template <typename T, int M, int N, int K>
inline void matrixMul_sycl_serial(T* c, const T* a, const T* b) {
#ifndef SKIP_FP16_BITS
#pragma unroll
  for (int i = 0; i < M; ++i)
#pragma unroll
    for (int j = 0; j < N; ++j) {
      T S = 0;
#pragma unroll
      for (int k = 0; k < K; ++k) S += a[i * K + k] * b[k * N + j];
      c[i * N + j] = S;
    }
#endif
}

template <typename T>
inline void FilterTransform4x4(T* transformed_filter, const T* filter) {
  // transform applied to filter (of size 3x3)
  T G[6 * 3] = {1.0f / 4,  0,         0,         -1.0f / 6,  -1.0f / 6,
                -1.0f / 6, -1.0f / 6, 1.0f / 6,  -1.0f / 6,  1.0f / 24,
                1.0f / 12, 1.0f / 6,  1.0f / 24, -1.0f / 12, 1.0f / 6,
                0,         0,         1};

  T Gt[3 * 6] = {1.0f / 4, -1.0f / 6, -1.0f / 6, 1.0f / 24, 1.0f / 24,  0,
                 0,        -1.0f / 6, 1.0f / 6,  1.0f / 12, -1.0f / 12, 0,
                 0,        -1.0f / 6, -1.0f / 6, 1.0f / 6,  1.0f / 6,   1};

  T temp_filter[6 * 3];
  matrixMul_sycl_serial<T, 6, 3, 3>(temp_filter, G, filter);
  matrixMul_sycl_serial<T, 6, 6, 3>(transformed_filter, temp_filter, Gt);
}

template <typename T>
inline void InputTransform4x4(T* transformedInput, const T* input) {
  // transform applied to input tile (of size 4x4)
  const T Bt[6 * 6] = {4, 0, -5, 0,  1, 0, 0, -4, -4, 1,  1, 0,
                       0, 4, -4, -1, 1, 0, 0, -2, -1, 2,  1, 0,
                       0, 2, -1, -2, 1, 0, 0, 4,  0,  -5, 0, 1};

  const T B[6 * 6] = {4,  0,  0,  0,  0,  0, 0, -4, 4,  -2, 2,  4,
                      -5, -4, -4, -1, -1, 0, 0, 1,  -1, 2,  -2, -5,
                      1,  1,  1,  1,  1,  0, 0, 0,  0,  0,  0,  1};

  T tempIp1[6 * 6];
  matrixMul_sycl_serial<T, 6, 6, 6>(tempIp1, Bt, input);
  matrixMul_sycl_serial<T, 6, 6, 6>(transformedInput, tempIp1, B);
}

template <typename T>
inline void OutputTransform4x4(T* output, const T* transformedOutput) {
  // transform applied to result
  const T At[4 * 6] = {1, 1, 1, 1, 1, 0, 0, 1, -1, 2, -2, 0,
                       0, 1, 1, 4, 4, 0, 0, 1, -1, 8, -8, 1};

  const T A[6 * 4] = {1, 0, 0, 0, 1, 1,  1, 1,  1, -1, 1, -1,
                      1, 2, 4, 8, 1, -2, 4, -8, 0, 0,  0, 1};

  T tempOp[4 * 6];
  matrixMul_sycl_serial<T, 4, 6, 6>(tempOp, At, transformedOutput);
  matrixMul_sycl_serial<T, 4, 4, 6>(output, tempOp, A);
}

// Index macros for SYCL kernels
#define FILTER_IDX_NCHW(k, c, h, w) ((k)*C * S * R + (c)*S * R + (h)*R + w)
#define INDEX_NCHW(n, c, h, w) ((n)*C * 8 * 8 + (c)*8 * 8 + (h)*8 + w)
#define INDEX_NHCW(n, c, h, w) ((n)*C * 8 * 8 + (h)*C * 8 + (c)*8 + w)
#define GemmN (N * 4)
#define TEMP_INDEX_HWNC(h, w, n, c) ((h)*6 * GemmN * C + (w)*GemmN * C + (n)*C + c)

// Helper function to do vector loads/stores with SYCL (replacing copyAs)
template <typename T>
inline void copyAs(void* dst, const void* src) {
  *((T*)(dst)) = *((const T*)(src));
}

}  // namespace sycl_backend
}  // namespace lczero