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

#include <CL/sycl.hpp>
#include "sycl_common.h"
#include "neural/tables/activation_function.h"

namespace lczero {
namespace sycl_backend {

// Adds two vectors (possibly of different sizes), also do optional
// activation (relu, tanh or sigmoid).
template <typename T>
void addVectors(T* c, T* a, T* b, int size, int asize, int bsize,
                ActivationFunction activation, sycl::queue& queue);

// Template instantiations
template void addVectors<float>(float* c, float* a, float* b, int size,
                                int asize, int bsize, ActivationFunction act,
                                sycl::queue& queue);
template void addVectors<half>(half* c, half* a, half* b, int size, int asize,
                                int bsize, ActivationFunction act,
                                sycl::queue& queue);

// Add vector from NHC (Number-Height-Channels) to HNC (Height-Number-Channels) layout
// and store result in HNC layout
template <typename T>
void addVectorsHNC_NHC(T* a, T* b, int N, int H, int C, sycl::queue& queue);

// Template instantiations for addVectorsHNC_NHC
template void addVectorsHNC_NHC<float>(float* a, float* b, int N, int H, int C,
                                       sycl::queue& queue);
template void addVectorsHNC_NHC<sycl::half>(half* a, half* b, int N, int H, int C,
                                          sycl::queue& queue);

// Add bias to batched tensor input/output
// Input/output tensors are Batch * N * C
// bias tensor is Batch * C (different bias for each Batch dimension)
template <typename T>
void addBiasBatched(T* output, const T* input, const T* bias, int Batch, int N,
                    int C, ActivationFunction activation, sycl::queue& queue);

// Version with Nstride support (for padded N dimension)
template <typename T>
void addBiasBatched(T* output, const T* input, const T* bias, int Batch, int N,
                    int C, int Nstride, ActivationFunction activation,
                    sycl::queue& queue);

// Template instantiations for addBiasBatched
template void addBiasBatched<float>(float* output, const float* input,
                                    const float* bias, int Batch, int N, int C,
                                    ActivationFunction activation,
                                    sycl::queue& queue);
template void addBiasBatched<half>(sycl::half* output, const sycl::half* input,
                                   const sycl::half* bias, int Batch, int N, int C,
                                   ActivationFunction activation,
                                   sycl::queue& queue);

// Template instantiations for addBiasBatched with Nstride
template void addBiasBatched<float>(float* output, const float* input,
                                    const float* bias, int Batch, int N, int C,
                                    int Nstride, ActivationFunction activation,
                                    sycl::queue& queue);
template void addBiasBatched<half>(sycl::half* output, const sycl::half* input,
                                   const sycl::half* bias, int Batch, int N, int C,
                                   int Nstride, ActivationFunction activation,
                                   sycl::queue& queue);

// Perform batch normalization.
template <typename T>
void batchNorm(sycl::queue& q, T* output, const T* input, const T* skipInput,
               int N, int C, int H, int W, const float* means,
               const float* varMultipliers, ActivationFunction activation);

// Template instantiations for batchNorm
template void batchNorm<float>(sycl::queue& q, float* output, const float* input,
                               const float* skipInput, int N, int C, int H, int W,
                               const float* means, const float* varMultipliers,
                               ActivationFunction activation);

template void batchNorm<half>(sycl::queue& q, half* output, const half* input,
                              const half* skipInput, int N, int C, int H, int W,
                              const float* means, const float* varMultipliers,
                              ActivationFunction activation);

// Preprocess for attention body
template <typename T>
void inputPreprocessForAttentionBody(sycl::queue& q, T* output, const T* input,
                                     const T* encoding, int N, int input_size,
                                     int encoding_size, bool is_pe_dense_embedding);

// Template instantiations for inputPreprocessForAttentionBody
template void inputPreprocessForAttentionBody<float>(sycl::queue& q, float* output, const float* input,
                                      const float* encoding, int N, int input_size,
                                      int encoding_size, bool is_pe_dense_embedding);

template void inputPreprocessForAttentionBody<sycl::half>(sycl::queue& q, half* output, const half* input,
                                         const half* encoding, int N, int input_size,
                                         int encoding_size, bool is_pe_dense_embedding);

// Input gating operation: output = input * mult + add
template <typename T>
void applyInputGating(sycl::queue& q, T* output, const T* input, const T* mult, const T* add,
                      int N, int HW, int C);

// Template instantiations for applyInputGating
template void applyInputGating<float>(sycl::queue& q, float* output, const float* input,
                                      const float* mult, const float* add,
                                      int N, int HW, int C);

template void applyInputGating<sycl::half>(sycl::queue& q, half* output, const half* input,
                                           const half* mult, const half* add,
                                           int N, int HW, int C);

// Add bias to convolution output (NCHW layout)
// Adds per-channel bias to each element in NCHW tensor and applies activation
template <typename T>
void addBias_NCHW(sycl::queue& q, T* c, const T* a, const T* b, int N, int C, int H, int W,
                 ActivationFunction activation);

// Template instantiations for addBias_NCHW
template void addBias_NCHW<float>(sycl::queue& q, float* c, const float* a, const float* b, int N, int C,
                                 int H, int W, ActivationFunction activation);
template void addBias_NCHW<sycl::half>(sycl::queue& q, sycl::half* c, const sycl::half* a, const sycl::half* b,
                                       int N, int C, int H, int W, ActivationFunction activation);

// Generate offset pointers for attention mechanism
template <typename T>
void genOffsetPointers(sycl::queue& q, T** offsets, int heads, int max_batch, int depth,
                      int d_model, T* k, T* q, T* b1, T* v, T* b2);

// Template instantiations for genOffsetPointers
template void genOffsetPointers<float>(sycl::queue& q, float** offsets, int heads, int max_batch,
                                     int depth, int d_model, float* k, float* q, float* b1,
                                     float* v, float* b2);
template void genOffsetPointers<sycl::half>(sycl::queue& q, sycl::half** offsets, int heads, int max_batch,
                                           int depth, int d_model, sycl::half* k, sycl::half* q,
                                           sycl::half* b1, sycl::half* v, sycl::half* b2);

// Fused Multi-Head Attention (MHA) kernel
void fusedMHA(void* output, void* mha_q, void* mha_k, void* mha_v, void* skip,
              int batch_size, int num_heads, int depth, sycl::queue& queue);

} // namespace sycl_backend
} // namespace lczero