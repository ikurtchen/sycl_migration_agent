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
#include <sycl/sycl.hpp>
#include "neural/tables/activation_function.h"

namespace lczero {
namespace sycl_backend {

// Adds two vectors (possibly of different sizes), also do optional
// activation (relu, tanh or sigmoid).
template <typename T>
void addVectors(sycl::queue& queue, T* c, T* a, T* b, int size, int asize, int bsize,
                ActivationFunction activation);

// Adds two vectors of equal size overwriting the first with the sum.
// This specialisation performs a transposition of the first 2 indexes
// of the second while performing the addition.
template <typename T>
void addVectorsHNC_NHC(sycl::queue& queue, T* a, T* b, int N, int H, int C);

// Optimized kernel to add bias to innermost dimension
// and perform optional activation (to be used with GEMMs/fully connected)
template <typename T>
void addBiasBatched(sycl::queue& queue, T* output, const T* input, const T* bias,
                    int Batch, int N, int C, ActivationFunction activation);

// Optimized kernel to add bias to innermost dimension
// and perform optional activation (to be used with GEMMs/fully connected)
template <typename T>
void addBiasBatched(sycl::queue& queue, T* output, const T* input, const T* bias,
                    int Batch, int N, int C, int Nstride, ActivationFunction activation);

// Add bias to convolution's output.
template <typename T>
void addBias_NCHW(sycl::queue& queue, T* c, T* a, T* b, int N, int C, int H, int W,
                  ActivationFunction activation);

// Conversion from NCHW to NHWC, can also change datatype depending on template
// params, also pad/un-pad elements from Batch or Channel dimensions
template <typename DstType, typename SrcType>
void convertNCHWtoNHWC(sycl::queue& queue, DstType* output_tensor, const SrcType* input_tensor,
                       int Nin, int Cin, int Nout, int Cout, int H, int W);

// Plain data-type conversion (no layout conversion).
template <typename DstType, typename SrcType>
void copyTypeConverted(sycl::queue& queue, DstType* op, SrcType* ip, int N);

// Perform batch normalization.
template <typename T>
void batchNorm(sycl::queue& queue, T* output, const T* input, const T* skipInput, int N, int C,
               int H, int W, float* means, float* var_multipliers,
               ActivationFunction activation);

// Unpack planes (input to network).
template <typename T>
void expandPlanes_NHWC(sycl::queue& queue, T* output, const uint64_t* masks, const T* values, int n);

template <typename T>
void expandPlanes_NCHW(sycl::queue& queue, T* output, const uint64_t* masks, const T* values, int n);

// Perform global avg pool.
template <typename T>
void globalAvgPool(sycl::queue& queue, int N, int C, T* output, const T* input,
                   const T* prevLayerBias, bool nhwc);

// Perform global scale.
template <typename T>
void globalScale(sycl::queue& queue, int N, int C, T* output, const T* input, const T* scaleBias,
                 const T* prevLayerBias, bool nhwc,
                 ActivationFunction activation);

// Perform Squeeze-and-Excitation (SE) in a single fused kernel.
// Returns false if the fused kernel can't handle the sizes.
bool Se_Fp16_NHWC(sycl::queue& queue, int N, int C, int numFc1Out, sycl::half* output, const sycl::half* skip,
                  const sycl::half* input, const sycl::half* w1, const sycl::half* b1,
                  const sycl::half* w2, const sycl::half* b2, const sycl::half* bPrev,
                  ActivationFunction activation);

template <typename T>
void PolicyMap(sycl::queue& queue, int N, T* output, const T* input, const short* indices,
               int inputSize, int usedSize, int outputSize);

// Custom winograd helper functions
template <typename T>
void FilterTransform(sycl::queue& queue, int N, int C, T* transformedFilter, const T* filter);

template <typename T, bool nhcw>
void InputTransform(sycl::queue& queue, int N, int C, T* transformedInput, const T* input);

template <typename T, bool use_se, ActivationFunction activation, bool use_bias,
          bool use_skip, bool skipInput_nhcw, bool output_nhcw>
void OutputTransform(sycl::queue& queue, int N, int C, int se_K, T* output, const T* input,
                     const T* skip, const T* bias, const T* w1, const T* b1,
                     const T* w2, const T* b2);

template <typename T, bool use_se, ActivationFunction activation, bool use_bias,
          bool use_skip>
void OutputInputTransform(sycl::queue& queue, int N, int C, int se_K, T* output, const T* input,
                          const T* skip, const T* bias, const T* w1,
                          const T* b1, const T* w2, const T* b2);

template <typename T>
void Softmax(sycl::queue& queue, int N, int C, T* output, const T* input, const T* input2);

template <typename T>
void LayerNorm(sycl::queue& queue, int N, int C, T* output, const T* input, const T* bias,
               const T* skip, const T* gammas, const T* betas, float ep,
               float alpha, ActivationFunction act);

template <typename T>
void ComputePromotionLogits(sycl::queue& queue, int N, int C, T* output, const T* keys,
                            const T* ppo, const T* policy_attn_logits);

template <typename T>
void inputPreprocessForAttentionBody(sycl::queue& queue, T* output, const T* input,
                                     const T* encoding, int N, int input_size,
                                     int encoding_size,
                                     bool is_pe_dense_embedding);

template <typename T>
void applyInputGating(sycl::queue& queue, T* output, const T* input, const T* mult, const T* add,
                      int N, int HW, int C);

template <typename T>
void genOffsetPointers(sycl::queue& queue, T** offsets, int heads, int max_batch, int depth,
                       int d_model, T* k, T* q, T* b1, T* v, T* b2);

void fusedMHA(sycl::queue& queue, void* output, void* mha_q, void* mha_k, void* mha_v, void* skip,
              int batch_size, int num_heads, int depth);

}  // namespace sycl_backend
}  // namespace lczero