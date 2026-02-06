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
#include "layers.h"

#include <cassert>
#include <cstring>
#include <vector>

#include "sycl_common.h"
#include "kernels.h"
#include "neural/network.h"
#include "neural/tables/attention_policy_map.h"
#include "utils/fp16_utils.h"

namespace lczero {

namespace sycl_backend {

// Basic template implementations - will be expanded as needed

template <typename DataType>
BaseLayer<DataType>::BaseLayer(int c, int h, int w, BaseLayer* ip)
    : input_(ip), C(c), H(h), W(w), nhwc_(true), use_gemm_ex_(false) {}

template <typename DataType>
BaseLayer<DataType>::BaseLayer(int c, int h, int w, BaseLayer* ip, bool nhwc)
    : input_(ip), C(c), H(h), W(w), nhwc_(nhwc), use_gemm_ex_(false) {}

template <typename DataType>
BaseLayer<DataType>::BaseLayer(int c, int h, int w, BaseLayer* ip, bool nhwc, bool use_gemm_ex)
    : input_(ip), C(c), H(h), W(w), nhwc_(nhwc), use_gemm_ex_(use_gemm_ex) {}


template <typename DataType>
void BaseLayer<DataType>::cublasRowMajorMatrixMul(const DataType* A, const DataType* B,
                                                  DataType* Out, int M, int N, int K,
                                                  int batchSize, void* blas_handle) {
  // Placeholder for SYCL matrix multiplication implementation
  // This will be implemented using SYCL BLAS or custom kernels
}

// FCLayer implementations
template <typename DataType>
FCLayer<DataType>::FCLayer(BaseLayer<DataType>* ip, int C, int H, int W, bool bias,
                           ActivationFunction activation)
    : BaseLayer<DataType>(C, H, W, ip, ip->isNHWC(), false),
      use_bias_(bias),
      act_(activation),
      weights_(nullptr),
      biases_(nullptr) {}

template <typename DataType>
FCLayer<DataType>::~FCLayer() {
  auto context = sycl_default_queue().get_context();
  if (weights_) sycl::free(weights_, context);
  if (biases_) sycl::free(biases_, context);
}

template <typename DataType>
void FCLayer<DataType>::LoadWeights(float* cpuWeight, float* cpuBias, void* scratch) {
  // Implementation for loading weights to SYCL device memory
  auto queue = sycl_default_queue();
  auto context = queue.get_context();
  auto device = queue.get_device();

  // Calculate sizes
  size_t input_dims = input_->GetC() * input_->GetH() * input_->GetW();
  size_t output_dims = C * H * W;
  size_t weight_size = sizeof(DataType) * input_dims * output_dims;

  // Allocate device memory
  weights_ = static_cast<DataType*>(sycl::malloc_device(weight_size, device, context));

  // Copy and convert weights
  std::vector<DataType> converted_weights(weight_size / sizeof(DataType));
  for (size_t i = 0; i < weight_size / sizeof(float); ++i) {
    converted_weights[i] = static_cast<DataType>(cpuWeight[i]);
  }
  queue.memcpy(weights_, converted_weights.data(), weight_size).wait();

  if (use_bias_ && cpuBias) {
    size_t bias_size = sizeof(DataType) * output_dims;
    biases_ = static_cast<DataType*>(sycl::malloc_device(bias_size, device, context));

    std::vector<DataType> converted_biases(bias_size / sizeof(DataType));
    for (size_t i = 0; i < bias_size / sizeof(float); ++i) {
      converted_biases[i] = static_cast<DataType>(cpuBias[i]);
    }
    queue.memcpy(biases_, converted_biases.data(), bias_size).wait();
  }
}

template <typename DataType>
void FCLayer<DataType>::Eval(int N, DataType* output, const DataType* input,
                             const DataType* input2, void* scratch, size_t scratch_size,
                             sycl::queue queue, void* blas_handle,
                             void* stream_handle, DataType*** offset_pointers) {
  // Basic SYCL implementation
  size_t input_dims = input_->GetC() * input_->GetH() * input_->GetW();
  size_t output_dims = C * H * W;

  for (int batch = 0; batch < N; ++batch) {
    const DataType* batch_input = input + batch * input_dims;
    DataType* batch_output = output + batch * output_dims;

    // Simple matmul kernel
    queue.submit([&](sycl::handler& cgh) {
      cgh.parallel_for(sycl::range<1>(output_dims), [=, this](sycl::id<1> idx) {
        size_t out_idx = idx[0];
        float acc = 0.0f;
        for (size_t i = 0; i < input_dims; ++i) {
          acc += static_cast<float>(batch_input[i]) * static_cast<float>(weights_[out_idx * input_dims + i]);
        }
        if (use_bias_) {
          acc += static_cast<float>(biases_[out_idx]);
        }
        // Apply activation function
        if (act_ == ACTIVATION_RELU) {
          acc = acc > 0 ? acc : 0;
        }
        batch_output[out_idx] = static_cast<DataType>(acc);
      });
    }).wait();
  }
}

// PolicyMapLayer implementations
template <typename DataType>
PolicyMapLayer<DataType>::PolicyMapLayer(BaseLayer<DataType>* ip, int C, int H, int W,
                                         int usedSize, bool attention)
    : BaseLayer<DataType>(C, H, W, ip, ip->isNHWC()),
      used_size_(usedSize),
      attention_map_(attention),
      weights_(nullptr) {}

template <typename DataType>
PolicyMapLayer<DataType>::~PolicyMapLayer() {
  auto context = sycl_default_queue().get_context();
  if (weights_) sycl::free(weights_, context);
}

template <typename DataType>
void PolicyMapLayer<DataType>::LoadWeights(const short* cpuWeight, void* scratch) {
  (void)scratch; // Suppress unused parameter warning
  auto queue = sycl_default_queue();
  auto context = queue.get_context();
  auto device = queue.get_device();
  size_t weights_size = sizeof(short) * used_size_ * C * H * W;

  weights_ = static_cast<short*>(sycl::malloc_device(weights_size, device, context));
  queue.memcpy(weights_, cpuWeight, weights_size).wait();
}

template <typename DataType>
void PolicyMapLayer<DataType>::Eval(int N, DataType* output, const DataType* input,
                                   const DataType* input2, void* scratch, size_t scratch_size,
                                   sycl::queue queue, void* blas_handle, void* stream_handle,
                                   DataType*** offset_pointers) {
  size_t input_dims = input_->GetC() * input_->GetH() * input_->GetW();
  size_t output_dims = C * H * W;

  // Copy member variables to local variables for kernel access
  const int local_used_size = used_size_;
  const short* local_weights = weights_;

  for (int batch = 0; batch < N; ++batch) {
    const DataType* batch_input = input + batch * input_dims;
    DataType* batch_output = output + batch * output_dims;

    queue.submit([&](sycl::handler& cgh) {
      cgh.parallel_for(sycl::range<1>(output_dims), [=](sycl::id<1> idx) {
        size_t out_idx = idx[0];
        float acc = 0.0f;
        for (size_t i = 0; i < local_used_size; ++i) {
          acc += static_cast<float>(batch_input[i]) * static_cast<float>(local_weights[out_idx * local_used_size + i]);
        }
        batch_output[out_idx] = static_cast<DataType>(acc);
      });
    }).wait();
  }
}

// SELayer implementations
template <typename DataType>
SELayer<DataType>::SELayer(BaseLayer<DataType>* ip, int numFc1Out, bool addPrevLayerBias,
                           ActivationFunction activation)
    : BaseLayer<DataType>(ip->GetC(), ip->GetH(), ip->GetW(), ip, ip->isNHWC()),
      w1_(nullptr), w1_t_(nullptr), b1_(nullptr),
      w2_(nullptr), w2_t_(nullptr), b2_(nullptr), bPrev_(nullptr),
      numFc1Out_(numFc1Out),
      addPrevLayerBias_(addPrevLayerBias),
      act_(activation) {}

template <typename DataType>
SELayer<DataType>::~SELayer() {
  auto context = sycl_default_queue().get_context();
  if (w1_) sycl::free(w1_, context);
  if (w1_t_) sycl::free(w1_t_, context);
  if (b1_) sycl::free(b1_, context);
  if (w2_) sycl::free(w2_, context);
  if (w2_t_) sycl::free(w2_t_, context);
  if (b2_) sycl::free(b2_, context);
  if (bPrev_) sycl::free(bPrev_, context);
}

template <typename DataType>
void SELayer<DataType>::LoadWeights(float* w1, float* b1, float* w2, float* b2,
                                    float* prevLayerBias, void* scratch) {
  // Implementation for SE layer weight loading
  // TODO: Implement based on the CUDA version
}

template <typename DataType>
void SELayer<DataType>::Eval(int N, DataType* output, const DataType* input,
                              const DataType* input2, void* scratch, size_t scratch_size,
                              sycl::queue queue, void* blas_handle, void* stream_handle,
                              DataType*** offset_pointers) {
  // Implementation for SE layer evaluation
  // TODO: Implement based on the CUDA version
}

// Template instantiations
template class BaseLayer<float>;
template class BaseLayer<sycl::half>;
template class FCLayer<float>;
template class FCLayer<sycl::half>;
template class PolicyMapLayer<float>;
template class PolicyMapLayer<sycl::half>;
template class SELayer<float>;
template class SELayer<sycl::half>;

}  // namespace sycl_backend
}  // namespace lczero