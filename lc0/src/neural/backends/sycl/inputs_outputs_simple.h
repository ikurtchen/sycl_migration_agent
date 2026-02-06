/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018 The LCZero Authors

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

#include <cassert>
#include <memory>
#include <sycl/sycl.hpp>

#include "sycl_common.h"
#include "neural/network.h"
#include "utils/bit.h"

namespace lczero {
namespace sycl_backend {

inline void ToType(float& dst, float src) { dst = src; }
inline void ToType(sycl::half& dst, float src) {
  dst = static_cast<sycl::half>(src);
}

inline float FromType(float src) { return src; }
inline float FromType(sycl::half src) {
  return static_cast<float>(src);
}

template <typename DataType>
struct SyclGraphCapture;

template <typename DataType>
struct SyclGraphExec {
  ~SyclGraphExec() {
    // SYCL graph cleanup if needed
  }

  SyclGraphExec& operator=(const SyclGraphCapture<DataType>&);
  explicit operator bool() const { return valid_; }

  void Launch(sycl::queue& queue) {
    // SYCL graph launch
  }
  bool valid_ = false;
};

template <typename DataType>
struct InputsOutputs {
  InputsOutputs(unsigned maxBatchSize, bool wdl, bool moves_left,
                size_t tensor_mem_size = 0, size_t scratch_size = 0,
                bool cublasDisableTensorCores = false) {
    // SYCL memory allocation
    auto queue = sycl_default_queue();
    auto context = sycl::get_default_context();

    // Input masks
    input_masks_mem_ = static_cast<uint64_t*>(sycl::malloc_host(maxBatchSize * kInputPlanes * sizeof(uint64_t), context));
    input_masks_mem_gpu_ = static_cast<uint64_t*>(sycl::malloc_device(maxBatchSize * kInputPlanes * sizeof(uint64_t), context));

    // Input values
    input_val_mem_ = static_cast<DataType*>(sycl::malloc_host(maxBatchSize * kInputPlanes * sizeof(DataType), context));
    input_val_mem_gpu_ = static_cast<DataType*>(sycl::malloc_device(maxBatchSize * kInputPlanes * sizeof(DataType), context));

    // Output policy
    op_policy_mem_ = static_cast<DataType*>(sycl::malloc_host(maxBatchSize * kNumOutputPolicy * sizeof(DataType), context));
    op_policy_mem_gpu_ = static_cast<DataType*>(sycl::malloc_device(maxBatchSize * kNumOutputPolicy * sizeof(DataType), context));

    // Output value
    op_value_mem_ = static_cast<DataType*>(sycl::malloc_host(maxBatchSize * (wdl ? 3 : 1) * sizeof(DataType), context));
    op_value_mem_gpu_ = static_cast<DataType*>(sycl::malloc_device(maxBatchSize * (wdl ? 3 : 1) * sizeof(DataType), context));

    // Moves left if needed
    if (moves_left) {
      op_moves_left_mem_ = static_cast<DataType*>(sycl::malloc_host(maxBatchSize * sizeof(DataType), context));
      op_moves_left_mem_gpu_ = static_cast<DataType*>(sycl::malloc_device(maxBatchSize * sizeof(DataType), context));
    }

    // Additional memory if needed
    if (tensor_mem_size) {
      tensor_mem_[0] = sycl::malloc_device(tensor_mem_size, context);
      tensor_mem_[1] = sycl::malloc_device(tensor_mem_size, context);
      tensor_mem_[2] = sycl::malloc_device(tensor_mem_size, context);
      scratch_mem_ = sycl::malloc_device(scratch_size, context);
    }
  }

  ~InputsOutputs() {
    auto context = sycl::get_default_context();

    sycl::free(input_masks_mem_, context);
    sycl::free(input_masks_mem_gpu_, context);
    sycl::free(input_val_mem_, context);
    sycl::free(input_val_mem_gpu_, context);
    sycl::free(op_policy_mem_, context);
    sycl::free(op_policy_mem_gpu_, context);
    sycl::free(op_value_mem_, context);
    sycl::free(op_value_mem_gpu_, context);

    if (op_moves_left_mem_) {
      sycl::free(op_moves_left_mem_, context);
      sycl::free(op_moves_left_mem_gpu_, context);
    }

    if (scratch_mem_) {
      sycl::free(tensor_mem_[0], context);
      sycl::free(tensor_mem_[1], context);
      sycl::free(tensor_mem_[2], context);
      sycl::free(scratch_mem_, context);
    }
  }

  uint64_t* input_masks_mem_;
  DataType* input_val_mem_;
  DataType* op_policy_mem_;
  DataType* op_value_mem_;
  DataType* op_moves_left_mem_ = nullptr;

  // Device memory copies
  uint64_t* input_masks_mem_gpu_;
  DataType* input_val_mem_gpu_;
  DataType* op_policy_mem_gpu_;
  DataType* op_value_mem_gpu_;
  DataType* op_moves_left_mem_gpu_ = nullptr;

  std::unique_ptr<float[]> wdl_cpu_softmax_;

  // Memory for network execution
  void* tensor_mem_[3] = {nullptr, nullptr, nullptr};
  void* scratch_mem_ = nullptr;
  void** offset_pointers_ = nullptr;
  void** head_offset_pointers_ = nullptr;

  // SYCL queue for execution
  sycl::queue compute_queue_;
  sycl::queue upload_queue_;
  sycl::queue download_queue_;

  // SYCL graph support
  std::unique_ptr<SyclGraphExec<DataType>[]> sycl_graphs_;
  sycl::event join_capture_event_;
};

template <typename DataType>
struct SyclGraphCapture {
  static constexpr int kMinimumFreeMemory = 100 * 1024 * 1024;

  SyclGraphCapture(InputsOutputs<DataType>& io, sycl::queue upload_queue,
                   sycl::queue download_queue)
      : io_(io), upload_queue_(upload_queue), download_queue_(download_queue) {
    // SYCL graph capture initialization
  }

  ~SyclGraphCapture() {
    // SYCL graph cleanup
  }

  static bool EnsureEnoughFreeMemory() {
    // For SYCL, we would query available memory
    return true; // Simplified for now
  }

  void EndCapture() {
    // SYCL graph capture end
  }

  InputsOutputs<DataType>& io_;
  sycl::queue upload_queue_;
  sycl::queue download_queue_;
};

template <typename DataType>
inline SyclGraphExec<DataType>& SyclGraphExec<DataType>::operator=(
    const SyclGraphCapture<DataType>& graph) {
  // SYCL graph instantiation
  valid_ = true;
  return *this;
}

}  // namespace sycl_backend
}  // namespace lczero