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

#include "sycl_inputs_outputs.h"
#include "sycl_common.h"
#include <cassert>

namespace lczero {
namespace sycl_backend {

template <typename DataType>
InputsOutputs<DataType>::InputsOutputs(int max_batch_size, bool wdl, bool moves_left,
                                        size_t tensor_mem_size, size_t scratch_size,
                                        bool has_tensor_cores)
    : max_batch_size_(max_batch_size), wdl_(wdl), moves_left_(moves_left) {

  // Initialize SYCL context and device
  sycl::queue queue(sycl::gpu_selector_v);
  device_ = queue.get_device();
  context_ = queue.get_context();

  // Allocate host memory for inputs
  input_masks_mem_.resize(max_batch_size * kInputPlanes);
  input_val_mem_.resize(max_batch_size * kInputPlanes);

  // Allocate device memory for inputs
  input_masks_mem_gpu_ = sycl::malloc_device<uint64_t>(
      max_batch_size * kInputPlanes * sizeof(uint64_t), device_, context_);
  input_val_mem_gpu_ = sycl::malloc_device<DataType>(
      max_batch_size * kInputPlanes * sizeof(DataType), device_, context_);

  // Allocate host memory for outputs
  op_policy_mem_.resize(max_batch_size * kNumOutputPolicy);
  op_value_mem_.resize(max_batch_size * 3); // WDL has 3 outputs
  op_moves_left_mem_.resize(max_batch_size);

  if (wdl_) {
    wdl_cpu_softmax_.resize(max_batch_size * 3);
  }

  // Allocate device memory for outputs
  op_policy_mem_gpu_ = sycl::malloc_device<DataType>(
      max_batch_size * kNumOutputPolicy * sizeof(DataType), device_, context_);
  op_value_mem_gpu_ = sycl::malloc_device<DataType>(
      max_batch_size * 3 * sizeof(DataType), device_, context_);
  op_moves_left_mem_gpu_ = sycl::malloc_device<DataType>(
      max_batch_size * sizeof(DataType), device_, context_);

  // Allocate tensor memory (3 buffers for intermediate results)
  tensor_mem_.resize(3);
  for (int i = 0; i < 3; i++) {
    tensor_mem_[i] = sycl::malloc_device<DataType>(
        tensor_mem_size / 3, device_, context_);
  }

  // Allocate scratch memory
  scratch_mem_ = sycl::malloc_device<void>(scratch_size, device_, context_);

  // Allocate offset pointers for attention layers (if needed)
  offset_pointers_ = sycl::malloc_device<DataType*>(
      32, device_, context_); // Placeholder size
  head_offset_pointers_ = sycl::malloc_device<DataType*>(
      32, device_, context_); // Placeholder size
}

template <typename DataType>
InputsOutputs<DataType>::~InputsOutputs() {
  // Free device memory
  if (input_masks_mem_gpu_) sycl::free(input_masks_mem_gpu_, context_);
  if (input_val_mem_gpu_) sycl::free(input_val_mem_gpu_, context_);
  if (op_policy_mem_gpu_) sycl::free(op_policy_mem_gpu_, context_);
  if (op_value_mem_gpu_) sycl::free(op_value_mem_gpu_, context_);
  if (op_moves_left_mem_gpu_) sycl::free(op_moves_left_mem_gpu_, context_);

  for (auto* tensor : tensor_mem_) {
    if (tensor) sycl::free(tensor, context_);
  }

  if (scratch_mem_) sycl::free(scratch_mem_, context_);
  if (offset_pointers_) sycl::free(offset_pointers_, context_);
  if (head_offset_pointers_) sycl::free(head_offset_pointers_, context_);
}

// Explicit template instantiations
template class InputsOutputs<float>;
template class InputsOutputs<sycl::half>;

}  // namespace sycl_backend
}  // namespace lczero