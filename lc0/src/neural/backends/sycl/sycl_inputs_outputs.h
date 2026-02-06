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
#include <cstddef>
#include <memory>
#include <vector>

namespace lczero {
namespace sycl_backend {

// Input/Output structure for SYCL neural network execution
template <typename DataType>
struct InputsOutputs {
  static constexpr int kInputPlanes = 112;
  static constexpr int kNumOutputPolicy = 18592;

  InputsOutputs(int max_batch_size, bool wdl, bool moves_left,
                size_t tensor_mem_size, size_t scratch_size,
                bool has_tensor_cores = false);

  ~InputsOutputs();

  // Input planes (host and device)
  std::vector<uint64_t> input_masks_mem_;           // Host
  std::vector<DataType> input_val_mem_;             // Host
  uint64_t* input_masks_mem_gpu_ = nullptr;         // Device
  DataType* input_val_mem_gpu_ = nullptr;           // Device

  // Outputs (host and device)
  std::vector<DataType> op_policy_mem_;             // Host
  std::vector<DataType> op_value_mem_;              // Host
  std::vector<DataType> op_moves_left_mem_;         // Host
  std::vector<float> wdl_cpu_softmax_;              // Host for WDL

  DataType* op_policy_mem_gpu_ = nullptr;          // Device
  DataType* op_value_mem_gpu_ = nullptr;           // Device
  DataType* op_moves_left_mem_gpu_ = nullptr;       // Device

  // Tensor memory for intermediate results
  std::vector<DataType*> tensor_mem_;
  void* scratch_mem_ = nullptr;
  DataType** offset_pointers_ = nullptr;
  DataType** head_offset_pointers_ = nullptr;

  // Batch size
  int max_batch_size_;
  bool wdl_;
  bool moves_left_;

 private:
  sycl::context context_;
  sycl::device device_;
};

} // namespace sycl_backend
} // namespace lczero