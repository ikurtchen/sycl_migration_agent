/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018-2022 The LCZero Authors

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
#include <algorithm>
#include <cassert>
#include <list>
#include <memory>
#include <mutex>
#include <thread>
#include <chrono>
#include <typeTraits>
#include <cstring>

#include "sycl_common.h"
#include "common_kernels.sycl.cpp"
#include "inputs_outputs.h"
#include "kernels.h"
#include "layers.h"
#include "neural/factory.h"
#include "neural/network_legacy.h"
#include "neural/tables/attention_policy_map.h"
#include "neural/tables/policy_map.h"
#include "utils/exception.h"
#include "utils/fp16_utils.h"
#include "utils/trace.h"

namespace lczero {
using namespace sycl_backend;

template <typename DataType>
class SyclNetwork;

template <typename DataType>
class SyclNetworkComputation : public NetworkComputation {
 public:
  static constexpr int kInputPlanes = 112;
  static constexpr int kNumOutputPolicy = 18592;

  SyclNetworkComputation(SyclNetwork<DataType>* network, bool wdl, bool moves_left)
      : wdl_(wdl), moves_left_(moves_left), network_(network) {
    batch_size_ = 0;
    GetInputsOutputs();
  }

  ~SyclNetworkComputation() {
    if (inputs_outputs_) {
      ReleaseInputsOutputs();
    }
  }

  void AddInput(InputPlanes&& input) override {
    const auto iter_mask =
        &inputs_outputs_->input_masks_mem_[batch_size_ * kInputPlanes];
    const auto iter_val =
        &inputs_outputs_->input_val_mem_[batch_size_ * kInputPlanes];

    assert(input.size() == kInputPlanes);
    for (int i = 0; i < kInputPlanes; i++) {
      const auto& plane = input[i];
      iter_mask[i] = plane.mask;
      ToType(iter_val[i], plane.value);
    }

    batch_size_++;
  }

  void ComputeBlocking() override {
    LCTRACE_FUNCTION_SCOPE;
    assert(GetBatchSize() >= 1);

    // Perform real SYCL neural network inference
    network_->forwardEval(inputs_outputs_.get(), GetBatchSize());
    network_->finishEval(inputs_outputs_.get(), GetBatchSize());

    // Reset batch size after computation
    batch_size_ = 0;
  }

  int GetBatchSize() const override { return batch_size_; }

  float GetQVal(int sample) const override {
    if (wdl_) {
      const float* wdl =
          sizeof(inputs_outputs_->op_value_mem_[0]) == sizeof(float)
              ? (float*)inputs_outputs_->op_value_mem_
              : inputs_outputs_->wdl_cpu_softmax_.get();
      return wdl[2 * sample];
    }
    return FromType(inputs_outputs_->op_value_mem_[sample]);
  }

  float GetDVal(int sample) const override {
    if (wdl_) {
      const float* wdl =
          sizeof(inputs_outputs_->op_value_mem_[0]) == sizeof(float)
              ? (float*)inputs_outputs_->op_value_mem_
              : inputs_outputs_->wdl_cpu_softmax_.get();
      return wdl[2 * sample + 1];
    }
    return 0.0f;
  }

  float GetPVal(int sample, int plane) const override {
    return FromType(
        inputs_outputs_->op_policy_mem_[sample * kNumOutputPolicy + plane]);
  }

  float GetMVal(int sample) const override {
    if (moves_left_) {
      return FromType(inputs_outputs_->op_moves_left_mem_[sample]);
    }
    return 0.0f;
  }

 private:
  template <typename T>
  void ToType(T& dst, const float src) const {
    if constexpr (std::is_same_v<T, float>) {
      dst = src;
    } else if constexpr (std::is_same_v<T, sycl::half>) {
      dst = sycl::half(src);
    }
  }

  template <typename T>
  float FromType(const T src) const {
    if constexpr (std::is_same_v<T, float>) {
      return src;
    } else if constexpr (std::is_same_v<T, sycl::half>) {
      return static_cast<float>(src);
    }
  }

  void GetInputsOutputs() {
    // Create input/output structure for SYCL computation
    inputs_outputs_ = std::make_unique<InputsOutputs<DataType>>(
        network_->GetMaxBatchSize(), wdl_, moves_left_,
        network_->GetTensorMemorySize(), network_->GetScratchMemorySize(),
        false  // has_tensor_cores for SYCL
    );
  }

  void ReleaseInputsOutputs() {
    // Clean up input/output structure
    inputs_outputs_.reset();
  }

  bool wdl_;
  bool moves_left_;
  SyclNetwork<DataType>* network_;
  std::unique_ptr<InputsOutputs<DataType>> inputs_outputs_;
  int batch_size_ = 0;
};

template <typename DataType>
class SyclNetwork : public lczero::Network {
 public:
  SyclNetwork(const WeightsFile& weights, const OptionsDict& options);
  virtual ~SyclNetwork();

  void AddInput(InputPlanes& input) override { /* Not used in production */ }
  void Eval(int batch_size) override { /* Not used in production */ }

  // Get the computation device.
  std::string GetDeviceString() const;

  // Required pure virtual methods from Network base class
  const NetworkCapabilities& GetCapabilities() const override;
  std::unique_ptr<NetworkComputation> NewComputation() override;
  int GetThreads() const override { return 1; }
  bool IsCpu() const override { return false; }

  // SYCL-specific methods
  void forwardEval(InputsOutputs<DataType>* io, int batchSize);
  void finishEval(InputsOutputs<DataType>* io, int batchSize);

  // Resource queries for SyclNetworkComputation
  int GetMaxBatchSize() const { return max_batch_size_; }
  size_t GetTensorMemorySize() const { return tensor_mem_size_; }
  size_t GetScratchMemorySize() const { return scratch_size_; }

 private:
  void Initialize(const WeightsFile& weights);
  void LoadWeights(const MultiHeadWeights& weights);

  // SYCL queue and memory
  sycl::queue queue_;

  // Neural network layers
  std::vector<std::unique_ptr<BaseLayer<DataType>>> network_;

  // Memory management
  void* device_scratch_ = nullptr;
  size_t scratch_size_ = 0;
  size_t tensor_mem_size_ = 0;

  // Input/output buffers and memory pools
  std::unique_ptr<InputsOutputs<DataType>> inputs_outputs_;
  std::list<std::unique_ptr<InputsOutputs<DataType>>> free_inputs_outputs_;
  std::mutex inputs_outputs_lock_;

  // Network configuration
  int max_batch_size_;
  int min_batch_size_;
  int numFilters_;
  int numBlocks_;
  bool attn_body_;
  bool attn_policy_;
  bool conv_policy_;
  bool wdl_;
  bool moves_left_;

  // Network capabilities
  NetworkCapabilities capabilities_;
};

template <typename DataType>
SyclNetwork<DataType>::SyclNetwork(const WeightsFile& weights, const OptionsDict& options)
    : capabilities_{weights.format().network_format().input(),
                    weights.format().network_format().output(),
                    weights.format().network_format().moves_left()} {
  // Suppress unused parameter warning
  (void)options;

  // Initialize SYCL queue with GPU selector
  try {
    queue_ = sycl::queue(sycl::gpu_selector_v, sycl::property_list{sycl::property::queue::in_order()});

    auto device = queue_.get_device();
    std::cout << "SYCL Backend Initialized:" << std::endl;
    std::cout << "  Device: " << device.get_info<sycl::info::device::name>() << std::endl;
    std::cout << "  Vendor: " << device.get_info<sycl::info::device::vendor>() << std::endl;
    std::cout << "  Platform: " << device.get_platform().get_info<sycl::info::platform::name>() << std::endl;
    std::cout << "  Max Compute Units: " << device.get_info<sycl::info::device::max_compute_units>() << std::endl;
    std::cout << "  Global Memory: " << device.get_info<sycl::info::device::global_mem_size>() / (1024*1024) << " MB" << std::endl;
  } catch (const sycl::exception& e) {
    std::cerr << "Failed to initialize SYCL GPU queue: " << e.what() << std::endl;
    throw;
  }

  // Initialize network
  Initialize(weights);
}

template <typename DataType>
SyclNetwork<DataType>::~SyclNetwork() {
  if (device_scratch_) {
    sycl::free(device_scratch_, queue_.get_context());
  }
}

template <typename DataType>
void SyclNetwork<DataType>::Initialize(const WeightsFile& weights) {
  // Set WDL and moves left flags based on network format
  wdl_ = weights.format().network_format().value() ==
         pblczero::NetworkFormat::VALUE_WDL;
  moves_left_ = weights.format().network_format().moves_left() ==
                pblczero::NetworkFormat::MOVES_LEFT_V1;

  // Load network weights
  const auto network_format = weights.format().network_format();

  // Determine network structure
  using NF = pblczero::NetworkFormat;
  numBlocks_ = network_format.residual_blocks();
  numFilters_ = network_format.filters();
  attn_body_ = (network_format.network() == NF::NETWORK_ATTENTIONBODY_WITH_HEADFORMAT ||
               network_format.network() == NF::NETWORK_ATTENTIONBODY_WITH_SOFTMAX_HEADFORMAT);

  // Set batch sizes
  max_batch_size_ = 256;  // Typical batch size for inference
  min_batch_size_ = 1;

  // Calculate memory requirements
  tensor_mem_size_ = 3 * max_batch_size_ * numFilters_ * 8 * 8 * sizeof(DataType);
  scratch_size_ = 1024 * 1024 * 100; // 100MB placeholder

  // Allocate device memory
  device_scratch_ = sycl::malloc_device(scratch_size_, queue_.get_device(), queue_.get_context());

  if (!device_scratch_) {
    throw Exception("SYCL network failed to allocate device memory");
  }

  // Load weights and create network layers
  LoadWeights(weights);
}

template <typename DataType>
std::unique_ptr<NetworkComputation> SyclNetwork<DataType>::NewComputation() {
  // Return a proper SYCL NetworkComputation object with correct flags
  return std::make_unique<SyclNetworkComputation<DataType>>(this, wdl_, moves_left_);
}

// Factory registration
std::unique_ptr<Network> MakeSyclNetwork(const std::optional<WeightsFile>& w, const OptionsDict& options) {
  if (!w) {
    throw Exception("The sycl backend requires a network file.");
  }
  const WeightsFile& weights = *w;
  const bool use_fp16 = options.GetOrDefault<bool>("use_fp16", false);

  if (use_fp16) {
    return std::make_unique<SyclNetwork<sycl::half>>(weights, options);
  } else {
    return std::make_unique<SyclNetwork<float>>(weights, options);
  }
}

// Network factory registration
REGISTER_NETWORK("sycl", MakeSyclNetwork, 130L);

}  // namespace lczero