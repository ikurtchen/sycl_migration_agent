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

#include <algorithm>
#include <cassert>
#include <iostream>
#include <stdexcept>
#include <sycl/sycl.hpp>
#include <cmath>

#include "sycl_common.h"
#include "neural/tables/activation_function.h"

namespace lczero {
namespace sycl_backend {
namespace {
constexpr int kInputPlanes = 112;
}  // namespace

/////////////////////////////////////////////////////////////////////////////
//          Simple SYCL kernels used by certain layers                     //
/////////////////////////////////////////////////////////////////////////////

template <typename T>
void addVectors_kernel(sycl::queue& q, T* c, T* a, T* b, int size, int asize,
                      int bsize, ActivationFunction activation) {
  // Launch configuration - equivalent to CUDA's <<<blocks, kBlockSize>>>
  const int kBlockSize = 256;
  int blocks = DivUp(size, kBlockSize);

  try {
    q.submit([&](sycl::handler& h) {
      h.parallel_for(sycl::nd_range<1>(
        sycl::range<1>(blocks * kBlockSize),  // Global range
        sycl::range<1>(kBlockSize)),        // Local range (work-group size)
        [=](sycl::nd_item<1> item) {
          // CUDA equivalent: int i = threadIdx.x + blockDim.x * blockIdx.x;
          int i = item.get_global_id(0);

          if (i < size) {
            float aVal = 0;
            float bVal = 0;
            if (a) aVal = static_cast<float>(a[i % asize]);
            if (b) bVal = static_cast<float>(b[i % bsize]);

            float cVal = aVal + bVal;

            // Apply activation function
            cVal = activate_device(cVal, activation);

            c[i] = static_cast<T>(cVal);
          }
        });
    }).wait();
  } catch (sycl::exception const& e) {
    std::cerr << "SYCL kernel error: " << e.what() << std::endl;
    throw std::runtime_error("SYCL kernel execution failed");
  }
}

// Adds two vectors (possibly of different sizes), also do optional relu
// activation.
template <typename T>
void addVectors(T* c, T* a, T* b, int size, int asize, int bsize,
                ActivationFunction activation, sycl::queue& q) {
  addVectors_kernel<T>(q, c, a, b, size, asize, bsize, activation);
}

// Template instantiation for float and half
template void addVectors<float>(float* c, float* a, float* b, int size,
                                int asize, int bsize, ActivationFunction act,
                                sycl::queue& q);
template void addVectors<sycl::half>(sycl::half* c, sycl::half* a, sycl::half* b, int size, int asize,
                                int bsize, ActivationFunction act,
                                sycl::queue& q);

template <typename T>
void addVectorsHNC_NHC_kernel(sycl::queue& q, T* a, T* b, int N, int H, int C) {
  const int kBlockSize = 256;
  int blocks = DivUp(N * H * C, kBlockSize);

  try {
    q.submit([&](sycl::handler& h) {
      h.parallel_for(sycl::nd_range<1>(
        sycl::range<1>(blocks * kBlockSize),  // Global range
        sycl::range<1>(kBlockSize)),         // Local range (work-group size)
        [=](sycl::nd_item<1> item) {
          // CUDA equivalent: int i = threadIdx.x + blockDim.x * blockIdx.x;
          int i = item.get_global_id(0);

          if (i < N * H * C) {
            int orig_i = i;
            int c = i % C;
            i /= C;
            int n = i % N;
            i /= N;
            int h = i;

            // Read a value from HNC layout (original position)
            float aVal = static_cast<float>(a[orig_i]);

            // Read b value from NHC layout
            // NHC index: n * H * C + h * C + c
            float bVal = static_cast<float>(b[n * H * C + h * C + c]);

            // Add and store back to HNC layout
            float cVal = aVal + bVal;
            a[orig_i] = static_cast<T>(cVal);
          }
        });
    }).wait();
  } catch (sycl::exception const& e) {
    std::cerr << "SYCL kernel error in addVectorsHNC_NHC: " << e.what() << std::endl;
    throw std::runtime_error("SYCL kernel execution failed");
  }
}

template <typename T>
void addVectorsHNC_NHC(T* a, T* b, int N, int H, int C, sycl::queue& q) {
  addVectorsHNC_NHC_kernel<T>(q, a, b, N, H, C);
}

// Template instantiation for float and half
template void addVectorsHNC_NHC<float>(float* a, float* b, int N, int H, int C,
                                       sycl::queue& q);
template void addVectorsHNC_NHC<sycl::half>(sycl::half* a, sycl::half* b, int N, int H, int C,
                                          sycl::queue& q);

template <typename T, ActivationFunction act>
void addBiasBatched_kernel(sycl::queue& q, T* output, const T* input, const T* bias,
                                      int N, int C) {
  // Check for unsupported filter sizes (same as CUDA version)
  if (C % 4 != 0) throw std::runtime_error("unsupported filter size");
  if (C > 4096) throw std::runtime_error("unsupported filter size");

  // Block dimensions - equivalent to CUDA's blockDim
  int block_dim_x = C / 4;
  int block_dim_y = std::min(std::max(512 / block_dim_x, 1), N);

  // Grid dimensions - equivalent to CUDA's gridDim
  // Note: In SYCL, we handle the batch dimension in the kernel itself
  int grid_dim_x = DivUp(N, block_dim_y);
  int batch_size = grid_dim_x * block_dim_y; // This will be passed as Batch

  try {
    q.submit([&](sycl::handler& h) {
      h.parallel_for(sycl::nd_range<2>(
        sycl::range<2>(grid_dim_x * block_dim_x, batch_size * block_dim_y),  // Global range
        sycl::range<2>(block_dim_x, block_dim_y)),                        // Local range (work-group size)
        [=](sycl::nd_item<2> item) {
          // CUDA equivalent mapping:
          // blockIdx.x -> item.get_group(0)
          // blockIdx.y -> item.get_group(1) / block_dim_y
          // threadIdx.x -> item.get_local_id(0)
          // threadIdx.y -> item.get_local_id(1)

          int batch = item.get_group(1) / block_dim_y;  // Extract batch from group coordinate
          int n = item.get_group(0) * block_dim_y + item.get_local_id(1);
          if (n >= N) return;
          int c = item.get_local_id(0) * 4;

          int biasIndex = batch * C + c;
          int tensorIndex = batch * N * C + n * C + c;

          float val[4];
          float b[4];

          // Load from memory
          const bool fp16 = std::is_same<sycl::half, T>::value;
          if (fp16) {
            // For fp16, use sycl::vec for efficient loads
            sycl::vec<sycl::half, 4> inp_vec = *reinterpret_cast<const sycl::vec<sycl::half, 4>*>(&input[tensorIndex]);
            sycl::vec<sycl::half, 4> bias_vec = *reinterpret_cast<const sycl::vec<sycl::half, 4>*>(&bias[biasIndex]);

            // Convert to float
            for (int i = 0; i < 4; i++) {
              val[i] = static_cast<float>(inp_vec[i]);
              b[i] = static_cast<float>(bias_vec[i]);
            }
          } else {
            // For fp32, use vec4 for 16-byte loads (4 float values)
            sycl::vec<float, 4> inp_vec = *reinterpret_cast<const sycl::vec<float, 4>*>(&input[tensorIndex]);
            sycl::vec<float, 4> bias_vec = *reinterpret_cast<const sycl::vec<float, 4>*>(&bias[biasIndex]);

            // Copy to float arrays
            for (int i = 0; i < 4; i++) {
              val[i] = inp_vec[i];
              b[i] = bias_vec[i];
            }
          }

          // Perform bias add and activation
          #pragma unroll
          for (int i = 0; i < 4; i++) {
            float x = val[i] + b[i];
            x = activate_device(x, act);
            val[i] = x;
          }

          // Write to memory
          if (fp16) {
            // Pack float values back to half precision
            sycl::vec<sycl::half, 4> out_vec;
            #pragma unroll
            for (int i = 0; i < 4; i++) out_vec[i] = static_cast<sycl::half>(val[i]);
            *reinterpret_cast<sycl::vec<sycl::half, 4>*>(&output[tensorIndex]) = out_vec;
          } else {
            // Direct float store using vec4
            sycl::vec<float, 4> out_vec;
            for (int i = 0; i < 4; i++) {
              out_vec[i] = val[i];
            }
            *reinterpret_cast<sycl::vec<float, 4>*>(&output[tensorIndex]) = out_vec;
          }
        });
    }).wait();
  } catch (sycl::exception const& e) {
    std::cerr << "SYCL addBiasBatched kernel error: " << e.what() << std::endl;
    throw std::runtime_error("SYCL addBiasBatched kernel execution failed");
  }
}

// Input/output tensors are Batch * N * C
// bias tensor is N * C (i.e, different bias for each Batch dimension)
template <typename T>
void addBiasBatched(T* output, const T* input, const T* bias, int Batch, int N,
                    int C, ActivationFunction activation, sycl::queue& q) {
  // Process 4 elements per thread to achieve close to peak memory bandwidth
  // (same as CUDA version requirements)
  if (C % 4 != 0) throw std::runtime_error("unsupported filter size");
  if (C > 4096) throw std::runtime_error("unsupported filter size");

  // Dispatch to appropriate templated kernel based on activation function
  switch (activation) {
    case ActivationFunction::ACTIVATION_NONE:
      addBiasBatched_kernel<T, ActivationFunction::ACTIVATION_NONE>(q, output, input, bias, N, C);
      break;
    case ActivationFunction::ACTIVATION_SELU:
      addBiasBatched_kernel<T, ActivationFunction::ACTIVATION_SELU>(q, output, input, bias, N, C);
      break;
    case ActivationFunction::ACTIVATION_MISH:
      addBiasBatched_kernel<T, ActivationFunction::ACTIVATION_MISH>(q, output, input, bias, N, C);
      break;
    case ActivationFunction::ACTIVATION_RELU:
      addBiasBatched_kernel<T, ActivationFunction::ACTIVATION_RELU>(q, output, input, bias, N, C);
      break;
    case ActivationFunction::ACTIVATION_SWISH:
      addBiasBatched_kernel<T, ActivationFunction::ACTIVATION_SWISH>(q, output, input, bias, N, C);
      break;
    case ActivationFunction::ACTIVATION_RELU_2:  // square relu
      addBiasBatched_kernel<T, ActivationFunction::ACTIVATION_RELU_2>(q, output, input, bias, N, C);
      break;
    default:
      throw std::runtime_error(
          "unsupported activation in addBiasBatched. Add in switch-case here");
  }
}

// Template instantiations for addBiasBatched
template void addBiasBatched<float>(float* output, const float* input,
                                    const float* bias, int Batch, int N, int C,
                                    ActivationFunction activation,
                                    sycl::queue& q);
template void addBiasBatched<sycl::half>(sycl::half* output, const sycl::half* input,
                                   const sycl::half* bias, int Batch, int N, int C,
                                   ActivationFunction activation,
                                   sycl::queue& q);

// Version with Nstride support (for padded N dimension)
template <typename T, ActivationFunction act>
void addBiasBatched_kernel_stride(sycl::queue& q, T* output, const T* input, const T* bias,
                                      int N, int C, int Nstride) {
  // Check for unsupported filter sizes (same as CUDA version)
  if (C % 4 != 0) throw std::runtime_error("unsupported filter size");
  if (C > 4096) throw std::runtime_error("unsupported filter size");

  // Block dimensions - equivalent to CUDA's blockDim
  int block_dim_x = C / 4;
  int block_dim_y = std::min(std::max(512 / block_dim_x, 1), N);

  // Grid dimensions - equivalent to CUDA's gridDim
  int grid_dim_x = DivUp(N, block_dim_y);
  int batch_size = grid_dim_x * block_dim_y; // This will be passed as Batch

  try {
    q.submit([&](sycl::handler& h) {
      h.parallel_for(sycl::nd_range<2>(
        sycl::range<2>(grid_dim_x * block_dim_x, batch_size * block_dim_y),  // Global range
        sycl::range<2>(block_dim_x, block_dim_y)),                        // Local range (work-group size)
        [=](sycl::nd_item<2> item) {
          int batch = item.get_group(1) / block_dim_y;  // Extract batch from group coordinate
          int n = item.get_group(0) * block_dim_y + item.get_local_id(1);
          if (n >= N) return;
          int c = item.get_local_id(0) * 4;

          int biasIndex = batch * C + c;
          int tensorIndex = batch * Nstride * C + n * C + c;  // Note: using Nstride here

          float val[4];
          float b[4];

          // Load from memory (same as regular version)
          const bool fp16 = std::is_same<sycl::half, T>::value;
          if (fp16) {
            // For fp16, use sycl::vec for efficient loads
            sycl::vec<sycl::half, 4> inp_vec = *reinterpret_cast<const sycl::vec<sycl::half, 4>*>(&input[tensorIndex]);
            sycl::vec<sycl::half, 4> bias_vec = *reinterpret_cast<const sycl::vec<sycl::half, 4>*>(&bias[biasIndex]);

            // Convert to float
            for (int i = 0; i < 4; i++) {
              val[i] = static_cast<float>(inp_vec[i]);
              b[i] = static_cast<float>(bias_vec[i]);
            }
          } else {
            // For fp32, use vec4 for 16-byte loads (4 float values)
            sycl::vec<float, 4> inp_vec = *reinterpret_cast<const sycl::vec<float, 4>*>(&input[tensorIndex]);
            sycl::vec<float, 4> bias_vec = *reinterpret_cast<const sycl::vec<float, 4>*>(&bias[biasIndex]);

            // Copy to float arrays
            for (int i = 0; i < 4; i++) {
              val[i] = inp_vec[i];
              b[i] = bias_vec[i];
            }
          }

          // Perform bias add and activation
          #pragma unroll
          for (int i = 0; i < 4; i++) {
            float x = val[i] + b[i];
            x = activate_device(x, act);
            val[i] = x;
          }

          // Write to memory
          if (fp16) {
            sycl::vec<sycl::half, 4> out_vec;
            #pragma unroll
            for (int i = 0; i < 4; i++) out_vec[i] = static_cast<sycl::half>(val[i]);
            *reinterpret_cast<sycl::vec<sycl::half, 4>*>(&output[tensorIndex]) = out_vec;
          } else {
            sycl::vec<float, 4> out_vec;
            for (int i = 0; i < 4; i++) {
              out_vec[i] = val[i];
            }
            *reinterpret_cast<sycl::vec<float, 4>*>(&output[tensorIndex]) = out_vec;
          }
        });
    }).wait();
  } catch (sycl::exception const& e) {
    std::cerr << "SYCL addBiasBatched kernel error: " << e.what() << std::endl;
    throw std::runtime_error("SYCL addBiasBatched kernel execution failed");
  }
}

// Version with Nstride support
template <typename T>
void addBiasBatched(T* output, const T* input, const T* bias, int Batch, int N,
                    int C, int Nstride, ActivationFunction activation,
                    sycl::queue& q) {
  // Process 4 elements per thread to achieve close to peak memory bandwidth
  if (C % 4 != 0) throw std::runtime_error("unsupported filter size");
  if (C > 4096) throw std::runtime_error("unsupported filter size");

  // Dispatch to appropriate templated kernel based on activation function
  switch (activation) {
    case ActivationFunction::ACTIVATION_NONE:
      addBiasBatched_kernel_stride<T, ActivationFunction::ACTIVATION_NONE>(q, output, input, bias, N, C, Nstride);
      break;
    case ActivationFunction::ACTIVATION_SELU:
      addBiasBatched_kernel_stride<T, ActivationFunction::ACTIVATION_SELU>(q, output, input, bias, N, C, Nstride);
      break;
    case ActivationFunction::ACTIVATION_MISH:
      addBiasBatched_kernel_stride<T, ActivationFunction::ACTIVATION_MISH>(q, output, input, bias, N, C, Nstride);
      break;
    case ActivationFunction::ACTIVATION_RELU:
      addBiasBatched_kernel_stride<T, ActivationFunction::ACTIVATION_RELU>(q, output, input, bias, N, C, Nstride);
      break;
    case ActivationFunction::ACTIVATION_SWISH:
      addBiasBatched_kernel_stride<T, ActivationFunction::ACTIVATION_SWISH>(q, output, input, bias, N, C, Nstride);
      break;
    case ActivationFunction::ACTIVATION_RELU_2:  // square relu
      addBiasBatched_kernel_stride<T, ActivationFunction::ACTIVATION_RELU_2>(q, output, input, bias, N, C, Nstride);
      break;
    default:
      throw std::runtime_error(
          "unsupported activation in addBiasBatched. Add in switch-case here");
  }
}

// Template instantiations for addBiasBatched with Nstride
template void addBiasBatched<float>(float* output, const float* input,
                                    const float* bias, int Batch, int N, int C,
                                    int Nstride, ActivationFunction activation,
                                    sycl::queue& q);
template void addBiasBatched<half>(sycl::half* output, const sycl::half* input,
                                   const sycl::half* bias, int Batch, int N, int C,
                                   int Nstride, ActivationFunction activation,
                                   sycl::queue& q);

// Use the activation function from the header to avoid conflicts

// Perform batch normalization.
template <typename T>
void batchNorm(sycl::queue& q, T* output, const T* input, const T* skipInput,
               int N, int C, int H, int W, const float* means,
               const float* varMultipliers, ActivationFunction activation) {
  const int total_elements = N * C * H * W;
  const int kBlockSize = 256;
  int blocks = DivUp(total_elements, kBlockSize);

  try {
    q.submit([&](sycl::handler& h) {
      h.parallel_for<class batchNorm_kernel<T>>(
          sycl::nd_range<1>(
              sycl::range<1>(blocks * kBlockSize),
              sycl::range<1>(kBlockSize)),
          [=](sycl::nd_item<1> item) {
            int index = item.get_global_id(0);

            if (index >= total_elements) return;

            int wIndex = 0;
            if (sizeof(T) == sizeof(float))
              wIndex = (index / (H * W)) % C;  // NCHW for fp32.
            else
              wIndex = index % C;  // NHWC for fp16.

            float el = static_cast<float>(input[index]);
            float mean = means[wIndex];
            float varMulti = varMultipliers[wIndex];

            // Batch normalization: (x - mean) * variance_multiplier
            // This computes: (x - mean) / sqrt(variance + epsilon)
            // where varMulti is pre-computed as 1/sqrt(variance + epsilon)
            el -= mean;
            el *= varMulti;

            // Add skip connection if present (residual connection)
            if (skipInput) el += static_cast<float>(skipInput[index]);

            // Apply activation function
            el = activate_device(el, activation);

            output[index] = static_cast<T>(el);
          });
    }).wait();
  } catch (sycl::exception const& e) {
    std::cerr << "SYCL batchNorm kernel error: " << e.what() << std::endl;
    throw std::runtime_error("SYCL batchNorm kernel execution failed");
  }
}

// Explicit template instantiations
template void batchNorm<float>(sycl::queue& q, float* output, const float* input,
                               const float* skipInput, int N, int C, int H, int W,
                               const float* means, const float* varMultipliers,
                               ActivationFunction activation);

template void batchNorm<sycl::half>(sycl::queue& q, sycl::half* output, const sycl::half* input,
                                   const sycl::half* skipInput, int N, int C, int H, int W,
                                   const float* means, const float* varMultipliers,
                                   ActivationFunction activation);

/////////////////////////////////////////////////////////////////////////////
//          Global Average Pooling kernels                                    //
/////////////////////////////////////////////////////////////////////////////

// Each thread reads 2 inputs (8x8/32), and each subgroup writes a single output.
// Replaces CUDA's __shfl_down_sync with SYCL subgroup operations.
template <typename T>
void globalAvgPool_kernel(sycl::queue& q, T* output, const T* input,
                         const T* prevLayerBias, int inputSize,
                         int outputSize, int N, int C) {
  const int elementsPerSubgroup = 64;  // 8x8 board size
  const int elementsPerThread = 2;

  // For optimal performance on Intel GPUs, use 32 threads per work-group
  // This maps directly to CUDA's warp size
  const int kWorkGroupSize = 32;
  const int kSubgroupsPerGroup = 8;
  const int kWorkGroupTotalSize = kSubgroupsPerGroup * kWorkGroupSize;

  const int kTotalSubgroups = N * C;
  const int kWorkGroups = DivUp(kTotalSubgroups, kSubgroupsPerGroup);

  try {
    q.submit([&](sycl::handler& h) {
      h.parallel_for<class globalAvgPool_kernel<T>>(
        sycl::nd_range<1>(
          sycl::range<1>(kWorkGroups * kWorkGroupTotalSize),  // Global range
          sycl::range<1>(kWorkGroupTotalSize)                 // Local range (work-group size)
        ),
        [=](sycl::nd_item<1> item) {
          // CUDA equivalent mapping:
          // tid = blockIdx.x * blockDim.x + threadIdx.x
          int tid = item.get_global_id(0);

          // Get subgroup information
          auto sg = item.get_sub_group();
          int sg_id = sg.get_group_id();           // Equivalent to warp index
          int sg_local_id = sg.get_local_id();    // Equivalent to laneId

          // Calculate thread's starting index for its assigned elements
          // Each thread processes 2 elements from the 64-element plane
          int threadStartIndex = sg_id * elementsPerSubgroup + sg_local_id * elementsPerThread;

          // Compute per-thread sum for elementsPerThread elements
          float S = 0;

          // Each thread processes 2 elements out of the 64 in the plane
          // Strided access pattern equivalent to CUDA's laneStartIndex + laneId + i
          for (int i = 0; i < elementsPerThread; i++) {
            int index = threadStartIndex + i;
            if (index < inputSize) {
              S += static_cast<float>(input[index]);
            }
          }

          // Compute subgroup-wide sum using SYCL reduction operations
          // This replaces CUDA's __shfl_down_sync loop
          S = sycl::reduce_over_group(sg, S, sycl::plus<float>());

          // Calculate average for the entire plane (64 elements)
          float avg = S / elementsPerSubgroup;

          // Only the first thread in each subgroup (lane 0) writes the output
          // This replaces CUDA's laneId == 0 condition
          if (sg_local_id == 0) {
            int outputIndex = sg_id;
            if (outputIndex < outputSize) {
              // Add bias from previous layer if present
              if (prevLayerBias) {
                avg += static_cast<float>(prevLayerBias[outputIndex % C]);
              }
              output[outputIndex] = static_cast<T>(avg);
            }
          }
        });
    }).wait();
  } catch (sycl::exception const& e) {
    std::cerr << "SYCL globalAvgPool kernel error: " << e.what() << std::endl;
    throw std::runtime_error("SYCL globalAvgPool kernel execution failed");
  }
}

// NHWC layout version for fp16 - each thread processes entire 8x8 board
// This version is optimized for the NHWC memory layout
void globalAvgPool_kernel_NHWC_fp16(sycl::queue& q, sycl::half* output,
                                   const sycl::half* input,
                                   const sycl::half* prevLayerBias,
                                   int inputSize, int outputSize, int N, int C) {
  const int elementsPerThread = 64;  // 8x8 board size

  // Launch configuration: N work-groups, each with C work-items
  // Equivalent to CUDA's <<<N, C>>>
  try {
    q.submit([&](sycl::handler& h) {
      h.parallel_for<class globalAvgPool_kernel_NHWC_fp16>(
        sycl::nd_range<2>(
          sycl::range<2>(N, C),  // Global range: N groups in x, C groups in y
          sycl::range<2>(1, 1)   // Local range: single thread per C channel
        ),
        [=](sycl::nd_item<2> item) {
          // CUDA equivalent mapping:
          // blockIdx.x -> item.get_group(0) (N dimension)
          // threadIdx.x -> item.get_group(1) (C dimension)

          int batch_idx = item.get_group(0);    // N index
          int channel_idx = item.get_group(1);   // C index

          // Calculate starting position for this thread's 64 elements
          // In NHWC: N * C * 64, where the 64 elements represent the spatial dimensions
          int blockStart = batch_idx * C * elementsPerThread + channel_idx;

          // Sum all 64 elements for this specific channel in this batch
          float S = 0;

          // Each element is spaced C apart in NHWC layout
          // This mimics CUDA's blockStart * elementsPerThread + localIndex
          for (int i = 0; i < elementsPerThread; i++) {
            int inputIndex = blockStart + i * C;
            if (inputIndex < inputSize) {
              S += static_cast<float>(input[inputIndex]);
            }
          }

          // Calculate average
          float avg = S / elementsPerThread;

          // Add bias from previous layer if present
          if (prevLayerBias) {
            avg += static_cast<float>(prevLayerBias[channel_idx]);
          }

          // Calculate output index
          int outputIndex = batch_idx * C + channel_idx;
          if (outputIndex < outputSize) {
            output[outputIndex] = static_cast<sycl::half>(avg);
          }
        });
    }).wait();
  } catch (sycl::exception const& e) {
    std::cerr << "SYCL globalAvgPool NHWC fp16 kernel error: " << e.what() << std::endl;
    throw std::runtime_error("SYCL globalAvgPool NHWC fp16 kernel execution failed");
  }
}

// Main global average pooling function - dispatches to appropriate kernel
template <typename T>
void globalAvgPool(sycl::queue& q, int N, int C, T* output, const T* input,
                   const T* prevLayerBias, bool nhwc) {
  const int kPlaneSize = 64;  // 8x8 board

  if (nhwc) {
    // For NHWC layout (used with fp16)
    // Verify we're using half precision as expected
    if (!std::is_same<sycl::half, T>::value) {
      throw std::runtime_error("NHWC layout requires half precision");
    }

    // Launch NHWC-specific kernel
    globalAvgPool_kernel_NHWC_fp16(q,
      static_cast<sycl::half*>(output),
      static_cast<const sycl::half*>(input),
      static_cast<const sycl::half*>(prevLayerBias),
      N * C * kPlaneSize,
      N * C, N, C);
  } else {
    // For NCHW layout (used with fp32)
    // Each subgroup processes a full plane (64 elements), and writes a single average
    // Launch N*C subgroups total

    globalAvgPool_kernel<T>(q, output, input, prevLayerBias,
                            N * C * kPlaneSize, N * C, N, C);
  }
}

// Explicit template instantiations for globalAvgPool
template void globalAvgPool<float>(sycl::queue& q, int N, int C, float* output,
                                   const float* input, const float* prevLayerBias,
                                   bool nhwc);

template void globalAvgPool<sycl::half>(sycl::queue& q, int N, int C, sycl::half* output,
                                        const sycl::half* input, const sycl::half* prevLayerBias,
                                        bool nhwc);

/////////////////////////////////////////////////////////////////////////////
//          Softmax kernels                                                 //
/////////////////////////////////////////////////////////////////////////////

namespace {
// Helper function for clamping values to prevent FP16 overflow
constexpr float kTwiceHalfMax = 131008.0f;  // Twice the max finite fp16 value

// SYCL helper function equivalent to CUDA's clamp
inline float clamp(float val, float low, float high) {
  if (std::isnan(val)) return val;
  return std::min(std::max(val, low), high);
}

// Helper function for vector type casting (equivalent to CUDA's copyAs)
template <typename DstT, typename SrcT>
inline void copyAs(DstT* dst, const SrcT* src) {
  *dst = static_cast<DstT>(*src);
}

// SYCL subgroup reduction operations (equivalent to CUDA's warpReduce)
inline float subgroupReduce(sycl::sub_group sg, float val) {
  // Use SYCL's built-in reduction over subgroup
  return sycl::reduce_over_group(sg, val, sycl::plus<float>());
}

// SYCL subgroup max operation (equivalent to CUDA's warpMax)
inline float subgroupMax(sycl::sub_group sg, float val) {
  return sycl::reduce_over_group(sg, val, sycl::maximum<float>());
}

// Atomic operations for shared memory in SYCL
inline float atomicMaxFloat(float* addr, float val) {
  // SYCL doesn't have direct atomic max for float, implement manually
  // Use sycl::atomic_ref with compare-and-swap loop
  sycl::atomic_ref<float, sycl::memory_order::relaxed,
                    sycl::memory_scope::work_group> atomic(*addr);
  float old = atomic.load();
  while (val > old && !atomic.compare_exchange_weak(old, val)) {
    // Continue until successful or no longer need to update
  }
  return old;
}
// SYCL device-side activation function (equivalent to CUDA version)
inline float activate_device(float cVal, ActivationFunction activation) {
  switch (activation) {
    case ActivationFunction::ACTIVATION_RELU:
      return (cVal < 0) ? 0 : cVal;
    case ActivationFunction::ACTIVATION_RELU_2:
      return (cVal < 0) ? 0 : (cVal * cVal);
    case ActivationFunction::ACTIVATION_TANH:
      return sycl::tanh(cVal);
    case ActivationFunction::ACTIVATION_SIGMOID:
      return 1.0f / (1.0f + sycl::exp(-cVal));
    case ActivationFunction::ACTIVATION_SELU: {
      constexpr float alpha = 1.67326324f;
      constexpr float scale = 1.05070098f;
      if (cVal > 0)
        return scale * cVal;
      else
        return scale * alpha * (sycl::exp(cVal) - 1.0f);
    }
    case ActivationFunction::ACTIVATION_MISH: {
      float e = sycl::exp(cVal);
      float n = e * e + 2.0f * e;
      float d = cVal / (n + 2.0f);
      if (cVal <= -0.6f) {
        return n * d;
      } else {
        return cVal - 2.0f * d;
      }
    }
    case ActivationFunction::ACTIVATION_SWISH:
      return cVal / (1.0f + sycl::exp(-cVal));
    case ActivationFunction::ACTIVATION_NONE:
      return cVal;
    case ActivationFunction::ACTIVATION_DEFAULT:
    case ActivationFunction::ACTIVATION_SOFTMAX:
      // This should not happen for activations
      return cVal;
  }
  return cVal;
}

// SYCL device-side mish activation function
inline float mishActivate_device(float el) {
  float e = sycl::exp(el);
  float n = e * e + 2.0f * e;
  float d = el / (n + 2.0f);
  if (el <= -0.6f) {
    return n * d;
  } else {
    return el - 2.0f * d;
  }
}

} // anonymous namespace

// Optimized softmax kernel for C=64 (equivalent to CUDA's softmax_opt_64_kernel)
// Uses SYCL subgroup operations instead of CUDA warp shuffle operations
template <typename T>
void softmax_opt_64_kernel(sycl::queue& q, T* output, const T* input,
                           const T* input2, int N) {
  const int kBlockSize = 256;  // Same as CUDA version
  int blocks = DivUp(N, kBlockSize);

  try {
    q.submit([&](sycl::handler& h) {
      h.parallel_for<class softmax_opt_64_kernel<T>>(
        sycl::nd_range<1>(
          sycl::range<1>(blocks * kBlockSize),  // Global range
          sycl::range<1>(kBlockSize)             // Local range
        ),
        [=](sycl::nd_item<1> item) {
          // CUDA equivalent mapping:
          // index = blockDim.x * blockIdx.x + threadIdx.x
          int index = item.get_global_id(0);
          if (index >= N) return;

          float x[4];
          float ex[2];

          // Load from memory
          const bool fp16 = std::is_same<sycl::half, T>::value;
          if (fp16) {
            // For fp16, each thread reads 2 elements (16 bytes total)
            sycl::vec<sycl::half, 2> inp_vec = *reinterpret_cast<const sycl::vec<sycl::half, 2>*>(&input[index * 2]);
            x[0] = static_cast<float>(inp_vec[0]);
            x[1] = static_cast<float>(inp_vec[1]);

            if (input2 != nullptr) {
              sycl::vec<sycl::half, 2> inp2_vec = *reinterpret_cast<const sycl::vec<sycl::half, 2>*>(&input2[index * 2]);
              x[2] = static_cast<float>(inp2_vec[0]);
              x[3] = static_cast<float>(inp2_vec[1]);
            }
          } else {
            // For fp32, each thread reads 2 elements (8 bytes total)
            sycl::vec<float, 2> inp_vec = *reinterpret_cast<const sycl::vec<float, 2>*>(&input[index * 2]);
            x[0] = inp_vec[0];
            x[1] = inp_vec[1];

            if (input2 != nullptr) {
              sycl::vec<float, 2> inp2_vec = *reinterpret_cast<const sycl::vec<float, 2>*>(&input2[index * 2]);
              x[2] = inp2_vec[0];
              x[3] = inp2_vec[1];
            }
          }

          // Add input2 if present
          if (input2 != nullptr) {
            x[0] += x[2];
            x[1] += x[3];
          }

          // Guard against Inf from fp16 overflow
          if (fp16) {
            x[0] = clamp(x[0], -kTwiceHalfMax, kTwiceHalfMax);
            x[1] = clamp(x[1], -kTwiceHalfMax, kTwiceHalfMax);
          }

          // Compute max across all elements in this 64-element softmax
          // Use SYCL subgroup operations to replace CUDA's warp shuffle
          auto sg = item.get_sub_group();
          float threadMax = std::max(x[0], x[1]);
          float maxval = subgroupMax(sg, threadMax);

          // For subgroups smaller than 32, we need additional reduction
          // This handles the case where subgroup size != 32
          if (sg.get_max_local_range()[0] < 32) {
            // Additional cross-subgroup reduction using shared memory
            sycl::local_accessor<float, 1> shared_max{32, h};
            int sg_id = sg.get_group_id();
            int sg_local_id = sg.get_local_id();

            if (sg_local_id == 0) {
              shared_max[sg_id] = maxval;
            }
            item.barrier();

            // Final max reduction by first thread in work-group
            if (item.get_local_id(0) == 0) {
              float final_max = shared_max[0];
              for (int i = 1; i < (32 + sg.get_max_local_range()[0] - 1) / sg.get_max_local_range()[0]; i++) {
                final_max = std::max(final_max, shared_max[i]);
              }
              shared_max[0] = final_max;
            }
            item.barrier();
            maxval = shared_max[0];
          }

          // Compute exp(x - maxval) for numerical stability
          ex[0] = sycl::exp(x[0] - maxval);
          ex[1] = sycl::exp(x[1] - maxval);

          // Compute sum across all elements
          float threadSum = ex[0] + ex[1];
          float Sum = subgroupReduce(sg, threadSum);

          // Additional cross-subgroup reduction if needed (same logic as max)
          if (sg.get_max_local_range()[0] < 32) {
            sycl::local_accessor<float, 1> shared_sum{32, h};
            int sg_id = sg.get_group_id();
            int sg_local_id = sg.get_local_id();

            if (sg_local_id == 0) {
              shared_sum[sg_id] = Sum;
            }
            item.barrier();

            if (item.get_local_id(0) == 0) {
              float final_sum = shared_sum[0];
              for (int i = 1; i < (32 + sg.get_max_local_range()[0] - 1) / sg.get_max_local_range()[0]; i++) {
                final_sum += shared_sum[i];
              }
              shared_sum[0] = final_sum;
            }
            item.barrier();
            Sum = shared_sum[0];
          }

          // Normalize: divide by sum
          ex[0] = ex[0] / Sum;
          ex[1] = ex[1] / Sum;

          // Store to memory
          if (fp16) {
            sycl::vec<sycl::half, 2> out_vec;
            out_vec[0] = static_cast<sycl::half>(ex[0]);
            out_vec[1] = static_cast<sycl::half>(ex[1]);
            *reinterpret_cast<sycl::vec<sycl::half, 2>*>(&output[index * 2]) = out_vec;
          } else {
            sycl::vec<float, 2> out_vec;
            out_vec[0] = ex[0];
            out_vec[1] = ex[1];
            *reinterpret_cast<sycl::vec<float, 2>*>(&output[index * 2]) = out_vec;
          }
        });
    }).wait();
  } catch (sycl::exception const& e) {
    std::cerr << "SYCL softmax_opt_64 kernel error: " << e.what() << std::endl;
    throw std::runtime_error("SYCL softmax_opt_64 kernel execution failed");
  }
}

// General softmax kernel for arbitrary C (equivalent to CUDA's softmax_kernel)
// Uses shared memory for reduction instead of warp operations
template <typename T>
void softmax_general_kernel(sycl::queue& q, T* output, const T* input,
                             const T* input2, int N, int C) {
  try {
    q.submit([&](sycl::handler& h) {
      // Shared memory for sum and max value per work-group
      sycl::local_accessor<float, 1> shared_sum{1, h};
      sycl::local_accessor<float, 1> shared_max{1, h};

      h.parallel_for<class softmax_general_kernel<T>>(
        sycl::nd_range<2>(
          sycl::range<2>(N, C),  // Global range: N batches, C channels
          sycl::range<2>(1, C)   // Local range: 1 thread per channel in work-group
        ),
        [=](sycl::nd_item<2> item) {
          // CUDA equivalent mapping:
          // blockIdx.x -> item.get_group(0) (N dimension)
          // threadIdx.x -> item.get_local_id(0) (C dimension)

          int n = item.get_group(0);   // Batch index
          int c = item.get_local_id(0); // Channel index
          int index = n * C + c;

          // Read input value
          float x = static_cast<float>(input[index]);
          if (input2 != nullptr) {
            x += static_cast<float>(input2[index]);
          }

          // Guard against Inf from fp16 overflow
          if (std::is_same<sycl::half, T>::value) {
            x = clamp(x, -kTwiceHalfMax, kTwiceHalfMax);
          }

          // Initialize shared memory (only first thread)
          if (c == 0) {
            shared_sum[0] = 0.0f;
            shared_max[0] = x;
          }
          item.barrier();

          // Get subgroup for efficient reduction
          auto sg = item.get_sub_group();
          int sg_local_id = sg.get_local_id();

          // Compute max across subgroup first
          float warp_max = subgroupMax(sg, x);

          // Update global max in shared memory (only first thread in subgroup)
          if (sg_local_id == 0) {
            atomicMaxFloat(&shared_max[0], warp_max);
          }
          item.barrier();

          // Compute exp(x - max) for numerical stability
          float ex = sycl::exp(x - shared_max[0]);

          // Compute sum across subgroup first
          float warp_sum = subgroupReduce(sg, ex);

          // Update global sum in shared memory (only first thread in subgroup)
          if (sg_local_id == 0) {
            sycl::atomic_ref<float, sycl::memory_order::relaxed,
                             sycl::memory_scope::work_group> atomic_sum(shared_sum[0]);
            atomic_sum.fetch_add(warp_sum);
          }
          item.barrier();

          // Final normalization
          float op = ex / shared_sum[0];
          output[index] = static_cast<T>(op);
        });
    }).wait();
  } catch (sycl::exception const& e) {
    std::cerr << "SYCL softmax_general kernel error: " << e.what() << std::endl;
    throw std::runtime_error("SYCL softmax_general kernel execution failed");
  }
}

// Main softmax function - dispatches to appropriate kernel
template <typename T>
void Softmax(sycl::queue& q, int N, int C, T* output, const T* input, const T* input2) {
  if (C == 64) {
    // Use optimized kernel for C=64
    int size = N * 32;  // Total threads needed (2 elements per thread)
    softmax_opt_64_kernel<T>(q, output, input, input2, size);
  } else {
    // Use general kernel for other sizes
    softmax_general_kernel<T>(q, output, input, input2, N, C);
  }
}

// Explicit template instantiations for Softmax
template void Softmax<float>(sycl::queue& q, int N, int C, float* output, const float* input,
                             const float* input2);

template void Softmax<sycl::half>(sycl::queue& q, int N, int C, sycl::half* output, const sycl::half* input,
                                  const sycl::half* input2);

/////////////////////////////////////////////////////////////////////////////
//          Policy Mapping kernel                                            //
/////////////////////////////////////////////////////////////////////////////

template <typename T>
void PolicyMap(int N, T* output, const T* input, const short* indices,
               int inputSize, int usedSize, int outputSize,
               sycl::queue& q) {
  // Each thread processes one input element
  // Only some of the threads (with valid mapping) write output
  const int kBlockSize = 256;
  const int totalElements = N * usedSize;
  const int kBlocks = DivUp(totalElements, kBlockSize);

  try {
    q.submit([&](sycl::handler& h) {
      h.parallel_for<class policyMap_kernel<T>>(
          sycl::nd_range<1>(
              sycl::range<1>(kBlocks * kBlockSize),  // Global range
              sycl::range<1>(kBlockSize)),         // Local range (work-group size)
          [=](sycl::nd_item<1> item) {
            // CUDA equivalent mapping:
            // int tid = blockIdx.x * blockDim.x + threadIdx.x;
            int tid = item.get_global_linear_id();

            // CUDA equivalent mapping:
            // int n = tid / usedSize;
            // int i = tid % usedSize;
            int n = tid / usedSize;
            int i = tid % usedSize;

            if (n >= N) return;

            // Read the output index from the mapping array
            int j = indices[i];

            // Only write if we have a valid mapping (j >= 0)
            if (j >= 0) {
              // Map from input to output using the indices array
              // CUDA: output[n * outputSize + j] = input[n * inputSize + i];
              output[n * outputSize + j] = input[n * inputSize + i];
            }
          });
    }).wait();
  } catch (sycl::exception const& e) {
    std::cerr << "SYCL policyMap kernel error: " << e.what() << std::endl;
    throw std::runtime_error("SYCL policyMap kernel execution failed");
  }
}

// Explicit template instantiations for PolicyMap
template void PolicyMap<float>(int N, float* output, const float* input,
                               const short* indices, int inputSize,
                               int usedSize, int outputSize,
                               sycl::queue& q);

template void PolicyMap<sycl::half>(int N, sycl::half* output, const sycl::half* input,
                                   const short* indices, int inputSize, int usedSize,
                                   int outputSize, sycl::queue& q);

///////////////////////////////////////////////////////////////////////////////
//                     Plane expansion kernels for NN input                    //
///////////////////////////////////////////////////////////////////////////////

template <typename T>
void expandPlanes_kernel_NHWC(sycl::queue& q, T* output, const uint64_t* masks,
                             const T* values, int n) {
  // Each thread writes a single element
  int threads = n * 8 * 8;
  const int kBlockSize = 256;
  int blocks = DivUp(threads, kBlockSize);

  try {
    q.submit([&](sycl::handler& h) {
      h.parallel_for(sycl::nd_range<1>(
        sycl::range<1>(blocks * kBlockSize),  // Global range
        sycl::range<1>(kBlockSize)),        // Local range (work-group size)
        [=](sycl::nd_item<1> item) {
          // CUDA equivalent: const int index = threadIdx.x + blockDim.x * blockIdx.x;
          const int index = item.get_global_id(0);
          if (index >= n * 8 * 8) return;

          // Decode index into plane, board, and square coordinates
          // CUDA: const int planeIndex = index % kInputPlanes;
          const int planeIndex = index % kInputPlanes;
          // CUDA: const int boardIndex = index / (kInputPlanes * 8 * 8);
          const int boardIndex = index / (kInputPlanes * 8 * 8);
          // CUDA: const int sqIndex = (index / kInputPlanes) & 0x3F;
          const int sqIndex = (index / kInputPlanes) & 0x3F;

          // Get the mask for this plane and board
          // CUDA: uint64_t mask = masks[boardIndex * kInputPlanes + planeIndex];
          uint64_t mask = masks[boardIndex * kInputPlanes + planeIndex];

          // Set output value based on whether the bit is set in the mask
          T op = static_cast<T>(0);
          bool set = !!(mask & (1ull << sqIndex));
          if (set) {
            // CUDA: op = values[boardIndex * kInputPlanes + planeIndex];
            op = values[boardIndex * kInputPlanes + planeIndex];
          }
          output[index] = op;
        });
    }).wait();
  } catch (sycl::exception const& e) {
    std::cerr << "SYCL expandPlanes_NHWC kernel error: " << e.what() << std::endl;
    throw std::runtime_error("SYCL expandPlanes_NHWC kernel execution failed");
  }
}

template <typename T>
void expandPlanes_NHWC(T* output, const uint64_t* masks, const T* values, int n,
                       sycl::queue& q) {
  expandPlanes_kernel_NHWC<T>(q, output, masks, values, n);
}

template <typename T>
void expandPlanes_kernel_NCHW(sycl::queue& q, T* output, const uint64_t* masks,
                             const T* values, unsigned n) {
  // Each thread writes two elements (optimized for memory bandwidth)
  unsigned threads = n * 8 * 8 / 2;
  const int blockSize = 256;
  unsigned blocks = DivUp(threads, blockSize);

  try {
    q.submit([&](sycl::handler& h) {
      h.parallel_for(sycl::nd_range<1>(
        sycl::range<1>(blocks * blockSize),  // Global range
        sycl::range<1>(blockSize)),          // Local range (work-group size)
        [=](sycl::nd_item<1> item) {
          // CUDA equivalent: unsigned index = threadIdx.x + blockDim.x * blockIdx.x;
          unsigned index = item.get_global_id(0);

          // Each thread processes 2 elements
          index *= 2;
          // CUDA: unsigned planeIndex = index >> 6;
          unsigned planeIndex = index >> 6;

          if (planeIndex >= n) return;

          // CUDA: uint64_t mask = masks[planeIndex];
          uint64_t mask = masks[planeIndex];

          // CUDA: int sqIndex = index & 0x3F;
          int sqIndex = index & 0x3F;
          T op[2] = {static_cast<T>(0), static_cast<T>(0)};

          // Check first square
          bool set = !!(mask & (1ull << sqIndex));
          if (set) {
            // CUDA: op[0] = values[planeIndex];
            op[0] = values[planeIndex];
          }

          // Check second square
          sqIndex++;
          set = !!(mask & (1ull << sqIndex));
          if (set) {
            // CUDA: op[1] = values[planeIndex];
            op[1] = values[planeIndex];
          }

          // CUDA: output[index + 0] = op[0];
          output[index + 0] = op[0];
          // CUDA: output[index + 1] = op[1];
          output[index + 1] = op[1];
        });
    }).wait();
  } catch (sycl::exception const& e) {
    std::cerr << "SYCL expandPlanes_NCHW kernel error: " << e.what() << std::endl;
    throw std::runtime_error("SYCL expandPlanes_NCHW kernel execution failed");
  }
}

template <typename T>
void expandPlanes_NCHW(T* output, const uint64_t* masks, const T* values,
                       int n, sycl::queue& q) {
  expandPlanes_kernel_NCHW<T>(q, output, masks, values, static_cast<unsigned>(n));
}

// Explicit template instantiations for expandPlanes
template void expandPlanes_NHWC<float>(float* output, const uint64_t* masks,
                                       const float* values, int n,
                                       sycl::queue& q);
template void expandPlanes_NHWC<sycl::half>(sycl::half* output, const uint64_t* masks,
                                            const sycl::half* values, int n,
                                            sycl::queue& q);

template void expandPlanes_NCHW<float>(float* output, const uint64_t* masks,
                                       const float* values, int n,
                                       sycl::queue& q);
template void expandPlanes_NCHW<sycl::half>(sycl::half* output, const uint64_t* masks,
                                            const sycl::half* values, int n,
                                            sycl::queue& q);

/////////////////////////////////////////////////////////////////////////////
//          Layer Normalization kernels                                       //
///////////////////////////////////////////////////////////////////////////////

// Helper function for shared memory reduction across C dimension
// Replaces CUDA's shared_sum_for_layer_norm which used shared memory and warp reductions
// SYCL version uses group-local memory and subgroup reductions for efficiency
template <typename T>
class LayerNormKernel;

template <typename T>
void layer_norm_kernel(sycl::queue& q, int N, int C, T* output, const T* input,
                      const T* bias, const T* skip, const T* gammas,
                      const T* betas, float ep, float alpha,
                      ActivationFunction act) {
  // Validate constraints (same as CUDA version)
  if (C % 16 != 0) throw std::runtime_error("unsupported filter size");
  if (C > 16384) throw std::runtime_error("unsupported filter size");

  // Calculate work-group and grid dimensions
  // Each work-group processes one batch (N) with multiple threads handling C dimension
  const int kWorkGroupSizeX = 32;    // Threads in X dimension (same as CUDA)
  const int kWorkGroupSizeY = DivUp(C / 16, 32);  // Threads in Y dimension
  const int kWorkGroupSizeZ = std::min(std::max(512 / (kWorkGroupSizeX * kWorkGroupSizeY), 1), N);

  const int kGridDimX = DivUp(N, kWorkGroupSizeZ);

  try {
    q.submit([&](sycl::handler& h) {
      // Shared memory for C-dimension reductions
      // Replace CUDA's static __shared__ array with dynamic local memory
      sycl::local_accessor<float, 2> shared_sums{sycl::range<2>(kWorkGroupSizeZ, kWorkGroupSizeY), h};

      h.parallel_for<class LayerNormKernel<T>>(
        sycl::nd_range<3>(
          sycl::range<3>(kGridDimX * kWorkGroupSizeX, kWorkGroupSizeY, kWorkGroupSizeZ),  // Global range
          sycl::range<3>(kWorkGroupSizeX, kWorkGroupSizeY, kWorkGroupSizeZ)               // Local range
        ),
        [=](sycl::nd_item<3> item) {
          // CUDA equivalent mapping:
          // int n = blockIdx.x * blockDim.z + threadIdx.z;
          int n = item.get_group(0) * item.get_local_range(2) + item.get_local_id(2);
          if (n >= N) return;

          // CUDA equivalent mapping:
          // int c = (threadIdx.y * 32 + threadIdx.x) * 16;
          int c = (item.get_local_id(1) * 32 + item.get_local_id(0)) * 16;
          bool oobThread = c >= C;

          int biasIndex = c;
          int tensorIndex = n * C + c;

          float val[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
          float oth[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

          const bool fp16 = std::is_same<sycl::half, T>::value;

          if (!oobThread) {
            // Load from memory (16 elements at a time)
            if (fp16) {
              // For fp16, use vector loads for efficiency
              // Load first 8 elements
              auto inp_vec1 = *reinterpret_cast<const sycl::vec<sycl::half, 8>*>(&input[tensorIndex]);
              auto bias_vec1 = *reinterpret_cast<const sycl::vec<sycl::half, 8>*>(&bias[biasIndex]);
              for (int i = 0; i < 8; i++) {
                val[i] = static_cast<float>(inp_vec1[i]);
                oth[i] = static_cast<float>(bias_vec1[i]);
              }

              // Load next 8 elements
              auto inp_vec2 = *reinterpret_cast<const sycl::vec<sycl::half, 8>*>(&input[tensorIndex + 8]);
              auto bias_vec2 = *reinterpret_cast<const sycl::vec<sycl::half, 8>*>(&bias[biasIndex + 8]);
              for (int i = 0; i < 8; i++) {
                val[i + 8] = static_cast<float>(inp_vec2[i]);
                oth[i + 8] = static_cast<float>(bias_vec2[i]);
              }
            } else {
              // For fp32, use 128-bit loads (4 floats at a time)
              // Load 16 elements using 4 vectorized loads
              auto inp_vec1 = *reinterpret_cast<const sycl::vec<float, 4>*>(&input[tensorIndex]);
              auto inp_vec2 = *reinterpret_cast<const sycl::vec<float, 4>*>(&input[tensorIndex + 4]);
              auto inp_vec3 = *reinterpret_cast<const sycl::vec<float, 4>*>(&input[tensorIndex + 8]);
              auto inp_vec4 = *reinterpret_cast<const sycl::vec<float, 4>*>(&input[tensorIndex + 12]);

              val[0] = inp_vec1[0]; val[1] = inp_vec1[1]; val[2] = inp_vec1[2]; val[3] = inp_vec1[3];
              val[4] = inp_vec2[0]; val[5] = inp_vec2[1]; val[6] = inp_vec2[2]; val[7] = inp_vec2[3];
              val[8] = inp_vec3[0]; val[9] = inp_vec3[1]; val[10] = inp_vec3[2]; val[11] = inp_vec3[3];
              val[12] = inp_vec4[0]; val[13] = inp_vec4[1]; val[14] = inp_vec4[2]; val[15] = inp_vec4[3];

              auto bias_vec1 = *reinterpret_cast<const sycl::vec<float, 4>*>(&bias[biasIndex]);
              auto bias_vec2 = *reinterpret_cast<const sycl::vec<float, 4>*>(&bias[biasIndex + 4]);
              auto bias_vec3 = *reinterpret_cast<const sycl::vec<float, 4>*>(&bias[biasIndex + 8]);
              auto bias_vec4 = *reinterpret_cast<const sycl::vec<float, 4>*>(&bias[biasIndex + 12]);

              oth[0] = bias_vec1[0]; oth[1] = bias_vec1[1]; oth[2] = bias_vec1[2]; oth[3] = bias_vec1[3];
              oth[4] = bias_vec2[0]; oth[5] = bias_vec2[1]; oth[6] = bias_vec2[2]; oth[7] = bias_vec2[3];
              oth[8] = bias_vec3[0]; oth[9] = bias_vec3[1]; oth[10] = bias_vec3[2]; oth[11] = bias_vec3[3];
              oth[12] = bias_vec4[0]; oth[13] = bias_vec4[1]; oth[14] = bias_vec4[2]; oth[15] = bias_vec4[3];
            }

            // Add bias to input
            for (int i = 0; i < 16; i++) {
              val[i] += oth[i];
            }
          }

          // Load skip connection if present
          if (!oobThread) {
            if (skip != nullptr) {
              if (fp16) {
                // Load skip data for fp16
                auto skip_vec1 = *reinterpret_cast<const sycl::vec<sycl::half, 8>*>(&skip[tensorIndex]);
                for (int i = 0; i < 8; i++) {
                  oth[i] = static_cast<float>(skip_vec1[i]);
                }
                auto skip_vec2 = *reinterpret_cast<const sycl::vec<sycl::half, 8>*>(&skip[tensorIndex + 8]);
                for (int i = 0; i < 8; i++) {
                  oth[i + 8] = static_cast<float>(skip_vec2[i]);
                }
              } else {
                // Load skip data for fp32
                auto skip_vec1 = *reinterpret_cast<const sycl::vec<float, 4>*>(&skip[tensorIndex]);
                auto skip_vec2 = *reinterpret_cast<const sycl::vec<float, 4>*>(&skip[tensorIndex + 4]);
                auto skip_vec3 = *reinterpret_cast<const sycl::vec<float, 4>*>(&skip[tensorIndex + 8]);
                auto skip_vec4 = *reinterpret_cast<const sycl::vec<float, 4>*>(&skip[tensorIndex + 12]);

                oth[0] = skip_vec1[0]; oth[1] = skip_vec1[1]; oth[2] = skip_vec1[2]; oth[3] = skip_vec1[3];
                oth[4] = skip_vec2[0]; oth[5] = skip_vec2[1]; oth[6] = skip_vec2[2]; oth[7] = skip_vec2[3];
                oth[8] = skip_vec3[0]; oth[9] = skip_vec3[1]; oth[10] = skip_vec3[2]; oth[11] = skip_vec3[3];
                oth[12] = skip_vec4[0]; oth[13] = skip_vec4[1]; oth[14] = skip_vec4[2]; oth[15] = skip_vec4[3];
              }
            }
          }

          // 1. Compute mean
          float s = 0;
          if (!oobThread) {
            if (skip != nullptr) {
              for (int i = 0; i < 16; i++) {
                // Apply activation, scale by alpha, add skip
                val[i] = activate_device(val[i], act) * alpha + oth[i];
                s += val[i];
              }
            } else {
              for (int i = 0; i < 16; i++) {
                // Apply activation and scale by alpha
                val[i] = activate_device(val[i], act) * alpha;
                s += val[i];
              }
            }
          }

          // Compute shared reduction across C dimension
          // This replaces CUDA's shared_sum_for_layer_norm function
          auto sg = item.get_sub_group();
          int sg_id = sg.get_group_id();
          int sg_local_id = sg.get_local_id();

          // First, reduce within each thread's 16 elements (already done in loop above)

          // Then, reduce across the work-group using subgroup operations
          float partial_sum = s;

          // Reduce across the X dimension (32 threads) using subgroup
          partial_sum = sycl::reduce_over_group(sg, partial_sum, sycl::plus<float>());

          // Copy per-subgroup partial sums to shared memory
          if (sg_local_id == 0) {
            shared_sums[item.get_local_id(2)][sg_id] = partial_sum;
          }
          item.barrier();

          // Combine across all subgroups in the Y dimension
          if (sg_local_id == 0 && item.get_local_id(0) == 0) {
            float cSum = 0;
            for (int j = 0; j < (C / 16 + 31) / 32; j++) {
              if (j < kWorkGroupSizeY) {
                cSum += shared_sums[item.get_local_id(2)][j];
              }
            }
            shared_sums[item.get_local_id(2)][0] = cSum;
          }
          item.barrier();

          // s now contains the sum across C dimension
          s = shared_sums[item.get_local_id(2)][0];
          float mean = s / C;

          // 2. Compute variance
          s = 0;
          if (!oobThread) {
            for (int i = 0; i < 16; i++) {
              float d = val[i] - mean;
              float d_sq = d * d;
              s += d_sq;
            }
          }

          // Same reduction pattern as above for variance
          partial_sum = s;
          partial_sum = sycl::reduce_over_group(sg, partial_sum, sycl::plus<float>());

          if (sg_local_id == 0) {
            shared_sums[item.get_local_id(2)][sg_id] = partial_sum;
          }
          item.barrier();

          if (sg_local_id == 0 && item.get_local_id(0) == 0) {
            float cSum = 0;
            for (int j = 0; j < (C / 16 + 31) / 32; j++) {
              if (j < kWorkGroupSizeY) {
                cSum += shared_sums[item.get_local_id(2)][j];
              }
            }
            shared_sums[item.get_local_id(2)][0] = cSum;
          }
          item.barrier();

          s = shared_sums[item.get_local_id(2)][0];
          float var = s / C;

          // Load gamma weights
          if (!oobThread) {
            if (fp16) {
              auto gamma_vec1 = *reinterpret_cast<const sycl::vec<sycl::half, 8>*>(&gammas[biasIndex]);
              auto gamma_vec2 = *reinterpret_cast<const sycl::vec<sycl::half, 8>*>(&gammas[biasIndex + 8]);
              for (int i = 0; i < 8; i++) {
                oth[i] = static_cast<float>(gamma_vec1[i]);
                oth[i + 8] = static_cast<float>(gamma_vec2[i]);
              }
            } else {
              auto gamma_vec1 = *reinterpret_cast<const sycl::vec<float, 4>*>(&gammas[biasIndex]);
              auto gamma_vec2 = *reinterpret_cast<const sycl::vec<float, 4>*>(&gammas[biasIndex + 4]);
              auto gamma_vec3 = *reinterpret_cast<const sycl::vec<float, 4>*>(&gammas[biasIndex + 8]);
              auto gamma_vec4 = *reinterpret_cast<const sycl::vec<float, 4>*>(&gammas[biasIndex + 12]);

              oth[0] = gamma_vec1[0]; oth[1] = gamma_vec1[1]; oth[2] = gamma_vec1[2]; oth[3] = gamma_vec1[3];
              oth[4] = gamma_vec2[0]; oth[5] = gamma_vec2[1]; oth[6] = gamma_vec2[2]; oth[7] = gamma_vec2[3];
              oth[8] = gamma_vec3[0]; oth[9] = gamma_vec3[1]; oth[10] = gamma_vec3[2]; oth[11] = gamma_vec3[3];
              oth[12] = gamma_vec4[0]; oth[13] = gamma_vec4[1]; oth[14] = gamma_vec4[2]; oth[15] = gamma_vec4[3];
            }
          }

          // 3. Normalize
          for (int i = 0; i < 16; i++) {
            float d = val[i] - mean;
            float norm = d / sycl::sqrt(var + ep);
            float op = norm * oth[i];  // Multiply by gamma
            val[i] = op;
          }

          // Load beta weights
          if (!oobThread) {
            if (fp16) {
              auto beta_vec1 = *reinterpret_cast<const sycl::vec<sycl::half, 8>*>(&betas[biasIndex]);
              auto beta_vec2 = *reinterpret_cast<const sycl::vec<sycl::half, 8>*>(&betas[biasIndex + 8]);
              for (int i = 0; i < 8; i++) {
                oth[i] = static_cast<float>(beta_vec1[i]);
                oth[i + 8] = static_cast<float>(beta_vec2[i]);
              }
            } else {
              auto beta_vec1 = *reinterpret_cast<const sycl::vec<float, 4>*>(&betas[biasIndex]);
              auto beta_vec2 = *reinterpret_cast<const sycl::vec<float, 4>*>(&betas[biasIndex + 4]);
              auto beta_vec3 = *reinterpret_cast<const sycl::vec<float, 4>*>(&betas[biasIndex + 8]);
              auto beta_vec4 = *reinterpret_cast<const sycl::vec<float, 4>*>(&betas[biasIndex + 12]);

              oth[0] = beta_vec1[0]; oth[1] = beta_vec1[1]; oth[2] = beta_vec1[2]; oth[3] = beta_vec1[3];
              oth[4] = beta_vec2[0]; oth[5] = beta_vec2[1]; oth[6] = beta_vec2[2]; oth[7] = beta_vec2[3];
              oth[8] = beta_vec3[0]; oth[9] = beta_vec3[1]; oth[10] = beta_vec3[2]; oth[11] = beta_vec3[3];
              oth[12] = beta_vec4[0]; oth[13] = beta_vec4[1]; oth[14] = beta_vec4[2]; oth[15] = beta_vec4[3];
            }
          }

          // Add beta to normalized values
          for (int i = 0; i < 16; i++) {
            val[i] += oth[i];
          }

          // Write to memory
          if (!oobThread) {
            if (fp16) {
              sycl::vec<sycl::half, 8> op_vec1, op_vec2;
              for (int i = 0; i < 8; i++) {
                op_vec1[i] = static_cast<sycl::half>(val[i]);
                op_vec2[i] = static_cast<sycl::half>(val[i + 8]);
              }
              *reinterpret_cast<sycl::vec<sycl::half, 8>*>(&output[tensorIndex]) = op_vec1;
              *reinterpret_cast<sycl::vec<sycl::half, 8>*>(&output[tensorIndex + 8]) = op_vec2;
            } else {
              sycl::vec<float, 4> out_vec1, out_vec2, out_vec3, out_vec4;
              out_vec1[0] = val[0]; out_vec1[1] = val[1]; out_vec1[2] = val[2]; out_vec1[3] = val[3];
              out_vec2[0] = val[4]; out_vec2[1] = val[5]; out_vec2[2] = val[6]; out_vec2[3] = val[7];
              out_vec3[0] = val[8]; out_vec3[1] = val[9]; out_vec3[2] = val[10]; out_vec3[3] = val[11];
              out_vec4[0] = val[12]; out_vec4[1] = val[13]; out_vec4[2] = val[14]; out_vec4[3] = val[15];

              *reinterpret_cast<sycl::vec<float, 4>*>(&output[tensorIndex]) = out_vec1;
              *reinterpret_cast<sycl::vec<float, 4>*>(&output[tensorIndex + 4]) = out_vec2;
              *reinterpret_cast<sycl::vec<float, 4>*>(&output[tensorIndex + 8]) = out_vec3;
              *reinterpret_cast<sycl::vec<float, 4>*>(&output[tensorIndex + 12]) = out_vec4;
            }
          }
        });
    }).wait();
  } catch (sycl::exception const& e) {
    std::cerr << "SYCL layer_norm kernel error: " << e.what() << std::endl;
    throw std::runtime_error("SYCL layer_norm kernel execution failed");
  }
}

// add (optional) skip connection to input, and then perform Layer normalization
// normalization is done across C dimension (i.e, sums and std deviations taken
// over elements in C dim)
template <typename T>
void LayerNorm(sycl::queue& q, int N, int C, T* output, const T* input, const T* bias,
               const T* skip, const T* gammas, const T* betas, float ep,
               float alpha, ActivationFunction act) {
  // Process 4 elements per thread to achieve close to peak memory bandwidth
  // (same constraints as CUDA version)
  if (C % 16 != 0) throw std::runtime_error("unsupported filter size");
  if (C > 16384) throw std::runtime_error("unsupported filter size");

  layer_norm_kernel<T>(q, N, C, output, input, bias, skip, gammas, betas, ep, alpha, act);
}

// Explicit template instantiations for LayerNorm
template void LayerNorm<float>(sycl::queue& q, int N, int C, float* output, const float* input,
                               const float* bias, const float* skip,
                               const float* gammas, const float* betas, float ep,
                               float alpha, ActivationFunction act);

template void LayerNorm<sycl::half>(sycl::queue& q, int N, int C, sycl::half* output, const sycl::half* input,
                                   const sycl::half* bias, const sycl::half* skip,
                                   const sycl::half* gammas, const sycl::half* betas, float ep,
                                   float alpha, ActivationFunction act);

/////////////////////////////////////////////////////////////////////////////
//          Type conversion kernel                                             //
///////////////////////////////////////////////////////////////////////////////

template <typename DstType, typename SrcType>
void copyTypeConverted_kernel(sycl::queue& q, DstType* op, SrcType* ip, int N) {
  // Launch configuration - equivalent to CUDA's <<<blocks, kBlockSize>>>
  const int kBlockSize = 256;
  int blocks = DivUp(N, kBlockSize);

  try {
    q.submit([&](sycl::handler& h) {
      h.parallel_for<class copyTypeConverted_kernel<DstType, SrcType>>(
        sycl::nd_range<1>(
          sycl::range<1>(blocks * kBlockSize),  // Global range
          sycl::range<1>(kBlockSize)             // Local range (work-group size)
        ),
        [=](sycl::nd_item<1> item) {
          // CUDA equivalent mapping:
          // int tid = blockIdx.x * blockDim.x + threadIdx.x;
          int tid = item.get_global_linear_id();

          if (tid >= N) return;

          // Convert from source type to destination type
          // CUDA: DstType el = (DstType)ip[tid];
          DstType el = static_cast<DstType>(ip[tid]);

          // Store the converted value
          // CUDA: op[tid] = el;
          op[tid] = el;
        });
    }).wait();
  } catch (sycl::exception const& e) {
    std::cerr << "SYCL copyTypeConverted kernel error: " << e.what() << std::endl;
    throw std::runtime_error("SYCL copyTypeConverted kernel execution failed");
  }
}

template <typename DstType, typename SrcType>
void copyTypeConverted(DstType* op, SrcType* ip, int N, sycl::queue& q) {
  copyTypeConverted_kernel<DstType, SrcType>(q, op, ip, N);
}

// Template instantiations for copyTypeConverted
// These match the CUDA template instantiations at lines 1342-1349
template void copyTypeConverted<sycl::half, float>(sycl::half* op, float* ip, int N, sycl::queue& q);
template void copyTypeConverted<float, sycl::half>(float* op, sycl::half* ip, int N, sycl::queue& q);
template void copyTypeConverted<float, float>(float* op, float* ip, int N, sycl::queue& q);
template void copyTypeConverted<sycl::half, sycl::half>(sycl::half* op, sycl::half* ip, int N, sycl::queue& q);

// /////////////////////////////////////////////////////////////////////////////
//           Global Scale kernels for residual blocks                        //
// /////////////////////////////////////////////////////////////////////////////

template <typename T>
void globalScale_kernel(sycl::queue& q, T* output, const T* input,
                        const T* scaleBias, const T* prevLayerBias,
                        int inputSize, int C, ActivationFunction activation) {
  const int kPlaneSize = 64;
  const int kBlockSize = 256;
  int kBlocks = DivUp(inputSize, kBlockSize);

  try {
    q.submit([&](sycl::handler& h) {
      h.parallel_for<class globalScale_kernel<T>>(
        sycl::nd_range<1>(
          sycl::range<1>(kBlocks * kBlockSize),  // Global range
          sycl::range<1>(kBlockSize)            // Local range (work-group size)
        ),
        [=](sycl::nd_item<1> item) {
          // CUDA equivalent: int tid = blockIdx.x * blockDim.x + threadIdx.x;
          int tid = item.get_global_linear_id();

          if (tid > inputSize) return;

          // Calculate n and c from thread index for NCHW layout
          // CUDA: int nc = tid / kPlaneSize; int n = nc / C; int c = nc % C;
          int nc = tid / kPlaneSize;
          int n = nc / C;
          int c = nc % C;

          // Load input and output values
          // CUDA: float val1 = input[tid]; float val2 = output[tid];
          float val1 = static_cast<float>(input[tid]);   // Output of residual block to be scaled
          float val2 = static_cast<float>(output[tid]);  // Skip connection to be added directly

          // Add previous layer bias if provided
          if (prevLayerBias) {
            // CUDA: val1 += (float)(prevLayerBias[c]);
            val1 += static_cast<float>(prevLayerBias[c]);
          }

          // Calculate scale and bias indices
          // CUDA: int startIdx = n * 2 * C; // Scale and bias interleaved
          int startIdx = n * 2 * C;  // Scale and bias interleaved

          // Load scale value and apply sigmoid
          // CUDA: float s = scaleBias[startIdx + c]; s = 1.0f / (1.0f + exp(-s));
          float s = static_cast<float>(scaleBias[startIdx + c]);
          s = 1.0f / (1.0f + sycl::exp(-s));  // Sigmoid on scale

          // Load bias value
          // CUDA: float b = scaleBias[startIdx + c + C];
          float b = static_cast<float>(scaleBias[startIdx + c + C]);

          // Apply scaling and addition
          // CUDA: float op = val1 * s + val2 + b; op = activate(op, activation);
          float op = val1 * s + val2 + b;
          op = activate(op, activation);

          // Store result
          // CUDA: output[tid] = (T)op;
          output[tid] = static_cast<T>(op);
        });
    }).wait();
  } catch (sycl::exception const& e) {
    std::cerr << "SYCL globalScale kernel error: " << e.what() << std::endl;
    throw std::runtime_error("SYCL globalScale kernel execution failed");
  }
}

// Specialized kernel for fp16 NHWC layout
void globalScale_kernel_fp16_nhwc(sycl::queue& q, sycl::half* output, const sycl::half* input,
                                 const sycl::half* scaleBias, const sycl::half* prevLayerBias,
                                 int inputSize, int C, int HWC, ActivationFunction activation) {
  const int kBlockSize = 256;
  int kBlocks = DivUp(inputSize, kBlockSize);

  try {
    q.submit([&](sycl::handler& h) {
      h.parallel_for<class globalScale_kernel_fp16_nhwc>(
        sycl::nd_range<1>(
          sycl::range<1>(kBlocks * kBlockSize),  // Global range
          sycl::range<1>(kBlockSize)            // Local range (work-group size)
        ),
        [=](sycl::nd_item<1> item) {
          // CUDA equivalent: int tid = blockIdx.x * blockDim.x + threadIdx.x;
          int tid = item.get_global_linear_id();

          if (tid > inputSize) return;

          // Calculate n and c from thread index for NHWC layout
          // CUDA: int c = tid % C; int n = tid / (HWC);
          int c = tid % C;
          int n = tid / HWC;

          // Load input and output values for NHWC layout
          // CUDA: float val1 = (float)input[tid]; float val2 = (float)output[tid];
          float val1 = static_cast<float>(input[tid]);   // Output of residual block to be scaled
          float val2 = static_cast<float>(output[tid]);  // Skip connection to be added directly

          // Add previous layer bias if provided (same as NCHW)
          if (prevLayerBias) {
            // CUDA: val1 += (float)prevLayerBias[c];
            val1 += static_cast<float>(prevLayerBias[c]);
          }

          // Calculate scale and bias indices (same as NCHW)
          // CUDA: int startIdx = n * 2 * C; // Scale and bias interleaved
          int startIdx = n * 2 * C;  // Scale and bias interleaved

          // Load scale value and apply sigmoid
          // CUDA: float s = scaleBias[startIdx + c]; s = 1.0f / (1.0f + exp(-s));
          float s = static_cast<float>(scaleBias[startIdx + c]);
          s = 1.0f / (1.0f + sycl::exp(-s));  // Sigmoid on scale

          // Load bias value
          // CUDA: float b = scaleBias[startIdx + c + C];
          float b = static_cast<float>(scaleBias[startIdx + c + C]);

          // Apply scaling and addition
          // CUDA: float op = val1 * s + val2 + b; op = activate(op, activation);
          float op = val1 * s + val2 + b;
          op = activate(op, activation);

          // Store result
          // CUDA: output[tid] = (half)op;
          output[tid] = static_cast<sycl::half>(op);
        });
    }).wait();
  } catch (sycl::exception const& e) {
    std::cerr << "SYCL globalScale kernel fp16 nhwc error: " << e.what() << std::endl;
    throw std::runtime_error("SYCL globalScale kernel fp16 nhwc execution failed");
  }
}

// Wrapper function for global scaling - matches CUDA template function at lines 684-701
template <typename T>
void globalScale(int N, int C, T* output, const T* input, const T* scaleBias,
                 const T* prevLayerBias, bool nhwc,
                 ActivationFunction activation, sycl::queue& q) {
  // Each thread writes one output.
  const int kBlockSize = 256;
  const int kBlocks = DivUp(N * 8 * 8 * C, kBlockSize);

  if (nhwc) {
    // For NHWC layout (fp16 only)
    static_assert(std::is_same<T, sycl::half>::value, "NHWC layout requires sycl::half type");
    globalScale_kernel_fp16_nhwc(
        q, static_cast<sycl::half*>(output),
        static_cast<const sycl::half*>(input),
        static_cast<const sycl::half*>(scaleBias),
        static_cast<const sycl::half*>(prevLayerBias),
        N * C * 8 * 8, C, 8 * 8 * C, activation);
  } else {
    // For NCHW layout (typically fp32)
    globalScale_kernel<T>(q, output, input, scaleBias, prevLayerBias,
                          N * C * 8 * 8, C, activation);
  }
}

// Template instantiations for globalScale - match CUDA at lines 1422-1431
template void globalScale<float>(int N, int C, float* output,
                                const float* input, const float* scaleBias,
                                const float* prevLayerBias, bool nhwc,
                                ActivationFunction activation, sycl::queue& q);
template void globalScale<sycl::half>(int N, int C, sycl::half* output,
                                      const sycl::half* input, const sycl::half* scaleBias,
                                      const sycl::half* prevLayerBias, bool nhwc,
                                      ActivationFunction activation, sycl::queue& q);

/////////////////////////////////////////////////////////////////////////////
//          NCHW to NHWC layout conversion kernel                            //
///////////////////////////////////////////////////////////////////////////////

namespace {
// Helper function to read from NCHW format tensor
template <typename dT, typename sT>
dT readNCHW(const sT* input_tensor, int n, int c, int h, int w,
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

  return (dT)(input_tensor[index]);
}
} // anonymous namespace

template <typename dT, typename sT>
class NCHWtoNHWC_kernel;

template <typename dT, typename sT>
void NCHWtoNHWC_kernel_impl(sycl::queue& q, dT* output_tensor, const sT* input_tensor,
                            int Nin, int Cin, int Nout, int Cout, int H, int W) {
  size_t numElements = Nout * Cout * H * W;
  const int blockSize = 256;
  int blocks = (numElements + blockSize - 1) / blockSize;

  try {
    q.submit([&](sycl::handler& h) {
      h.parallel_for<class NCHWtoNHWC_kernel<dT, sT>>(
        sycl::nd_range<1>(blocks * blockSize, blockSize),
        [=](sycl::nd_item<1> item) {
          int tid = item.get_global_id(0);

          if (tid >= Nout * Cout * H * W) return;

          int index = tid;

          int c = (index % Cout);
          index /= Cout;
          int w = index % W;
          index /= W;
          int h = index % H;
          index /= H;
          int n = index;

          output_tensor[tid] = readNCHW<dT, sT>(input_tensor, n, c, h, w, Nin, Cin, H, W);
        });
    }).wait();
  } catch (sycl::exception const& e) {
    std::cerr << "SYCL exception in NCHWtoNHWC kernel: " << e.what() << std::endl;
    throw;
  }
}

template <typename DstType, typename SrcType>
void convertNCHWtoNHWC(sycl::queue& q, DstType* output_tensor, const SrcType* input_tensor,
                       int Nin, int Cin, int Nout, int Cout, int H, int W) {
  NCHWtoNHWC_kernel_impl<DstType, SrcType>(q, output_tensor, input_tensor,
                                           Nin, Cin, Nout, Cout, H, W);
}

// Preprocess for attention body kernel
template <typename T>
void preprocess_for_attention_body_kernel_impl(
    sycl::queue& q, T* output, const T* input, const T* encoding,
    int input_size, int encoding_size, bool is_pe_dense_embedding,
    int N) {

  // N x 64 grid, input_size + encoding_size threads per block
  // In SYCL: 2D nd_range with (N, 64) global range and (1, 64) local range

  try {
    q.submit([&](sycl::handler& h) {
      h.parallel_for<class preprocess_for_attention_body_kernel<T>>(
        sycl::nd_range<2>(
          sycl::range<2>(N, 64),                 // Global range: N blocks x 64 spatial positions
          sycl::range<2>(1, 64)                  // Local range: 1 x 64 (work-group size)
        ),
        [=](sycl::nd_item<2> item) {
          // Map CUDA indices to SYCL
          // CUDA: blockIdx.x -> item.get_group(0) (batch index n)
          // CUDA: blockIdx.y -> item.get_group(1) (spatial position hw)
          // CUDA: threadIdx.x -> item.get_local_id(1) (channel index c)
          int n = item.get_group(0);
          int hw = item.get_group(1);
          int c = item.get_local_id(1);

          // Check bounds - make sure we don't exceed the required channels
          int total_channels = input_size + encoding_size;
          if (c >= total_channels) return;

          T op;
          if (c >= input_size) {
            // concatenate from position encoding array
            if (is_pe_dense_embedding) {
              // Dense embedding encoding: n * 64 * encoding_size + hw * encoding_size + (c - input_size)
              int encoding_index = n * 64 * encoding_size + hw * encoding_size + (c - input_size);
              op = static_cast<T>(encoding[encoding_index]);
            } else {
              // Standard encoding: 64 * hw + (c - input_size)
              int encoding_index = 64 * hw + (c - input_size);
              op = static_cast<T>(encoding[encoding_index]);
            }
          } else {
            // Input data in NCHW format: n * input_size * 64 + c * 64 + hw
            int input_index = n * input_size * 64 + c * 64 + hw;
            op = input[input_index];
          }

          // Convert to NHWC format and store
          // NHWC index: n * 64 * outputC + hw * outputC + c
          int output_index = n * 64 * total_channels + hw * total_channels + c;
          output[output_index] = op;
        });
    }).wait();
  } catch (sycl::exception const& e) {
    std::cerr << "SYCL exception in preprocess_for_attention_body kernel: " << e.what() << std::endl;
    throw;
  }
}

template <typename T>
void inputPreprocessForAttentionBody(sycl::queue& q, T* output, const T* input,
                                     const T* encoding, int N, int input_size,
                                     int encoding_size,
                                     bool is_pe_dense_embedding) {
  preprocess_for_attention_body_impl<T>(q, output, input, encoding,
                                       input_size, encoding_size,
                                       is_pe_dense_embedding, N);
}

// Template instantiation
template void inputPreprocessForAttentionBody<float>(sycl::queue& q, float* output, const float* input,
                                      const float* encoding, int N, int input_size,
                                      int encoding_size, bool is_pe_dense_embedding);

template void inputPreprocessForAttentionBody<sycl::half>(sycl::queue& q, sycl::half* output, const sycl::half* input,
                                         const sycl::half* encoding, int N, int input_size,
                                         int encoding_size, bool is_pe_dense_embedding);

// Template instantiation
template void convertNCHWtoNHWC<sycl::half, float>(sycl::queue& q, sycl::half* output_tensor,
                                             const float* input_tensor, int Nin,
                                             int Cin, int Nout, int Cout, int H,
                                             int W);
template void convertNCHWtoNHWC<float, float>(sycl::queue& q, float* output_tensor,
                                              const float* input_tensor,
                                              int Nin, int Cin, int Nout,
                                              int Cout, int H, int W);
template void convertNCHWtoNHWC<sycl::half, sycl::half>(sycl::queue& q, sycl::half* output_tensor,
                                           const sycl::half* input_tensor, int Nin,
                                           int Cin, int Nout, int Cout, int H,
                                           int W);

template <typename T>
void input_gating_impl(sycl::queue& q, T* output, const T* input, const T* mult,
                       const T* add, int N, int HW, int C) {
  try {
    // Calculate optimal work-group sizes based on Intel GPU architecture
    // For Intel GPUs, work-groups of 256-512 are typically optimal
    const int max_threads_per_block = 1024;
    int block_x = DivUp(max_threads_per_block, HW);
    if (block_x > 256) block_x = 256;  // Cap for efficiency
    int block_y = HW;

    sycl::range<3> local_range(block_x, block_y, 1);
    sycl::range<3> global_range(DivUp(C, block_x) * block_x, 1, N);

    q.submit([&](sycl::handler& h) {
      h.parallel_for(
        sycl::nd_range<3>(global_range, local_range),
        [=](sycl::nd_item<3> item) {
          // Map CUDA thread indexing to SYCL
          // blockIdx.z -> item.get_group(2)  (batch dimension)
          // blockIdx.x -> item.get_group(0)  (channel section)
          // threadIdx.x -> item.get_local_id(0)  (thread in x)
          // threadIdx.y -> item.get_local_id(1)  (thread in y)

          int n_offset = item.get_group(2) * HW * C;

          // Calculate input index (equivalent to threadIdx.y * C + blockIdx.x * blockDim.x + threadIdx.x)
          int idx = item.get_local_id(1) * C + item.get_group(0) * item.get_local_range(0) + item.get_local_id(0);

          // Calculate transposed weight index (equivalent to (blockIdx.x * blockDim.x + threadIdx.x) * HW + threadIdx.y)
          int idxT = (item.get_group(0) * item.get_local_range(0) + item.get_local_id(0)) * HW + item.get_local_id(1);

          if (idx < HW * C) {
            // Combine multiply gating, add gating and weights transpose
            // This is the core input gating operation: output = input * mult + add
            float op = static_cast<float>(input[n_offset + idx]) * static_cast<float>(mult[idxT]) + static_cast<float>(add[idxT]);
            output[n_offset + idx] = static_cast<T>(op);
          }
        });
    }).wait();
  } catch (sycl::exception const& e) {
    std::cerr << "SYCL exception in input_gating kernel: " << e.what() << std::endl;
    throw;
  }
}

template <typename T>
void applyInputGating(sycl::queue& q, T* output, const T* input, const T* mult, const T* add,
                      int N, int HW, int C) {
  // Multiple blocks to fit into each input area / volume
  // Block x position indicates horizontal section of area
  // Block y position indicates batch (handled in SYCL via z-dimension)
  // Each thread computes a single output element

  input_gating_impl<T>(q, output, input, mult, add, N, HW, C);
}

// Template instantiation for input gating
template void applyInputGating<float>(sycl::queue& q, float* output, const float* input,
                                      const float* mult, const float* add,
                                      int N, int HW, int C);

template void applyInputGating<sycl::half>(sycl::queue& q, sycl::half* output, const sycl::half* input,
                                           const sycl::half* mult, const sycl::half* add,
                                           int N, int HW, int C);

/////////////////////////////////////////////////////////////////////////////
//          Promotion Logits Kernel Implementation                        //
/////////////////////////////////////////////////////////////////////////////

// Compute promotion logits in a single kernel
// keys matrix is of N * 64 * C (but we use only last 8 from the 'rows'
// dimension, so N * 8 * C)
// ppo matrix is 4 * C (weights for dense layer / matrix multiplication)
// policy_attn_logits matrix is N * 64 * 64, but we use only 8x8 part of it
// from each batch dimension (so, N * 8 * 8)
// output matrix (promotion logits) is of N * 8 * 24 size
template <typename T>
void promotion_logits_kernel(sycl::queue& q, int C, T* output, const T* keys,
                            const T* ppo, const T* policy_attn_logits, int N) {
  constexpr int output_stride = 64 * 64 + 8 * 24;

  // Launch N work-groups, each with 8 * 24 work-items
  // In SYCL: 2D nd_range with (N*8, 24) global range and (8, 24) local range
  auto range = sycl::nd_range<2>(
      sycl::range<2>(N * 8, 24),  // Global range: N*8 groups in y, 24 in x
      sycl::range<2>(8, 24)       // Local range (work-group size): 8x24
  );

  try {
    q.submit([&](sycl::handler& h) {
      // Allocate local memory (shared memory) for promotion_offsets[4][8]
      sycl::local_accessor<float, 2> promotion_offsets(sycl::range<2>(4, 8), h);

      h.parallel_for<class promotion_logits_kernel<T>>(range, [=](sycl::nd_item<2> item) {
        // Map CUDA indices to SYCL
        // CUDA: blockIdx.x -> N dimension
        // CUDA: threadIdx.y -> y dimension [0..8)
        // CUDA: threadIdx.x -> x dimension [0..24)

        int n = item.get_group(0) / 8;           // Block index / 8
        int y = item.get_group(0) % 8;            // Block index % 8 = [0..8)
        int x = item.get_local_id(1);             // [0..24)

        int threadInGroup = y * 24 + x;

        // phase 1 : compute promotion_offsets by multiplying keys and ppo matrices
        const T* keys_start =
            keys + n * 64 * C + C * 56;  // we are interested only in last 8 out of 64
                                         // 'rows' of keys matrix

        // only 32 threads out of 192 in the group are active in this phase, and each
        // thread computes one element of the promotion_offsets matrix
        if (threadInGroup < 32) {
          int x_local = threadInGroup % 4;
          int y_local = threadInGroup / 4;

          float S = 0;
          for (int i = 0; i < C; i++) {
            float a = static_cast<float>(keys_start[y_local * C + i]);
            float b = static_cast<float>(ppo[x_local * C + i]);  // weight matrix is transposed (col major)
            S += a * b;
          }

          // write the product (promotion_offsets) in local memory
          promotion_offsets[x_local][y_local] = S;
        }

        // Synchronize work-group (equivalent to CUDA's __syncthreads())
        item.barrier();

        // phase 2: add the last "row" to the other 3
        // #knight offset is added to the other three
        // promotion_offsets = promotion_offsets[:, :3, :] + promotion_offsets[:, 3:4, :]
        // Only 24 threads in the group are active in this phase
        if (threadInGroup < 32) {
          int x_local = threadInGroup % 4;
          int y_local = threadInGroup / 4;
          if (x_local < 3) {
            promotion_offsets[x_local][y_local] += promotion_offsets[3][y_local];
          }
        }

        // Synchronize work-group again
        item.barrier();

        // phase 3: add 8x8 chunk of policy_attn_logits matrix to promotion offsets
        //          the output is 3x8x8 (written as 8 * 24)
        // All threads are active in this phase and they compute one element each
        int w = x / 3;
        int c = x % 3;

        // n_promo_logits = matmul_qk[:, -16:-8, -8:]  # default traversals from rank
        // 7 to rank 8
        float n_promo_logit =
            static_cast<float>(policy_attn_logits[n * output_stride + (48 + y) * 64 + (56 + w)]);
        float promo_offset = promotion_offsets[c][w];

        float op = n_promo_logit + promo_offset;

        output[n * output_stride + threadInGroup] = static_cast<T>(op);
      });
    }).wait();
  } catch (sycl::exception const& e) {
    std::cerr << "SYCL promotion_logits kernel error: " << e.what() << std::endl;
    throw std::runtime_error("SYCL promotion_logits kernel execution failed");
  }
}

template <typename T>
void addBias_NCHW_kernel(sycl::queue& q, T* c, const T* a, const T* b, int N, int C, int H,
                         int W, ActivationFunction activation) {
  int total_elements = N * C * H * W;
  const int kBlockSize = 256;
  int num_blocks = (total_elements + kBlockSize - 1) / kBlockSize;

  q.submit([&](sycl::handler& h) {
    h.parallel_for(sycl::nd_range<1>(
      sycl::range<1>(num_blocks * kBlockSize),
      sycl::range<1>(kBlockSize)),
      [=](sycl::nd_item<1> item) {
        int i = item.get_global_id(0);
        if (i < total_elements) {
          float aVal = static_cast<float>(a[i]);

          // Calculate bias index: the channel index for current element
          // NCHW layout: index = n*C*H*W + c*H*W + h*W + w
          // biasIndex = (i / (H * W)) % C - this gets the channel index
          int biasIndex = (i / (H * W)) % C;
          float bVal = static_cast<float>(b[biasIndex]);

          float cVal = aVal + bVal;
          cVal = activate(cVal, activation);

          c[i] = static_cast<T>(cVal);
        }
      });
  }).wait();
}

template <typename T>
void addBias_NCHW(sycl::queue& q, T* c, const T* a, const T* b, int N, int C, int H, int W,
                 ActivationFunction activation) {
  try {
    addBias_NCHW_kernel(q, c, a, b, N, C, H, W, activation);
  } catch (sycl::exception const& e) {
    std::cerr << "SYCL exception in addBias_NCHW: " << e.what() << std::endl;
    throw;
  }
}

template <typename T>
void ComputePromotionLogits(sycl::queue& q, int N, int C, T* output, const T* keys,
                            const T* ppo, const T* policy_attn_logits) {
  // N work-groups
  // 8 * 24 work-items per group
  // Each work-item computes a single output element
  promotion_logits_kernel<T>(q, C, output, keys, ppo, policy_attn_logits, N);
}

// Template instantiation for addBias_NCHW - matches CUDA at lines 1392-1398
template void addBias_NCHW<float>(sycl::queue& q, float* c, const float* a, const float* b, int N, int C,
                                 int H, int W, ActivationFunction activation);
template void addBias_NCHW<sycl::half>(sycl::queue& q, sycl::half* c, const sycl::half* a, const sycl::half* b,
                                       int N, int C, int H, int W, ActivationFunction activation);

// Template instantiation for ComputePromotionLogits - matches CUDA at lines 1606-1613
template void ComputePromotionLogits<sycl::half>(sycl::queue& q, int N, int C, sycl::half* output,
                                           const sycl::half* keys, const sycl::half* ppo,
                                           const sycl::half* policy_attn_logits);
template void ComputePromotionLogits<float>(sycl::queue& q, int N, int C, float* output,
                                            const float* keys, const float* ppo,
                                            const float* policy_attn_logits);

// Generate offset pointers for attention mechanism
template <typename T, int kWorkPerThread>
void genOffsetPointers_kernel(sycl::queue& q, T** offsets, int heads, int block_size,
                             int depth, int d_model, T* k, T* q, T* b1, T* v, T* b2) {
  const int kWorkGroupSize = 128;

  // Calculate total work items needed
  const int total_work = (block_size + kWorkPerThread - 1) / kWorkPerThread;

  q.submit([&](sycl::handler& h) {
    h.parallel_for<class genOffsetPointers>(
        sycl::nd_range<1>(
            sycl::range<1>((total_work + kWorkGroupSize - 1) / kWorkGroupSize * kWorkGroupSize),
            sycl::range<1>(kWorkGroupSize)),
        [=](sycl::nd_item<1> item) {
          const int base_idx = (item.get_global_id(0)) * kWorkPerThread;
          if (base_idx >= block_size) return;

          const int h = base_idx % heads;
          const int n = base_idx / heads;

          // Process kWorkPerThread elements per thread
          for (int w = 0; w < kWorkPerThread; ++w) {
            const int i = base_idx + w;
            if (i >= block_size) break;

            // Generate pointers for k
            T* k_ptr = k + h * depth + 64 * d_model * n + w * depth;
            offsets[i + w] = k_ptr;
          }

          // Process pointers for q
          for (int w = 0; w < kWorkPerThread; ++w) {
            const int i = base_idx + w;
            if (i >= block_size) break;

            T* q_ptr = q + h * depth + 64 * d_model * n + w * depth;
            offsets[i + w + block_size] = q_ptr;
          }

          // Process pointers for b1
          for (int w = 0; w < kWorkPerThread; ++w) {
            const int i = base_idx + w;
            if (i >= block_size) break;

            T* b1_ptr = b1 + i * 64 * 64 + w * 64 * 64;
            offsets[i + w + 2 * block_size] = b1_ptr;
          }

          // Process pointers for v
          for (int w = 0; w < kWorkPerThread; ++w) {
            const int i = base_idx + w;
            if (i >= block_size) break;

            T* v_ptr = v + h * depth + 64 * d_model * n + w * depth;
            offsets[i + w + 3 * block_size] = v_ptr;
          }

          // Process pointers for b2
          for (int w = 0; w < kWorkPerThread; ++w) {
            const int i = base_idx + w;
            if (i >= block_size) break;

            T* b2_ptr = b2 + h * depth + 64 * d_model * n + w * depth;
            offsets[i + w + 4 * block_size] = b2_ptr;
          }
        });
  }).wait();
}

template <typename T>
void genOffsetPointers(sycl::queue& q, T** offsets, int heads, int max_batch, int depth,
                      int d_model, T* k, T* q, T* b1, T* v, T* b2) {
  const int block_size = heads * max_batch;
  // Process two elements per thread to use 128 bit store instructions.
  constexpr int kWorkPerThread = 2;

  if (block_size % kWorkPerThread != 0) {
    // Handle odd block sizes.
    genOffsetPointers_kernel<T, 1>(q, offsets, heads, block_size, depth, d_model, k, q, b1, v, b2);
  } else {
    // Handle even block size
    genOffsetPointers_kernel<T, kWorkPerThread>(q, offsets, heads, block_size, depth, d_model, k, q, b1, v, b2);
  }
}

// Template instantiation for genOffsetPointers - matches CUDA at lines 1649-1656
template void genOffsetPointers<float>(sycl::queue& q, float** offsets, int heads, int max_batch,
                                     int depth, int d_model, float* k, float* q, float* b1,
                                     float* v, float* b2);
template void genOffsetPointers<half>(sycl::queue& q, half** offsets, int heads, int max_batch,
                                           int depth, int d_model, half* k, half* q,
                                           half* b1, half* v, half* b2);

}  // namespace sycl_backend
}  // namespace lczero