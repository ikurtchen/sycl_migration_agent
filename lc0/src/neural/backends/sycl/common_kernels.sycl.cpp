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

#include <algorithm>
#include <cassert>

#include "sycl_common.h"
#include "sycl_helper.inc"
#include "neural/tables/activation_function.h"
#include "neural/tables/attention_policy_map.h"
#include "utils/exception.h"

namespace lczero {
namespace sycl_backend {
namespace {
constexpr int kInputPlanes = 112;
}  // namespace

/////////////////////////////////////////////////////////////////////////////
//          Simple SYCL kernels used by certain layers                     //
/////////////////////////////////////////////////////////////////////////////

// Adds two vectors (possibly of different sizes), also do optional
// activation (relu, tanh or sigmoid).
template <typename T>
void addVectors(sycl::queue& queue, T* c, T* a, T* b, int size, int asize, int bsize,
                ActivationFunction activation) {
  const int kBlockSize = 256;
  int blocks = DivUp(size, kBlockSize);

  sycl::range<1> global_range(blocks * kBlockSize);
  sycl::range<1> local_range(kBlockSize);

  queue.submit([&](sycl::handler& h) {
    h.parallel_for(sycl::nd_range<1>(global_range, local_range),
      [=](sycl::nd_item<1> item) {
        int i = item.get_global_id(0);
        if (i < size) {
          float aVal = 0;
          float bVal = 0;
          if (a) aVal = static_cast<float>(a[i % asize]);
          if (b) bVal = static_cast<float>(b[i % bsize]);

          float cVal = aVal + bVal;
          cVal = activate(cVal, activation);
          c[i] = static_cast<T>(cVal);
        }
      });
  });

  queue.wait_and_throw();
}

// Adds two vectors of equal size overwriting the first with the sum.
// This specialization performs a transposition of the first 2 indexes
// of the second while performing the addition.
template <typename T>
void addVectorsHNC_NHC(sycl::queue& queue, T* a, T* b, int N, int H, int C) {
  const int kBlockSize = 256;
  int blocks = DivUp(N * H * C, kBlockSize);

  sycl::range<1> global_range(blocks * kBlockSize);
  sycl::range<1> local_range(kBlockSize);

  queue.submit([&](sycl::handler& h) {
    h.parallel_for(sycl::nd_range<1>(global_range, local_range),
      [=](sycl::nd_item<1> item) {
        int i = item.get_global_id(0);
        if (i < N * H * C) {
          int orig_i = i;
          int c = i % C;
          i /= C;
          int n = i % N;
          i /= N;
          int h = i;

          float aVal = static_cast<float>(a[orig_i]);
          float bVal = static_cast<float>(b[n * H * C + h * C + c]);

          float cVal = aVal + bVal;
          a[orig_i] = static_cast<T>(cVal);
        }
      });
  });

  queue.wait_and_throw();
}

// Optimized kernel to add bias to innermost dimension
// and perform optional activation (to be used with GEMMs/fully connected)
template <typename T>
void addBiasBatched(sycl::queue& queue, T* output, const T* input, const T* bias,
                    int Batch, int N, int C, int Nstride, ActivationFunction activation) {
  // process 4 elements per thread to achieve close to peak memory bandwidth
  if (C % 4 != 0) throw Exception("unsupported filter size");
  if (C > 4096) throw Exception("unsupported filter size");

  sycl::range<3> global_range(C / 4, DivUp(N, std::min(std::max(512 / (C / 4), 1), N)), Batch);
  sycl::range<3> local_range(C / 4, std::min(std::max(512 / (C / 4), 1), N), 1);

  switch (activation) {
    case ACTIVATION_NONE:
      queue.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<3>(global_range, local_range),
          [=](sycl::nd_item<3> item) {
            int batch = item.get_global_id(2);
            int n = item.get_global_id(1);
            int c = item.get_local_id(0) * 4;
            if (n >= N) return;

            int biasIndex = batch * C + c;
            int tensorIndex = batch * Nstride * C + n * C + c;

            // Load from memory
            const bool fp16 = std::is_same<sycl::half, T>::value;

            // Vectorized load for better performance
            sycl::vec<T, 4> input_vec, bias_vec;
            if (c + 3 < C) {
              input_vec = *reinterpret_cast<const sycl::vec<T, 4>*>(&input[tensorIndex]);
              bias_vec = *reinterpret_cast<const sycl::vec<T, 4>*>(&bias[biasIndex]);
            } else {
              // Handle edge case for non-multiple of 4
              for (int i = 0; i < 4 && c + i < C; ++i) {
                input_vec[i] = input[tensorIndex + i];
                bias_vec[i] = bias[biasIndex + i];
              }
            }

            // Perform bias add and activation
            sycl::vec<float, 4> val;
            for (int i = 0; i < 4; i++) {
              float x = static_cast<float>(input_vec[i]) + static_cast<float>(bias_vec[i]);
              x = activate(x, activation);
              val[i] = x;
            }

            // Write to memory
            if (c + 3 < C) {
              *reinterpret_cast<sycl::vec<T, 4>*>(&output[tensorIndex]) =
                sycl::vec<T, 4>(static_cast<T>(val[0]), static_cast<T>(val[1]),
                                 static_cast<T>(val[2]), static_cast<T>(val[3]));
            } else {
              for (int i = 0; i < 4 && c + i < C; ++i) {
                output[tensorIndex + i] = static_cast<T>(val[i]);
              }
            }
          });
      });
      break;

    case ACTIVATION_RELU:
      // Similar pattern for other activations...
      queue.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<3>(global_range, local_range),
          [=](sycl::nd_item<3> item) {
            int batch = item.get_global_id(2);
            int n = item.get_global_id(1);
            int c = item.get_local_id(0) * 4;
            if (n >= N) return;

            int biasIndex = batch * C + c;
            int tensorIndex = batch * Nstride * C + n * C + c;

            const bool fp16 = std::is_same<sycl::half, T>::value;
            sycl::vec<T, 4> input_vec, bias_vec;

            if (c + 3 < C) {
              input_vec = *reinterpret_cast<const sycl::vec<T, 4>*>(&input[tensorIndex]);
              bias_vec = *reinterpret_cast<const sycl::vec<T, 4>*>(&bias[biasIndex]);
            } else {
              for (int i = 0; i < 4 && c + i < C; ++i) {
                input_vec[i] = input[tensorIndex + i];
                bias_vec[i] = bias[biasIndex + i];
              }
            }

            sycl::vec<float, 4> val;
            for (int i = 0; i < 4; i++) {
              float x = static_cast<float>(input_vec[i]) + static_cast<float>(bias_vec[i]);
              x = activate(x, ACTIVATION_RELU);  // ReLU: x = max(0, x)
              val[i] = x;
            }

            if (c + 3 < C) {
              *reinterpret_cast<sycl::vec<T, 4>*>(&output[tensorIndex]) =
                sycl::vec<T, 4>(static_cast<T>(val[0]), static_cast<T>(val[1]),
                                 static_cast<T>(val[2]), static_cast<T>(val[3]));
            } else {
              for (int i = 0; i < 4 && c + i < C; ++i) {
                output[tensorIndex + i] = static_cast<T>(val[i]);
              }
            }
          });
      });
      break;

    // Add other activation cases as needed...
    default:
      throw Exception("unsupported activation in addBiasBatched. Add in switch-case here");
  }

  queue.wait_and_throw();
}

// Add bias to convolution's output.
template <typename T>
void addBias_NCHW(sycl::queue& queue, T* c, T* a, T* b, int N, int C, int H, int W,
                  ActivationFunction activation) {
  int size = N * C * H * W;
  const int kBlockSize = 256;
  int blocks = DivUp(size, kBlockSize);

  sycl::range<1> global_range(blocks * kBlockSize);
  sycl::range<1> local_range(kBlockSize);

  queue.submit([&](sycl::handler& h) {
    h.parallel_for(sycl::nd_range<1>(global_range, local_range),
      [=](sycl::nd_item<1> item) {
        int i = item.get_global_id(0);
        if (i < size) {
          float aVal = static_cast<float>(a[i]);

          // All this math can be optimized, but the kernel is memory bound anyway.
          int biasIndex = (i / (H * W)) % C;
          float bVal = static_cast<float>(b[biasIndex]);

          float cVal = aVal + bVal;
          cVal = activate(cVal, activation);

          c[i] = static_cast<T>(cVal);
        }
      });
  });

  queue.wait_and_throw();
}

// Conversion from NCHW to NHWC
template <typename DstType, typename SrcType>
void convertNCHWtoNHWC(sycl::queue& queue, DstType* output_tensor, const SrcType* input_tensor,
                       int Nin, int Cin, int Nout, int Cout, int H, int W) {
  size_t numElements = Nout * Cout * H * W;
  const int blockSize = 256;
  int blocks = DivUp(numElements, blockSize);

  sycl::range<1> global_range(blocks * blockSize);
  sycl::range<1> local_range(blockSize);

  queue.submit([&](sycl::handler& h) {
    h.parallel_for(sycl::nd_range<1>(global_range, local_range),
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

        if (n >= Nin || c >= Cin) {
          output_tensor[tid] = 0;
        } else {
          int input_index = n;
          input_index *= Cin;
          input_index += c;
          input_index *= H;
          input_index += h;
          input_index *= W;
          input_index += w;

          output_tensor[tid] = static_cast<DstType>(input_tensor[input_index]);
        }
      });
  });

  queue.wait_and_throw();
}

// Plain data-type conversion (no layout conversion).
template <typename DstType, typename SrcType>
void copyTypeConverted(sycl::queue& queue, DstType* op, SrcType* ip, int N) {
  const int blockSize = 256;
  int blocks = DivUp(N, blockSize);

  sycl::range<1> global_range(blocks * blockSize);
  sycl::range<1> local_range(blockSize);

  queue.submit([&](sycl::handler& h) {
    h.parallel_for(sycl::nd_range<1>(global_range, local_range),
      [=](sycl::nd_item<1> item) {
        int tid = item.get_global_id(0);
        if (tid >= N) return;
        op[tid] = static_cast<DstType>(ip[tid]);
      });
  });

  queue.wait_and_throw();
}

// Unpack planes for NHWC layout
template <typename T>
void expandPlanes_NHWC(sycl::queue& queue, T* output, const uint64_t* masks, const T* values, int n) {
  int threads = n * 8 * 8;  // Each thread writes a single element.
  const int kBlockSize = 256;
  int blocks = DivUp(threads, kBlockSize);

  sycl::range<1> global_range(blocks * kBlockSize);
  sycl::range<1> local_range(kBlockSize);

  queue.submit([&](sycl::handler& h) {
    h.parallel_for(sycl::nd_range<1>(global_range, local_range),
      [=](sycl::nd_item<1> item) {
        int index = item.get_global_id(0);
        if (index >= n * 8 * 8) return;

        const int planeIndex = index % kInputPlanes;
        const int boardIndex = index / (kInputPlanes * 8 * 8);
        const int sqIndex = (index / kInputPlanes) & 0x3F;

        uint64_t mask = masks[boardIndex * kInputPlanes + planeIndex];

        T op_val = 0;
        bool set = !!(mask & (1ull << sqIndex));
        if (set) {
          op_val = values[boardIndex * kInputPlanes + planeIndex];
        }
        output[index] = op_val;
      });
  });

  queue.wait_and_throw();
}

// Unpack planes for NCHW layout
template <typename T>
void expandPlanes_NCHW(sycl::queue& queue, T* output, const uint64_t* masks, const T* values, int n) {
  unsigned threads = n * 8 * 8 / 2;  // each thread writes two elements.
  const int blockSize = 256;
  unsigned blocks = DivUp(threads, blockSize);

  sycl::range<1> global_range(blocks * blockSize);
  sycl::range<1> local_range(blockSize);

  queue.submit([&](sycl::handler& h) {
    h.parallel_for(sycl::nd_range<1>(global_range, local_range),
      [=](sycl::nd_item<1> item) {
        unsigned index = item.get_global_id(0);
        index *= 2;
        unsigned planeIndex = index >> 6;

        if (planeIndex >= n) return;

        uint64_t mask = masks[planeIndex];

        int sqIndex = index & 0x3F;
        T op[2] = {0, 0};

        bool set = !!(mask & (1ull << sqIndex));
        if (set) {
          op[0] = values[planeIndex];
        }
        sqIndex++;
        set = !!(mask & (1ull << sqIndex));
        if (set) {
          op[1] = values[planeIndex];
        }
        output[index + 0] = op[0];
        output[index + 1] = op[1];
      });
  });

  queue.wait_and_throw();
}

// Global Average Pool kernel for NHWC with FP16 optimization
template <typename T>
void globalAvgPool_NHWC_fp16_kernel(sycl::queue& queue, sycl::half* output, const sycl::half* input,
                                   const sycl::half* prevLayerBias, int inputSize, int outputSize) {
  const int elementsPerThread = 64;  // 8x8 board.

  sycl::range<1> global_range(inputSize);
  sycl::range<1> local_range(elementsPerThread);

  queue.submit([&](sycl::handler& h) {
    h.parallel_for(sycl::nd_range<1>(global_range, local_range),
      [=](sycl::nd_item<1> item) {
        int localIndex = item.get_local_id(0);
        int globalIndex = item.get_global_id(0);

        if (globalIndex >= inputSize) return;

        float S = 0;

        // Each thread processes elementsPerThread elements
        for (int i = 0; i < elementsPerThread; i++) {
          int inputIndex = item.get_group(0) * elementsPerThread * elementsPerThread + i * elementsPerThread + localIndex;
          if (inputIndex < inputSize) {
            S += static_cast<float>(input[inputIndex]);
          }
        }

        float avg = S / elementsPerThread;

        // Add bias from previous layer.
        if (prevLayerBias && localIndex < outputSize / item.get_group_range(0)) {
          int biasIndex = item.get_group(0) * elementsPerThread + localIndex;
          if (biasIndex < outputSize) {
            avg += static_cast<float>(prevLayerBias[biasIndex]);
          }
        }

        int opIndex = item.get_group(0) * elementsPerThread + localIndex;
        if (opIndex < outputSize) {
          output[opIndex] = sycl::half(avg);
        }
      });
  });
}

// Global Average Pool kernel with SYCL subgroup optimization for NCHW
template <typename T>
void globalAvgPool_kernel(sycl::queue& queue, T* output, const T* input,
                         const T* prevLayerBias, int inputSize, int outputSize, int C) {
  const int elementsPerSubgroup = 64;
  const int elementsPerThread = 2;
  const int subGroupSize = getSubGroupSize(queue);

  sycl::range<2> global_range(DivUp(inputSize, elementsPerThread * subGroupSize) * subGroupSize, subGroupSize);
  sycl::range<2> local_range(subGroupSize, subGroupSize);

  queue.submit([&](sycl::handler& h) {
    h.parallel_for(sycl::nd_range<2>(global_range, local_range),
      [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(16)]] {
        auto sg = item.get_sub_group();
        int lid = sg.get_local_id()[0];
        int gid = item.get_global_id(0);

        int laneStartIndex = (gid - lid) * elementsPerThread;

        // Compute per-thread sum for elementsPerThread elements.
        float S = 0;

        for (int i = 0; i < elementsPerSubgroup; i += subGroupSize) {
          int index = laneStartIndex + lid + i;
          if (index < inputSize) {
            S += static_cast<float>(input[index]);
          }
        }

        // Compute subgroup-wide sum using SYCL reduction
        float subgroup_sum = subgroup_reduce(sg, S);

        float avg = subgroup_sum / elementsPerSubgroup;
        int opIndex = gid / subGroupSize;

        // First thread in subgroup has the sum, write it in output.
        if (lid == 0 && opIndex < outputSize) {
          output[opIndex] = static_cast<T>(avg);
        }
      });
  });
}

template <typename T>
void globalAvgPool(sycl::queue& queue, int N, int C, T* output, const T* input,
                   const T* prevLayerBias, bool nhwc) {
  const int kPlaneSize = 64;

  if (nhwc) {
    assert((std::is_same<sycl::half, T>::value));
    // For NHWC fp16, simply launch N blocks, each with C threads.
    globalAvgPool_NHWC_fp16_kernel(queue,
        reinterpret_cast<sycl::half*>(output),
        reinterpret_cast<const sycl::half*>(input),
        reinterpret_cast<const sycl::half*>(prevLayerBias),
        N * C * kPlaneSize, N * C);
  } else {
    // For NCHW layout (used with fp32),
    // each subgroup processes a full plane (64 elements), and writes a single average
    const int kTotalSubgroups = N * C;
    const int kSubgroupsPerWorkGroup = 8;
    const int kWorkGroupSize = kSubgroupsPerWorkGroup * 16; // 16 is typical subgroup size

    int blocks = DivUp(kTotalSubgroups, kSubgroupsPerWorkGroup);
    globalAvgPool_kernel(queue, output, input, prevLayerBias,
                        N * C * kPlaneSize, N * C, C);
  }

  queue.wait_and_throw();
}

// Policy mapping kernel - chess-specific optimization
template <typename T>
void PolicyMap(sycl::queue& queue, int N, T* output, const T* input, const short* indices,
               int inputSize, int usedSize, int outputSize) {
  // Each thread processes one input element
  // Only some of the threads (with valid mapping) write output
  const int kBlockSize = 256;
  const int kBlocks = DivUp(N * usedSize, kBlockSize);

  sycl::range<1> global_range(kBlocks * kBlockSize);
  sycl::range<1> local_range(kBlockSize);

  queue.submit([&](sycl::handler& h) {
    h.parallel_for(sycl::nd_range<1>(global_range, local_range),
      [=](sycl::nd_item<1> item) {
        int tid = item.get_global_id(0);

        int n = tid / usedSize;
        int i = tid % usedSize;

        if (n >= N) return;

        int j = indices[i];

        if (j >= 0) {
          output[n * outputSize + j] = input[n * inputSize + i];
        }
      });
  });

  queue.wait_and_throw();
}

// Specialized Softmax kernel with SYCL optimization
template <typename T>
void softmax_kernel(sycl::queue& queue, T* output, const T* input, const T* input2, int N, int C) {
  // Softmax computation with shared memory reduction optimized for SYCL
  sycl::range<2> global_range(N, C);
  sycl::range<2> local_range(1, C);  // Each work group handles one C-dimension

  queue.submit([&](sycl::handler& h) {
    // Local memory for reduction
    auto sum_acc = sycl::local_accessor<float, 1>(1, h);
    auto maxval_acc = sycl::local_accessor<float, 1>(1, h);

    h.parallel_for(sycl::nd_range<2>(global_range, local_range),
      [=](sycl::nd_item<2> item) {
        auto sg = item.get_sub_group();
        int n = item.get_group(0);
        int c = item.get_local_id(1);

        int index = n * C + c;

        // softmax = tf.exp(logits) / tf.reduce_sum(tf.exp(logits), axis)
        float x = static_cast<float>(input[index]);
        if (input2 != nullptr) x += static_cast<float>(input2[index]);

        if (std::is_same<sycl::half, T>::value) {
          // Guard against Inf from fp16 overflow.
          x = clamp_for_fp16(x);
        }

        // Initialize shared memory
        if (c == 0) {
          sum_acc[0] = 0;
          maxval_acc[0] = x;
        }

        item.barrier(sycl::access::fence_space::local_space);

        // Get max across subgroup first, and then update across C dimension
        float subgroup_max = subgroup_max(sg, x);

        // Update global max using atomic operation or reduction
        if (sg.get_local_id()[0] == 0) {
          // Use atomic compare-and-swap for max
          sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::work_group,
                          sycl::access::address_space::local_space> max_ref(maxval_acc[0]);
          float current = max_ref.load();
          float expected = current;
          do {
            expected = current;
            current = sycl::atomic_compare_exchange(max_ref, expected, subgroup_max > expected ? subgroup_max : expected);
          } while (current != expected);
        }

        item.barrier(sycl::access::fence_space::local_space);

        float max_val = maxval_acc[0];
        float ex = sycl::exp(x - max_val);

        // Compute subgroup sum first
        float subgroup_sum = subgroup_reduce(sg, ex);

        // Update global sum
        if (sg.get_local_id()[0] == 0) {
          sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::work_group,
                          sycl::access::address_space::local_space> sum_ref(sum_acc[0]);
          sum_ref.fetch_add(subgroup_sum);
        }

        item.barrier(sycl::access::fence_space::local_space);

        float sum = sum_acc[0];
        float op = ex / sum;

        output[index] = static_cast<T>(op);
      });
  });

  queue.wait_and_throw();
}

template <typename T>
void Softmax(sycl::queue& queue, int N, int C, T* output, const T* input, const T* input2) {
  int size = N * C;
  const int kBlockSize = 256;

  if (C == 1858) {  // Policy head specific optimization
    int blocks = DivUp(size, kBlockSize);
    // Note: A specialized optimized kernel for C=1858 would go here
    // For now, using the generic kernel
    softmax_kernel(queue, output, input, input2, N, C);
  } else {
    softmax_kernel(queue, output, input, input2, N, C);
  }
}

// Layer normalization kernel with SYCL optimization
template <typename T>
void layer_norm_kernel(sycl::queue& queue, int N, int C, T* output, const T* input,
                       const T* bias, const T* skip, const T* gammas, const T* betas,
                       float ep, float alpha, ActivationFunction act) {
  // Each thread processes 4 elements
  // 1. Perform Bias add, and skip add
  // 2. Perform layer norm (normalize across C dimension)

  // Get optimal work-group size for the device
  int wg_size = getOptimalWorkGroupSize(queue);
  int subgroup_size = getSubGroupSize(queue);

  // Work group dimensions: (C/16) threads in X direction, subgroups in Y direction
  sycl::range<3> global_range(DivUp(C, 16) * subgroup_size, DivUp(N, wg_size / DIVUP(C, 16)), 1);
  sycl::range<3> local_range(subgroup_size, wg_size / subgroup_size, 1);

  queue.submit([&](sycl::handler& h) {
    // Local memory for reduction across C dimension
    auto sum_acc = sycl::local_accessor<float, 2>(sycl::range<2>(16, 16), h);

    h.parallel_for(sycl::nd_range<3>(global_range, local_range),
      [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(16)]] {
        auto sg = item.get_sub_group();
        int n = item.get_global_id(2);
        if (n >= N) return;

        int c = (item.get_local_id(1) * subgroup_size + item.get_local_id(0)) * 16;
        bool oobThread = c >= C;

        int biasIndex = c;
        int tensorIndex = n * C + c;

        sycl::vec<float, 16> val{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        sycl::vec<float, 16> oth{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

        const bool fp16 = std::is_same<sycl::half, T>::value;

        if (!oobThread) {
          // Load from memory (16 elements at a time) with vectorization
          if (c + 15 < C) {
            auto input_vec = *reinterpret_cast<const sycl::vec<T, 16>*>(&input[tensorIndex]);
            auto bias_vec = *reinterpret_cast<const sycl::vec<T, 16>*>(&bias[biasIndex]);

            for (int i = 0; i < 16; i++) {
              val[i] = static_cast<float>(input_vec[i]);
              oth[i] = static_cast<float>(bias_vec[i]);
              val[i] += oth[i];  // Add bias
            }
          } else {
            // Handle edge case
            for (int i = 0; i < 16 && c + i < C; i++) {
              val[i] = static_cast<float>(input[tensorIndex + i]);
              oth[i] = static_cast<float>(bias[biasIndex + i]);
              val[i] += oth[i];
            }
          }

          // Add skip connection if available
          if (skip) {
            if (c + 15 < C) {
              auto skip_vec = *reinterpret_cast<const sycl::vec<T, 16>*>(&skip[tensorIndex]);
              for (int i = 0; i < 16; i++) {
                val[i] += static_cast<float>(skip_vec[i]);
              }
            } else {
              for (int i = 0; i < 16 && c + i < C; i++) {
                val[i] += static_cast<float>(skip[tensorIndex + i]);
              }
            }
          }

          // Compute per-thread sum and sum of squares for layer norm
          float thread_sum = 0;
          float thread_sum_sq = 0;

          for (int i = 0; i < 16; i++) {
            thread_sum += val[i];
            thread_sum_sq += val[i] * val[i];
          }

          // Compute subgroup-wide sum
          float subgroup_sum = subgroup_reduce(sg, thread_sum);
          float subgroup_sum_sq = subgroup_reduce(sg, thread_sum_sq);

          // Store partial sum in shared memory
          if (sg.get_local_id()[0] == 0) {
            sum_acc[item.get_global_id(1)][item.get_local_id(1)] = subgroup_sum;
          }

          item.barrier(sycl::access::fence_space::local_space);

          // Compute total sum across all subgroups
          if (item.get_local_id(0) == 0 && item.get_local_id(1) == 0) {
            float total_sum = 0;
            for (int j = 0; j < item.get_local_range(1); j++) {
              total_sum += sum_acc[item.get_global_id(1)][j];
            }
            sum_acc[item.get_global_id(1)][0] = total_sum;
          }

          item.barrier(sycl::access::fence_space::local_space);

          // Now compute mean and variance
          float total_sum = sum_acc[item.get_global_id(1)][0];
          float mean = total_sum / C;

          // Compute variance
          if (sg.get_local_id()[0] == 0) {
            sum_acc[item.get_global_id(1)][1] = subgroup_sum_sq;
          }

          item.barrier(sycl::access::fence_space::local_space);

          if (item.get_local_id(0) == 0 && item.get_local_id(1) == 0) {
            float total_sum_sq = 0;
            for (int j = 0; j < item.get_local_range(1); j++) {
              total_sum_sq += sum_acc[item.get_global_id(1)][j + 1];
            }
            sum_acc[item.get_global_id(1)][0] = total_sum_sq;
            sum_acc[item.get_global_id(1)][1] = total_sum_sq / C - mean * mean + ep;
          }

          item.barrier(sycl::access::fence_space::local_space);

          float var = sum_acc[item.get_global_id(1)][1];
          float std_dev = sycl::sqrt(var);

          // Apply layer norm: y = ((x - mean) / std_dev) * gamma + beta
          for (int i = 0; i < 16 && c + i < C; i++) {
            float normalized = (val[i] - mean) / std_dev;
            if (gammas) normalized *= static_cast<float>(gammas[c + i]);
            if (betas) normalized += static_cast<float>(betas[c + i]);

            // Apply activation if needed
            normalized = activate(normalized, act);

            val[i] = normalized;
          }

          // Write to memory with vectorization
          if (c + 15 < C) {
            *reinterpret_cast<sycl::vec<T, 16>*>(&output[tensorIndex]) =
              sycl::vec<T, 16>(static_cast<T>(val[0]), static_cast<T>(val[1]),
                               static_cast<T>(val[2]), static_cast<T>(val[3]),
                               static_cast<T>(val[4]), static_cast<T>(val[5]),
                               static_cast<T>(val[6]), static_cast<T>(val[7]),
                               static_cast<T>(val[8]), static_cast<T>(val[9]),
                               static_cast<T>(val[10]), static_cast<T>(val[11]),
                               static_cast<T>(val[12]), static_cast<T>(val[13]),
                               static_cast<T>(val[14]), static_cast<T>(val[15]));
          } else {
            for (int i = 0; i < 16 && c + i < C; i++) {
              output[tensorIndex + i] = static_cast<T>(val[i]);
            }
          }
        }
      });
  });

  queue.wait_and_throw();
}

template <typename T>
void LayerNorm(sycl::queue& queue, int N, int C, T* output, const T* input, const T* bias,
               const T* skip, const T* gammas, const T* betas, float ep,
               float alpha, ActivationFunction act) {
  // Each thread processes 16 elements to maximize vectorization
  int wg_size = getOptimalWorkGroupSize(queue);

  sycl::range<3> global_range(DivUp(C, 16) * 32, DivUp(N, wg_size / 32), 1);
  sycl::range<3> local_range(32, wg_size / 32, 1);

  layer_norm_kernel(queue, N, C, output, input, bias, skip, gammas, betas, ep, alpha, act);
}

// Add more overloaded versions as needed for different signatures...

}  // namespace sycl_backend
}  // namespace lczero