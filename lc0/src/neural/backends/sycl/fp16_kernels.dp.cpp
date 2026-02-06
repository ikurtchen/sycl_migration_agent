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

#include "sycl_common.h"
#include "neural/tables/activation_function.h"
#include "utils/exception.h"
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

namespace lczero {
namespace sycl_backend {

// Missing macros from CUDA version
#define TEMP_INDEX_HWNC(y, x, n, c, H, W, C) ((n) * (C) * (H) * (W) + (c) * (H) * (W) + (y) * (W) + (x))
#define INDEX_NHCW(n, c, h, w, C, H, W) ((n) * (C) * (H) * (W) + (c) * (H) * (W) + (h) * (W) + (w))

// Activate function helper (use apply_activation from sycl_common.h)
inline float activate(float x, ActivationFunction activation) {
  return apply_activation(x, activation);
}

/////////////////////////////////////////////////////////////////////////////
//          fp16-specific kernels used by certain layers                   //
/////////////////////////////////////////////////////////////////////////////

// SE layer implementation using single fused kernel.
// SYCL version with Intel GPU optimizations

template <int C, int K>
void se_layer_nhwc_kernel(sycl::queue& queue, sycl::half* output,
                          const sycl::half* skip, const sycl::half* input,
                          const sycl::half* w1, const sycl::half* b1,
                          const sycl::half* w2, const sycl::half* b2,
                          const sycl::half* bPrev, ActivationFunction activation,
                          int n_blocks) {

  const int elementsPerThread = 64;  // 8x8 board
  const int se_K = K;

  queue.submit([&](sycl::handler& cgh) {
    // Local memory for averaging data
    sycl::local_accessor<sycl::half, 1> sharedData{sycl::range<1>(C), cgh};

    cgh.parallel_for(sycl::nd_range<1>{
      sycl::range<1>(n_blocks * C),
      sycl::range<1>(C)
    }, [=](sycl::nd_item<1> item) [[intel::reqd_sub_group_size(16)]] {
      int n = item.get_group(0);
      int c = item.get_local_id(0);

      sycl::vec<sycl::half, 2> localData[elementsPerThread];
      sycl::half S = sycl::half(0.0f);

      sycl::half bias = sycl::half(0.0f);
      if (bPrev) bias = bPrev[c];

      // 1. Global avg (1 avg per thread)
      for (int i = 0; i < elementsPerThread; i++) {
        int localIndex = i * C + c;
        int inputIndex = n * C * elementsPerThread + localIndex;
        localData[i][0] = input[inputIndex] + bias;
        localData[i][1] = skip[inputIndex];
        S += localData[i][0];
      }

      sycl::half avg = S / sycl::half(elementsPerThread);
      sharedData[c] = avg;

      sycl::group_barrier(item.get_group());

      // 2. First fully connected layer
      if (c < K) {
        S = sycl::half(0.0f);

        for (int i = 0; i < C; i++) {
          // readw1 equivalent - using weight matrix logic
          S += sharedData[i] * w1[i * K + c];
        }

        S += b1[c];
        S = sycl::half(activate(float(S), activation));
        sharedData[c] = S;
      }

      sycl::group_barrier(item.get_group());

      // 3. Second fully connected layer
      S = sycl::half(0.0f);
      sycl::half B = sycl::half(0.0f);
      for (int i = 0; i < K; i++) {
        sycl::half val = sharedData[i];
        S += val * w2[i * C + c];
        B += val * w2[i * C + (c + C)];
      }
      S += b2[c];
      B += b2[c + C];

      // Sigmoid (only on the scale part)
      S = sycl::half(1.0f / (1.0f + sycl::exp(-float(S))));

      // 4. Scale, and add skip connection, perform relu, and write to output
      for (int i = 0; i < elementsPerThread; i++) {
        int localIndex = i * C + c;
        int inputIndex = n * C * elementsPerThread + localIndex;
        sycl::half val = localData[i][1] + localData[i][0] * S + B;

        // Relu activation function
        val = sycl::half(activate(float(val), activation));

        output[inputIndex] = val;
      }
    });
  });
}

bool Se_Fp16_NHWC(int N, int C, int numFc1Out, sycl::half* output, const sycl::half* skip,
                  const sycl::half* input, const sycl::half* w1, const sycl::half* b1,
                  const sycl::half* w2, const sycl::half* b2, const sycl::half* bPrev,
                  ActivationFunction activation, sycl::queue& queue) {

  switch (numFc1Out) {
    case 16:
      if (C == 64) {
        se_layer_nhwc_kernel<64, 16>(queue, output, skip, input, w1, b1, w2, b2, bPrev, activation, N);
      } else {
        throw Exception("channel count unsupported by SE layer");
      }
      break;
    case 32:
      switch (C) {
        case 64:
          se_layer_nhwc_kernel<64, 32>(queue, output, skip, input, w1, b1, w2, b2, bPrev, activation, N);
          break;
        case 128:
          se_layer_nhwc_kernel<128, 32>(queue, output, skip, input, w1, b1, w2, b2, bPrev, activation, N);
          break;
        case 192:
          se_layer_nhwc_kernel<192, 32>(queue, output, skip, input, w1, b1, w2, b2, bPrev, activation, N);
          break;
        case 256:
          se_layer_nhwc_kernel<256, 32>(queue, output, skip, input, w1, b1, w2, b2, bPrev, activation, N);
          break;
        case 320:
          se_layer_nhwc_kernel<320, 32>(queue, output, skip, input, w1, b1, w2, b2, bPrev, activation, N);
          break;
        case 352:
          se_layer_nhwc_kernel<352, 32>(queue, output, skip, input, w1, b1, w2, b2, bPrev, activation, N);
          break;
        case 384:
          se_layer_nhwc_kernel<384, 32>(queue, output, skip, input, w1, b1, w2, b2, bPrev, activation, N);
          break;
        default:
          return false;
      }
      break;
    case 64:
      switch (C) {
        case 64:
          se_layer_nhwc_kernel<64, 64>(queue, output, skip, input, w1, b1, w2, b2, bPrev, activation, N);
          break;
        case 128:
          se_layer_nhwc_kernel<128, 64>(queue, output, skip, input, w1, b1, w2, b2, bPrev, activation, N);
          break;
        case 192:
          se_layer_nhwc_kernel<192, 64>(queue, output, skip, input, w1, b1, w2, b2, bPrev, activation, N);
          break;
        case 256:
          se_layer_nhwc_kernel<256, 64>(queue, output, skip, input, w1, b1, w2, b2, bPrev, activation, N);
          break;
        case 320:
          se_layer_nhwc_kernel<320, 64>(queue, output, skip, input, w1, b1, w2, b2, bPrev, activation, N);
          break;
        case 384:
          se_layer_nhwc_kernel<384, 64>(queue, output, skip, input, w1, b1, w2, b2, bPrev, activation, N);
          break;
        default:
          return false;
      }
      break;
    default:
      return false;
  }

  return true;
}

// Get board for this thread from local memory.
// We are just using local memory to store local thread data in this kernel to
// help reduce some register pressure and spills to local memory.
#define BOARD(y, x) shboard[(y)*8 + (x)]

template <ActivationFunction activation, bool use_bias, bool use_skip>
void output_input_transform_fp16_shmem_board_kernel(sycl::queue& queue, int N, int C, int se_K,
                                                  sycl::half* output, const sycl::half* input,
                                                  const sycl::half* skip, const sycl::half* bias,
                                                  const sycl::half* w1, const sycl::half* b1,
                                                  const sycl::half* w2, const sycl::half* b2) {

  queue.submit([&](sycl::handler& cgh) {
    // Extent of shared memory: 72 elements per thread to reduce bank conflicts
    const int shared_mem_size = 72 * C;
    sycl::local_accessor<sycl::half, 1> shared_mem{sycl::range<1>(shared_mem_size), cgh};

    // Additional shared memory for SE computations
    sycl::local_accessor<float, 1> shared_data{sycl::range<1>(C), cgh};
    sycl::local_accessor<float, 1> shared_sums{sycl::range<1>(C), cgh};

    cgh.parallel_for(sycl::nd_range<1>{
      sycl::range<1>(N * C),
      sycl::range<1>(C)
    }, [=](sycl::nd_item<1> item) [[intel::reqd_sub_group_size(16)]] {
      int k = item.get_local_id(0);
      int n = item.get_group(0);

      // Allocate board memory for this thread
      sycl::half* shboard = &shared_mem[0] + k * 72;
      sycl::half b = bias[k];

      // Process 4x4 tiles
      for (int hStart = 0; hStart < 8; hStart += 4) {
        for (int wStart = 0; wStart < 8; wStart += 4) {
          // Read to registers (output transform)
          int shln = n * 4 + (hStart / 4) * 2 + (wStart / 4);
          sycl::half outElTransformed[6][6];

          for (int y = 0; y < 6; y++) {
            for (int x = 0; x < 6; x++) {
              outElTransformed[y][x] = input[TEMP_INDEX_HWNC(y, x, shln, k, 6, 6, C)];
            }
          }

          // Transform it
          sycl::half outEl[4][4];
          // OutputTransform4x4 equivalent
          // (Implementation would be in sycl_helper.inc)

          for (int y = 0; y < 4; y++) {
            for (int x = 0; x < 4; x++) {
              BOARD(hStart + y, wStart + x) = outEl[y][x];
            }
          }
        }
      }

      // Add bias, and compute the average for SE
      float S = 0.0f;
      float B = 0.0f;

      for (int y = 0; y < 8; y++) {
        for (int x = 0; x < 8; x++) {
          if (use_bias) BOARD(y, x) += b;
          S += float(BOARD(y, x));
        }
      }

      float avg = S / 64.0f;
      shared_data[k] = avg;

      sycl::group_barrier(item.get_group());

      // First fully-connected layer for SE
      if (k < se_K) {
        S = 0.0f;
        for (int i = 0; i < C; i++) {
          S += shared_data[i] * float(w1[i * se_K + k]);
        }

        S += float(b1[k]);
        S = activate(S, activation);
        shared_data[k] = S;
      }

      sycl::group_barrier(item.get_group());

      // Second fully-connected layer for SE
      S = 0.0f;
      for (int i = 0; i < se_K; i++) {
        float val = shared_data[i];
        S += val * float(w2[i * C + k]);
        B += val * float(w2[i * C + (k + C)]);
      }
      S += float(b2[k]);
      B += float(b2[k + C]);

      // Sigmoid (only on the scale part)
      S = 1.0f / (1.0f + sycl::exp(-S));

      // Scale/bias, add skip connection, perform activation
      for (int h = 0; h < 8; h++) {
        for (int w = 0; w < 8; w++) {
          BOARD(h, w) = sycl::half(float(BOARD(h, w)) * S + B);
        }

        // Residual add
        if (use_skip) {
          for (int w = 0; w < 8; w++) {
            BOARD(h, w) += skip[INDEX_NHCW(n, k, h, w, C, 8, 8)];
          }
        }

        // Activation
        if (activation != ACTIVATION_NONE) {
          for (int w = 0; w < 8; w++) {
            BOARD(h, w) = sycl::half(activate(float(BOARD(h, w)), activation));
          }
        }
      }

      // Input transform - simplified version
      // Full Winograd transform implementation would be here
      for (int y = 0; y < 6; y++) {
        for (int x = 0; x < 6; x++) {
          // InputTransform4x4 equivalent
          // Four quadrants: top-left, top-right, bottom-left, bottom-right
          output[TEMP_INDEX_HWNC(y, x, n * 4 + 0, k, 6, 6, C)] = sycl::half(0.0f); // placeholder
          output[TEMP_INDEX_HWNC(y, x, n * 4 + 1, k, 6, 6, C)] = sycl::half(0.0f); // placeholder
          output[TEMP_INDEX_HWNC(y, x, n * 4 + 2, k, 6, 6, C)] = sycl::half(0.0f); // placeholder
          output[TEMP_INDEX_HWNC(y, x, n * 4 + 3, k, 6, 6, C)] = sycl::half(0.0f); // placeholder
        }
      }
    });
  });
}

template <typename T, bool use_se, ActivationFunction activation,
          bool use_bias, bool use_skip>
void OutputInputTransform(int N, int C, int se_K, T* output, const T* input,
                        const T* skip, const T* bias, const T* w1,
                        const T* b1, const T* w2, const T* b2,
                        sycl::queue& queue) {
  // Each thread processes entire chess board
  if (use_se == false) {
    // Use basic transform without SE
    // Implementation would call basic OutputTransform/InputTransform kernels
  } else {
    // Use special kernel with SE
    output_input_transform_fp16_shmem_board_kernel<activation, use_bias, use_skip>(
        queue, N, C, se_K, output, input, skip, bias, w1, b1, w2, b2);
  }
}

// Template instantiations
template void OutputInputTransform<sycl::half, true, ACTIVATION_RELU, true, true>(
    int N, int C, int se_K, sycl::half* output, const sycl::half* input,
    const sycl::half* skip, const sycl::half* bias, const sycl::half* w1,
    const sycl::half* b1, const sycl::half* w2, const sycl::half* b2,
    sycl::queue& queue);

template void OutputInputTransform<sycl::half, false, ACTIVATION_RELU, true, true>(
    int N, int C, int se_K, sycl::half* output, const sycl::half* input,
    const sycl::half* skip, const sycl::half* bias, const sycl::half* w1,
    const sycl::half* b1, const sycl::half* w2, const sycl::half* b2,
    sycl::queue& queue);

template void OutputInputTransform<sycl::half, true, ACTIVATION_MISH, true, true>(
    int N, int C, int se_K, sycl::half* output, const sycl::half* input,
    const sycl::half* skip, const sycl::half* bias, const sycl::half* w1,
    const sycl::half* b1, const sycl::half* w2, const sycl::half* b2,
    sycl::queue& queue);

template void OutputInputTransform<sycl::half, false, ACTIVATION_MISH, true, true>(
    int N, int C, int se_K, sycl::half* output, const sycl::half* input,
    const sycl::half* skip, const sycl::half* bias, const sycl::half* w1,
    const sycl::half* b1, const sycl::half* w2, const sycl::half* b2,
    sycl::queue& queue);

}  // namespace sycl_backend
}  // namespace lczero