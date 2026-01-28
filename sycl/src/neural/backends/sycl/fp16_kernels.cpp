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

namespace lczero {
namespace sycl_backend {

/////////////////////////////////////////////////////////////////////////////
//          fp16-specific kernels used by certain layers                   //
/////////////////////////////////////////////////////////////////////////////

// SE layer implementation using single fused kernel - SYCL version
// N blocks.
// C threads per block.
// 'HWC' input data processed by thread block.
// Each thread processes 8x8 elements.
// K is the no. of outputs of first fully connected layer (same as no. of inputs
// for second fully connected layer).
// The kernel assumes K <= C.

template <int C, int K>
class SE_Layer_NHWCKernel;

template <int C, int K>
void SE_Layer_NHWC(sycl::queue& q, sycl::half* output, const sycl::half* skip,
                   const sycl::half* input, const sycl::half* w1, const sycl::half* b1,
                   const sycl::half* w2, const sycl::half* b2, const sycl::half* bPrev,
                   ActivationFunction activation) {
  const int elementsPerThread = 64;  // 8x8 board
  const int se_K = K;

  q.submit([&](sycl::handler& h) {
    // Allocate shared memory for channel averages
    sycl::local_accessor<sycl::half, 1> sharedData(C, h);

    // Kernel range
    sycl::range<1> globalRange(C);  // C threads per block
    sycl::range<1> localRange(C);   // All threads in one work-group

    h.parallel_for<class SE_Layer_NHWCKernel<C, K>>(
        sycl::nd_range<1>(globalRange, localRange),
        [=](sycl::nd_item<1> item) {
          int n = item.get_group(0);  // blockIdx.x equivalent
          int c = item.get_local_id(0);  // threadIdx.x equivalent

          sycl::half2 localData[elementsPerThread];
          sycl::half S = 0;

          sycl::half bias = 0;
          if (bPrev) bias = bPrev[c];

          // 1. Global avg (1 avg per thread).
          #pragma unroll
          for (int i = 0; i < elementsPerThread; i++) {
            int localIndex = i * C + c;
            int inputIndex = n * C * elementsPerThread + localIndex;

            // Store input and skip in half2 for vectorized access
            sycl::half in_val = input[inputIndex] + bias;
            sycl::half skip_val = skip[inputIndex];
            localData[i] = sycl::half2(in_val, skip_val);
            S += in_val;
          }

          sycl::half avg = S / (sycl::half)elementsPerThread;
          sharedData[c] = avg;

          item.barrier();  // __syncthreads() equivalent

          // 2. First fully connected layer.
          if (c < K) {
            S = 0;

            #pragma unroll
            for (int i = 0; i < C; i++) {
              S += sharedData[i] * w1[i * se_K + c];  // readw1(i, c) equivalent
            }

            S += b1[c];
            S = static_cast<sycl::half>(activate(static_cast<float>(S), activation));

            sharedData[c] = S;
          }
          item.barrier();  // __syncthreads() equivalent

          // 3. Second fully connected layer.
          S = 0;
          sycl::half B = 0;
          #pragma unroll
          for (int i = 0; i < K; i++) {
            sycl::half val = sharedData[i];
            S += val * w2[i * 2 * C + c];      // readw2(i, c) equivalent
            B += val * w2[i * 2 * C + (c + C)]; // readw2(i, c + C) equivalent
          }
          S += b2[c];
          B += b2[c + C];

          // Sigmoid (only on the scale part).
          S = static_cast<sycl::half>(1.0f / (1.0f + exp(-static_cast<float>(S))));

          // 4. Scale, and add skip connection, perform relu, and write to output.
          #pragma unroll
          for (int i = 0; i < elementsPerThread; i++) {
            int localIndex = i * C + c;
            int inputIndex = n * C * elementsPerThread + localIndex;

            sycl::half val = localData[i].y() + localData[i].x() * S + B;

            // Relu activation function.
            val = static_cast<sycl::half>(activate(static_cast<float>(val), activation));

            output[inputIndex] = val;
          }
        });
  }).wait();
}

bool Se_Fp16_NHWC(sycl::queue& q, int N, int C, int numFc1Out, sycl::half* output,
                  const sycl::half* skip, const sycl::half* input, const sycl::half* w1,
                  const sycl::half* b1, const sycl::half* w2, const sycl::half* b2,
                  const sycl::half* bPrev, ActivationFunction activation) {
  try {
    // TODO: Think of more elegant way to avoid this hardcoding :-/
    if (numFc1Out == 16) {
      if (C == 64) {
        SE_Layer_NHWC<64, 16>(q, output, skip, input, w1, b1, w2, b2, bPrev, activation);
      } else {
        // TODO: support other channel counts.
        throw Exception("channel count unsupported by SE layer");
      }
    } else if (numFc1Out == 32) {
      if (C == 64) {
        SE_Layer_NHWC<64, 32>(q, output, skip, input, w1, b1, w2, b2, bPrev, activation);
      } else if (C == 128) {
        SE_Layer_NHWC<128, 32>(q, output, skip, input, w1, b1, w2, b2, bPrev, activation);
      } else if (C == 192) {
        SE_Layer_NHWC<192, 32>(q, output, skip, input, w1, b1, w2, b2, bPrev, activation);
      } else if (C == 256) {
        SE_Layer_NHWC<256, 32>(q, output, skip, input, w1, b1, w2, b2, bPrev, activation);
      } else if (C == 320) {
        SE_Layer_NHWC<320, 32>(q, output, skip, input, w1, b1, w2, b2, bPrev, activation);
      } else if (C == 352) {
        SE_Layer_NHWC<352, 32>(q, output, skip, input, w1, b1, w2, b2, bPrev, activation);
      } else if (C == 384) {
        SE_Layer_NHWC<384, 32>(q, output, skip, input, w1, b1, w2, b2, bPrev, activation);
      } else {
        // TODO: support other channel counts.
        return false;
      }
    } else if (numFc1Out == 64) {
      if (C == 64) {
        SE_Layer_NHWC<64, 64>(q, output, skip, input, w1, b1, w2, b2, bPrev, activation);
      } else if (C == 128) {
        SE_Layer_NHWC<128, 64>(q, output, skip, input, w1, b1, w2, b2, bPrev, activation);
      } else if (C == 192) {
        SE_Layer_NHWC<192, 64>(q, output, skip, input, w1, b1, w2, b2, bPrev, activation);
      } else if (C == 256) {
        SE_Layer_NHWC<256, 64>(q, output, skip, input, w1, b1, w2, b2, bPrev, activation);
      } else if (C == 320) {
        SE_Layer_NHWC<320, 64>(q, output, skip, input, w1, b1, w2, b2, bPrev, activation);
      } else if (C == 384) {
        SE_Layer_NHWC<384, 64>(q, output, skip, input, w1, b1, w2, b2, bPrev, activation);
      } else {
        // TODO: support other channel counts.
        return false;
      }
    } else {
      // TODO: support other sizes.
      return false;
    }
  } catch (sycl::exception const& e) {
    std::cerr << "SYCL exception caught: " << e.what() << std::endl;
    return false;
  }
  return true;
}

}  // namespace sycl_backend
}  // namespace lczero