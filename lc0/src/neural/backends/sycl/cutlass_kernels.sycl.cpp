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
#include "utils/exception.h"
#include <sycl/sycl.hpp>
#include <math.h>

namespace lczero {
namespace sycl_backend {

/////////////////////////////////////////////////////////////////////////////
//          Multi-Head Attention Implementation (SYCL)                     //
/////////////////////////////////////////////////////////////////////////////

// Multi-head attention kernel implementation from scratch
// Since CUTLASS doesn't have a SYCL equivalent, we implement from first principles

// Basic MHA implementation without CUTLASS dependencies
template <bool bias>
class fused_mha_kernel {
public:
  fused_mha_kernel(sycl::queue& queue, void* output, void* q, void* k, void* v,
                   void* skip, int batch_size, int num_heads, int depth) {
    constexpr int kQueriesPerBlock = 64;
    constexpr int kKeysPerBlock = 64;
    constexpr int kSingleValueIteration = true;

    sycl::half* mha_q = static_cast<sycl::half*>(q);
    sycl::half* mha_k = static_cast<sycl::half*>(k);
    sycl::half* mha_v = static_cast<sycl::half*>(v);
    sycl::half* mha_output = static_cast<sycl::half*>(output);
    sycl::half* mha_skip = static_cast<sycl::half*>(skip);

    // Key parameters from CUTLASS version
    const float scale = 1.0f / sqrt(static_cast<float>(depth));
    const int num_queries = 64;
    const int num_keys = 64;

    // Strides for BMHK shapes
    const int q_stride_h = depth;
    const int k_stride_h = depth;
    const int v_stride_h = depth;
    const int q_stride_m = depth * num_heads;
    const int k_stride_m = depth * num_heads;
    const int v_stride_m = depth * num_heads;
    const int q_stride_b = q_stride_m * num_queries;
    const int k_stride_b = k_stride_m * num_keys;
    const int v_stride_b = v_stride_m * num_keys;
    const int o_stride_m = depth * num_heads;

    const int bias_stride_h = 64 * 64;
    const int bias_stride_m = 64;
    const int bias_stride_b = num_heads * bias_stride_h;

    // Launch kernel for each batch and head
    for (int batch = 0; batch < batch_size; batch++) {
      for (int head = 0; head < num_heads; head++) {
        launch_mha_block(queue, mha_output, mha_q, mha_k, mha_v, mha_skip,
                        batch, head, num_heads, depth, scale,
                        q_stride_h, k_stride_h, v_stride_h,
                        q_stride_m, k_stride_m, v_stride_m,
                        q_stride_b, k_stride_b, v_stride_b,
                        o_stride_m, bias_stride_h, bias_stride_m, bias_stride_b);
      }
    }
  }

private:
  void launch_mha_block(sycl::queue& queue,
                       sycl::half* output, sycl::half* q, sycl::half* k, sycl::half* v, sycl::half* skip,
                       int batch, int head, int num_heads, int head_dim, float scale,
                       int q_stride_h, int k_stride_h, int v_stride_h,
                       int q_stride_m, int k_stride_m, int v_stride_m,
                       int q_stride_b, int k_stride_b, int v_stride_b,
                       int o_stride_m, int bias_stride_h, int bias_stride_m, int bias_stride_b) {

    queue.submit([&](sycl::handler& cgh) {
      // Local memory for attention scores and intermediate results
      sycl::local_accessor<sycl::half, 1> local_attention{sycl::range<1>(64 * 64), cgh};
      sycl::local_accessor<float, 1> local_scores{sycl::range<1>(64 * 64), cgh};

      cgh.parallel_for(sycl::nd_range<3>{
        sycl::range<3>(1, 1, 64),  // 64 queries
        sycl::range<3>(1, 1, 1)
      }, [=](sycl::nd_item<3> item) [[intel::reqd_sub_group_size(16)]] {
        int query_idx = item.get_global_id(2);
        const int num_queries = 64;
        const int num_keys = 64;

        // Compute attention scores for this query
        for (int key_idx = 0; key_idx < num_keys; key_idx++) {
          // Compute dot product between query and key
          float score = 0.0f;

          for (int d = 0; d < head_dim; d++) {
            // Q[batch, query, head, dim] * K[batch, key, head, dim]
            float q_val = static_cast<float>(q[batch * q_stride_b + head * q_stride_h + query_idx * q_stride_m + d]);
            float k_val = static_cast<float>(k[batch * k_stride_b + head * k_stride_h + key_idx * k_stride_m + d]);
            score += q_val * k_val;
          }

          // Apply scale
          score *= scale;

          // Add bias if present
          if (bias && skip) {
            int bias_idx = head * bias_stride_h + query_idx * bias_stride_m + key_idx;
            score += static_cast<float>(skip[batch * bias_stride_b + bias_idx]);
          }

          local_scores[query_idx * num_keys + key_idx] = score;
        }

        item.barrier(sycl::access::fence_space::local_space);

        // Softmax across keys
        float max_score = local_scores[query_idx * num_keys];
        for (int key_idx = 1; key_idx < num_keys; key_idx++) {
          max_score = sycl::max(max_score, local_scores[query_idx * num_keys + key_idx]);
        }

        // Use subgroup reduction for max
        auto sg = item.get_sub_group();
        max_score = sycl::reduce_over_group(sg, max_score, sycl::ext::oneapi::maximum<float>());

        // Compute exp and sum
        float sum_exp = 0.0f;
        for (int key_idx = 0; key_idx < num_keys; key_idx++) {
          float exp_score = sycl::exp(local_scores[query_idx * num_keys + key_idx] - max_score);
          local_scores[query_idx * num_keys + key_idx] = exp_score;
          sum_exp += exp_score;
        }

        sum_exp = sycl::reduce_over_group(sg, sum_exp, sycl::plus<float>());
        sum_exp = sycl::max(sum_exp, 1e-6f); // Avoid division by zero

        // Normalize to get attention weights
        for (int key_idx = 0; key_idx < num_keys; key_idx++) {
          local_scores[query_idx * num_keys + key_idx] /= sum_exp;
        }

        item.barrier(sycl::access::fence_space::local_space);

        // Weighted sum of values
        for (int d = 0; d < head_dim; d++) {
          float weighted_sum = 0.0f;

          for (int key_idx = 0; key_idx < num_keys; key_idx++) {
            float weight = local_scores[query_idx * num_keys + key_idx];
            float v_val = static_cast<float>(v[batch * v_stride_b + head * v_stride_h + key_idx * v_stride_m + d]);
            weighted_sum += weight * v_val;
          }

          // Use subgroup reduction for final accumulation
          weighted_sum = sycl::reduce_over_group(sg, weighted_sum, sycl::plus<float>());

          // Write to output
          // Output[batch, query, head, dim]
          output[batch * o_stride_m * num_queries + head * o_stride_m + query_idx * o_stride_m + d] =
              static_cast<sycl::half>(weighted_sum);
        }
      });
    });
  }
};

// Public interface functions
template <bool bias>
void fusedMHACutlass(sycl::queue& queue, void* output, void* q, void* k, void* v,
                    void* skip, int batch_size, int num_heads, int depth) {
  try {
    fused_mha_kernel<bias> kernel(queue, output, q, k, v, skip, batch_size, num_heads, depth);
  } catch (const sycl::exception& e) {
    throw Exception("SYCL MHA kernel failed: " + std::string(e.what()));
  }
}

void fusedMHA(sycl::queue& queue, void* output, void* mha_q, void* mha_k, void* mha_v,
              void* skip, int batch_size, int num_heads, int depth) {
  if (skip == nullptr) {
    fusedMHACutlass<false>(queue, output, mha_q, mha_k, mha_v, skip, batch_size, num_heads, depth);
  } else {
    fusedMHACutlass<true>(queue, output, mha_q, mha_k, mha_v, skip, batch_size, num_heads, depth);
  }
}

// Advanced MHA with optimizations for Intel GPUs
// This version uses more sophisticated memory access patterns and subgroup operations

template <bool bias>
class optimized_mha_kernel {
public:
  optimized_mha_kernel(sycl::queue& queue, void* output, void* q, void* k, void* v,
                      void* skip, int batch_size, int num_heads, int depth) {
    constexpr int kTileSize = 16;  // Tile size for better memory access
    sycl::half* mha_q = static_cast<sycl::half*>(q);
    sycl::half* mha_k = static_cast<sycl::half*>(k);
    sycl::half* mha_v = static_cast<sycl::half*>(v);
    sycl::half* mha_output = static_cast<sycl::half*>(output);
    sycl::half* mha_skip = static_cast<sycl::half*>(skip);

    const float scale = 1.0f / sqrt(static_cast<float>(depth));
    const int num_queries = 64;
    const int num_keys = 64;

    // Launch with 2D tiling for better utilization
    queue.submit([&](sycl::handler& cgh) {
      // Local memory for tiled computation
      sycl::local_accessor<sycl::half, 2> tiled_q{sycl::range<2>(kTileSize, depth), cgh};
      sycl::local_accessor<sycl::half, 2> tiled_k{sycl::range<2>(kTileSize, depth), cgh};
      sycl::local_accessor<float, 1> attention_weights{sycl::range<1>(kTileSize * num_keys), cgh};

      cgh.parallel_for(sycl::nd_range<2>{
        sycl::range<2>((num_queries + kTileSize - 1) / kTileSize * kTileSize,
                      (num_keys + kTileSize - 1) / kTileSize * kTileSize),
        sycl::range<2>(kTileSize, kTileSize)
      }, [=](sycl::nd_item<2> item) [[intel::reqd_sub_group_size(16)]] {
        int query_tile = item.get_group(0);
        int key_tile = item.get_group(1);
        int local_q = item.get_local_id(0);
        int local_k = item.get_local_id(1);

        int global_q = query_tile * kTileSize + local_q;
        int global_k = key_tile * kTileSize + local_k;

        if (global_q >= num_queries || global_k >= num_keys) return;

        // Load Q and K tiles into local memory
        for (int d = 0; d < depth; d++) {
          if (global_q < num_queries) {
            tiled_q[local_q][d] = mha_q[0 * num_queries * depth + global_q * depth + d];
          }
          if (global_k < num_keys) {
            tiled_k[local_k][d] = mha_k[0 * num_keys * depth + global_k * depth + d];
          }
        }

        item.barrier(sycl::access::fence_space::local_space);

        // Compute one tile of attention matrix
        if (global_q < num_queries && global_k < num_keys) {
          float score = 0.0f;

          // Vectorized dot product
          for (int d = 0; d < depth; d++) {
            score += static_cast<float>(tiled_q[local_q][d]) * static_cast<float>(tiled_k[local_k][d]);
          }

          score *= scale;

          // Bias addition if needed
          if (bias && skip) {
            score += static_cast<float>(skip[global_q * num_keys + global_k]);
          }

          attention_weights[local_q * num_keys + global_k] = score;
        }

        item.barrier(sycl::access::fence_space::local_space);

        // Only proceed for the first tile to compute softmax (simplified)
        if (key_tile == 0) {
          // Softmax computation using subgroups
          for (int q = 0; q < kTileSize && (global_q + q) < num_queries; q++) {
            float max_score = attention_weights[q * num_keys];

            // Subgroup-wide max reduction
            auto sg = item.get_sub_group();
            max_score = sycl::reduce_over_group(sg, max_score, sycl::ext::oneapi::maximum<float>());

            float sum_exp = 0.0f;
            for (int k = 0; k < num_keys; k++) {
              float exp_score = sycl::exp(attention_weights[q * num_keys + k] - max_score);
              attention_weights[q * num_keys + k] = exp_score;
              sum_exp += exp_score;
            }

            sum_exp = sycl::reduce_over_group(sg, sum_exp, sycl::plus<float>());

            // Normalize
            for (int k = 0; k < num_keys; k++) {
              attention_weights[q * num_keys + k] /= (sum_exp + 1e-6f);
            }
          }
        }

        item.barrier(sycl::access::fence_space::local_space);

        // Compute weighted sum with values
        for (int d = 0; d < depth; d++) {
          float weighted_sum = 0.0f;

          for (int k = 0; k < num_keys; k++) {
            float weight = attention_weights[local_q * num_keys + k];
            float v_val = static_cast<float>(mha_v[0 * num_keys * depth + k * depth + d]);
            weighted_sum += weight * v_val;
          }

          if (global_q < num_queries) {
            mha_output[0 * num_queries * depth + global_q * depth + d] =
                static_cast<sycl::half>(weighted_sum);
          }
        }
      });
    });
  }
};

// Alternative optimized implementation
template <bool bias>
void optimizedMHACutlass(sycl::queue& queue, void* output, void* q, void* k, void* v,
                         void* skip, int batch_size, int num_heads, int depth) {
  try {
    optimized_mha_kernel<bias> kernel(queue, output, q, k, v, skip, batch_size, num_heads, depth);
  } catch (const sycl::exception& e) {
    throw Exception("Optimized SYCL MHA kernel failed: " + std::string(e.what()));
  }
}

// The main API that matches CUDA interface
void fusedMHAOptimized(sycl::queue& queue, void* output, void* mha_q, void* mha_k, void* mha_v,
                      void* skip, int batch_size, int num_heads, int depth) {
  if (skip == nullptr) {
    optimizedMHACutlass<false>(queue, output, mha_q, mha_k, mha_v, skip, batch_size, num_heads, depth);
  } else {
    optimizedMHACutlass<true>(queue, output, mha_q, mha_k, mha_v, skip, batch_size, num_heads, depth);
  }
}

// Fallback basic implementation for testing
void fusedMHABasic(sycl::queue& queue, void* output, void* q, void* k, void* v,
                  void* skip, int batch_size, int num_heads, int depth) {
  fusedMHA(queue, output, q, k, v, skip, batch_size, num_heads, depth);
}

}  // namespace sycl_backend
}  // namespace lczero