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
*/

#include "neural/backends/sycl/sycl_common.h"

namespace lczero {
namespace sycl_backend {

// Fused Multi-Head Attention implementation using SYCL-TLA approach
template <bool bias>
void fusedMHASycl(sycl::queue& q, void* output, void* q_ptr, void* k_ptr, void* v_ptr,
                 void* skip, int batch_size, int num_heads, int depth) {
  using data_type = sycl::half;

  // Convert pointers to appropriate types
  data_type* query = static_cast<data_type*>(q_ptr);
  data_type* key = static_cast<data_type*>(k_ptr);
  data_type* value = static_cast<data_type*>(v_ptr);
  data_type* out = static_cast<data_type*>(output);
  data_type* attn_bias = static_cast<data_type*>(skip);

  // Configuration parameters (matching CUTLASS MHA)
  constexpr int kQueriesPerBlock = 64;
  constexpr int kKeysPerBlock = 64;
  constexpr int kNumQueries = 64;
  constexpr int kNumKeys = 64;

  // Compute scale factor (1/sqrt(d_k))
  const float scale = 1.0f / sycl::sqrt(static_cast<float>(depth));

  // Compute tensor dimensions
  const int head_dim = depth;
  const int head_dim_value = depth;
  const int q_stride_h = head_dim;
  const int k_stride_h = head_dim;
  const int v_stride_h = head_dim;
  const int q_stride_m = head_dim * num_heads;
  const int k_stride_m = head_dim * num_heads;
  const int v_stride_m = head_dim * num_heads;
  const int q_stride_b = q_stride_m * kNumQueries;
  const int k_stride_b = k_stride_m * kNumKeys;
  const int v_stride_b = v_stride_m * kNumKeys;
  const int o_stride_m = head_dim_value * num_heads;
  const int bias_stride_h = kNumKeys * kNumQueries;
  const int bias_stride_m = kNumQueries;
  const int bias_stride_b = num_heads * bias_stride_h;

  try {
    // Submit kernel for execution
    q.submit([&](sycl::handler& h) {
      // Local memory for tile computations
      sycl::local_accessor<data_type, 2> q_tile(sycl::range<2>(kQueriesPerBlock, head_dim), h);
      sycl::local_accessor<data_type, 2> k_tile(sycl::range<2>(kKeysPerBlock, head_dim), h);
      sycl::local_accessor<data_type, 2> v_tile(sycl::range<2>(kKeysPerBlock, head_dim_value), h);
      sycl::local_accessor<data_type, 2> attn_tile(sycl::range<2>(kQueriesPerBlock, kKeysPerBlock), h);

      // ND-range for kernel launch
      h.parallel_for(sycl::nd_range<3>(
        sycl::range<3>(batch_size, num_heads, kQueriesPerBlock),  // Global range
        sycl::range<3>(1, 1, kQueriesPerBlock)),                  // Local range
        [=](sycl::nd_item<3> item) [[intel::reqd_sub_group_size(16)]] {
          // Compute indices
          const int batch_idx = item.get_global_id(0);
          const int head_idx = item.get_global_id(1);
          const int query_idx = item.get_global_id(2);

          // Early exit if query index is out of bounds
          if (query_idx >= kNumQueries) return;

          const int local_query_idx = item.get_local_id(2);

          // Pointer offsets for this batch and head
          const data_type* q_base = query + batch_idx * q_stride_b + head_idx * q_stride_h;
          const data_type* k_base = key + batch_idx * k_stride_b + head_idx * k_stride_h;
          const data_type* v_base = value + batch_idx * v_stride_b + head_idx * v_stride_h;
          const data_type* bias_base = bias ? attn_bias + batch_idx * bias_stride_b + head_idx * bias_stride_h : nullptr;
          data_type* out_base = out + batch_idx * o_stride_m * kNumQueries + head_idx * o_stride_m;

          // Accumulator for attention weights and output
          data_type attn_weights[kKeysPerBlock];
          data_type output_acc[head_dim_value];

          // Initialize output accumulator
          for (int d = 0; d < head_dim_value; ++d) {
            output_acc[d] = sycl::half(0.0f);
          }

          // Load query tile into shared memory
          for (int d = 0; d < head_dim; ++d) {
            q_tile[local_query_idx][d] = q_base[query_idx * q_stride_m + d];
          }

          // Process keys in tiles
          for (int key_tile_idx = 0; key_tile_idx < kNumKeys; key_tile_idx += kKeysPerBlock) {
            // Load key tile into shared memory
            for (int local_key_idx = 0; local_key_idx < kKeysPerBlock; ++local_key_idx) {
              int global_key_idx = key_tile_idx + local_key_idx;
              if (global_key_idx < kNumKeys) {
                for (int d = 0; d < head_dim; ++d) {
                  k_tile[local_key_idx][d] = k_base[global_key_idx * k_stride_m + d];
                }
              }
            }

            item.barrier();

            // Compute attention scores
            for (int local_key_idx = 0; local_key_idx < kKeysPerBlock; ++local_key_idx) {
              int global_key_idx = key_tile_idx + local_key_idx;
              if (global_key_idx >= kNumKeys) {
                attn_weights[local_key_idx] = sycl::half(-65504.0f);  // Max negative half value
                continue;
              }

              // Compute dot product: Q · K^T
              data_type score = sycl::half(0.0f);
              for (int d = 0; d < head_dim; ++d) {
                score += q_tile[local_query_idx][d] * k_tile[local_key_idx][d];
              }

              // Apply scaling
              score = score * sycl::half(scale);

              // Add bias if provided
              if (bias) {
                score += bias_base[query_idx * bias_stride_m + global_key_idx];
              }

              attn_weights[local_key_idx] = score;
            }

            item.barrier();

            // Softmax computation (in-place)
            // Find max for numerical stability
            data_type max_val = attn_weights[0];
            for (int i = 1; i < kKeysPerBlock; ++i) {
              if (attn_weights[i] > max_val) max_val = attn_weights[i];
            }

            // Compute exp and sum
            float sum = 0.0f;
            for (int i = 0; i < kKeysPerBlock; ++i) {
              float x = (float)attn_weights[i] - (float)max_val;
              attn_weights[i] = sycl::half(sycl::exp(x));
              sum += (float)attn_weights[i];
            }

            // Normalize
            for (int i = 0; i < kKeysPerBlock; ++i) {
              attn_weights[i] = attn_weights[i] * sycl::half(1.0f / sum);
            }

            // Load value tile into shared memory
            for (int local_key_idx = 0; local_key_idx < kKeysPerBlock; ++local_key_idx) {
              int global_key_idx = key_tile_idx + local_key_idx;
              if (global_key_idx < kNumKeys) {
                for (int d = 0; d < head_dim_value; ++d) {
                  v_tile[local_key_idx][d] = v_base[global_key_idx * v_stride_m + d];
                }
              }
            }

            item.barrier();

            // Apply attention weights to values and accumulate
            for (int d = 0; d < head_dim_value; ++d) {
              data_type weighted_val = sycl::half(0.0f);
              for (int local_key_idx = 0; local_key_idx < kKeysPerBlock; ++local_key_idx) {
                int global_key_idx = key_tile_idx + local_key_idx;
                if (global_key_idx < kNumKeys) {
                  weighted_val += attn_weights[local_key_idx] * v_tile[local_key_idx][d];
                }
              }
              output_acc[d] += weighted_val;
            }

            item.barrier();
          }

          // Write final output
          for (int d = 0; d < head_dim_value; ++d) {
            out_base[query_idx * o_stride_m + d] = output_acc[d];
          }
        });
    });

    // Wait for kernel completion
    q.wait();

  } catch (const sycl::exception& e) {
    throw std::runtime_error("SYCL MHA kernel failed: " + std::string(e.what()));
  }
}

// Unified entry point that handles bias/no-bias cases
void fusedMHA(void* output, void* mha_q, void* mha_k, void* mha_v, void* skip,
              int batch_size, int num_heads, int depth, sycl::queue& queue) {
  // Validate input parameters
  if (!output || !mha_q || !mha_k || !mha_v) {
    throw std::invalid_argument("Null pointer provided for required tensors");
  }

  if (batch_size <= 0 || num_heads <= 0 || depth <= 0) {
    throw std::invalid_argument("Invalid dimensions provided");
  }

  // Call appropriate template instantiation
  if (skip == nullptr) {
    fusedMHASycl<false>(queue, output, mha_q, mha_k, mha_v, skip, batch_size, num_heads, depth);
  } else {
    fusedMHASycl<true>(queue, output, mha_q, mha_k, mha_v, skip, batch_size, num_heads, depth);
  }
}

} // namespace sycl_backend
} // namespace lczero