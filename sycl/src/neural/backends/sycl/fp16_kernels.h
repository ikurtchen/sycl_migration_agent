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

#include <sycl/sycl.hpp>
#include "neural/tables/activation_function.h"

namespace lczero {
namespace sycl_backend {

// Perform Squeeze-and-Excitation (SE) in a single fused kernel.
bool Se_Fp16_NHWC(sycl::queue& q, int N, int C, int numFc1Out, sycl::half* output,
                  const sycl::half* skip, const sycl::half* input, const sycl::half* w1,
                  const sycl::half* b1, const sycl::half* w2, const sycl::half* b2,
                  const sycl::half* bPrev, ActivationFunction activation);

}  // namespace sycl_backend
}  // namespace lczero