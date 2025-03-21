
/* Copyright 2024, NVIDIA Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once
#include <tuple>
#include "../../cpp_utils/cpp_utils.h"
#include "legateboost.h"
namespace legateboost {

struct RbfOp {
  using ArgsT = std::tuple<double>;
  double sigma;
  explicit RbfOp(double sigma) : sigma(sigma) {}

  template <typename T>
  __host__ __device__ auto operator()(T const& v) const -> T
  {
    return std::exp(-v / (2.0 * sigma * sigma));
  }
};

class RbfTask : public Task<RbfTask, RBF> {
 public:
  static inline const auto TASK_CONFIG = legate::TaskConfig{legate::LocalTaskID{RBF}};
  static void cpu_variant(legate::TaskContext context);
#ifdef LEGATEBOOST_USE_CUDA
  static void gpu_variant(legate::TaskContext context);
#endif
};
}  // namespace legateboost
