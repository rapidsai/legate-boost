/* Copyright 2023 NVIDIA Corporation
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
 *
 */
#pragma once
#include "legate_library.h"
#include "legateboost.h"

namespace legateboost {

struct GPair {
  double grad = 0.0;
  double hess = 0.0;

  GPair& operator+=(const GPair& b)
  {
    this->grad += b.grad;
    this->hess += b.hess;
    return *this;
  }
};

class BuildTreeTask : public Task<BuildTreeTask, BUILD_TREE> {
 public:
  static void cpu_variant(legate::TaskContext& context);
#ifdef LEGATEBOOST_USE_CUDA
  static void gpu_variant(legate::TaskContext& context);
#endif
};

}  // namespace legateboost
