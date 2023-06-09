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
#include "legate_library.h"
#include "legateboost.h"
#include "utils.h"

namespace legateboost {

struct predict_fn {
  template <legate::Type::Code CODE>
  void operator()(legate::TaskContext& context)
  {
    using T          = legate::legate_type_of<CODE>;
    auto& X          = context.inputs().at(0);
    auto X_shape     = context.inputs().at(0).shape<2>();
    auto X_accessor  = context.inputs().at(0).read_accessor<T, 2>();
    auto leaf_value  = context.inputs().at(1).read_accessor<double, 1>();
    auto feature     = context.inputs().at(2).read_accessor<int32_t, 1>();
    auto split_value = context.inputs().at(3).read_accessor<double, 1>();

    auto pred = context.outputs().at(0).write_accessor<double, 1>();

    for (int64_t i = X_shape.lo[0]; i <= X_shape.hi[0]; i++) {
      int pos = 0;
      while (feature[pos] != -1) {
        auto x = X_accessor[{i, feature[pos]}];
        pos    = x <= split_value[pos] ? pos * 2 + 1 : pos * 2 + 2;
      }
      auto leaf = leaf_value[pos];
      pred[i]   = leaf;
    }
  }
};

class PredictTask : public Task<PredictTask, PREDICT> {
 public:
  static void cpu_variant(legate::TaskContext& context)
  {
    const auto& X = context.inputs().at(0);
    type_dispatch_float(X.code(), predict_fn(), context);
  }
};
}  // namespace legateboost

namespace  // unnamed
{
static void __attribute__((constructor)) register_tasks(void)
{
  legateboost::PredictTask::register_variants();
}
}  // namespace
