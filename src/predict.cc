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
#include "predict.h"
#include "utils.h"

namespace legateboost {

namespace {
struct predict_fn {
  template <legate::Type::Code CODE>
  void operator()(legate::TaskContext& context)
  {
    using T         = legate::legate_type_of<CODE>;
    auto& X         = context.inputs().at(0);
    auto X_shape    = context.inputs().at(0).shape<2>();
    auto X_accessor = context.inputs().at(0).read_accessor<T, 2>();

    // The tree structure stores all have 1 extra 'dummy' dimension
    // due to broadcasting
    auto leaf_value  = context.inputs().at(1).read_accessor<double, 2>();
    auto feature     = context.inputs().at(2).read_accessor<int32_t, 1>();
    auto split_value = context.inputs().at(3).read_accessor<double, 1>();

    auto& pred         = context.outputs().at(0);
    auto pred_shape    = pred.shape<2>();
    auto pred_accessor = pred.write_accessor<double, 2>();
    auto n_outputs     = pred.shape<2>().hi[1] - pred.shape<2>().lo[1] + 1;

    // We should have one output prediction per row of X
    EXPECT_AXIS_ALIGNED(0, X_shape, pred_shape);

    // We should have the whole tree
    EXPECT_IS_BROADCAST(context.inputs().at(1).shape<2>());
    EXPECT_IS_BROADCAST(context.inputs().at(2).shape<1>());
    EXPECT_IS_BROADCAST(context.inputs().at(3).shape<1>());

    for (int64_t i = X_shape.lo[0]; i <= X_shape.hi[0]; i++) {
      int pos = 0;
      // Use a max depth of 100 to avoid infinite loops
      for (int depth = 0; depth < 100; depth++) {
        if (feature[pos] == -1) break;
        auto x = X_accessor[{i, feature[pos]}];
        pos    = x <= split_value[pos] ? pos * 2 + 1 : pos * 2 + 2;
      }
      for (int64_t j = 0; j < n_outputs; j++) { pred_accessor[{i, j}] = leaf_value[{pos, j}]; }
    }
  }
};
}  // namespace

/*static*/ void PredictTask::cpu_variant(legate::TaskContext& context)
{
  const auto& X = context.inputs().at(0);
  type_dispatch_float(X.code(), predict_fn(), context);
}

}  // namespace legateboost

namespace  // unnamed
{
static void __attribute__((constructor)) register_tasks(void)
{
  legateboost::PredictTask::register_variants();
}
}  // namespace
