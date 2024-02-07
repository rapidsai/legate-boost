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
#include "../../cpp_utils/cpp_utils.h"

namespace legateboost {

namespace {
struct predict_fn {
  template <typename T>
  void operator()(legate::TaskContext context)
  {
    auto X          = context.input(0).data();
    auto X_shape    = X.shape<3>();
    auto X_accessor = X.read_accessor<T, 3>();

    auto leaf_value  = context.input(1).data().read_accessor<double, 2>();
    auto feature     = context.input(2).data().read_accessor<int32_t, 1>();
    auto split_value = context.input(3).data().read_accessor<double, 1>();

    auto pred          = context.output(0).data();
    auto pred_shape    = pred.shape<3>();
    auto pred_accessor = pred.write_accessor<double, 3>();

    // We should have one output prediction per row of X
    EXPECT_AXIS_ALIGNED(0, X_shape, pred_shape);

    // We should have the whole tree
    EXPECT_IS_BROADCAST(context.input(1).data().shape<2>());
    EXPECT_IS_BROADCAST(context.input(2).data().shape<1>());
    EXPECT_IS_BROADCAST(context.input(3).data().shape<1>());

    for (int64_t i = X_shape.lo[0]; i <= X_shape.hi[0]; i++) {
      int pos = 0;
      // Use a max depth of 100 to avoid infinite loops
      for (int depth = 0; depth < 100; depth++) {
        if (feature[pos] == -1) break;
        auto x = X_accessor[{i, feature[pos], 0}];
        pos    = x <= split_value[pos] ? pos * 2 + 1 : pos * 2 + 2;
      }
      for (int64_t j = pred_shape.lo[2]; j <= pred_shape.hi[2]; j++) {
        pred_accessor[{i, 0, j}] = leaf_value[{pos, j}];
      }
    }
  }
};
}  // namespace

/*static*/ void PredictTask::cpu_variant(legate::TaskContext context)
{
  const auto& X = context.input(0).data();
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
