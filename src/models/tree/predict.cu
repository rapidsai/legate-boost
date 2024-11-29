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
#include "legate/utilities/dispatch.h"
#include "../../cpp_utils/cpp_utils.cuh"
#include "../../cpp_utils/cpp_utils.h"
#include "predict.h"

namespace legateboost {

namespace {
struct predict_fn {
  template <typename T>
  void operator()(legate::TaskContext context)
  {
    const auto& X   = context.input(0).data();
    auto X_shape    = X.shape<3>();
    auto X_accessor = X.read_accessor<T, 3>();
    EXPECT_DENSE_ROW_MAJOR(X_accessor.accessor, X_shape);

    auto leaf_value  = context.input(1).data().read_accessor<double, 2>();
    auto feature     = context.input(2).data().read_accessor<int32_t, 1>();
    auto split_value = context.input(3).data().read_accessor<double, 1>();

    auto pred          = context.output(0).data();
    auto pred_shape    = pred.shape<3>();
    auto pred_accessor = pred.write_accessor<double, 3>();
    auto n_outputs     = pred_shape.hi[2] - pred_shape.lo[2] + 1;

    EXPECT(pred_shape.lo[2] == 0, "Expect all outputs to be present");
    // We should have one output prediction per row of X
    EXPECT_AXIS_ALIGNED(0, X_shape, pred_shape);

    // We should have the whole tree
    EXPECT_IS_BROADCAST(context.input(1).data().shape<2>());
    EXPECT_IS_BROADCAST(context.input(2).data().shape<1>());
    EXPECT_IS_BROADCAST(context.input(3).data().shape<1>());

    // rowwise kernel
    auto prediction_lambda = [=] __device__(size_t idx) {
      int64_t pos              = 0;
      legate::Point<3> x_point = {X_shape.lo[0] + static_cast<int64_t>(idx), 0, 0};

      // Use a max depth of 100 to avoid infinite loops
      const int max_depth = 100;
      for (int depth = 0; depth < max_depth; depth++) {
        if (feature[pos] == -1) break;
        x_point[1]         = feature[pos];
        double const X_val = X_accessor[x_point];
        pos                = X_val <= split_value[pos] ? pos * 2 + 1 : pos * 2 + 2;
      }
      for (int64_t j = 0; j < n_outputs; j++) {
        pred_accessor[{X_shape.lo[0] + static_cast<int64_t>(idx), 0, j}] = leaf_value[{pos, j}];
      }
    };  // NOLINT(readability/braces)

    auto stream = context.get_task_stream();
    LaunchN(X_shape.hi[0] - X_shape.lo[0] + 1, stream, prediction_lambda);

    CHECK_CUDA_STREAM(stream);
  }
};
}  // namespace

/*static*/ void PredictTask::gpu_variant(legate::TaskContext context)
{
  auto X = context.input(0).data();
  type_dispatch_float(X.code(), predict_fn(), context);
}

}  // namespace legateboost
