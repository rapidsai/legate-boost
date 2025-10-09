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
    auto n_rows     = X_shape.hi[0] - X_shape.lo[0] + 1;
    // n_rows can be negative. God help us.
    if (n_rows < 1) return;
    EXPECT_DENSE_ROW_MAJOR(X_accessor.accessor, X_shape);

    auto pred          = context.output(0).data();
    auto pred_shape    = pred.shape<3>();
    auto pred_accessor = pred.write_accessor<double, 3>();
    auto n_outputs     = pred_shape.hi[2] - pred_shape.lo[2] + 1;

    // Zero the predictions
    auto* stream = context.get_task_stream();
    LaunchN(n_rows, stream, [=] __device__(size_t idx) {
      for (int64_t j = 0; j < n_outputs; j++) {
        pred_accessor[{X_shape.lo[0] + static_cast<int64_t>(idx), 0, j}] = 0.0;
      }
    });

    EXPECT(pred_shape.lo[2] == 0, "Expect all outputs to be present");
    // We should have one output prediction per row of X
    EXPECT_AXIS_ALIGNED(0, X_shape, pred_shape);

    // Loop over each tree
    for (int i = 1; i < context.inputs().size(); i += 3) {
      auto leaf_accessor    = context.input(i).data().read_accessor<double, 2>();
      auto feature_accessor = context.input(i + 1).data().read_accessor<int32_t, 1>();
      auto split_accessor   = context.input(i + 2).data().read_accessor<double, 1>();

      // rowwise kernel
      auto prediction_lambda = [=] __device__(size_t idx) {
        int64_t pos              = 0;
        legate::Point<3> x_point = {X_shape.lo[0] + static_cast<int64_t>(idx), 0, 0};

        // Use a max depth of 100 to avoid infinite loops
        const int max_depth = 100;
        for (int depth = 0; depth < max_depth; depth++) {
          if (feature_accessor[pos] == -1) { break; }
          x_point[1]         = feature_accessor[pos];
          double const X_val = X_accessor[x_point];
          pos                = X_val <= split_accessor[pos] ? (pos * 2) + 1 : (pos * 2) + 2;
        }
        for (int64_t j = 0; j < n_outputs; j++) {
          pred_accessor[{X_shape.lo[0] + static_cast<int64_t>(idx), 0, j}] +=
            leaf_accessor[{pos, j}];
        }
      };  // NOLINT(readability/braces)

      LaunchN(n_rows, stream, prediction_lambda);
    }
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
