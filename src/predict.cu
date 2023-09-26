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
#include "cuda_help.h"
#include "kernel_helper.cuh"
#include "utils.h"
#include "predict.h"

namespace legateboost {

namespace {
template <typename T>
struct predict_fn {
  void operator()(legate::TaskContext& context)
  {
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

    // rowwise kernel
    auto prediction_lambda = [=] __device__(size_t idx) {
      int64_t pos              = 0;
      legate::Point<2> x_point = {X_shape.lo[0] + (int64_t)idx, 0};

      // Use a max depth of 100 to avoid infinite loops
      for (int depth = 0; depth < 100; depth++) {
        if (feature[pos] == -1) break;
        x_point[1]   = feature[pos];
        double X_val = X_accessor[x_point];
        pos          = X_val <= split_value[pos] ? pos * 2 + 1 : pos * 2 + 2;
      }
      for (int64_t j = 0; j < n_outputs; j++) {
        pred_accessor[{X_shape.lo[0] + (int64_t)idx, j}] = leaf_value[{pos, j}];
      }
    };

    auto stream = legate::cuda::StreamPool::get_stream_pool().get_stream();
    LaunchN(X_shape.hi[0] - X_shape.lo[0] + 1, stream, prediction_lambda);

    CHECK_CUDA_STREAM(stream);
  }
};
}  // namespace

/*static*/ void PredictTask::gpu_variant(legate::TaskContext& context)
{
  const auto& X = context.inputs().at(0);
  dispatch_dtype_float(X.code(), [&](auto t) { predict_fn<decltype(t)>{}(context); });
}

}  // namespace legateboost
