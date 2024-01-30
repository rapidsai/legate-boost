
/* Copyright 2024 NVIDIA Corporation
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
#include "../cpp_utils/cpp_utils.cuh"
#include "../cpp_utils/cpp_utils.h"
#include "gather.h"

namespace legateboost {

namespace {
struct gather_fn {
  template <legate::Type::Code CODE>
  void operator()(legate::TaskContext context)
  {
    using T                   = legate::type_of<CODE>;
    const auto& X             = context.input(0).data();
    auto X_shape              = X.shape<2>();
    auto X_accessor           = X.read_accessor<T, 2>();
    auto sample_rows          = context.input(1).data();
    auto sample_rows_shape    = sample_rows.shape<1>();
    auto sample_rows_accessor = sample_rows.read_accessor<int64_t, 1>();
    auto n_samples            = sample_rows_shape.hi[0] - sample_rows_shape.lo[0] + 1;
    auto split_proposals      = context.reduction(0).data();
    EXPECT_IS_BROADCAST(split_proposals.shape<2>());
    auto n_features = split_proposals.shape<2>().hi[1] - split_proposals.shape<2>().lo[1] + 1;
    auto split_proposals_accessor =
      split_proposals.reduce_accessor<legate::SumReduction<T>, true, 2>();
    auto stream = legate::cuda::StreamPool::get_stream_pool().get_stream();
    LaunchN(n_features * n_samples, stream, [=] __device__(auto idx) {
      auto i   = idx / n_features;
      auto j   = idx % n_features;
      auto row = sample_rows_accessor[i];
      if (row >= X_shape.lo[0] && row <= X_shape.hi[0] && j >= X_shape.lo[1] &&
          j <= X_shape.hi[1]) {
        split_proposals_accessor.reduce({i, j}, X_accessor[{row, j}]);
      }
    });

    CHECK_CUDA_STREAM(stream);
  }
};
}  // namespace

/*static*/ void GatherTask::gpu_variant(legate::TaskContext context)
{
  auto X = context.input(0).data();
  type_dispatch_float(X.code(), gather_fn(), context);
}

}  // namespace legateboost
