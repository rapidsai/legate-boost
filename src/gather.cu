
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
#include "cuda_help.h"
#include "kernel_helper.cuh"
#include "utils.h"
#include "gather.h"
#include "linalg.h"

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
    auto n_features           = context.scalar(0).value<int32_t>();
    auto split_proposals_tmp  = legate::create_buffer<T, 2>({n_samples, n_features});
    auto stream               = legate::cuda::StreamPool::get_stream_pool().get_stream();
    LaunchN(n_features * n_samples, stream, [=] __device__(auto idx) {
      auto sample_idx  = idx / n_features;
      auto feature_idx = idx % n_features;
      auto row         = sample_rows_accessor[sample_idx];
      if (row >= X_shape.lo[0] && row <= X_shape.hi[0] && feature_idx >= X_shape.lo[1] &&
          feature_idx <= X_shape.hi[1]) {
        split_proposals_tmp[{sample_idx, feature_idx}] = X_accessor[{row, feature_idx}];
      } else {
        split_proposals_tmp[{sample_idx, feature_idx}] = 0;
      }
    });

    SumAllReduce(context, split_proposals_tmp.ptr({0, 0}), n_features * n_samples, stream);

    auto split_proposals          = context.output(0).data();
    auto split_proposals_shape    = split_proposals.shape<2>();
    auto split_proposals_accessor = split_proposals.write_accessor<T, 2>();
    LaunchN(split_proposals_shape.volume(), stream, [=] __device__(auto idx) {
      int width = split_proposals_shape.hi[1] - split_proposals_shape.lo[1] + 1;
      legate::Point<2> p(idx / width + split_proposals_shape.lo[0],
                         idx % width + split_proposals_shape.lo[1]);
      split_proposals_accessor[p] = split_proposals_tmp[p];
    });
    split_proposals_tmp.destroy();
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
