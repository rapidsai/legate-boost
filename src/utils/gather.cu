
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
  template <typename T>
  void operator()(legate::TaskContext context)
  {
    const auto& X   = context.input(0).data();
    auto X_shape    = X.shape<2>();
    auto X_accessor = X.read_accessor<T, 2>();
    EXPECT_DENSE_ROW_MAJOR(X_accessor.accessor, X_shape);
    auto split_proposals       = context.output(0).data();
    auto split_proposals_shape = split_proposals.shape<2>();
    EXPECT_IS_BROADCAST(split_proposals_shape);
    auto split_proposals_accessor = split_proposals.write_accessor<T, 2>();
    EXPECT_DENSE_ROW_MAJOR(split_proposals_accessor.accessor, split_proposals_shape);
    EXPECT_AXIS_ALIGNED(1, X_shape, split_proposals_shape);
    auto n_features = split_proposals_shape.hi[1] - split_proposals_shape.lo[1] + 1;

    auto stream = legate::cuda::StreamPool::get_stream_pool().get_stream();

    // we can retrieve sample ids via argument(host) or legate_store(device)
    const int64_t* sample_row_ptr;
    int64_t n_samples = 0;
    const int64_t* sample_row_host_ptr;
    bool host_samples = context.scalars().size() > 0;
    if (host_samples) {
      auto sample_rows_span = context.scalar(0).values<int64_t>();
      n_samples             = sample_rows_span.size();
      sample_row_host_ptr   = &sample_rows_span[0];
    }
    auto samples_buffer = legate::create_buffer<int64_t, 1>(n_samples);
    if (host_samples) {
      CHECK_CUDA(cudaMemcpyAsync(samples_buffer.ptr(0),
                                 sample_row_host_ptr,
                                 n_samples * sizeof(int64_t),
                                 cudaMemcpyHostToDevice,
                                 stream));
      sample_row_ptr = samples_buffer.ptr(0);
    } else {
      auto sample_rows       = context.input(1).data();
      auto sample_rows_shape = sample_rows.shape<1>();
      EXPECT_IS_BROADCAST(sample_rows_shape);
      auto sample_rows_accessor = sample_rows.read_accessor<int64_t, 1>();
      n_samples                 = sample_rows_shape.hi[0] - sample_rows_shape.lo[0] + 1;
      sample_row_ptr            = sample_rows_accessor.ptr(0);
    }

    // fill with local data
    LaunchN(n_features * n_samples, stream, [=] __device__(auto idx) {
      auto i                           = idx / n_features;
      auto j                           = idx % n_features;
      auto row                         = sample_row_ptr[i];
      bool has_data                    = row >= X_shape.lo[0] && row <= X_shape.hi[0];
      split_proposals_accessor[{i, j}] = has_data ? X_accessor[{row, j}] : T(0);
    });

    // use NCCL for reduction
    SumAllReduce(context,
                 reinterpret_cast<T*>(split_proposals_accessor.ptr({0, 0})),
                 n_features * n_samples,
                 stream);

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
