
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
#include <legate.h>
#include <cstdint>
#include <tcb/span.hpp>
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

    // we can retrieve sample ids via argument(host) or legate store (host)
    tcb::span<const int64_t> sample_rows{};
    if (!context.scalars().empty()) {
      auto legate_span = context.scalar(0).values<int64_t>();
      sample_rows      = {legate_span.ptr(), legate_span.size()};
    } else {
      auto [store, shape, accessor] = GetInputStore<int64_t, 1>(context.input(1).data());
      EXPECT_IS_BROADCAST(shape);
      sample_rows = {accessor.ptr(shape.lo), shape.volume()};
    }

    for (auto i = 0; i < sample_rows.size(); i++) {
      auto row            = sample_rows[i];
      const bool has_data = row >= X_shape.lo[0] && row <= X_shape.hi[0];
      for (int j = 0; j < n_features; j++) {
        split_proposals_accessor[{i, j}] = has_data ? X_accessor[{row, j}] : T(0);
      }
    }

    SumAllReduce(context,
                 tcb::span(split_proposals_accessor.ptr({0, 0}), sample_rows.size() * n_features));
  }
};

}  // namespace

/*static*/ void GatherTask::cpu_variant(legate::TaskContext context)
{
  const auto& X = context.input(0).data();
  type_dispatch_float(X.code(), gather_fn(), context);
}

}  // namespace legateboost

namespace  // unnamed
{
const auto reg_tasks_ = []() -> char {
  legateboost::GatherTask::register_variants();
  return 0;
}();
}  // namespace
