
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
#include "legate.h"
#include "legate_library.h"
#include "legateboost.h"
#include "utils.h"
#include "gather.h"

namespace legateboost {

namespace {
struct gather_fn {
  template <legate::Type::Code CODE>
  void operator()(legate::TaskContext context)
  {
    using T                   = legate::type_of<CODE>;
    auto X                    = context.input(0).data();
    auto X_shape              = X.shape<2>();
    auto X_accessor           = X.read_accessor<T, 2>();
    auto sample_rows          = context.input(1).data();
    auto sample_rows_shape    = sample_rows.shape<1>();
    auto sample_rows_accessor = sample_rows.read_accessor<int64_t, 1>();
    auto n_samples            = sample_rows_shape.hi[0] - sample_rows_shape.lo[0] + 1;
    auto n_features           = context.scalar(0).value<int>();
    auto split_proposals_tmp  = legate::create_buffer<T, 2>({n_samples, n_features});

    for (int i = sample_rows_shape.lo[0]; i <= sample_rows_shape.hi[0]; i++) {
      auto row = sample_rows_accessor[i];
      for (int j = 0; j < n_features; j++) {
        if (row >= X_shape.lo[0] && row <= X_shape.hi[0] && j >= X_shape.lo[1] &&
            j <= X_shape.hi[1]) {
          split_proposals_tmp[{i, j}] = X_accessor[{row, j}];
        } else {
          split_proposals_tmp[{i, j}] = 0;
        }
      }
    }
    SumAllReduce(context, split_proposals_tmp.ptr({0, 0}), n_samples * n_features);

    auto split_proposals          = context.output(0).data();
    auto split_proposals_shape    = split_proposals.shape<2>();
    auto split_proposals_accessor = split_proposals.write_accessor<T, 2>();
    for (legate::PointInRectIterator<2> it(split_proposals_shape, false /*fortran_order*/);
         it.valid();
         ++it) {
      auto p                      = *it;
      split_proposals_accessor[p] = split_proposals_tmp[p];
    }
    split_proposals_tmp.destroy();
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
static void __attribute__((constructor)) register_tasks(void)
{
  legateboost::GatherTask::register_variants();
}
}  // namespace
