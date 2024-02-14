
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
#include "../../cpp_utils/cpp_utils.h"
#include "l2.h"
#include <tuple>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

namespace legateboost {

namespace {
struct l2_fn {
  template <typename T>
  void operator()(legate::TaskContext context)
  {
    auto X           = context.input(0).data();
    auto Y           = context.input(1).data();
    auto L2          = context.output(0).data();
    auto X_shape     = X.shape<3>();
    auto Y_shape     = Y.shape<3>();
    auto L2_shape    = L2.shape<3>();
    auto X_accessor  = X.read_accessor<T, 3>();
    auto Y_accessor  = Y.read_accessor<T, 3>();
    auto L2_accessor = L2.write_accessor<T, 3>();

    auto stream = legate::cuda::StreamPool::get_stream_pool().get_stream();
    legate::Rect<2> out_shape({L2_shape.lo[0], L2_shape.lo[1]}, {L2_shape.hi[0], L2_shape.hi[1]});
    LaunchNWarps(out_shape.volume(), stream, [=] __device__(size_t idx) {
      auto tile  = cg::tiled_partition<32>(cg::this_thread_block());
      auto tid   = tile.thread_rank();
      auto p     = UnravelIndex(idx, out_shape);
      T result   = 0.0;
      auto x_ptr = X_accessor.ptr({p[0], 0, X_shape.lo[2]});
      auto y_ptr = Y_accessor.ptr({0, p[1], Y_shape.lo[2]});
      for (int i = X_shape.lo[2] + tid; i <= X_shape.hi[2]; i += 32) {
        auto diff = x_ptr[i] - y_ptr[i];
        result += diff * diff;
      }
      result = cg::reduce(tile, result, cg::plus<T>());

      if (tid == 0) { L2_accessor[{p[0], p[1], 0}] = result; }
    });
  }
};
}  // namespace

/*static*/ void L2Task::gpu_variant(legate::TaskContext context)
{
  auto X = context.input(0).data();
  type_dispatch_float(X.code(), l2_fn(), context);
}

}  // namespace legateboost
namespace  // unnamed
{
static void __attribute__((constructor)) register_tasks(void)
{
  legateboost::L2Task::register_variants();
}
}  // namespace
