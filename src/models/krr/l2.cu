
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

    thrust::for_each_n(
      thrust::device, UnravelIter(L2_shape), L2_shape.volume(), [=] __device__(auto p) {
        T result = 0.0;
        for (int i = X_shape.lo[2]; i <= X_shape.hi[2]; i++) {
          auto x    = X_accessor[{p[0], p[1], i}];
          auto y    = Y_accessor[{p[0], p[1], i}];
          auto diff = x - y;
          result += diff * diff;
        }
        L2_accessor[p] = result;
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
