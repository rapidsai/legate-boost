
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
#include "../cpp_utils/cpp_utils.h"
#include "preround.h"

namespace legateboost {

struct abssum_fn {
  template <typename T>
  void operator()(legate::TaskContext context)
  {
    auto x               = context.input(0).data();
    auto reduce          = context.reduction(0).data();
    auto x_accessor      = x.read_accessor<T, 1>();
    auto x_shape         = x.shape<1>();
    auto reduce_accessor = reduce.reduce_accessor<legate::SumReduction<T>, true, 1>();
    for (int i = x_shape.lo[0]; i <= x_shape.hi[0]; i++) {
      reduce_accessor.reduce(i, std::abs(x_accessor[i]));
    }
  }
};

/*static*/ void AbsSumTask::cpu_variant(legate::TaskContext context)
{
  const auto& X = context.input(0).data();
  type_dispatch_float(X.code(), abssum_fn(), context);
}

struct preround_fn {
  template <typename T>
  void operator()(legate::TaskContext context)
  {
    auto x          = context.input(0).data();
    auto x_accessor = x.read_accessor<T, 1>();
    T m             = context.input(1).data().read_accessor<T, 1>()[0];
    int64_t n       = context.scalar(0).value<int64_t>();
    T eps           = std::numeric_limits<T>::epsilon();
    T delta         = std::floor(m / (1.0 - 2.0 * n * eps));
    T M             = std::pow(2.0, std::ceil(std::log2(delta)));

    auto x_shape        = x.shape<1>();
    auto x_out          = context.output(0).data();
    auto x_out_accessor = x_out.write_accessor<T, 1>();
    for (int i = x_shape.lo[0]; i <= x_shape.hi[0]; i++) {
      x_out_accessor[i] = (x_accessor[i] + M) - M;
    }
  }
};

/*static*/ void PreroundTask::cpu_variant(legate::TaskContext context)
{
  const auto& X = context.input(0).data();
  type_dispatch_float(X.code(), preround_fn(), context);
}

/*static*/ void PreroundNCCLTask::cpu_variant(legate::TaskContext context)
{
  const auto& X = context.input(0).data();
  type_dispatch_float(X.code(), preround_fn(), context);
}

}  // namespace legateboost

namespace  // unnamed
{
static void __attribute__((constructor)) register_tasks(void)
{
  legateboost::PreroundTask::register_variants();
  legateboost::AbsSumTask::register_variants();
  legateboost::PreroundNCCLTask::register_variants();
}
}  // namespace
