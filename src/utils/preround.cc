
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

namespace  // unnamed
{
struct preround_fn {
  template <typename T>
  void operator()(legate::TaskContext context)
  {
    auto n_inputs = context.inputs().size();
    EXPECT(n_inputs == context.outputs().size(), "Inputs do not match outputs");
    std::vector<T> sum(n_inputs);

    for (int i = 0; i < n_inputs; i++) {
      auto x          = context.input(i).data();
      auto x_accessor = x.read_accessor<T, 1, true>();
      auto x_shape    = x.shape<1>();
      T s             = 0;
      for (int j = x_shape.lo[0]; j <= x_shape.hi[0]; j++) { s += std::abs(x_accessor[j]); }
      sum[i] = s;
    }

    SumAllReduce(context, sum.data(), n_inputs);

    for (int i = 0; i < n_inputs; i++) {
      auto x              = context.input(i).data();
      auto x_accessor     = x.read_accessor<T, 1>();
      int64_t n           = context.scalar(i).value<int64_t>();
      auto x_shape        = x.shape<1>();
      auto x_out          = context.output(i).data();
      auto x_out_accessor = x_out.write_accessor<T, 1>();
      T m                 = sum[i];
      T eps               = std::numeric_limits<T>::epsilon();
      T delta             = std::floor(m / (1.0 - 2.0 * n * eps));
      T M                 = std::pow(2.0, std::ceil(std::log2(delta)));
      for (int j = x_shape.lo[0]; j <= x_shape.hi[0]; j++) {
        auto v            = x_accessor[j];
        x_out_accessor[j] = (v + M) - M;
      }
    }
  }
};
}  // unnamed namespace

/*static*/ void PreroundTask::cpu_variant(legate::TaskContext context)
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
}
}  // namespace
