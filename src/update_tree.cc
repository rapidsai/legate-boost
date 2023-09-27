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
#include "utils.h"

namespace legateboost {

struct update_tree_fn {
  template <legate::Type::Code CODE>
  void operator()(legate::TaskContext& context)
  {
    using T           = legate::legate_type_of<CODE>;
    const auto& X     = context.inputs().at(0);
    auto X_shape      = X.shape<2>();
    auto X_accessor   = X.read_accessor<T, 2>();
    auto num_features = X_shape.hi[1] - X_shape.lo[1] + 1;
    auto num_rows     = X_shape.hi[0] - X_shape.lo[0] + 1;
    const auto& g     = context.inputs().at(1);
    const auto& h     = context.inputs().at(2);
    EXPECT_AXIS_ALIGNED(0, X.shape<2>(), g.shape<2>());
    EXPECT_AXIS_ALIGNED(0, g.shape<2>(), h.shape<2>());
    EXPECT_AXIS_ALIGNED(1, g.shape<2>(), h.shape<2>());
    auto g_shape     = context.inputs().at(1).shape<2>();
    auto num_outputs = g.shape<2>().hi[1] - g.shape<2>().lo[1] + 1;
    auto g_accessor  = g.read_accessor<double, 2>();
    auto h_accessor  = h.read_accessor<double, 2>();

    // Tree structure
    auto feature     = context.inputs().at(3).read_accessor<int32_t, 1>();
    auto split_value = context.inputs().at(4).read_accessor<double, 1>();

    // We should have the whole tree
    EXPECT_IS_BROADCAST(context.inputs().at(3).shape<1>());
    EXPECT_IS_BROADCAST(context.inputs().at(4).shape<1>());

    auto feature_shape  = context.inputs().at(3).shape<1>();
    auto num_nodes      = feature_shape.hi[0] - feature_shape.lo[0] + 1;
    auto new_leaf_value = legate::create_buffer<double, 2>({num_nodes, num_outputs});
    auto new_gradient   = legate::create_buffer<double, 2>({num_nodes, num_outputs});
    auto new_hessian    = legate::create_buffer<double, 2>({num_nodes, num_outputs});

    for (int i = 0; i < num_nodes; i++) {
      for (int j = 0; j < num_outputs; j++) {
        new_leaf_value[{i, j}] = 0.0;
        new_gradient[{i, j}]   = 0.0;
        new_hessian[{i, j}]    = 0.0;
      }
    }

    // Walk through the tree and add the new statistics
    for (int64_t i = X_shape.lo[0]; i <= X_shape.hi[0]; i++) {
      int pos = 0;
      // Use a max depth of 100 to avoid infinite loops
      for (int depth = 0; depth < 100; depth++) {
        for (int k = 0; k < num_outputs; k++) {
          new_gradient[{pos, k}] += g_accessor[{i, k}];
          new_hessian[{pos, k}] += h_accessor[{i, k}];
        }
        if (feature[pos] == -1) break;
        auto x = X_accessor[{i, feature[pos]}];
        pos    = x <= split_value[pos] ? pos * 2 + 1 : pos * 2 + 2;
      }
    }

    // Sync the new statistics
    SumAllReduce(context, new_gradient.ptr({0, 0}), num_nodes * num_outputs);
    SumAllReduce(context, new_hessian.ptr({0, 0}), num_nodes * num_outputs);

    // Update tree
    for (int i = 0; i < num_nodes; i++) {
      for (int j = 0; j < num_outputs; j++) {
        auto H = new_hessian[{i, j}];
        if (H > 0.0) {
          new_leaf_value[{i, j}] = -new_gradient[{i, j}] / H;
        } else {
          new_leaf_value[{i, j}] = 0.0;
        }
        new_hessian[{i, j}] = new_hessian[{i, j}];
      }
    }

    if (context.get_task_index()[0] == 0) {
      auto leaf_value_out = context.outputs().at(0).write_accessor<double, 2>();
      std::copy(new_leaf_value.ptr({0, 0}),
                new_leaf_value.ptr({0, 0}) + num_nodes * num_outputs,
                leaf_value_out.ptr({0, 0}));

      auto hessian_out = context.outputs().at(1).write_accessor<double, 2>();
      std::copy(new_hessian.ptr({0, 0}),
                new_hessian.ptr({0, 0}) + num_nodes * num_outputs,
                hessian_out.ptr({0, 0}));
    }
  }
};

class UpdateTreeTask : public Task<UpdateTreeTask, UPDATE_TREE> {
 public:
  static void cpu_variant(legate::TaskContext& context)
  {
    const auto& X = context.inputs().at(0);
    type_dispatch_float(X.code(), update_tree_fn(), context);
  }
};

}  // namespace legateboost

namespace  // unnamed
{
static void __attribute__((constructor)) register_tasks(void)
{
  legateboost::UpdateTreeTask::register_variants();
}
}  // namespace
