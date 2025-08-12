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
#include <legate.h>
#include <cuda/std/span>
#include <cstdint>
#include "legate_library.h"
#include "legateboost.h"
#include "build_tree.h"
#include "../../cpp_utils/cpp_utils.h"

namespace legateboost {

namespace {
template <typename T, int DIM>
void WriteOutput(const legate::PhysicalStore& out, const legate::Buffer<T, DIM>& x)
{
  auto shape = out.shape<DIM>();
  auto write = out.write_accessor<T, DIM>();
  for (legate::PointInRectIterator<DIM> it(shape); it.valid(); ++it) { write[*it] = x[*it]; }
}
}  // namespace

struct update_tree_fn {
  template <typename T>
  void operator()(legate::TaskContext context)
  {
    const auto& X   = context.input(0).data();
    auto X_shape    = X.shape<3>();  // 3rd dimension is unused
    auto X_accessor = X.read_accessor<T, 3>();
    EXPECT_DENSE_ROW_MAJOR(X_accessor.accessor, X_shape);
    auto num_features = X_shape.hi[1] - X_shape.lo[1] + 1;
    auto num_rows     = X_shape.hi[0] - X_shape.lo[0] + 1;
    const auto& g     = context.input(1).data();  // 2nd dimension is unused
    const auto& h     = context.input(2).data();  // 2nd dimension is unused
    EXPECT_AXIS_ALIGNED(0, X.shape<3>(), g.shape<3>());
    EXPECT_AXIS_ALIGNED(0, g.shape<3>(), h.shape<3>());
    EXPECT_AXIS_ALIGNED(1, g.shape<3>(), h.shape<3>());
    auto g_shape                = context.input(1).data().shape<3>();
    auto num_outputs            = g.shape<3>().hi[2] - g.shape<3>().lo[2] + 1;
    auto g_accessor             = g.read_accessor<double, 3>();
    auto h_accessor             = h.read_accessor<double, 3>();
    const auto& split_proposals = context.input(3).data();
    EXPECT(g_shape.lo[2] == 0, "Expect all outputs to be present");

    auto l1_regularization = context.scalars().at(0).value<double>();
    auto l2_regularization = context.scalars().at(1).value<double>();

    // Tree structure
    auto feature     = context.input(3).data().read_accessor<int32_t, 1>();
    auto split_value = context.input(4).data().read_accessor<double, 1>();

    // We should have the whole tree
    EXPECT_IS_BROADCAST(context.input(3).data().shape<1>());
    EXPECT_IS_BROADCAST(context.input(4).data().shape<1>());

    auto feature_shape  = context.input(3).data().shape<1>();
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
      const int max_depth = 100;
      for (int depth = 0; depth < max_depth; depth++) {
        for (int k = 0; k < num_outputs; k++) {
          new_gradient[{pos, k}] += g_accessor[{i, 0, k}];
          new_hessian[{pos, k}] += h_accessor[{i, 0, k}];
        }
        if (feature[pos] == -1) { break; }
        auto x = X_accessor[{i, feature[pos], 0}];
        pos    = x <= split_value[pos] ? (pos * 2) + 1 : (pos * 2) + 2;
      }
    }

    // Sync the new statistics
    SumAllReduce(context,
                 cuda::std::span<double>(new_gradient.ptr({0, 0}), num_nodes * num_outputs));
    SumAllReduce(context,
                 cuda::std::span<double>(new_hessian.ptr({0, 0}), num_nodes * num_outputs));

    // Update tree
    for (int i = 0; i < num_nodes; i++) {
      for (int j = 0; j < num_outputs; j++) {
        auto H = new_hessian[{i, j}];
        if (H > 0.0) {
          new_leaf_value[{i, j}] =
            CalculateLeafValue(new_gradient[{i, j}], H, l1_regularization, l2_regularization);
        } else {
          new_leaf_value[{i, j}] = 0.0;
        }
        new_hessian[{i, j}] = new_hessian[{i, j}];
      }
    }

    WriteOutput(context.output(0).data(), new_leaf_value);
    WriteOutput(context.output(1).data(), new_hessian);
  }
};

class UpdateTreeTask : public Task<UpdateTreeTask, UPDATE_TREE> {
 public:
  static constexpr auto CPU_VARIANT_OPTIONS = legate::VariantOptions{}.with_has_allocations(true);

  static void cpu_variant(legate::TaskContext context)
  {
    const auto& X = context.input(0).data();
    type_dispatch_float(X.code(), update_tree_fn(), context);
  }
};

}  // namespace legateboost

namespace  // unnamed
{
void __attribute__((constructor)) register_tasks()
{
  legateboost::UpdateTreeTask::register_variants();
}
}  // namespace
