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
#include "predict.h"
#include <cstdint>
#include "../../cpp_utils/cpp_utils.h"
#include "matrix_types.h"

namespace legateboost {

namespace {
template <typename MatrixT>
void PredictRows(const MatrixT& X,
                 legate::AccessorWO<double, 3> pred_accessor,
                 legate::Rect<3, legate::coord_t> pred_shape,
                 legate::AccessorRO<double, 1> split_value,
                 legate::AccessorRO<int32_t, 1> feature,
                 legate::AccessorRO<double, 2> leaf_value)
{
  for (int64_t i = X.RowRange().lo[0]; i <= X.RowRange().hi[0]; i++) {
    int pos = 0;
    // Use a max depth of 100 to avoid infinite loops
    const int max_depth = 100;
    for (int depth = 0; depth < max_depth; depth++) {
      if (feature[pos] == -1) { break; }
      auto x = X.Get(i, feature[pos]);
      pos    = x <= split_value[pos] ? (pos * 2) + 1 : (pos * 2) + 2;
    }
    for (int64_t j = pred_shape.lo[2]; j <= pred_shape.hi[2]; j++) {
      pred_accessor[{i, 0, j}] = leaf_value[{pos, j}];
    }
  }
}
struct predict_dense_fn {
  template <typename T>
  void operator()(legate::TaskContext context)
  {
    auto X          = context.input(0).data();
    auto X_shape    = X.shape<3>();
    auto X_accessor = X.read_accessor<T, 3>();
    EXPECT_DENSE_ROW_MAJOR(X_accessor.accessor, X_shape);

    auto leaf_value  = context.input(1).data().read_accessor<double, 2>();
    auto feature     = context.input(2).data().read_accessor<int32_t, 1>();
    auto split_value = context.input(3).data().read_accessor<double, 1>();

    auto pred          = context.output(0).data();
    auto pred_shape    = pred.shape<3>();
    auto pred_accessor = pred.write_accessor<double, 3>();

    // We should have one output prediction per row of X
    EXPECT_AXIS_ALIGNED(0, X_shape, pred_shape);

    // We should have the whole tree
    EXPECT_IS_BROADCAST(context.input(1).data().shape<2>());
    EXPECT_IS_BROADCAST(context.input(2).data().shape<1>());
    EXPECT_IS_BROADCAST(context.input(3).data().shape<1>());

    PredictRows(DenseXMatrix<T>(X_accessor, X_shape),
                pred_accessor,
                pred_shape,
                split_value,
                feature,
                leaf_value);
  }
};

struct predict_csr_fn {
  template <typename T>
  void operator()(legate::TaskContext context)
  {
    auto [X_vals, X_vals_shape, X_vals_accessor] = GetInputStore<T, 1>(context.input(0).data());
    auto [X_coords, X_coords_shape, X_coords_accessor] =
      GetInputStore<int64_t, 1>(context.input(1).data());
    auto [X_offsets, X_offsets_shape, X_offsets_accessor] =
      GetInputStore<legate::Rect<1, legate::coord_t>, 1>(context.input(2).data());

    auto leaf_value  = context.input(3).data().read_accessor<double, 2>();
    auto feature     = context.input(4).data().read_accessor<int32_t, 1>();
    auto split_value = context.input(5).data().read_accessor<double, 1>();

    auto pred          = context.output(0).data();
    auto pred_shape    = pred.shape<3>();
    auto pred_accessor = pred.write_accessor<double, 3>();

    auto num_features = context.scalars().at(0).value<int32_t>();
    CSRXMatrix<T> X(
      X_vals_accessor, X_coords_accessor, X_offsets_accessor, X_offsets_shape, num_features);

    EXPECT_AXIS_ALIGNED(0, X_offsets_shape, pred_shape);

    PredictRows(X, pred_accessor, pred_shape, split_value, feature, leaf_value);
  }
};
}  // namespace

/*static*/ void PredictTreeTask::cpu_variant(legate::TaskContext context)
{
  const auto& X = context.input(0).data();
  type_dispatch_float(X.code(), predict_dense_fn(), context);
}

/*static*/ void PredictTreeCSRTask::cpu_variant(legate::TaskContext context)
{
  const auto& X = context.input(0).data();
  type_dispatch_float(X.code(), predict_csr_fn(), context);
}

}  // namespace legateboost

namespace  // unnamed
{
void __attribute__((constructor)) register_tasks()
{
  legateboost::PredictTreeTask::register_variants();
  legateboost::PredictTreeCSRTask::register_variants();
}
}  // namespace
