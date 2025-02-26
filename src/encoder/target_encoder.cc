/* Copyright 2025 NVIDIA Corporation
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
#include "target_encoder.h"
#include <vector>
#include "../cpp_utils/cpp_utils.h"

namespace legateboost {
struct target_encoder_mean_fn {
  template <typename T>
  void operator()(legate::TaskContext context)
  {
    auto [X_store, X_shape, X_accessor] = GetInputStore<T, 3>(context.input(0).data());
    auto [y_store, y_shape, y_accessor] = GetInputStore<T, 3>(context.input(1).data());

    auto [categories, categories_shape, categories_accessor] =
      GetInputStore<T, 1>(context.input(2).data());
    auto [row_pointers, row_pointers_shape, row_pointers_accessor] =
      GetInputStore<int64_t, 1>(context.input(3).data());

    // Create a mapping of feature/value pairs to category indices
    std::vector<std::unordered_map<T, int>> categories_map;
    for (int feature_idx = row_pointers_shape.lo[0]; feature_idx <= row_pointers_shape.hi[0] - 1;
         feature_idx++) {
      auto feature_start = row_pointers_accessor[feature_idx];
      auto feature_end   = row_pointers_accessor[feature_idx + 1];
      categories_map.push_back(std::unordered_map<T, int>());
      for (auto category_idx = feature_start; category_idx < feature_end; category_idx++) {
        categories_map[feature_idx][categories_accessor[category_idx]] = category_idx;
      }
    }

    auto out_statistics =
      context.reduction(0).data().reduce_accessor<legate::SumReduction<double>, true, 3>();
    // Iterate through the data and accumulate labels
    for (auto row_idx = X_shape.lo[0]; row_idx <= X_shape.hi[0]; row_idx++) {
      for (auto feature_idx = X_shape.lo[1]; feature_idx <= X_shape.hi[1]; feature_idx++) {
        for (auto output_idx = X_shape.lo[2]; output_idx <= X_shape.hi[2]; output_idx++) {
          auto label         = y_accessor[{row_idx, output_idx}];
          auto feature_value = X_accessor[{row_idx, feature_idx, output_idx}];
          if (categories_map[feature_idx].count(feature_value) == 0) { continue; }
          auto category_idx = categories_map[feature_idx][feature_value];
          out_statistics.reduce({category_idx, output_idx, 0}, label);
          out_statistics.reduce({category_idx, output_idx, 1}, 1);
        }
      }
    }
  }
};

/*static*/ void TargetEncoderMeanTask::cpu_variant(legate::TaskContext context)
{
  const auto& X = context.input(0).data();
  type_dispatch_float(X.code(), target_encoder_mean_fn(), context);
}
}  // namespace legateboost
namespace  // unnamed
{
void __attribute__((constructor)) register_tasks()
{
  legateboost::TargetEncoderMeanTask::register_variants();
}
}  // namespace
