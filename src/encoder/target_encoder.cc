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
#include <unordered_map>
#include <random>
#include "../cpp_utils/cpp_utils.h"

namespace legateboost {

// If n_splits is 0
// Both the test and train set contain all samples
struct CV {
  int64_t seed;
  int64_t n_splits;
  bool shuffle;
  int64_t partition_size;
  CV(int64_t seed, int64_t n_splits, bool shuffle, int64_t global_rows)
    : seed(seed), n_splits(n_splits), shuffle(shuffle)
  {
    // We must round up here to ensure that we don't miss any rows
    partition_size = std::ceil(static_cast<double>(global_rows) / n_splits);
  }
  // TODO(Rory): this is not fast, use Philox
  bool is_train(int64_t fold, int64_t row_idx) const
  {
    if (n_splits == 0) return true;
    if (!shuffle) return (row_idx / partition_size) != fold;
    auto rng = std::mt19937(seed);
    rng.discard(row_idx);
    auto dist = std::uniform_int_distribution<int64_t>(0, n_splits - 1);
    return dist(rng) != fold;
  }
  bool is_test(int64_t fold, int64_t row_idx) const
  {
    if (n_splits == 0) return true;
    return !is_train(fold, row_idx);
  }
};

// Create a mapping of feature/value pairs to category indices
template <typename T>
std::vector<std::unordered_map<T, int>> create_categories_map(
  const legate::AccessorRO<T, 1>& categories,
  const legate::AccessorRO<int64_t, 1>& row_pointers,
  const legate::Rect<1>& row_pointers_shape)
{
  std::vector<std::unordered_map<T, int>> categories_map;
  for (int feature_idx = row_pointers_shape.lo[0]; feature_idx <= row_pointers_shape.hi[0] - 1;
       feature_idx++) {
    auto feature_start = row_pointers[feature_idx];
    auto feature_end   = row_pointers[feature_idx + 1];
    categories_map.push_back(std::unordered_map<T, int>());
    for (auto category_idx = feature_start; category_idx < feature_end; category_idx++) {
      categories_map[feature_idx][categories[category_idx]] = category_idx;
    }
  }
  return categories_map;
}

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

    auto categories_map =
      create_categories_map(categories_accessor, row_pointers_accessor, row_pointers_shape);

    auto seed        = context.scalars().at(0).value<int64_t>();
    auto n_splits    = context.scalars().at(1).value<int64_t>();
    auto shuffle     = context.scalars().at(2).value<bool>();
    auto fold        = context.scalars().at(3).value<int64_t>();
    auto global_rows = context.scalars().at(4).value<int64_t>();
    auto cv          = CV{seed, n_splits, shuffle, global_rows};

    auto means =
      context.reduction(0).data().reduce_accessor<legate::SumReduction<double>, true, 3>();
    // Iterate through the data and accumulate labels
    for (auto row_idx = X_shape.lo[0]; row_idx <= X_shape.hi[0]; row_idx++) {
      bool is_train = cv.is_train(fold, row_idx);
      // Only use the training data
      if (!is_train) { continue; }
      for (auto feature_idx = X_shape.lo[1]; feature_idx <= X_shape.hi[1]; feature_idx++) {
        auto feature_value = X_accessor[{row_idx, feature_idx, 0}];  // Last index is broadcast
        for (auto output_idx = X_shape.lo[2]; output_idx <= X_shape.hi[2]; output_idx++) {
          auto label = y_accessor[{row_idx, 0, output_idx}];
          if (categories_map[feature_idx].count(feature_value) == 0) { continue; }
          auto category_idx = categories_map[feature_idx][feature_value];
          means.reduce({category_idx, output_idx, 0}, label);
          means.reduce({category_idx, output_idx, 1}, 1);
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

struct target_encoder_variance_fn {
  template <typename T>
  void operator()(legate::TaskContext context)
  {
    auto [X_store, X_shape, X_accessor] = GetInputStore<T, 3>(context.input(0).data());
    auto [y_store, y_shape, y_accessor] = GetInputStore<T, 3>(context.input(1).data());

    auto [categories, categories_shape, categories_accessor] =
      GetInputStore<T, 1>(context.input(2).data());
    auto [row_pointers, row_pointers_shape, row_pointers_accessor] =
      GetInputStore<int64_t, 1>(context.input(3).data());

    // Sum and count per each category and output
    auto [mean, mean_shape, mean_accessor] = GetInputStore<double, 3>(context.input(4).data());
    // Mean of the target variable for this train fold over all categories
    auto [y_mean, y_mean_shape, y_mean_accessor] =
      GetInputStore<double, 1>(context.input(5).data());

    auto categories_map =
      create_categories_map(categories_accessor, row_pointers_accessor, row_pointers_shape);

    auto seed        = context.scalars().at(0).value<int64_t>();
    auto n_splits    = context.scalars().at(1).value<int64_t>();
    auto shuffle     = context.scalars().at(2).value<bool>();
    auto fold        = context.scalars().at(3).value<int64_t>();
    auto global_rows = context.scalars().at(4).value<int64_t>();
    auto cv          = CV{seed, n_splits, shuffle, global_rows};

    auto variances =
      context.reduction(0).data().reduce_accessor<legate::SumReduction<double>, true, 2>();
    auto y_variance =
      context.reduction(1).data().reduce_accessor<legate::SumReduction<double>, true, 1>();

    for (auto row_idx = X_shape.lo[0]; row_idx <= X_shape.hi[0]; row_idx++) {
      bool is_train = cv.is_train(fold, row_idx);
      // Only use the training data
      if (!is_train) { continue; }
      for (auto feature_idx = X_shape.lo[1]; feature_idx <= X_shape.hi[1]; feature_idx++) {
        auto feature_value = X_accessor[{row_idx, feature_idx, 0}];  // Last index is broadcast
        for (auto output_idx = X_shape.lo[2]; output_idx <= X_shape.hi[2]; output_idx++) {
          auto label = y_accessor[{row_idx, 0, output_idx}];
          y_variance.reduce(
            {output_idx},
            (label - y_mean_accessor[output_idx]) * (label - y_mean_accessor[output_idx]));
          if (categories_map[feature_idx].count(feature_value) == 0) { continue; }
          auto category_idx = categories_map[feature_idx][feature_value];
          auto mean_value   = mean_accessor[{category_idx, output_idx, 0}] /
                            mean_accessor[{category_idx, output_idx, 1}];
          auto squared_diff = (label - mean_value) * (label - mean_value);
          variances.reduce({category_idx, output_idx}, squared_diff);
        }
      }
    }
  }
};

/*static*/ void TargetEncoderVarianceTask::cpu_variant(legate::TaskContext context)
{
  const auto& X = context.input(0).data();
  type_dispatch_float(X.code(), target_encoder_variance_fn(), context);
}

struct target_encoder_encode_fn {
  template <typename T>
  void operator()(legate::TaskContext context)
  {
    auto [X_in, X_in_shape, X_in_accessor] = GetInputStore<T, 3>(context.input(0).data());
    auto [categories, categories_shape, categories_accessor] =
      GetInputStore<T, 1>(context.input(1).data());
    auto [row_pointers, row_pointers_shape, row_pointers_accessor] =
      GetInputStore<int64_t, 1>(context.input(2).data());
    auto [encodings, encodings_shape, encodings_accessor] =
      GetInputStore<double, 2>(context.input(3).data());

    auto [y_mean, y_mean_shape, y_mean_accessor] =
      GetInputStore<double, 1>(context.input(4).data());

    auto X_out          = context.output(0).data();
    auto X_out_shape    = X_out.shape<3>();
    auto X_out_accessor = X_out.write_accessor<T, 3, true>();

    auto seed        = context.scalars().at(0).value<int64_t>();
    auto n_splits    = context.scalars().at(1).value<int64_t>();
    auto shuffle     = context.scalars().at(2).value<bool>();
    auto fold        = context.scalars().at(3).value<int64_t>();
    auto global_rows = context.scalars().at(4).value<int64_t>();
    auto cv          = CV{seed, n_splits, shuffle, global_rows};

    auto categories_map =
      create_categories_map(categories_accessor, row_pointers_accessor, row_pointers_shape);

    for (auto row_idx = X_in_shape.lo[0]; row_idx <= X_in_shape.hi[0]; row_idx++) {
      bool is_test = cv.is_test(fold, row_idx);
      // Only write the test set
      if (!is_test) { continue; }
      for (auto feature_idx = X_in_shape.lo[1]; feature_idx <= X_in_shape.hi[1]; feature_idx++) {
        auto feature_value = X_in_accessor[{row_idx, feature_idx, 0}];  // Last index is broadcast
        for (auto output_idx = X_in_shape.lo[2]; output_idx <= X_in_shape.hi[2]; output_idx++) {
          if (categories_map[feature_idx].count(feature_value) == 0) {
            X_out_accessor[{row_idx, feature_idx, output_idx}] = y_mean_accessor[output_idx];
          } else {
            auto category_idx = categories_map[feature_idx][feature_value];
            X_out_accessor[{row_idx, feature_idx, output_idx}] =
              encodings_accessor[{category_idx, output_idx}];
          }
        }
      }
    }
  }
};

/*static*/ void TargetEncoderEncodeTask::cpu_variant(legate::TaskContext context)
{
  const auto& X = context.input(0).data();
  type_dispatch_float(X.code(), target_encoder_encode_fn(), context);
}
}  // namespace legateboost
namespace  // unnamed
{
void __attribute__((constructor)) register_tasks()
{
  legateboost::TargetEncoderMeanTask::register_variants();
  legateboost::TargetEncoderEncodeTask::register_variants();
  legateboost::TargetEncoderVarianceTask::register_variants();
}
}  // namespace
