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
#include <thrust/find.h>
#include <vector>
#include <unordered_map>
#include <random>
#include <tcb/span.hpp>
#include "../cpp_utils/cpp_utils.cuh"

namespace legateboost {

template <typename T>
struct CategoriesMap {
  tcb::span<const T> categories;
  tcb::span<const int64_t> row_pointers;
  __device__ int64_t GetIndex(T x, int64_t feature_idx) const
  {
    const auto* begin  = categories.begin() + row_pointers[feature_idx];
    const auto* end    = categories.begin() + row_pointers[feature_idx + 1];
    const auto* result = thrust::find(thrust::seq, begin, end, x);
    if (result == end) { return -1; }
    return result - categories.begin();
  }
};

struct target_encoder_mean_fn {
  template <typename T>
  void operator()(legate::TaskContext context)
  {
    // Can't use structured bindings with lambdas until C++20 :(
    auto X_store    = context.input(0).data();
    auto X_shape    = X_store.shape<3>();
    auto X_accessor = X_store.read_accessor<T, 3, true>();
    auto y_store    = context.input(1).data();
    auto y_shape    = y_store.shape<3>();
    auto y_accessor = y_store.read_accessor<T, 3, true>();

    auto [categories, categories_shape, categories_accessor] =
      GetInputStore<T, 1>(context.input(2).data());
    auto [row_pointers, row_pointers_shape, row_pointers_accessor] =
      GetInputStore<int64_t, 1>(context.input(3).data());

    const CategoriesMap<T> categories_map{
      {categories_accessor.ptr(0), categories_shape.volume()},
      {row_pointers_accessor.ptr(0), row_pointers_shape.volume()}};
    auto cv_fold = context.scalars().at(0).value<int64_t>();
    auto do_cv   = context.scalars().at(1).value<bool>();

    // 2nd and 3rd dimensions broadcast to align with X
    legate::AccessorRO<int64_t, 3> cv_indices;
    if (do_cv) {
      auto [cv_indices_store, cv_indices_shape, cv_indices_accessor] =
        GetInputStore<int64_t, 3>(context.input(4).data());
      cv_indices = cv_indices_accessor;
    }

    auto means =
      context.reduction(0).data().reduce_accessor<legate::SumReduction<double>, true, 3>();

    auto* stream      = context.get_task_stream();
    auto thrust_alloc = ThrustAllocator(legate::Memory::GPU_FB_MEM);
    auto policy       = DEFAULT_POLICY(thrust_alloc).on(stream);
    thrust::for_each_n(
      policy, UnravelIter(X_shape), X_shape.volume(), [=] __device__(const legate::Point<3>& p) {
        if (do_cv && cv_indices[p] == cv_fold) { return; }
        auto feature_idx   = p[1];
        auto output_idx    = p[2];
        auto feature_value = X_accessor[p];
        auto label         = y_accessor[p];
        auto category_idx  = categories_map.GetIndex(feature_value, feature_idx);
        if (category_idx == -1) { return; }
        atomicAdd(means.ptr({category_idx, output_idx, 0}), label);
        atomicAdd(means.ptr({category_idx, output_idx, 1}), 1);
      });
  }
};

/*static*/ void TargetEncoderMeanTask::gpu_variant(legate::TaskContext context)
{
  const auto& X = context.input(0).data();
  type_dispatch_float(X.code(), target_encoder_mean_fn(), context);
}

struct target_encoder_variance_fn {
  template <typename T>
  void operator()(legate::TaskContext context)
  {
    // Can't use structured bindings with lambdas until C++20 :(
    auto X_store    = context.input(0).data();
    auto X_shape    = X_store.shape<3>();
    auto X_accessor = X_store.read_accessor<T, 3, true>();
    auto y_store    = context.input(1).data();
    auto y_shape    = y_store.shape<3>();
    auto y_accessor = y_store.read_accessor<T, 3, true>();

    auto [categories, categories_shape, categories_accessor] =
      GetInputStore<T, 1>(context.input(2).data());
    auto [row_pointers, row_pointers_shape, row_pointers_accessor] =
      GetInputStore<int64_t, 1>(context.input(3).data());

    const CategoriesMap<T> categories_map{
      {categories_accessor.ptr(0), categories_shape.volume()},
      {row_pointers_accessor.ptr(0), row_pointers_shape.volume()}};

    // Sum and count per each category and output
    auto mean          = context.input(4).data();
    auto mean_shape    = mean.shape<3>();
    auto mean_accessor = mean.read_accessor<double, 3, true>();

    // Mean of the target variable for this train fold over all categories
    auto y_mean          = context.input(5).data();
    auto y_mean_shape    = y_mean.shape<1>();
    auto y_mean_accessor = y_mean.read_accessor<double, 1, true>();

    auto cv_fold = context.scalars().at(0).value<int64_t>();
    auto do_cv   = context.scalars().at(1).value<bool>();

    // 2nd and 3rd dimensions broadcast to align with X
    legate::AccessorRO<int64_t, 3> cv_indices;
    if (do_cv) {
      auto [cv_indices_store, cv_indices_shape, cv_indices_accessor] =
        GetInputStore<int64_t, 3>(context.input(6).data());
      cv_indices = cv_indices_accessor;
    }

    auto variances =
      context.reduction(0).data().reduce_accessor<legate::SumReduction<double>, true, 2, true>();
    auto y_variance =
      context.reduction(1).data().reduce_accessor<legate::SumReduction<double>, true, 1, true>();

    auto* stream      = context.get_task_stream();
    auto thrust_alloc = ThrustAllocator(legate::Memory::GPU_FB_MEM);
    auto policy       = DEFAULT_POLICY(thrust_alloc).on(stream);
    thrust::for_each_n(
      policy, UnravelIter(X_shape), X_shape.volume(), [=] __device__(const legate::Point<3>& p) {
        if (do_cv && cv_indices[p] == cv_fold) { return; }
        auto feature_idx   = p[1];
        auto output_idx    = p[2];
        auto feature_value = X_accessor[p];
        auto label         = y_accessor[p];
        atomicAdd(y_variance.ptr({output_idx}),
                  (label - y_mean_accessor[output_idx]) * (label - y_mean_accessor[output_idx]));
        auto category_idx = categories_map.GetIndex(feature_value, feature_idx);
        if (category_idx == -1) { return; }
        auto mean_value = mean_accessor[{category_idx, output_idx, 0}] /
                          mean_accessor[{category_idx, output_idx, 1}];
        auto squared_diff = (label - mean_value) * (label - mean_value);
        atomicAdd(variances.ptr({category_idx, output_idx}), squared_diff);
      });
  }
};

/*static*/ void TargetEncoderVarianceTask::gpu_variant(legate::TaskContext context)
{
  const auto& X = context.input(0).data();
  type_dispatch_float(X.code(), target_encoder_variance_fn(), context);
}

struct target_encoder_encode_fn {
  template <typename T>
  void operator()(legate::TaskContext context)
  {
    auto X_in          = context.input(0).data();
    auto X_in_shape    = X_in.shape<3>();
    auto X_in_accessor = X_in.read_accessor<T, 3, true>();
    auto [categories, categories_shape, categories_accessor] =
      GetInputStore<T, 1>(context.input(1).data());
    auto [row_pointers, row_pointers_shape, row_pointers_accessor] =
      GetInputStore<int64_t, 1>(context.input(2).data());

    auto encodings          = context.input(3).data();
    auto encodings_shape    = encodings.shape<2>();
    auto encodings_accessor = encodings.read_accessor<double, 2, true>();

    auto y_mean          = context.input(4).data();
    auto y_mean_shape    = y_mean.shape<1>();
    auto y_mean_accessor = y_mean.read_accessor<double, 1, true>();

    auto X_out          = context.output(0).data();
    auto X_out_shape    = X_out.shape<3>();
    auto X_out_accessor = X_out.write_accessor<T, 3, true>();

    auto cv_fold = context.scalars().at(0).value<int64_t>();
    auto do_cv   = context.scalars().at(1).value<bool>();

    // 2nd and 3rd dimensions broadcast to align with X
    legate::AccessorRO<int64_t, 3> cv_indices;
    if (do_cv) {
      auto [cv_indices_store, cv_indices_shape, cv_indices_accessor] =
        GetInputStore<int64_t, 3>(context.input(5).data());
      cv_indices = cv_indices_accessor;
    }

    const CategoriesMap<T> categories_map{
      {categories_accessor.ptr(0), categories_shape.volume()},
      {row_pointers_accessor.ptr(0), row_pointers_shape.volume()}};
    auto* stream      = context.get_task_stream();
    auto thrust_alloc = ThrustAllocator(legate::Memory::GPU_FB_MEM);
    auto policy       = DEFAULT_POLICY(thrust_alloc).on(stream);
    thrust::for_each_n(policy,
                       UnravelIter(X_in_shape),
                       X_in_shape.volume(),
                       [=] __device__(const legate::Point<3>& p) {
                         if (do_cv && cv_indices[p] != cv_fold) { return; }
                         auto category_idx = categories_map.GetIndex(X_in_accessor[p], p[1]);
                         if (category_idx == -1) {
                           X_out_accessor[p] = y_mean_accessor[p[2]];
                         } else {
                           X_out_accessor[p] = encodings_accessor[{category_idx, p[2]}];
                         }
                       });
  }
};

void TargetEncoderEncodeTask::gpu_variant(legate::TaskContext context)
{
  const auto& X = context.input(0).data();
  type_dispatch_float(X.code(), target_encoder_encode_fn(), context);
}
}  // namespace legateboost
