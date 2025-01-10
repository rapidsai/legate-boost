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
#pragma once
#include <legate.h>
#include <cstdint>
#include <utility>
#ifdef __CUDACC__
#include <thrust/binary_search.h>
#else
#include <algorithm>
#endif

namespace legateboost {
// Create a uniform interface to two matrix formats
// Dense and CSR
template <typename T>
class DenseXMatrix {
 public:
  using value_type = T;

 private:
  legate::AccessorRO<T, 3> x;
  legate::Rect<3> shape;

 public:
  DenseXMatrix(const legate::AccessorRO<T, 3>& x, legate::Rect<3> shape)
    : x(std::move(x)), shape(shape)
  {
  }
  // Global row index refers to the index across partitions
  // For features, each worker has every feature so the global is the same as the local index
  [[nodiscard]] __host__ __device__ auto Get(std::size_t global_row_idx, uint32_t feature_idx) const
    -> T
  {
    return x[legate::Point<3>{global_row_idx, feature_idx, 0}];
  }
  [[nodiscard]] __host__ __device__ auto NumFeatures() const -> int
  {
    return shape.hi[1] - shape.lo[1] + 1;
  }
  [[nodiscard]] __host__ __device__ auto RowSubset() const -> legate::Rect<1, legate::coord_t>
  {
    return {shape.lo[0], shape.hi[0]};
  }
};

template <typename T>
class CSRXMatrix {
 public:
  using value_type = T;

  legate::AccessorRO<T, 1> values;
  legate::Rect<1, legate::coord_t> vals_shape;
  legate::AccessorRO<int64_t, 1> column_indices;
  legate::AccessorRO<legate::Rect<1, legate::coord_t>, 1> row_ranges;
  legate::Rect<1, legate::coord_t> row_ranges_shape;
  int num_features;
  std::size_t nnz;  // The number of nnz in ths local partition

  CSRXMatrix(const legate::AccessorRO<T, 1>& values,
             const legate::AccessorRO<int64_t, 1>& column_indices,
             const legate::AccessorRO<legate::Rect<1, legate::coord_t>, 1>& row_ranges,
             legate::Rect<1, legate::coord_t> vals_shape,
             legate::Rect<1, legate::coord_t> row_ranges_shape,
             int num_features,
             std::size_t nnz)
    : values(std::move(values)),
      column_indices(column_indices),
      row_ranges(row_ranges),
      num_features(num_features),
      vals_shape(vals_shape),
      row_ranges_shape(row_ranges_shape),
      nnz(nnz)
  {
  }

  // Global row index refers to the index across partitions
  // For features, each worker has every feature so the global is the same as the local index
  // This method is less efficient than its Dense counterpart due to the need to search for the
  // feature
  [[nodiscard]] __host__ __device__ auto Get(std::size_t global_row_idx, uint32_t feature_idx) const
    -> T
  {
    auto row_range = row_ranges[narrow<int64_t>(global_row_idx)];

    tcb::span<const int64_t> const column_indices_span(column_indices.ptr(row_range.lo),
                                                       row_range.volume());

#ifdef __CUDACC__
    const auto* result = thrust::lower_bound(
      thrust::seq, column_indices_span.begin(), column_indices_span.end(), feature_idx);
#else
    const auto* result =
      std::lower_bound(column_indices_span.begin(), column_indices_span.end(), feature_idx);
#endif

    if (result != column_indices_span.end() && *result == feature_idx) {
      return values[row_range.lo + (result - column_indices_span.begin())];
    }
    return 0;
  }

  [[nodiscard]] auto NNZ() const { return nnz; }

  [[nodiscard]] __host__ __device__ auto NumFeatures() const -> int { return num_features; }
  [[nodiscard]] __host__ __device__ auto RowSubset() const -> legate::Rect<1, legate::coord_t>
  {
    return row_ranges_shape;
  }
};
}  // namespace legateboost
