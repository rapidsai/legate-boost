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
  DenseXMatrix(legate::AccessorRO<T, 3> x, legate::Rect<3> shape) : x(x), shape(shape) {}
  __host__ __device__ T Get(uint32_t i, uint32_t j) const { return x[legate::Point<3>{i, j, 0}]; }
  __host__ __device__ int NumFeatures() const { return shape.hi[1] - shape.lo[1] + 1; }
  __host__ __device__ legate::Rect<1, legate::coord_t> RowRange() const
  {
    return {shape.lo[0], shape.hi[0]};
  }
};

template <typename T>
class CSRXMatrix {
 public:
  using value_type = T;

 private:
  legate::AccessorRO<T, 1> values;
  legate::AccessorRO<int64_t, 1> column_indices;
  legate::AccessorRO<legate::Rect<1, legate::coord_t>, 1> row_ranges;
  legate::Rect<1, legate::coord_t> row_ranges_shape;
  int num_features;

 public:
  CSRXMatrix(legate::AccessorRO<T, 1> values,
             legate::AccessorRO<int64_t, 1> column_indices,
             legate::AccessorRO<legate::Rect<1, legate::coord_t>, 1> row_ranges,
             legate::Rect<1, legate::coord_t> row_ranges_shape,
             int num_features)
    : values(values),
      column_indices(column_indices),
      row_ranges(row_ranges),
      num_features(num_features),
      row_ranges_shape(row_ranges_shape)
  {
  }

  // Slower than dense due to search for column index
  __host__ __device__ T Get(uint32_t i, uint32_t j) const
  {
    auto row_range = row_ranges[i];
    // TODO(Rory): Binary search?
    for (int64_t k = row_range.lo; k <= row_range.hi; k++) {
      if (column_indices[k] == j) return values[k];
      if (column_indices[k] > j) return 0;
    }
    return 0;
  }
  __host__ __device__ int NumFeatures() const { return num_features; }
  __host__ __device__ legate::Rect<1, legate::coord_t> RowRange() const { return row_ranges_shape; }
};
