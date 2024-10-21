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
#pragma once
#include "../../cpp_utils/cpp_utils.h"
#include "legate_library.h"
#include "legateboost.h"

namespace legateboost {

template <typename T>
struct Matrix {
  T* data;
  std::shared_ptr<legate::Buffer<T, 2>> buffer;
  std::array<int64_t, 2> extent;

  Matrix(T* data, std::array<int64_t, 2> extent) : data(data)
  {
    this->extent[0] = std::max(extent[0], 0L);
    this->extent[1] = std::max(extent[1], 0L);
  }

  __host__ __device__ std::int64_t size() const { return extent[0] * extent[1]; }

  __host__ __device__ T& operator[](std::array<int64_t, 2> idx) const
  {
    return data[idx[0] * extent[1] + idx[1]];
  }

  static Matrix<T> From1dStore(legate::PhysicalStore store)
  {
    auto shape = store.shape<1>();
    T* data    = store.read_accessor<T, 1, true>().ptr(shape.lo);
    return Matrix<T>(data, {shape.hi[0] - shape.lo[0] + 1, 1});
  }
  static Matrix<T> From1dOutputStore(legate::PhysicalStore store)
  {
    auto shape = store.shape<1>();
    T* data    = store.read_write_accessor<T, 1, true>().ptr(shape.lo);
    return Matrix<T>(data, {shape.hi[0] - shape.lo[0] + 1, 1});
  }

  static Matrix<T> From2dStore(legate::PhysicalStore store)
  {
    auto shape = store.shape<2>();
    T* data    = store.read_accessor<T, 2, true>().ptr(shape.lo);
    return Matrix<T>(data, {shape.hi[0] - shape.lo[0] + 1, shape.hi[1] - shape.lo[1] + 1});
  }
  static Matrix<T> From2dOutputStore(legate::PhysicalStore store)
  {
    auto shape = store.shape<2>();
    T* data    = store.read_write_accessor<T, 2, true>().ptr(shape.lo);
    return Matrix<T>(data, {shape.hi[0] - shape.lo[0] + 1, shape.hi[1] - shape.lo[1] + 1});
  }

  // Take a 3d store, remove a broadcast dimension and return a 2d Matrix
  static Matrix<T> Project3dStore(legate::PhysicalStore store, int broadcast_dimension)
  {
    auto shape = store.shape<3>();
    auto data  = store.read_accessor<T, 3, true>().ptr(shape.lo);
    std::array<int64_t, 2> extent;
    if (broadcast_dimension == 0) {
      extent = {shape.hi[1] - shape.lo[1] + 1, shape.hi[2] - shape.lo[2] + 1};
    } else if (broadcast_dimension == 1) {
      extent = {shape.hi[0] - shape.lo[0] + 1, shape.hi[2] - shape.lo[2] + 1};
    } else {
      extent = {shape.hi[0] - shape.lo[0] + 1, shape.hi[1] - shape.lo[1] + 1};
    }
    return Matrix<T>(const_cast<T*>(data), extent);
  }

  static Matrix<T> Create(std::array<int64_t, 2> extent)
  {
    auto deleter = [](legate::Buffer<T, 2>* ptr) {
      ptr->destroy();
      delete ptr;
    };
    std::shared_ptr<legate::Buffer<T, 2>> buffer(
      new legate::Buffer<T, 2>(legate::create_buffer<T>(legate::Point<2>{extent[0], extent[1]})),
      deleter);
    auto t   = Matrix<T>(buffer->ptr({0, 0}), extent);
    t.buffer = buffer;
    return t;
  }
};

class LearningMonitor {
  const int max_iter;
  const int max_iterations_no_progress = 5;
  const int verbose;
  const double gtol;
  int iteration              = 0;
  int iterations_no_progress = 0;
  double old_cost            = std::numeric_limits<double>::max();

 public:
  LearningMonitor(int max_iter, int verbose, double gtol)
    : max_iter(max_iter), verbose(verbose), gtol(gtol)
  {
  }

  template <typename T>
  bool IsConverged(T cost, T grad_norm)
  {
    iterations_no_progress = old_cost - cost < 1e-16 ? iterations_no_progress + 1 : 0;
    old_cost               = cost;

    if (verbose && iteration % verbose == 0) {
      logger.print() << "L-BFGS Iteration: " << iteration << " Cost: " << cost
                     << " Grad Norm: " << grad_norm;
    }

    if (iteration >= max_iter) {
      if (verbose) { logger.print() << "L-BFGS: Maximum number of iterations reached."; }
      return true;
    }

    if (iterations_no_progress >= max_iterations_no_progress) {
      if (verbose) {
        logger.print() << "No progress in " << max_iterations_no_progress
                       << " iterations. Stopping.";
      }
      return true;
    }

    if (grad_norm < gtol) {
      if (verbose) {
        if (verbose) logger.print() << "Gradient norm below tolerance " << gtol << ". Stopping.";
      }
      return true;
    }

    iteration++;
    return false;
  };
};

class BuildNNTask : public Task<BuildNNTask, BUILD_NN> {
 public:
  static void cpu_variant(legate::TaskContext context);
#ifdef LEGATEBOOST_USE_CUDA
  static void gpu_variant(legate::TaskContext context);
#endif
};

}  // namespace legateboost
