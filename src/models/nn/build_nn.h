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
#include <algorithm>
#include <memory>
#include <limits>
#include <tcb/span.hpp>
#include "../../cpp_utils/cpp_utils.h"
#include "legate_library.h"
#include "legateboost.h"

namespace legateboost {

template <typename T>
struct Matrix {
  std::array<int64_t, 2> extent{};
  tcb::span<T> data;
  std::shared_ptr<legate::Buffer<T, 2>> buffer;

  Matrix(tcb::span<T> data, std::array<int64_t, 2> extent)
    : extent({std::max(extent[0], 0L), std::max(extent[1], 0L)}), data(data)
  {
    if (extent[0] * extent[1] != data.size()) {
      throw std::runtime_error("Matrix extent does not match data size");
    }
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

  static Matrix<T> From1dOutputStore(const legate::PhysicalStore& store)
  {
    auto shape = store.shape<1>();
    tcb::span<T> data(store.read_write_accessor<T, 1, true>().ptr(shape.lo), shape.volume());
    return Matrix<T>(data, {shape.hi[0] - shape.lo[0] + 1, 1});
  }

  static Matrix<T> From2dStore(legate::PhysicalStore store)
  {
    auto shape = store.shape<2>();
    tcb::span<T> data(store.read_accessor<T, 2, true>().ptr(shape.lo), shape.volume());
    return Matrix<T>(data, {shape.hi[0] - shape.lo[0] + 1, shape.hi[1] - shape.lo[1] + 1});
  }
  static Matrix<T> From2dOutputStore(const legate::PhysicalStore& store)
  {
    auto shape = store.shape<2>();
    tcb::span<T> data(store.read_write_accessor<T, 2, true>().ptr(shape.lo), shape.volume());
    return Matrix<T>(data, {shape.hi[0] - shape.lo[0] + 1, shape.hi[1] - shape.lo[1] + 1});
  }

  // Take a 3d store, remove a broadcast dimension and return a 2d Matrix
  static Matrix<T> Project3dStore(const legate::PhysicalStore& store, int broadcast_dimension)
  {
    auto shape = store.shape<3>();
    auto data  = store.read_accessor<T, 3, true>().ptr(shape.lo);
    std::array<int64_t, 2> extent{};
    if (broadcast_dimension == 0) {
      extent = {shape.hi[1] - shape.lo[1] + 1, shape.hi[2] - shape.lo[2] + 1};
    } else if (broadcast_dimension == 1) {
      extent = {shape.hi[0] - shape.lo[0] + 1, shape.hi[2] - shape.lo[2] + 1};
    } else {
      extent = {shape.hi[0] - shape.lo[0] + 1, shape.hi[1] - shape.lo[1] + 1};
    }
    return Matrix<T>({const_cast<T*>(data), narrow<std::size_t>(extent[0] * extent[1])}, extent);
  }

  static Matrix<T> Create(std::array<int64_t, 2> extent)
  {
    auto deleter = [](legate::Buffer<T, 2>* ptr) {
      ptr->destroy();
      // Clang tidy suggests not deleting a pointer and using a smart pointer instead
      // But this is a deleter for a smart pointer anyway
      delete ptr;  // NOLINT(cppcoreguidelines-owning-memory)
    };
    std::shared_ptr<legate::Buffer<T, 2>> buffer(
      new legate::Buffer<T, 2>(legate::create_buffer<T>(legate::Point<2>{extent[0], extent[1]})),
      deleter);
    auto t   = Matrix<T>({buffer->ptr({0, 0}), narrow<std::size_t>(extent[0] * extent[1])}, extent);
    t.buffer = buffer;
    return t;
  }
};

class LearningMonitor {
  static constexpr int max_iterations_no_progress = 5;
  static constexpr double min_progress            = 1e-16;
  int max_iter;
  int verbose;
  double gtol;
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
    iterations_no_progress = old_cost - cost < min_progress ? iterations_no_progress + 1 : 0;
    old_cost               = cost;

    if (verbose && iteration % verbose == 0) {
      GetLogger().print() << "L-BFGS Iteration: " << iteration << " Cost: " << cost
                          << " Grad Norm: " << grad_norm;
    }

    if (iteration >= max_iter) {
      if (verbose) { GetLogger().print() << "L-BFGS: Maximum number of iterations reached."; }
      return true;
    }

    if (iterations_no_progress >= max_iterations_no_progress) {
      if (verbose) {
        GetLogger().print() << "No progress in " << max_iterations_no_progress
                            << " iterations. Stopping.";
      }
      return true;
    }

    if (grad_norm < gtol) {
      if (verbose) {
        if (verbose)
          GetLogger().print() << "Gradient norm below tolerance " << gtol << ". Stopping.";
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
