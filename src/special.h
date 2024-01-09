/* Copyright 2024, NVIDIA Corporation
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
 */
#pragma once

#include <cstdint>                              // for int32_t
#include <type_traits>                          // for is_same_v
#include <legate/core/task/task_context.h>      // for TaskContext
#include <legate/core/task/exception.h>         // for TaskException
#include <legate/core/data/physical_array.h>    // for PhysicalArray
#include "legateboost.h"                        // for ERF
#include "legate_library.h"                     // for Task
#include "linalg.h"                             // for ToExtents, UnravelIndex
#include <thrust/iterator/counting_iterator.h>  // for make_counting_iterator
#include <thrust/for_each.h>                    // for for_each_n
#include <cmath>                                // for lgamma, erf
#include "math.h"                               // for digamma, trigamma

namespace legateboost {
/**
 * @brief Implementation for elementwise special functions.
 */
struct SpecialFn {
  template <typename Policy, std::int32_t kDim, typename DType, typename Fn>
  static void DispatchDType(legate::TaskContext& context,
                            legate::PhysicalArray const& in,
                            Policy& policy,
                            Fn fn)
  {
    legate::PhysicalArray out = context.output(0);
    if (out.dim() != in.dim()) { throw legate::TaskException{"Dimension mismatch."}; }

    auto in_accessor  = in.data().read_accessor<DType, kDim>();
    auto out_accessor = out.data().write_accessor<DType, kDim>();

    auto in_shape  = in.shape<kDim>();
    auto out_shape = out.shape<kDim>();

    auto v = out_shape.volume();

    // If we use `in_accessor.ptr(in_shape)` instead of accessors, there's an
    // error from legate when repeating tests with high dimension inputs:
    //
    // ERROR: Illegal request for pointer of non-dense rectangle
    auto cnt = thrust::make_counting_iterator(static_cast<std::int64_t>(0));
    std::int64_t shape[kDim];
    detail::ToExtents<kDim>(in_shape, shape);
    thrust::for_each_n(policy, cnt, v, [=] __host__ __device__(std::int64_t i) {
      auto idx = UnravelIndex<kDim>(i, shape);
      static_assert(std::tuple_size_v<decltype(idx)> == kDim);
      out_accessor[detail::ToPoint(idx) + out_shape.lo] =
        fn(in_accessor[detail::ToPoint(idx) + in_shape.lo]);
    });
  }

  struct DispatchDimOp {
    template <std::int32_t kDim, typename Policy, typename Fn>
    void operator()(legate::TaskContext& context,
                    legate::PhysicalArray const& in,
                    Policy& policy,
                    Fn fn)
    {
      dispatch_dtype_float(in.type().code(), [&](auto t) {
        using T = decltype(t);
        if constexpr (std::is_same_v<T, __half>) {
          throw legate::TaskException{"half is not supported."};
        } else {
          DispatchDType<Policy, kDim, T>(context, in, policy, fn);
        }
      });
    }
  };

  template <typename Policy, typename Fn>
  static void Impl(legate::TaskContext& context, Policy& policy, Fn fn)
  {
    auto const& in = context.input(0);
    legate::dim_dispatch(in.dim(), DispatchDimOp{}, context, in, policy, fn);
  }
};

/**
 * @brief Task for calculating erf function on ndarray.
 */
class ErfTask : public Task<ErfTask, ERF> {
 public:
  struct ErfOp {
    template <typename T>
    __host__ __device__ T operator()(T const& v) const
    {
      return std::erf(v);
    }
  };

 public:
  static void cpu_variant(legate::TaskContext context);
  static void gpu_variant(legate::TaskContext context);
};

/**
 * @brief Task for calculating log-gamma function on ndarray.
 */
class LgammaTask : public Task<LgammaTask, LGAMMA> {
 public:
  struct LgammaOp {
    template <typename T>
    __host__ __device__ T operator()(T const& v) const
    {
      return std::lgamma(v);
    }
  };

 public:
  static void cpu_variant(legate::TaskContext context);
  static void gpu_variant(legate::TaskContext context);
};

class TgammaTask : public Task<TgammaTask, TGAMMA> {
 public:
  struct TgammaOp {
    template <typename T>
    __host__ __device__ T operator()(T const& v) const
    {
      return std::tgamma(v);
    }
  };

 public:
  static void cpu_variant(legate::TaskContext context);
  static void gpu_variant(legate::TaskContext context);
};

class DigammaTask : public Task<DigammaTask, DIGAMMA> {
 public:
  struct DigammaOp {
    template <typename T>
    __host__ __device__ T operator()(T const& v) const
    {
      return calc_digamma(v);
    }
  };

 public:
  static void cpu_variant(legate::TaskContext context);
  static void gpu_variant(legate::TaskContext context);
};
}  // namespace legateboost
