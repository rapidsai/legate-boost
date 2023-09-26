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
#include "legateboost.h"
#include "legate_library.h"  // for Task
#include "utils.h"
#include "linalg.h"

#include <thrust/iterator/counting_iterator.h>
#include <thrust/for_each.h>
#include <type_traits>

namespace legateboost {
template <typename Tuple, std::int32_t kDim = std::tuple_size<Tuple>::value>
__host__ __device__ legate::Point<kDim, legate::coord_t> ToPoint(Tuple const& tup)
{
  if constexpr (kDim == 1) {
    Realm::Point<kDim, legate::coord_t> p{static_cast<legate::coord_t>(std::get<0>(tup))};
    return p;
  } else if constexpr (kDim == 2) {
    legate::Point<kDim, legate::coord_t> p{static_cast<legate::coord_t>(std::get<0>(tup)),
                                           static_cast<legate::coord_t>(std::get<1>(tup))};
    return p;
  } else if constexpr (kDim == 3) {
    legate::Point<kDim, legate::coord_t> p{static_cast<legate::coord_t>(std::get<0>(tup)),
                                           static_cast<legate::coord_t>(std::get<1>(tup)),
                                           static_cast<legate::coord_t>(std::get<2>(tup))};
    return p;
  } else {
    std::terminate();
  }
  Realm::Point<kDim, legate::coord_t> p;
  return p;
}

/**
 * @brief Task for calculating erf function on ndarray.
 */
class ErfTask : public Task<ErfTask, ERF> {
 public:
  template <typename Policy, std::int32_t kDim, typename DType>
  static void DispatchDType(legate::TaskContext& context, legate::Store const& in, Policy& policy)
  {
    legate::Store const& out = context.outputs().at(0);
    if (out.dim() != in.dim()) { throw legate::TaskException{"Dimension mismatch."}; }
    auto in_accessor  = in.read_accessor<DType, kDim>();
    auto out_accessor = out.write_accessor<DType, kDim>();
    auto in_shape     = in.shape<kDim>();
    auto out_shape    = out.shape<kDim>();
    auto v            = out_shape.volume();

    auto cnt = thrust::make_counting_iterator(static_cast<std::int64_t>(0));
    std::int64_t shape[kDim]{};
    for (std::int64_t i = 0; i < kDim; ++i) { shape[i] = in_shape.hi[i] - in_shape.lo[i] + 1; }
    thrust::for_each_n(policy, cnt, v, [=] __host__ __device__(std::int64_t i) {
      auto idx = UnravelIndex(i, shape);
      out_accessor[ToPoint(idx) + out_shape.lo] =
        std::erf(static_cast<DType>(in_accessor[ToPoint(idx) + in_shape.lo]));
    });
  }

  template <typename Policy, std::int32_t kDim>
  static void DispatchDim(legate::TaskContext& context, legate::Store const& in, Policy& policy)
  {
    dispatch_dtype_float(in.code(), [&](auto t) {
      using T = decltype(t);
      if constexpr (std::is_same_v<T, __half>) {
        throw legate::TaskException{"half is not supported."};
      } else {
        DispatchDType<Policy, kDim, T>(context, in, policy);
      }
    });
  }

  template <typename Policy>
  static void Impl(legate::TaskContext& context, Policy& policy)
  {
    auto const& in = context.inputs().at(0);
    switch (in.dim()) {
      case 1: {
        DispatchDim<Policy, 1>(context, in, policy);
        break;
      }
      case 2: {
        DispatchDim<Policy, 2>(context, in, policy);
        break;
      }
      case 3: {
        DispatchDim<Policy, 3>(context, in, policy);
        break;
      }
      default: throw legate::TaskException{"Too many dimensions."};
    }
  }

 public:
  static void cpu_variant(legate::TaskContext& context);
#ifdef LEGATEBOOST_USE_CUDA
  static void gpu_variant(legate::TaskContext& context);
#endif
};
}  // namespace legateboost
