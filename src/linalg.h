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
 *
 */
#pragma once

#include <cstddef>                           // for size_t
#include <cstdint>                           // for int32_t
#include <tuple>                             // for tuple
#include <utility>                           // for tuple_size_v, index_sequence
#include <legate/core/utilities/typedefs.h>  // for coord_t
#include <thrust/detail/config.h>            // for __host__ __device__
#include <legate/core/type/type_info.h>      // for Type
#include <legate/core/task/exception.h>      // for TaskException

namespace legateboost {
namespace detail {
template <typename Tuple, std::size_t... idx, std::int32_t kDim = std::tuple_size_v<Tuple>>
__host__ __device__ auto ToPointImpl(Tuple const& tup, std::index_sequence<idx...>)
{
  static_assert(sizeof...(idx) == kDim);
  return legate::Point<kDim, legate::coord_t>{static_cast<legate::coord_t>(std::get<idx>(tup))...};
}

template <typename Tuple, std::size_t kDim = std::tuple_size_v<Tuple>>
__host__ __device__ auto ToPoint(Tuple const& tup)
{
  return ToPointImpl(tup, std::make_index_sequence<kDim>{});
}

template <std::size_t... idx, std::int32_t kDim = sizeof...(idx)>
auto ToTupleImpl(Realm::Rect<kDim, legate::coord_t> const& p, std::index_sequence<idx...>)
{
  return std::make_tuple((p.hi[idx] - p.lo[idx] + 1)...);
}

template <std::size_t kDim, std::size_t idx, typename Head>
auto MakeArray(std::int64_t (&out)[kDim], Head&& head)
{
  assert(idx == kDim - 1 && "Invalid index");
  out[idx] = head;
}

template <std::size_t kDim, std::size_t idx, typename Head, typename... Args>
auto MakeArray(std::int64_t (&out)[kDim], Head&& head, Args&&... args)
{
  out[idx] = head;
  MakeArray<kDim, idx + 1>(out, std::forward<Args>(args)...);
}

template <typename Tuple, std::size_t... idx, std::size_t kDim = std::tuple_size_v<Tuple>>
auto ToArrayImpl(Tuple const& tup, std::int64_t (&out)[kDim], std::index_sequence<idx...>)
{
  static_assert(sizeof...(idx) == kDim);
  MakeArray<kDim, 0>(out, std::get<idx>(tup)...);
}

template <std::size_t kDim>
auto ToExtents(legate::Rect<static_cast<std::int32_t>(kDim), legate::coord_t> const& p,
               std::int64_t (&shape)[kDim])
{
  auto tup = ToTupleImpl(p, std::make_index_sequence<kDim>{});
  return ToArrayImpl(tup, shape, std::make_index_sequence<kDim>{});
}

template <class T, std::size_t N, std::size_t... Idx>
constexpr auto ArrayToTuple(T (&arr)[N], std::index_sequence<Idx...>)
{
  return std::make_tuple(arr[Idx]...);
}

/**
 * @brief Convert C-styple array to std::tuple.
 */
template <class T, std::size_t N>
constexpr auto ArrayToTuple(T (&arr)[N])
{
  return ArrayToTuple(arr, std::make_index_sequence<N>{});
}

template <typename T>
int32_t NativePopc(T v)
{
  int c = 0;
  for (; v != 0; v &= v - 1) c++;
  return c;
}

inline __host__ __device__ int Popc(uint32_t v)
{
#if defined(__CUDA_ARCH__)
  return __popc(v);
#elif defined(__GNUC__) || defined(__clang__)
  return __builtin_popcount(v);
#elif defined(_MSC_VER)
  return __popcnt(v);
#else
  return NativePopc(v);
#endif  // compiler
}

inline __host__ __device__ int Popc(uint64_t v)
{
#if defined(__CUDA_ARCH__)
  return __popcll(v);
#elif defined(__GNUC__) || defined(__clang__)
  return __builtin_popcountll(v);
#elif defined(_MSC_VER) && defined(_M_X64)
  return __popcnt64(v);
#else
  return NativePopc(v);
#endif  // compiler
}

// uint division optimization inspired by the CIndexer in cupy.  Division operation is
// slow on both CPU and GPU, especially 64 bit integer.  So here we first try to avoid 64
// bit when the index is smaller, then try to avoid division when it's exp of 2.
template <typename I, std::int32_t D>
__host__ __device__ auto UnravelImpl(I idx, std::int64_t const* shape)
{
  std::size_t index[D]{0};
  static_assert(std::is_signed<decltype(D)>::value,
                "Don't change the type without changing the for loop.");
  for (std::int32_t dim = D; --dim > 0;) {
    auto s = static_cast<std::remove_const_t<std::remove_reference_t<I>>>(shape[dim]);
    if (s & (s - 1)) {
      auto t     = idx / s;
      index[dim] = idx - t * s;
      idx        = t;
    } else {  // exp of 2
      index[dim] = idx & (s - 1);
      idx >>= Popc(s - 1);
    }
  }
  index[0] = idx;
  return ArrayToTuple(index);
}
}  // namespace detail

/**
 * @brief Turns linear index into multi-dimension index.  Similar to numpy unravel.
 */
template <std::int32_t D>
__host__ __device__ auto UnravelIndex(std::size_t idx, std::int64_t const (&shape)[D])
{
  if (idx > std::numeric_limits<uint32_t>::max()) {
    return detail::UnravelImpl<std::uint64_t, D>(static_cast<uint64_t>(idx), shape);
  } else {
    return detail::UnravelImpl<std::uint32_t, D>(static_cast<uint32_t>(idx), shape);
  }
}

/**
 * @brief Dispatch floating-point types.
 *
 * @param code Legate type code
 * @param f    Function that accepts a default value of dispatched type.
 */
template <typename Fn>
constexpr decltype(auto) dispatch_dtype_float(legate::Type::Code code, Fn&& f)
{
  switch (code) {
    case legate::Type::Code::FLOAT16: {
      return f(__half{});
    }
    case legate::Type::Code::FLOAT32: {
      return f(float{});
    }
    case legate::Type::Code::FLOAT64: {
      return f(double{});
    }
    default: throw legate::TaskException{"Invalid type."}; break;
  }
  return f(float{});
}
}  // namespace legateboost
