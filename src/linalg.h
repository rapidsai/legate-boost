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

#include <tuple>    // for tuple
#include <cstdint>  // for int32_t, int64_t, uint32_t, uint64_t
#include <utility>  // for index_sequence

namespace legateboost {
namespace detail {
template <typename T>
int32_t NativePopc(T v)
{
  int c = 0;
  for (; v != 0; v &= v - 1) c++;
  return c;
}

inline __host__ __device__ std::int32_t Popc(std::uint32_t v)
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

inline __host__ __device__ std::int32_t Popc(std::uint64_t v)
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

template <class T, std::size_t N, std::size_t... Idx>
constexpr auto ArrToTuple(T (&arr)[N], std::index_sequence<Idx...>)
{
  return std::make_tuple(arr[Idx]...);
}

/**
 * \brief Convert C-styple array to std::tuple.
 */
template <class T, std::size_t N>
constexpr auto ArrToTuple(T (&arr)[N])
{
  return ArrToTuple(arr, std::make_index_sequence<N>{});
}

// uint division optimization inspired by the CIndexer in cupy.  Division operation is
// slow on both CPU and GPU, especially 64 bit integer.  So here we first try to avoid 64
// bit when the index is smaller, then try to avoid division when it's exp of 2.
template <typename I, std::int32_t D>
__host__ __device__ auto UnravelImpl(I idx, std::int64_t const (&shape)[D])
{
  std::int64_t index[D]{0};
  static_assert(std::is_signed<decltype(D)>::value,
                "Don't change the type without changing the for loop.");
  for (int32_t dim = D; --dim > 0;) {
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
  return ArrToTuple(index);
}
}  // namespace detail

template <std::int32_t D>
__host__ __device__ auto UnravelIndex(std::int64_t idx, std::int64_t const (&shape)[D])
{
  if (idx > std::numeric_limits<uint32_t>::max()) {
    return detail::UnravelImpl<std::uint64_t, D>(static_cast<uint64_t>(idx), shape);
  } else {
    return detail::UnravelImpl<std::uint32_t, D>(static_cast<uint32_t>(idx), shape);
  }
}
}  // namespace legateboost
