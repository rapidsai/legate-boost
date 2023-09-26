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
#include "legate_library.h"
#include <core/type/type_info.h>
#include <core/utilities/dispatch.h>

namespace legateboost {

extern Legion::Logger logger;

inline void expect(bool condition, std::string message, std::string file, int line)
{
  if (!condition) { throw std::runtime_error(file + "(" + std::to_string(line) + "): " + message); }
}
#define EXPECT(condition, message) (expect(condition, message, __FILE__, __LINE__))

template <int AXIS, typename ShapeAT, typename ShapeBT>
void expect_axis_aligned(const ShapeAT& a, const ShapeBT& b, std::string file, int line)
{
  expect((a.lo[AXIS] == b.lo[AXIS]) && (a.hi[AXIS] == b.hi[AXIS]),
         "Inconsistent axis alignment.",
         file,
         line);
}
#define EXPECT_AXIS_ALIGNED(axis, shape_a, shape_b) \
  (expect_axis_aligned<axis>(shape_a, shape_b, __FILE__, __LINE__))

template <typename ShapeT>
void expect_is_broadcast(const ShapeT& shape, std::string file, int line)
{
  for (int i = 0; i < sizeof(shape.lo.x) / sizeof(shape.lo[0]); i++) {
    std::stringstream ss;
    ss << "Expected a broadcast store. Got shape: " << shape << ".";
    expect(shape.lo[i] == 0, ss.str(), file, line);
  }
}
#define EXPECT_IS_BROADCAST(shape) (expect_is_broadcast(shape, __FILE__, __LINE__))

template <typename Functor, typename... Fnargs>
constexpr decltype(auto) type_dispatch_float(legate::Type::Code code, Functor f, Fnargs&&... args)
{
  switch (code) {
    case legate::Type::Code::FLOAT16: {
      return f.template operator()<legate::Type::Code::FLOAT16>(std::forward<Fnargs>(args)...);
    }
    case legate::Type::Code::FLOAT32: {
      return f.template operator()<legate::Type::Code::FLOAT32>(std::forward<Fnargs>(args)...);
    }
    case legate::Type::Code::FLOAT64: {
      return f.template operator()<legate::Type::Code::FLOAT64>(std::forward<Fnargs>(args)...);
    }
    default: break;
  }
  EXPECT(false, "Expected floating point data.");
  return f.template operator()<legate::Type::Code::FLOAT32>(std::forward<Fnargs>(args)...);
}

template <typename Fn>
constexpr decltype(auto) dispatch_dtype(legate::Type::Code code, Fn&& f)
{
  switch (code) {
    case legate::Type::Code::FLOAT16: {
#if LEGATEBOOST_USE_CUDA
      return f(float2{});
#else
      throw legate::TaskException{"half type is not supported."};
#endif
    }
    case legate::Type::Code::FLOAT32: {
      return f(float{});
    }
    case legate::Type::Code::FLOAT64: {
      return f(double{});
    }
    case legate::Type::Code::INT8: {
      return f(std::int8_t{});
    }
    case legate::Type::Code::INT16: {
      return f(std::int16_t{});
    }
    case legate::Type::Code::INT32: {
      return f(std::int32_t{});
    }
    case legate::Type::Code::INT64: {
      return f(std::int64_t{});
    }
    case legate::Type::Code::UINT8: {
      return f(std::uint8_t{});
    }
    case legate::Type::Code::UINT16: {
      return f(std::uint16_t{});
    }
    case legate::Type::Code::UINT32: {
      return f(std::uint32_t{});
    }
    case legate::Type::Code::UINT64: {
      return f(std::uint64_t{});
    }
    default: logger.error("Invalid dtype."); break;
  }
  return f(float{});
}

template <typename Fn>
constexpr decltype(auto) dispatch_dtype_float(legate::Type::Code code, Fn&& f)
{
  switch (code) {
    case legate::Type::Code::FLOAT16: {
#if LEGATEBOOST_USE_CUDA
      return f(float2{});
#else
      throw legate::TaskException{"half type is not supported."};
#endif
    }
    case legate::Type::Code::FLOAT32: {
      return f(float{});
    }
    case legate::Type::Code::FLOAT64: {
      return f(double{});
    }
    default: logger.error("Invalid dtype."); break;
  }
  return f(float{});
}

void SumAllReduce(legate::TaskContext& context, double* x, int count);
}  // namespace legateboost
