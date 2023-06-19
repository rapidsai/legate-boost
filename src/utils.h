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

namespace legateboost {

extern Legion::Logger logger;

inline void expect(bool condition, std::string message, std::string file, int line)
{
  if (!condition) { throw std::runtime_error(file + "(" + std::to_string(line) + "): " + message); }
}
#define EXPECT(condition, message) (expect(condition, message, __FILE__, __LINE__))

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

}  // namespace legateboost
