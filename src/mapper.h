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

#include <vector>
#include "legate_library.h"

namespace legateboost {

class LegateboostMapper : public legate::mapping::Mapper {
 public:
  LegateboostMapper();
  ~LegateboostMapper() override                                      = default;
  LegateboostMapper(const LegateboostMapper& rhs)                    = delete;
  LegateboostMapper(LegateboostMapper&& rhs)                         = delete;
  auto operator=(const LegateboostMapper& rhs) -> LegateboostMapper& = delete;
  auto operator=(LegateboostMapper&& rhs) -> LegateboostMapper&      = delete;

  // Legate mapping functions

  auto store_mappings(const legate::mapping::Task& task,
                      const std::vector<legate::mapping::StoreTarget>& options)
    -> std::vector<legate::mapping::StoreMapping> override;
  auto tunable_value(legate::TunableID tunable_id) -> legate::Scalar override;
};

}  // namespace legateboost
