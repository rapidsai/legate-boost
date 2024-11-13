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

#include <set>
#include <vector>
#include "mapper.h"
#include "legateboost.h"

namespace legateboost {

LegateboostMapper::LegateboostMapper() {}

legate::mapping::TaskTarget LegateboostMapper::task_target(
  const legate::mapping::Task& /*task*/, const std::vector<legate::mapping::TaskTarget>& options)
{
  return *options.begin();
}

legate::Scalar LegateboostMapper::tunable_value(legate::TunableID /*tunable_id*/)
{
  return legate::Scalar{};
}

std::vector<legate::mapping::StoreMapping> LegateboostMapper::store_mappings(
  const legate::mapping::Task& task, const std::vector<legate::mapping::StoreTarget>& options)
{
  auto task_id = task.task_id();
  // Enforce c-ordering for these tasks
  std::set<LegateBoostOpCode> row_major_only = {BUILD_TREE};
  std::vector<legate::mapping::StoreMapping> mappings;
  if (row_major_only.count(static_cast<LegateBoostOpCode>(task_id))) {
    for (auto input : task.inputs()) {
      mappings.push_back(
        legate::mapping::StoreMapping::default_mapping(input.data(), options.front()));
      mappings.back().policy().ordering.set_c_order();
      mappings.back().policy().exact = true;
    }
    return mappings;
  }
  return mappings;
}

}  // namespace legateboost
