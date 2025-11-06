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

#include <legate.h>
#include <set>
#include <vector>
#include "mapper.h"
#include "legateboost.h"

namespace legateboost {

LegateboostMapper::LegateboostMapper() = default;

auto LegateboostMapper::allocation_pool_size(const legate::mapping::Task& /* task */,
                                             legate::mapping::StoreTarget memory_kind)
  -> std::optional<std::size_t>
{
  if (memory_kind == legate::mapping::StoreTarget::ZCMEM) { return 0; }
  // TODO(seberg): nullopt means we give no upper bound.  For tasks that use
  // `legate::VariantOptions{}.with_has_allocations(true);` giving a bound
  // may improve parallelism.
  return std::nullopt;
}

auto LegateboostMapper::tunable_value(legate::TunableID /*tunable_id*/) -> legate::Scalar
{
  return legate::Scalar{};
}

auto LegateboostMapper::store_mappings(const legate::mapping::Task& task,
                                       const std::vector<legate::mapping::StoreTarget>& options)
  -> std::vector<legate::mapping::StoreMapping>
{
  auto task_id = task.task_id();
  // Enforce c-ordering for these tasks
  const std::set<LegateBoostOpCode> row_major_only = {BUILD_TREE};
  std::vector<legate::mapping::StoreMapping> mappings;
  if (row_major_only.count(static_cast<LegateBoostOpCode>(task_id)) != 0U) {
    for (auto input : task.inputs()) {
      mappings.push_back(
        legate::mapping::StoreMapping::default_mapping(input.data(), options.front()));
      mappings.back().policy().ordering = legate::mapping::DimOrdering::c_order();
      mappings.back().policy().exact = true;
    }
    return mappings;
  }
  return mappings;
}

}  // namespace legateboost
