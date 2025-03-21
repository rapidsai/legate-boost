/* Copyright 2023-2024, NVIDIA Corporation
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
#include "special/special.h"
#include "../cpp_utils/cpp_utils.h"
namespace legateboost {

/*static*/ void ErfTask::cpu_variant(legate::TaskContext context)
{
  legateboost::UnaryOp<ErfOp>::cpu_variant(context);
}

/*static*/ void LgammaTask::cpu_variant(legate::TaskContext context)
{
  legateboost::UnaryOp<LgammaOp>::cpu_variant(context);
}

/*static*/ void TgammaTask::cpu_variant(legate::TaskContext context)
{
  legateboost::UnaryOp<TgammaOp>::cpu_variant(context);
}

/*static*/ void DigammaTask::cpu_variant(legate::TaskContext context)
{
  legateboost::UnaryOp<DigammaOp>::cpu_variant(context);
}

/*static*/ void ZetaTask::cpu_variant(legate::TaskContext context)
{
  legateboost::UnaryOp<ZetaOp>::cpu_variant(context);
}

}  // namespace legateboost

namespace {
const auto reg_id_ = []() -> char {
  legateboost::ErfTask::register_variants();
  legateboost::LgammaTask::register_variants();
  legateboost::TgammaTask::register_variants();
  legateboost::DigammaTask::register_variants();
  legateboost::ZetaTask::register_variants();
  return 0;
}();
}  // namespace
