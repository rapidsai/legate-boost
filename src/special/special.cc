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
namespace  // unnamed
{
struct ErfTask : legateboost::ErfTask {};
struct LgammaTask : legateboost::LgammaTask {};
struct TgammaTask : legateboost::TgammaTask {};
struct DigammaTask : legateboost::DigammaTask {};
struct ZetaTask : legateboost::ZetaTask {};

static const auto reg_id_ = []() -> char {
  ErfTask::register_variants();
  LgammaTask::register_variants();
  TgammaTask::register_variants();
  DigammaTask::register_variants();
  ZetaTask::register_variants();
  return 0;
}();
}  // namespace
