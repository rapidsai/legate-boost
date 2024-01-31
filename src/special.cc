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
#include <legate/core/task/task_context.h>  // for TaskContext
#include <thrust/execution_policy.h>        // for host

#include "special.h"

namespace legateboost {
void ErfTask::cpu_variant(legate::TaskContext context)
{
  SpecialFn::Impl(context, thrust::host, ErfOp{});
}

void LgammaTask::cpu_variant(legate::TaskContext context)
{
  SpecialFn::Impl(context, thrust::host, LgammaOp{});
}

void TgammaTask::cpu_variant(legate::TaskContext context)
{
  SpecialFn::Impl(context, thrust::host, TgammaOp{});
}

void DigammaTask::cpu_variant(legate::TaskContext context)
{
  SpecialFn::Impl(context, thrust::host, DigammaOp{});
}

void ZetaTask::cpu_variant(legate::TaskContext context)
{
  // we convert it to double in Python
  auto x = context.scalar(0).value<double>();
  SpecialFn::Impl(context, thrust::host, ZetaOp{x});
}
}  // namespace legateboost

namespace  // unnamed
{
static void __attribute__((constructor)) register_tasks(void)
{
  legateboost::ErfTask::register_variants();
  legateboost::LgammaTask::register_variants();
  legateboost::TgammaTask::register_variants();
  legateboost::DigammaTask::register_variants();
  legateboost::ZetaTask::register_variants();
}
}  // namespace