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
#include "legateboost.h"
#include "metrics.h"
#include "thrust/execution_policy.h"

namespace legateboost {
void ErfTask::cpu_variant(legate::TaskContext& context)
{
  auto policy = thrust::host;
  ErfTask::Impl(context, policy);
}
}  // namespace legateboost

namespace {
static void __attribute__((constructor)) register_tasks()
{
  legateboost::ErfTask::register_variants();
}
}  // namespace
