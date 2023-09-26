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

#include "cuda_help.h"
namespace legateboost {
void ErfTask::gpu_variant(legate::TaskContext& context)
{
  auto stream = legate::cuda::StreamPool::get_stream_pool().get_stream();
  auto alloc  = ThrustAllocator(legate::Memory::GPU_FB_MEM);
  auto policy = DEFAULT_POLICY(alloc).on(stream);

  ErfTask::Impl(context, policy);
}
}  // namespace legateboost
