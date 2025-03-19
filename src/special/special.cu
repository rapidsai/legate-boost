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
#include "special.h"
#include <legate.h>
#include "../cpp_utils/cpp_utils.cuh"

namespace legateboost {
/*static*/ void ErfTask::gpu_variant(legate::TaskContext context)
{
  UnaryOp<ErfOp>::gpu_variant(context);
}

/*static*/ void LgammaTask::gpu_variant(legate::TaskContext context)
{
  UnaryOp<LgammaOp>::gpu_variant(context);
}

/*static*/ void TgammaTask::gpu_variant(legate::TaskContext context)
{
  UnaryOp<TgammaOp>::gpu_variant(context);
}

/*static*/ void DigammaTask::gpu_variant(legate::TaskContext context)
{
  UnaryOp<DigammaOp>::gpu_variant(context);
}

/*static*/ void ZetaTask::gpu_variant(legate::TaskContext context)
{
  UnaryOp<ZetaOp>::gpu_variant(context);
}

}  // namespace legateboost
