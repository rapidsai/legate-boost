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

#include <cuda_help.h>

namespace legateboost {

template <typename L>
__global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  LaunchNKernel(size_t size, L lambda)
{
  for (auto i = blockDim.x * blockIdx.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
    lambda(i);
  }
}

template <int ITEMS_PER_THREAD = 8, typename L>
inline void LaunchN(size_t n, cudaStream_t stream, L lambda)
{
  if (n == 0) { return; }
  const int GRID_SIZE = static_cast<int>((n + ITEMS_PER_THREAD * THREADS_PER_BLOCK - 1) /
                                         (ITEMS_PER_THREAD * THREADS_PER_BLOCK));
  LaunchNKernel<<<GRID_SIZE, THREADS_PER_BLOCK, 0, stream>>>(n, lambda);
}

}  // namespace legateboost
