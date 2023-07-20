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

#pragma once

#include "legate.h"
#include "core/cuda/cuda_help.h"
#include "core/cuda/stream_pool.h"
#include <nccl.h>

#define THREADS_PER_BLOCK 128
#define MIN_CTAS_PER_SM 4

#define CHECK_NCCL(expr)                    \
  do {                                      \
    ncclResult_t result = (expr);           \
    check_nccl(result, __FILE__, __LINE__); \
  } while (false)

namespace legateboost {

#if __CUDA_ARCH__ < 600
__device__ inline double atomicAdd(double* address, double val)
{
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old             = *address_as_ull, assumed;

  do {
    assumed = old;
    old =
      atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
  } while (assumed != old);

  return __longlong_as_double(old);
}
#endif

__host__ inline void check_nccl(ncclResult_t error, const char* file, int line)
{
  if (error != ncclSuccess) {
    fprintf(stderr,
            "Internal NCCL failure with error %s in file %s at line %d\n",
            ncclGetErrorString(error),
            file,
            line);
    exit(error);
  }
}

#if THRUST_VERSION >= 101600
#define DEFAULT_POLICY thrust::cuda::par_nosync
#else
#define DEFAULT_POLICY thrust::cuda::par
#endif

class ThrustAllocator : public legate::ScopedAllocator {
 public:
  using value_type = char;

  ThrustAllocator(legate::Memory::Kind kind) : legate::ScopedAllocator(kind) {}

  char* allocate(size_t num_bytes)
  {
    return static_cast<char*>(ScopedAllocator::allocate(num_bytes));
  }

  void deallocate(char* ptr, size_t n) { ScopedAllocator::deallocate(ptr); }
};

}  // namespace legateboost
