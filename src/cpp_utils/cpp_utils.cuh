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
#include "cpp_utils.h"
#include "core/cuda/cuda_help.h"
#include "core/cuda/stream_pool.h"
#include <cooperative_groups.h>
#include <nccl.h>

namespace legateboost {

#define THREADS_PER_BLOCK 128
#define MIN_CTAS_PER_SM 4

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

#define CHECK_NCCL(expr)                    \
  do {                                      \
    ncclResult_t result = (expr);           \
    check_nccl(result, __FILE__, __LINE__); \
  } while (false)

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
  const int GRID_SIZE = min(static_cast<int>((n + ITEMS_PER_THREAD * THREADS_PER_BLOCK - 1) /
                                             (ITEMS_PER_THREAD * THREADS_PER_BLOCK)),
                            128);
  LaunchNKernel<<<GRID_SIZE, THREADS_PER_BLOCK, 0, stream>>>(n, lambda);
}

template <int kTHREADS_PER_BLOCK, typename L>
__global__ void __launch_bounds__(kTHREADS_PER_BLOCK) LaunchNWarpsKernel(size_t size, L lambda)
{
  // block idx * warps per block + local warp idx
  auto block_idx  = blockIdx.x;
  auto warp_idx   = threadIdx.x / 32;
  auto block_size = blockDim.x / 32;
  auto num_blocks = gridDim.x;
  for (auto i = block_size * block_idx + warp_idx; i < size; i += block_size * num_blocks) {
    lambda(i);
  }
}

template <typename L>
inline void LaunchNWarps(size_t n, cudaStream_t stream, L lambda)
{
  if (n == 0) { return; }
  const int block_threads = 512;
  const int GRID_SIZE =
    min(static_cast<int>((n + (THREADS_PER_BLOCK / 32) - 1) / (block_threads / 32)), 2048);

  LaunchNWarpsKernel<block_threads><<<GRID_SIZE, block_threads, 0, stream>>>(n, lambda);
}
template <typename T>
void SumAllReduce(legate::TaskContext context, T* x, int count, cudaStream_t stream)
{
  auto domain      = context.get_launch_domain();
  size_t num_ranks = domain.get_volume();
  EXPECT(num_ranks == 1 || context.num_communicators() > 0,
         "Expected a GPU communicator for multi-rank task.");
  if (context.num_communicators() == 0) return;
  auto comm             = context.communicator(0);
  ncclComm_t* nccl_comm = comm.get<ncclComm_t*>();

  if (num_ranks > 1) {
    if (std::is_same<T, float>::value) {
      CHECK_NCCL(ncclAllReduce(x, x, count, ncclFloat, ncclSum, *nccl_comm, stream));
    } else if (std::is_same<T, double>::value) {
      CHECK_NCCL(ncclAllReduce(x, x, count, ncclDouble, ncclSum, *nccl_comm, stream));
    } else {
      EXPECT(false, "Unsupported type for all reduce.");
    }
    CHECK_CUDA_STREAM(stream);
  }
}

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

template <typename F, int OpCode>
void UnaryOpTask<F, OpCode>::gpu_variant(legate::TaskContext context)
{
  auto const& in          = context.input(0);
  auto stream             = legate::cuda::StreamPool::get_stream_pool().get_stream();
  auto thrust_alloc       = ThrustAllocator(legate::Memory::GPU_FB_MEM);
  auto thrust_exec_policy = DEFAULT_POLICY(thrust_alloc).on(stream);
  legate::dim_dispatch(in.dim(), DispatchDimOp{}, context, in, thrust_exec_policy);
}

}  // namespace legateboost
