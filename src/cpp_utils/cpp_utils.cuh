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

#include "cpp_utils.h"
#include <nccl.h>
#include <cstdio>
#include <utility>
#include <string>
#include "legate.h"
#include "legate/cuda/cuda.h"

namespace legateboost {

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

inline void throw_on_cuda_error(cudaError_t code, const char* file, int line)
{
  if (code != cudaSuccess) {
    std::stringstream ss;
    ss << file << "(" << line << ")";
    std::string file_and_line;
    ss >> file_and_line;
    throw thrust::system_error(code, thrust::cuda_category(), file_and_line);
  }
}

#define CHECK_CUDA(expr) throw_on_cuda_error(expr, __FILE__, __LINE__)

#ifdef DEBUG
#define CHECK_CUDA_STREAM(stream)              \
  do {                                         \
    CHECK_CUDA(cudaStreamSynchronize(stream)); \
    CHECK_CUDA(cudaPeekAtLastError());         \
  } while (false)
#else
#define CHECK_CUDA_STREAM(stream) CHECK_CUDA(cudaPeekAtLastError())
#endif

#define CHECK_NCCL(expr)                    \
  do {                                      \
    ncclResult_t result = (expr);           \
    check_nccl(result, __FILE__, __LINE__); \
  } while (false)

template <typename L, int BLOCK_THREADS>
__global__ void __launch_bounds__(BLOCK_THREADS)
  LaunchNKernel(size_t size, L lambda)  // NOLINT(performance-unnecessary-value-param)
{
  for (auto i = blockDim.x * blockIdx.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
    lambda(i);
  }
}

template <int ITEMS_PER_THREAD = 8, typename L>
inline void LaunchN(size_t n, cudaStream_t stream, const L& lambda)
{
  if (n == 0) { return; }
  const int kBlockThreads = 256;
  const int GRID_SIZE     = static_cast<int>((n + ITEMS_PER_THREAD * kBlockThreads - 1) /
                                         (ITEMS_PER_THREAD * kBlockThreads));
  LaunchNKernel<L, kBlockThreads><<<GRID_SIZE, kBlockThreads, 0, stream>>>(n, lambda);
}

template <typename T>
void AllReduce(legate::TaskContext context, tcb::span<T> x, ncclRedOp_t op, cudaStream_t stream)
{
  const auto& domain = context.get_launch_domain();
  size_t num_ranks   = domain.get_volume();
  EXPECT(num_ranks == 1 || context.num_communicators() > 0,
         "Expected a GPU communicator for multi-rank task.");
  if (context.num_communicators() == 0) return;
  const auto& comm      = context.communicator(0);
  ncclComm_t* nccl_comm = comm.get<ncclComm_t*>();

  if (num_ranks > 1) {
    if (std::is_same<T, float>::value) {
      CHECK_NCCL(ncclAllReduce(&x[0], &x[0], x.size(), ncclFloat, op, *nccl_comm, stream));
    } else if (std::is_same<T, double>::value) {
      CHECK_NCCL(ncclAllReduce(&x[0], &x[0], x.size(), ncclDouble, op, *nccl_comm, stream));
    } else if (std::is_same<T, int64_t>::value) {
      CHECK_NCCL(ncclAllReduce(&x[0], &x[0], x.size(), ncclInt64, op, *nccl_comm, stream));
    } else if (std::is_same<T, int32_t>::value) {
      CHECK_NCCL(ncclAllReduce(&x[0], &x[0], x.size(), ncclInt, op, *nccl_comm, stream));
    } else {
      EXPECT(false, "Unsupported type for all reduce.");
    }
    CHECK_CUDA_STREAM(stream);
  }
}

template <typename T>
void SumAllReduce(legate::TaskContext context, tcb::span<T> x, cudaStream_t stream)
{
  AllReduce(context, x, ncclSum, stream);
}

#if THRUST_VERSION >= 101600
#define DEFAULT_POLICY thrust::cuda::par_nosync
#else
#define DEFAULT_POLICY thrust::cuda::par
#endif

__device__ inline uint32_t ballot(bool inFlag, uint32_t mask = 0xffffffffu)
{
  return __ballot_sync(mask, inFlag);
}

template <typename T>
__device__ inline T shfl(T val, int srcLane, int width = 32, uint32_t mask = 0xffffffffu)
{
  return __shfl_sync(mask, val, srcLane, width);
}

class ThrustAllocator : public legate::ScopedAllocator {
 public:
  using value_type = char;

  explicit ThrustAllocator(legate::Memory::Kind kind) : legate::ScopedAllocator(kind) {}

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
  auto stream             = context.get_task_stream();
  auto thrust_alloc       = ThrustAllocator(legate::Memory::GPU_FB_MEM);
  auto thrust_exec_policy = DEFAULT_POLICY(thrust_alloc).on(stream);
  legate::dim_dispatch(in.dim(), DispatchDimOp{}, context, in, thrust_exec_policy);
}

}  // namespace legateboost
