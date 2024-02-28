
/* Copyright 2024 NVIDIA Corporation
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
#include "legate_library.h"
#include "legateboost.h"
#include "preround.h"
#include "../cpp_utils/cpp_utils.cuh"
#include "core/cuda/cuda_help.h"
#include "core/cuda/stream_pool.h"
#include <cub/cub.cuh>
#include <thrust/iterator/transform_iterator.h>

namespace legateboost {

struct abssum_fn {
  template <typename T>
  void operator()(legate::TaskContext context)
  {
    auto x                  = context.input(0).data();
    auto reduce             = context.reduction(0).data();
    auto x_accessor         = x.read_accessor<T, 1>();
    auto x_shape            = x.shape<1>();
    auto reduce_accessor    = reduce.reduce_accessor<legate::SumReduction<T>, true, 1>();
    auto stream             = legate::cuda::StreamPool::get_stream_pool().get_stream();
    auto thrust_alloc       = ThrustAllocator(legate::Memory::GPU_FB_MEM);
    auto thrust_exec_policy = DEFAULT_POLICY(thrust_alloc).on(stream);
    auto sum                = legate::create_buffer<T, 1>(1);
    auto sum_ptr            = sum.ptr(0);

    std::size_t temp_storage_bytes = 0;
    auto abs                       = thrust::make_transform_iterator(x_accessor.ptr(0),
                                               [] __device__(T x) { return std::abs(x); });

    cub::DeviceReduce::Sum(nullptr, temp_storage_bytes, abs, sum_ptr, x_shape.volume(), stream);
    auto temp_storage = legate::create_buffer<char, 1>(temp_storage_bytes);
    cub::DeviceReduce::Sum(
      temp_storage.ptr(0), temp_storage_bytes, abs, sum_ptr, x_shape.volume(), stream);

    LaunchN(1, stream, [=] __device__(int idx) {
      reduce_accessor.reduce(legate::Point<1>{idx}, sum_ptr[0]);
    });
  }
};

/*static*/ void AbsSumTask::gpu_variant(legate::TaskContext context)
{
  const auto& X = context.input(0).data();
  type_dispatch_float(X.code(), abssum_fn(), context);
}

struct preround_fn {
  template <typename T>
  void operator()(legate::TaskContext context)
  {
    auto x              = context.input(0).data();
    auto x_accessor     = x.read_accessor<T, 1>();
    auto m_accessor     = context.input(1).data().read_accessor<T, 1>();
    int64_t n           = context.scalar(0).value<int64_t>();
    auto x_shape        = x.shape<1>();
    auto x_out          = context.output(0).data();
    auto x_out_accessor = x_out.write_accessor<T, 1>();

    auto stream             = legate::cuda::StreamPool::get_stream_pool().get_stream();
    auto thrust_alloc       = ThrustAllocator(legate::Memory::GPU_FB_MEM);
    auto thrust_exec_policy = DEFAULT_POLICY(thrust_alloc).on(stream);

    thrust::for_each_n(thrust_exec_policy,
                       UnravelIter(x_shape),
                       x_shape.volume(),
                       [=] __host__ __device__(const legate::Point<1>& p) {
                         T m               = m_accessor[0];
                         T eps             = std::numeric_limits<T>::epsilon();
                         T delta           = std::floor(m / (1.0 - 2.0 * n * eps));
                         T M               = std::pow(2.0, std::ceil(std::log2(delta)));
                         auto x            = x_accessor[p];
                         x_out_accessor[p] = (x + M) - M;
                       });
  }
};

/*static*/ void PreroundTask::gpu_variant(legate::TaskContext context)
{
  const auto& X = context.input(0).data();
  type_dispatch_float(X.code(), preround_fn(), context);
}

struct preround_nccl_fn {
  template <typename T>
  void operator()(legate::TaskContext context)
  {
    auto x              = context.input(0).data();
    auto x_accessor     = x.read_accessor<T, 1>();
    int64_t n           = context.scalar(0).value<int64_t>();
    auto x_shape        = x.shape<1>();
    auto x_out          = context.output(0).data();
    auto x_out_accessor = x_out.write_accessor<T, 1>();

    auto stream             = legate::cuda::StreamPool::get_stream_pool().get_stream();
    auto thrust_alloc       = ThrustAllocator(legate::Memory::GPU_FB_MEM);
    auto thrust_exec_policy = DEFAULT_POLICY(thrust_alloc).on(stream);

    auto sum     = legate::create_buffer<T, 1>(1);
    auto sum_ptr = sum.ptr(0);

    std::size_t temp_storage_bytes = 0;

    auto abs = thrust::make_transform_iterator(x_accessor.ptr(0),
                                               [] __device__(T x) { return std::abs(x); });

    cub::DeviceReduce::Sum(nullptr, temp_storage_bytes, abs, sum_ptr, x_shape.volume(), stream);
    auto temp_storage = legate::create_buffer<char, 1>(temp_storage_bytes);
    cub::DeviceReduce::Sum(
      temp_storage.ptr(0), temp_storage_bytes, abs, sum_ptr, x_shape.volume(), stream);

    SumAllReduce(context, sum_ptr, 1, stream);

    thrust::for_each_n(thrust_exec_policy,
                       UnravelIter(x_shape),
                       x_shape.volume(),
                       [=] __host__ __device__(const legate::Point<1>& p) {
                         T m               = sum_ptr[0];
                         T eps             = std::numeric_limits<T>::epsilon();
                         T delta           = std::floor(m / (1.0 - 2.0 * n * eps));
                         T M               = std::pow(2.0, std::ceil(std::log2(delta)));
                         auto x            = x_accessor[p];
                         x_out_accessor[p] = (x + M) - M;
                       });
  }
};

/*static*/ void PreroundNCCLTask::gpu_variant(legate::TaskContext context)
{
  const auto& X = context.input(0).data();
  type_dispatch_float(X.code(), preround_nccl_fn(), context);
}

}  // namespace legateboost
