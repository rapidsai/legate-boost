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
#include "legate_library.h"
#include <legate/type/type_info.h>
#include "legate/comm/coll.h"
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/execution_policy.h>  // for host

namespace legateboost {

extern Legion::Logger logger;

inline void expect(bool condition, std::string message, std::string file, int line)
{
  if (!condition) { throw std::runtime_error(file + "(" + std::to_string(line) + "): " + message); }
}
#define EXPECT(condition, message) (expect(condition, message, __FILE__, __LINE__))

template <int AXIS, typename ShapeAT, typename ShapeBT>
void expect_axis_aligned(const ShapeAT& a, const ShapeBT& b, std::string file, int line)
{
  expect((a.lo[AXIS] == b.lo[AXIS]) && (a.hi[AXIS] == b.hi[AXIS]),
         "Inconsistent axis alignment.",
         file,
         line);
}
#define EXPECT_AXIS_ALIGNED(axis, shape_a, shape_b) \
  (expect_axis_aligned<axis>(shape_a, shape_b, __FILE__, __LINE__))

template <int DIM>
void expect_is_broadcast(const legate::Rect<DIM>& shape, std::string file, int line)
{
  for (int i = 0; i < DIM; i++) {
    std::stringstream ss;
    ss << "Expected a broadcast store. Got shape: " << shape << ".";
    expect(shape.lo[i] == 0, ss.str(), file, line);
  }
}
#define EXPECT_IS_BROADCAST(shape) (expect_is_broadcast(shape, __FILE__, __LINE__))

template <typename T, int NDIM, bool assert_row_major = true>
std::tuple<legate::PhysicalStore, legate::Rect<NDIM>, legate::AccessorRO<T, NDIM>> GetInputStore(
  legate::PhysicalStore store)
{
  auto shape    = store.shape<NDIM>();
  auto accessor = store.read_accessor<T, NDIM, true>();
  return std::make_tuple(store, shape, accessor);
}

template <typename Functor, typename... Fnargs>
constexpr decltype(auto) type_dispatch_impl(legate::Type::Code code, Functor&& f, Fnargs&&... args)
{
  throw std::runtime_error("Unsupported type.");
}

template <typename AccessorT, int N, typename T>
void expect_dense_row_major(const AccessorT& accessor,
                            const legate::Rect<N, T>& shape,
                            std::string file,
                            int line)
{
  // workaround to check 'row-major' for more than 2 dimensions, with
  // dim[i] being promoted with stride[i] == 0 for i > 1
  auto shape_mod = shape;
  for (int i = 2; i < N; ++i) shape_mod.hi[i] = 0;

  expect(shape_mod.empty() || accessor.is_dense_row_major(shape_mod),
         "Expected a dense row major store",
         file,
         line);
}
#define EXPECT_DENSE_ROW_MAJOR(accessor, shape) \
  (expect_dense_row_major(accessor, shape, __FILE__, __LINE__))

template <typename T, typename... Types, typename Functor, typename... Fnargs>
constexpr decltype(auto) type_dispatch_impl(legate::Type::Code code, Functor&& f, Fnargs&&... args)
{
  if (code == legate::type_code_of_v<T>) {
    return f.template operator()<T>(std::forward<Fnargs>(args)...);
  }
  return type_dispatch_impl<Types...>(code, f, std::forward<Fnargs>(args)...);
}

template <typename... Types, typename Functor, typename... Fnargs>
constexpr decltype(auto) type_dispatch(legate::Type::Code code, Functor&& f, Fnargs&&... args)
{
  return type_dispatch_impl<Types...>(code, f, std::forward<Fnargs>(args)...);
}

template <typename Functor, typename... Fnargs>
constexpr decltype(auto) type_dispatch_float(legate::Type::Code code, Functor&& f, Fnargs&&... args)
{
  type_dispatch<float, double>(code, f, std::forward<Fnargs>(args)...);
}

template <typename T, typename OpT>
void AllReduce(legate::TaskContext context, T* x, int count, OpT op)
{
  auto domain      = context.get_launch_domain();
  size_t num_ranks = domain.get_volume();
  EXPECT(num_ranks == 1 || context.num_communicators() > 0,
         "Expected a CPU communicator for multi-rank task.");
  if (count == 0 || context.num_communicators() == 0) return;
  auto comm = context.communicator(0);
  legate::comm::coll::CollDataType type;
  if (std::is_same<T, float>::value)
    type = legate::comm::coll::CollDataType::CollFloat;
  else if (std::is_same<T, double>::value)
    type = legate::comm::coll::CollDataType::CollDouble;
  else
    EXPECT(false, "Unsupported type.");
  auto comm_ptr = comm.get<legate::comm::coll::CollComm>();
  EXPECT(comm_ptr != nullptr, "CPU communicator is null.");
  size_t items_per_rank = (count + num_ranks - 1) / num_ranks;
  std::vector<T> data(items_per_rank * num_ranks);
  std::copy(x, x + count, data.begin());
  std::vector<T> recvbuf(items_per_rank * num_ranks);

  legate::comm::coll::collAlltoall(data.data(), recvbuf.data(), items_per_rank, type, comm_ptr);

  // Sum partials
  std::vector<T> partials(items_per_rank, 0.0);
  for (size_t j = 0; j < items_per_rank; j++) {
    for (size_t i = 0; i < num_ranks; i++) {
      partials[j] = op(partials[j], recvbuf[i * items_per_rank + j]);
    }
  }

  legate::comm::coll::collAllgather(
    partials.data(), recvbuf.data(), items_per_rank, type, comm_ptr);
  std::copy(recvbuf.begin(), recvbuf.begin() + count, x);
}

template <typename T>
void SumAllReduce(legate::TaskContext context, T* x, int count)
{
  AllReduce(context, x, count, std::plus<T>());
}

/**
 * @brief Turns linear index into multi-dimension index.  Similar to numpy unravel.
 */
template <std::int32_t D>
__host__ __device__ auto UnravelIndex(std::size_t idx,
                                      legate::Rect<D, legate::coord_t> const& in_shape)
{
  auto extent = in_shape.hi - in_shape.lo;
  for (std::int32_t dim = 0; dim < D; dim++) { extent[dim] += 1; }
  // First find the point in sub-rectangle
  legate::Point<D, legate::coord_t> sub_p;
  static_assert(std::is_signed<decltype(D)>::value,
                "Don't change the type without changing the for loop.");
  for (std::int32_t dim = D; --dim > 0;) {
    auto s     = extent[dim];
    auto t     = idx / s;
    sub_p[dim] = idx - t * s;
    idx        = t;
  }
  sub_p[0] = idx;
  // Re-align the point to the original rectangle
  return sub_p + in_shape.lo;
}

template <int DIM>
class UnravelIter {
 public:
  using iterator_category = std::random_access_iterator_tag;
  using value_type        = legate::Point<DIM, legate::coord_t>;
  using difference_type   = std::int64_t;
  using pointer           = value_type*;
  using reference         = value_type&;
  legate::Rect<DIM, legate::coord_t> shape_;
  difference_type current_ = 0;
  __host__ __device__ UnravelIter(legate::Rect<DIM, legate::coord_t> shape) : shape_(shape) {}
  __host__ __device__ UnravelIter& operator++()
  {
    current_++;
    return *this;
  }
  template <typename DistanceT>
  __host__ __device__ UnravelIter& operator+=(DistanceT n)
  {
    current_ += n;
    return *this;
  }
  template <typename DistanceT>
  __host__ __device__ UnravelIter operator+(DistanceT n)
  {
    UnravelIter copy = *this;
    copy += n;
    return copy;
  }

  template <typename DistanceT>
  __host__ __device__ value_type operator[](DistanceT n) const
  {
    return UnravelIndex(current_ + n, shape_);
  }

  __host__ __device__ value_type operator*() const { return UnravelIndex(current_, shape_); }
};

template <size_t I = 0, typename... Tp>
void extract_scalars(std::tuple<Tp...>& t, legate::TaskContext context)
{
  using T        = typename std::tuple_element<I, std::tuple<Tp...>>::type;
  std::get<I>(t) = context.scalar(I).value<T>();
  if constexpr (I + 1 != sizeof...(Tp)) extract_scalars<I + 1>(t);
}
inline void extract_scalars(std::tuple<>& t, legate::TaskContext context) {}

template <typename F, int OpCode>
class UnaryOpTask : public Task<UnaryOpTask<F, OpCode>, OpCode> {
 public:
  template <std::int32_t kDim, typename Policy>
  struct DispatchTypeOp {
    template <typename T>
    void operator()(legate::TaskContext& context, legate::PhysicalArray const& in, Policy& policy)
    {
      typename F::ArgsT op_args;
      extract_scalars(op_args, context);
      auto f                    = std::make_from_tuple<F>(op_args);
      legate::PhysicalArray out = context.output(0);
      if (out.dim() != in.dim()) { throw legate::TaskException{"Dimension mismatch."}; }

      auto in_accessor  = in.data().read_accessor<T, kDim>();
      auto out_accessor = out.data().write_accessor<T, kDim>();

      auto in_shape  = in.shape<kDim>();
      auto out_shape = out.shape<kDim>();

      auto v = out_shape.volume();

      // If we use `in_accessor.ptr(in_shape)` instead of accessors, there's an
      // error from legate when repeating tests with high dimension inputs:
      //
      // ERROR: Illegal request for pointer of non-dense rectangle
      thrust::for_each_n(
        policy, UnravelIter(in_shape), v, [=] __host__ __device__(const legate::Point<kDim>& p) {
          out_accessor[p] = f(in_accessor[p]);
        });
    }
  };
  struct DispatchDimOp {
    template <std::int32_t kDim, typename Policy>
    void operator()(legate::TaskContext& context, legate::PhysicalArray const& in, Policy& policy)
    {
      type_dispatch_float(in.type().code(), DispatchTypeOp<kDim, Policy>{}, context, in, policy);
    }
  };
  static void cpu_variant(legate::TaskContext context)
  {
    auto const& in = context.input(0);
    legate::dim_dispatch(in.dim(), DispatchDimOp{}, context, in, thrust::host);
  }
#ifdef LEGATEBOOST_USE_CUDA
  static void gpu_variant(legate::TaskContext context);
#endif
};

}  // namespace legateboost

#ifdef __CUDACC__
#include "../cpp_utils/cpp_utils.cuh"
#endif
