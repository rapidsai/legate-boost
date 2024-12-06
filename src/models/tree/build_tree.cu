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
#include <cuda/std/tuple>
#include <cuda/functional>
#include <thrust/execution_policy.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/sort.h>
#include <thrust/random.h>
#include <thrust/unique.h>
#include <cstddef>
#include <numeric>
#include <limits>
#include <vector>
#include <algorithm>
#include "legate_library.h"
#include "legateboost.h"
#include "../../cpp_utils/cpp_utils.h"
#include "../../cpp_utils/cpp_utils.cuh"
#include "legate/comm/coll.h"
#include "build_tree.h"
#include "matrix_types.h"

namespace legateboost {

namespace {

struct NodeBatch {
  int32_t node_idx_begin{};
  int32_t node_idx_end{};
  tcb::span<cuda::std::tuple<int32_t, int32_t>> instances;
  __host__ __device__ auto InstancesInBatch() const -> std::size_t { return instances.size(); }
  __host__ __device__ auto NodesInBatch() const -> std::size_t
  {
    return node_idx_end - node_idx_begin;
  }
};

class GradientQuantiser {
  GPair scale;
  GPair inverse_scale;

 public:
  struct GetAbsGPair {
    int num_outputs;
    legate::AccessorRO<double, 3> g;
    legate::AccessorRO<double, 3> h;
    __device__ auto operator()(int n) const -> GPair
    {
      legate::Point<3> const p = {n / num_outputs, 0, n % num_outputs};
      return GPair{abs(g[p]), abs(h[p])};
    }
  };

  // Calculate scale from upper bound on data
  GradientQuantiser(legate::TaskContext context,
                    const legate::AccessorRO<double, 3>& g,
                    const legate::AccessorRO<double, 3>& h,
                    legate::Rect<3> g_shape,
                    cudaStream_t stream)
  {
    auto thrust_alloc = ThrustAllocator(legate::Memory::GPU_FB_MEM);
    auto policy       = DEFAULT_POLICY(thrust_alloc).on(stream);
    auto counting     = thrust::make_counting_iterator(0);
    auto num_outputs  = g_shape.hi[2] - g_shape.lo[2] + 1;
    auto n            = (g_shape.hi[0] - g_shape.lo[0] + 1) * num_outputs;
    auto zip_gpair =
      thrust::make_transform_iterator(counting, GetAbsGPair{narrow<int>(num_outputs), g, h});
    GPair local_abs_sum =
      thrust::reduce(policy, zip_gpair, zip_gpair + n, GPair{0.0, 0.0}, thrust::plus<GPair>());

    auto local_abs_sum_device = legate::create_buffer<GPair, 1>(1);
    CHECK_CUDA(cudaMemcpyAsync(
      local_abs_sum_device.ptr(0), &local_abs_sum, sizeof(GPair), cudaMemcpyHostToDevice, stream));
    // Take the max of the local sums
    AllReduce(context,
              // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
              tcb::span<double>{reinterpret_cast<double*>(local_abs_sum_device.ptr(0)), 2},
              ncclMax,
              stream);
    CHECK_CUDA(cudaMemcpyAsync(
      &local_abs_sum, local_abs_sum_device.ptr(0), sizeof(GPair), cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));

    // We will quantise values between -max_int and max_int
    int64_t const max_int = std::numeric_limits<int32_t>::max();
    scale.grad         = local_abs_sum.grad == 0 ? 1 : narrow<double>(max_int) / local_abs_sum.grad;
    scale.hess         = local_abs_sum.hess == 0 ? 1 : narrow<double>(max_int) / local_abs_sum.hess;
    inverse_scale.grad = 1.0 / scale.grad;
    inverse_scale.hess = 1.0 / scale.hess;
  }

  // Round gradient and hessian using stochastic rounding
  // Thus the expected value of the quantised value is unbiased
  // Also the expected error grows as O(1/sqrt(n)) where n is the number of samples
  // Vs. O(1/n) for round nearest
  // The seed here should be unique for each gpair over each boosting iteration
  // Use a hash combine function to generate the seed
  __device__ auto QuantiseStochasticRounding(GPair value, int64_t seed) const -> IntegerGPair
  {
    thrust::default_random_engine eng(seed);
    thrust::uniform_real_distribution<double> dist(0.0, 1.0);
    auto scaled_grad            = value.grad * scale.grad;
    auto scaled_hess            = value.hess * scale.hess;
    double const grad_remainder = scaled_grad - floor(scaled_grad);
    double const hess_remainder = scaled_hess - floor(scaled_hess);
    // We won't check for overflow here as this is performance critical
    // If our calculation of the scale factor is correct we should never overflow
    // NOLINTBEGIN(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
    IntegerGPair::value_type const grad_quantised =
      floor(scaled_grad) + static_cast<double>(dist(eng) < grad_remainder);
    IntegerGPair::value_type const hess_quantised =
      floor(scaled_hess) + static_cast<double>(dist(eng) < hess_remainder);
    // NOLINTEND(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
    return IntegerGPair{grad_quantised, hess_quantised};
  }

  __device__ auto Dequantise(IntegerGPair value) const -> GPair
  {
    GPair result;
    // NOLINTBEGIN(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
    result.grad = value.grad * inverse_scale.grad;
    result.hess = value.hess * inverse_scale.hess;
    // NOLINTEND(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
    return result;
  }
};

// Hash function fmix64 from MurmurHash3
__device__ auto hash(int64_t k) -> int64_t
{
  // We will assume murmurhash is correct here and ignore clang-tidy warnings
  // NOLINTBEGIN(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
  k ^= k >> 33;
  k *= 0xff51afd7ed558ccd;
  k ^= k >> 33;
  k *= 0xc4ceb9fe1a85ec53;
  k ^= k >> 33;
  // NOLINTEND(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
  return k;
}

__device__ auto hash_combine(int64_t seed) -> int64_t { return seed; }

// Hash combine from boost
// This function is used to combine several random seeds e.g. a 3d index
template <typename... Rest>
__device__ auto hash_combine(int64_t seed, const int64_t& v, Rest... rest) -> int64_t
{
  seed ^= hash(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return hash_combine(seed, rest...);
}

// NOLINTBEGIN(performance-unnecessary-value-param)
template <int BLOCK_THREADS>
__global__ void __launch_bounds__(BLOCK_THREADS)
  reduce_base_sums(legate::AccessorRO<double, 3> g,
                   legate::AccessorRO<double, 3> h,
                   size_t n_local_samples,
                   int64_t sample_offset,
                   legate::Buffer<IntegerGPair, 2> node_sums,
                   GradientQuantiser quantiser,
                   int64_t seed)
{
  using BlockReduce = cub::BlockReduce<IntegerGPair, BLOCK_THREADS>;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  auto output = blockIdx.y;

  int64_t const sample_id = threadIdx.x + (blockDim.x * blockIdx.x);

  legate::Point<3> p = {sample_id + sample_offset, 0, output};
  double const grad  = sample_id < n_local_samples ? g[p] : 0.0;
  double const hess  = sample_id < n_local_samples ? h[p] : 0.0;

  auto quantised =
    quantiser.QuantiseStochasticRounding({grad, hess}, hash_combine(seed, p[0], p[2]));
  IntegerGPair const blocksum = BlockReduce(temp_storage).Sum(quantised);

  if (threadIdx.x == 0) {
    // Need to reinterpret cast here because cuda has no atomicAdd for int64_t, only unsigned long
    // long NOLINTBEGIN(cppcoreguidelines-pro-type-reinterpret-cast)
    atomicAdd(
      reinterpret_cast<Histogram<IntegerGPair>::atomic_add_type*>(&node_sums[{0, output}].grad),
      blocksum.grad);
    atomicAdd(
      reinterpret_cast<Histogram<IntegerGPair>::atomic_add_type*>(&node_sums[{0, output}].hess),
      blocksum.hess);
    // NOLINTEND(cppcoreguidelines-pro-type-reinterpret-cast)
  }
}
// NOLINTEND(performance-unnecessary-value-param)

using SharedMemoryHistogramType = GPairBase<int32_t>;
// NOTE: changes to the below should be reflected in the python Tree learner constructor and its
// documentation
const int kMaxSharedBins = 2048;  // 16KB shared memory. More is not helpful and creates more cache
                                  // misses for binary search in split_proposals.

template <typename MatrixT,
          int kBlockThreads,
          int kItemsPerThread,
          int kItemsPerTile = kBlockThreads * kItemsPerThread>
struct HistogramAgent {
  using T                      = typename MatrixT::value_type;
  static const int kImpureTile = -1;  // Special value for a tile that is not pure (contains
                                      // multiple nodes)
  struct SharedMemoryHistogram {
    SharedMemoryHistogramType* data;
    // -1 means no node is currently being processed
    int current_node = kImpureTile;
    int begin_idx    = 0;
    int end_idx      = 0;

    __device__ SharedMemoryHistogram(SharedMemoryHistogramType* shared, int begin_idx, int end_idx)
      : data(shared), begin_idx(begin_idx), end_idx(end_idx)
    {
    }
    // Write out to global memory
    __device__ void Flush(Histogram<IntegerGPair>& histogram, int output)
    {
      if (current_node == kImpureTile) { return; }
      __syncthreads();

      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
      auto* src_ptr = reinterpret_cast<SharedMemoryHistogramType::value_type*>(data);
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
      auto* dest_ptr = reinterpret_cast<Histogram<IntegerGPair>::atomic_add_type*>(
        &histogram[{current_node, output, begin_idx}]);

      for (auto i = narrow_cast<int>(threadIdx.x); i < (end_idx - begin_idx) * 2;
           i += kBlockThreads) {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
        atomicAdd(&dest_ptr[i], src_ptr[i]);
      }
    }
    __device__ void LazyInit(int node_id, Histogram<IntegerGPair>& histogram, int output)
    {
      if (current_node == node_id) { return; }
      this->Flush(histogram, output);
      current_node = node_id;
      __syncthreads();
      // Zero data
      for (int i = narrow_cast<int>(threadIdx.x); i < end_idx - begin_idx; i += kBlockThreads) {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
        data[i] = SharedMemoryHistogramType{0, 0};
      }
      __syncthreads();
    }
    __device__ void Add(int bin_idx, IntegerGPair gpair)
    {
      using AddType = typename SharedMemoryHistogramType::value_type;
      // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic,cppcoreguidelines-pro-type-reinterpret-cast)
      auto* addPosition = reinterpret_cast<AddType*>(data + (bin_idx - begin_idx));
      // These gradients will not overflow as they are quantised such that the sum cannot exceed
      // int32_t max
      atomicAdd(addPosition, narrow_cast<AddType>(gpair.grad));
      // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
      atomicAdd(addPosition + 1, narrow_cast<AddType>(gpair.hess));
    }
  };

  const MatrixT& X;
  const legate::AccessorRO<double, 3>& g;
  const legate::AccessorRO<double, 3>& h;
  const size_t& n_outputs;
  const SparseSplitProposals<T>& split_proposals;
  const NodeBatch& batch;
  Histogram<IntegerGPair>& histogram;
  const legate::Buffer<IntegerGPair, 2>& node_sums;
  const GradientQuantiser& quantiser;
  const int64_t& seed;
  int output;
  int feature_begin;
  int feature_end;
  int feature_stride;
  SharedMemoryHistogram shared_histogram;

  __device__ HistogramAgent(const MatrixT& X,
                            const legate::AccessorRO<double, 3>& g,
                            const legate::AccessorRO<double, 3>& h,
                            const size_t& n_outputs,
                            const SparseSplitProposals<T>& split_proposals,
                            const NodeBatch& batch,
                            Histogram<IntegerGPair>& histogram,
                            const legate::Buffer<IntegerGPair, 2>& node_sums,
                            const GradientQuantiser& quantiser,
                            const legate::Buffer<int>& feature_groups,
                            const int64_t& seed,
                            SharedMemoryHistogramType* shared_memory)
    : X(X),
      g(g),
      h(h),
      n_outputs(n_outputs),
      split_proposals(split_proposals),
      batch(batch),
      histogram(histogram),
      node_sums(node_sums),
      quantiser(quantiser),
      seed(seed),
      feature_begin(feature_groups[blockIdx.y]),
      feature_end(feature_groups[blockIdx.y + 1]),
      feature_stride(feature_end - feature_begin),
      output(narrow_cast<int>(blockIdx.z)),
      shared_histogram(shared_memory,
                       split_proposals.row_pointers[feature_groups[blockIdx.y]],
                       split_proposals.row_pointers[feature_groups[blockIdx.y + 1]])
  {
  }

  HistogramAgent(const HistogramAgent&)                    = delete;
  HistogramAgent(HistogramAgent&&)                         = delete;
  auto operator=(const HistogramAgent&) -> HistogramAgent& = delete;
  auto operator=(HistogramAgent&&) -> HistogramAgent&      = delete;
  ~HistogramAgent()                                        = default;

  __device__ void ProcessPartialTileGlobal(std::size_t offset, std::size_t end)
  {
    for (std::size_t idx = offset + threadIdx.x; idx < end; idx += kBlockThreads) {
      uint32_t const row_index             = idx / feature_stride;
      int const feature                    = feature_begin + (idx % feature_stride);
      auto [sample_node, local_sample_idx] = (row_index < batch.InstancesInBatch())
                                               ? batch.instances[row_index]
                                               : cuda::std::make_tuple(-1, -1);

      const bool computeHistogram =
        row_index < batch.InstancesInBatch() &&
        ComputeHistogramBin(
          sample_node, node_sums, histogram.ContainsNode(BinaryTree::Parent(sample_node)));
      if (!computeHistogram) { continue; }

      auto x = X.Get(X.RowRange().lo[0] + local_sample_idx, feature);
      // int bin_idx = shared_split_proposals.FindBin(x, feature);
      int const bin_idx = split_proposals.FindBin(x, feature);

      legate::Point<3> p = {X.RowRange().lo[0] + local_sample_idx, 0, output};
      auto gpair_quantised =
        quantiser.QuantiseStochasticRounding({g[p], h[p]}, hash_combine(seed, p[0], p[2]));
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
      auto* addPosition = reinterpret_cast<typename IntegerGPair::value_type*>(
        // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
        &histogram[{sample_node, output, bin_idx}]);

      if (bin_idx != SparseSplitProposals<T>::NOT_FOUND) {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        auto* addPosition = reinterpret_cast<Histogram<IntegerGPair>::atomic_add_type*>(
          &histogram[{sample_node, output, bin_idx}]);
        atomicAdd(addPosition, gpair_quantised.grad);
        // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
        atomicAdd(addPosition + 1, gpair_quantised.hess);
      }
    }
  }

  __device__ void ProcessTileShared(std::size_t offset, int node_id)
  {
    // Disable this check as cuda loop unrolling is fairly standard and safe
    // NOLINTBEGIN(cppcoreguidelines-pro-bounds-constant-array-index)

    // If this whole tile has a node that we don't need to compute
    // Early exit
    if (!ComputeHistogramBin(
          node_id, node_sums, histogram.ContainsNode(BinaryTree::Parent(node_id)))) {
      return;
    }

    shared_histogram.LazyInit(node_id, histogram, output);

    std::array<int, kItemsPerThread> sample_node{};
    std::array<int, kItemsPerThread> local_sample_idx{};
    std::array<int, kItemsPerThread> feature{};
#pragma unroll
    for (int i = 0; i < kItemsPerThread; i++) {
      auto idx = offset + (static_cast<std::size_t>(i * kBlockThreads)) + threadIdx.x;
      uint32_t const row_index                            = idx / feature_stride;
      feature[i]                                          = feature_begin + idx % feature_stride;
      cuda::std::tie(sample_node[i], local_sample_idx[i]) = batch.instances[row_index];
    }

    std::array<T, kItemsPerThread> x{};
#pragma unroll
    for (int i = 0; i < kItemsPerThread; i++) {
      x[i] = X.Get(X.RowRange().lo[0] + local_sample_idx[i], feature[i]);
    }

    std::array<int, kItemsPerThread> bin_idx{};
#pragma unroll
    for (int i = 0; i < kItemsPerThread; i++) {
      bin_idx[i] = split_proposals.FindBin(x[i], feature[i]);
    }
    std::array<IntegerGPair, kItemsPerThread> gpair{};
#pragma unroll
    for (int i = 0; i < kItemsPerThread; i++) {
      legate::Point<3> p = {X.RowRange().lo[0] + local_sample_idx[i], 0, output};
      gpair[i] =
        bin_idx[i] != SparseSplitProposals<T>::NOT_FOUND
          ? quantiser.QuantiseStochasticRounding({g[p], h[p]}, hash_combine(seed, p[0], p[2]))
          : IntegerGPair{0, 0};
    }
#pragma unroll
    for (int i = 0; i < kItemsPerThread; i++) {
      if (bin_idx[i] != SparseSplitProposals<T>::NOT_FOUND) {
        shared_histogram.Add(bin_idx[i], gpair[i]);
      }
    }
    // NOLINTEND(cppcoreguidelines-pro-bounds-constant-array-index)
  }

  __device__ auto GetTileNode(std::size_t offset) -> int
  {
    // Check the first and last element here and see if they are in the same node
    int const begin_ridx            = offset / feature_stride;
    int const end_ridx              = (offset + kItemsPerTile - 1) / feature_stride;
    auto [begin_node, begin_sample] = batch.instances[begin_ridx];
    auto [end_node, end_sample]     = batch.instances[end_ridx];
    return begin_node == end_node ? begin_node : kImpureTile;
  }

  __device__ void BuildHistogram()
  {
    // Loop across 1st grid dimension
    // Second dimension is feature groups
    // Third dimension is output
    std::size_t const n_elements = batch.InstancesInBatch() * feature_stride;
    auto offset                  = static_cast<std::size_t>(blockIdx.x) * kItemsPerTile;
    while (offset + kItemsPerTile <= n_elements) {
      int const tile_node_id = this->GetTileNode(offset);
      // If all threads here have the same node we can use shared memory
      if (tile_node_id == kImpureTile) {
        ProcessPartialTileGlobal(offset, offset + kItemsPerTile);
      } else {
        ProcessTileShared(offset, tile_node_id);
      }
      offset += static_cast<std::size_t>(kItemsPerTile * gridDim.x);
    }
    ProcessPartialTileGlobal(offset, n_elements);

    // Flush any remaining sums to the global histogram
    shared_histogram.Flush(histogram, output);
  }
};

// NOLINTBEGIN(performance-unnecessary-value-param)
template <typename MatrixT, int kBlockThreads, int kItemsPerThread>
__global__ void __launch_bounds__(kBlockThreads)
  fill_histogram_shared(MatrixT X,
                        legate::AccessorRO<double, 3> g,
                        legate::AccessorRO<double, 3> h,
                        size_t n_outputs,
                        SparseSplitProposals<typename MatrixT::value_type> split_proposals,
                        NodeBatch batch,
                        Histogram<IntegerGPair> histogram,
                        legate::Buffer<IntegerGPair, 2> node_sums,
                        GradientQuantiser quantiser,
                        legate::Buffer<int> feature_groups,
                        int64_t seed)
{
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays,hicpp-avoid-c-arrays)
  __shared__ char shared_char[kMaxSharedBins * sizeof(SharedMemoryHistogramType)];
  // Allocate as char and then cast to type. This is because we dont want to initialise the type.
  auto* shared_memory =
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    reinterpret_cast<SharedMemoryHistogramType*>(shared_char);
  HistogramAgent<MatrixT, kBlockThreads, kItemsPerThread> agent(X,
                                                                g,
                                                                h,
                                                                n_outputs,
                                                                split_proposals,
                                                                batch,
                                                                histogram,
                                                                node_sums,
                                                                quantiser,
                                                                feature_groups,
                                                                seed,
                                                                shared_memory);
  agent.BuildHistogram();
}
// NOLINTEND(performance-unnecessary-value-param)

// Manage the launch parameters for histogram kernel
template <typename MatrixT, std::int32_t kBlockThreads = 1024, std::int32_t kItemsPerThread = 4>
struct HistogramKernel {
  using T                                 = typename MatrixT::value_type;
  static const std::int32_t kItemsPerTile = kBlockThreads * kItemsPerThread;
  legate::Buffer<int> feature_groups;
  int num_groups{};
  int maximum_blocks_for_occupancy;
  HistogramKernel(const SparseSplitProposals<T>& split_proposals, cudaStream_t stream)
  {
    int device = 0;
    CHECK_CUDA(cudaGetDevice(&device));
    std::int32_t n_mps = 0;
    CHECK_CUDA(cudaDeviceGetAttribute(&n_mps, cudaDevAttrMultiProcessorCount, device));
    std::int32_t n_blocks_per_mp = 0;
    CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &n_blocks_per_mp,
      fill_histogram_shared<MatrixT, kBlockThreads, kItemsPerThread>,
      kBlockThreads,
      0));
    this->maximum_blocks_for_occupancy = n_blocks_per_mp * n_mps;
    FindFeatureGroups(split_proposals, stream);
  }

  void FindFeatureGroups(const SparseSplitProposals<T>& split_proposals, cudaStream_t stream)
  {
    // Find feature groups
    // This is a bin packing problem
    // We want to pack as many features as possible into a group
    std::vector<int> split_proposal_row_pointers(split_proposals.num_features + 1);
    CHECK_CUDA(cudaMemcpyAsync(split_proposal_row_pointers.data(),
                               split_proposals.row_pointers.ptr(0),
                               (split_proposals.num_features + 1) * sizeof(int),
                               cudaMemcpyDeviceToHost,
                               stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    std::vector<int> feature_groups({0});
    int current_bins_in_group = 0;
    for (int i = 0; i < split_proposals.num_features; i++) {
      int const bins_in_feature =
        split_proposal_row_pointers[i + 1] - split_proposal_row_pointers[i];
      EXPECT(bins_in_feature <= kMaxSharedBins, "Too many bins in a feature");
      if (current_bins_in_group + bins_in_feature > kMaxSharedBins) {
        feature_groups.push_back(i);
        current_bins_in_group = 0;
      }
      current_bins_in_group += bins_in_feature;
    }
    feature_groups.push_back(split_proposals.num_features);
    num_groups = narrow<int>(feature_groups.size() - 1);
    EXPECT(num_groups * kMaxSharedBins >= split_proposals.histogram_size, "Too few feature groups");
    this->feature_groups = legate::create_buffer<int>(num_groups + 1);
    CHECK_CUDA(cudaMemcpyAsync(this->feature_groups.ptr(0),
                               feature_groups.data(),
                               (num_groups + 1) * sizeof(int),
                               cudaMemcpyHostToDevice,
                               stream));
  }

  void BuildHistogram(const MatrixT& X,
                      const legate::AccessorRO<double, 3>& g,
                      const legate::AccessorRO<double, 3>& h,
                      size_t n_outputs,
                      const SparseSplitProposals<T>& split_proposals,
                      NodeBatch batch,
                      const Histogram<IntegerGPair>& histogram,
                      const legate::Buffer<IntegerGPair, 2>& node_sums,
                      GradientQuantiser quantiser,
                      int64_t seed,
                      cudaStream_t stream)
  {
    int const average_features_per_group = split_proposals.num_features / num_groups;
    std::size_t const average_elements_per_group =
      batch.InstancesInBatch() * average_features_per_group;
    auto min_blocks  = (average_elements_per_group + kItemsPerTile - 1) / kItemsPerTile;
    auto x_grid_size = std::min(static_cast<uint64_t>(maximum_blocks_for_occupancy), min_blocks);
    // Launch the kernel
    fill_histogram_shared<MatrixT, kBlockThreads, kItemsPerThread>
      <<<dim3(x_grid_size, num_groups, n_outputs), kBlockThreads, 0, stream>>>(X,
                                                                               g,
                                                                               h,
                                                                               n_outputs,
                                                                               split_proposals,
                                                                               batch,
                                                                               histogram,
                                                                               node_sums,
                                                                               quantiser,
                                                                               feature_groups,
                                                                               seed);
  }
};

// The only way to achieve this AFAIK is with reinterpret cast
// NOLINTBEGIN(cppcoreguidelines-pro-type-reinterpret-cast)
__device__ auto vectorised_load(const IntegerGPair* ptr) -> IntegerGPair
{
  static_assert(sizeof(IntegerGPair) == sizeof(int4), "size inconsistent");
  auto load = *reinterpret_cast<const int4*>(ptr);
  return *reinterpret_cast<const IntegerGPair*>(&load);
}

__device__ void vectorised_store(IntegerGPair* ptr, IntegerGPair value)
{
  static_assert(sizeof(IntegerGPair) == sizeof(int4), "size inconsistent");
  auto* store = reinterpret_cast<int4*>(ptr);
  *store      = *reinterpret_cast<int4*>(&value);
}
// NOLINTEND(cppcoreguidelines-pro-type-reinterpret-cast)

// NOLINTBEGIN(performance-unnecessary-value-param)
template <typename T, int BLOCK_THREADS>
__global__ void __launch_bounds__(BLOCK_THREADS)
  scan_kernel(Histogram<IntegerGPair> histogram,
              legate::Buffer<IntegerGPair, 2> node_sums,
              int n_features,
              SparseSplitProposals<T> split_proposals,
              NodeBatch batch)

{
  const int kWarpThreads = 32;
  const auto lane_idx    = narrow_cast<int>(threadIdx.x % kWarpThreads);
  auto const rank        = narrow<int>((blockIdx.x * blockDim.x + threadIdx.x) / kWarpThreads);
  auto const num_nodes   = narrow<int>(batch.NodesInBatch());
  int const i            = rank / num_nodes;
  int const j            = rank % num_nodes;
  auto const output      = narrow_cast<int>(blockIdx.y);

  // Specialize WarpScan for type int
  using WarpScan = cub::WarpScan<IntegerGPair>;

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays,hicpp-avoid-c-arrays)
  __shared__ typename WarpScan::TempStorage temp_storage[BLOCK_THREADS / kWarpThreads];

  int const scan_node_idx = batch.node_idx_begin + j;
  int const parent        = BinaryTree::Parent(scan_node_idx);
  // Exit if we didn't compute this histogram
  if (node_sums[{scan_node_idx, output}].hess <= 0) { return; }
  if (!ComputeHistogramBin(scan_node_idx, node_sums, histogram.ContainsNode(parent))) { return; }
  if (i >= n_features || scan_node_idx >= batch.node_idx_end) { return; }

  const int feature_idx             = i;
  auto [feature_begin, feature_end] = split_proposals.FeatureRange(feature_idx);
  const int num_bins                = feature_end - feature_begin;
  const int num_tiles               = (num_bins + kWarpThreads - 1) / kWarpThreads;

  IntegerGPair aggregate;
  // Scan left side
  for (int tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
    const int bin_idx              = feature_begin + tile_idx * kWarpThreads + lane_idx;
    bool const thread_participates = bin_idx < feature_end;
    auto e = thread_participates ? vectorised_load(&histogram[{scan_node_idx, output, bin_idx}])
                                 : IntegerGPair{0, 0};
    IntegerGPair tile_aggregate;
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-constant-array-index)
    WarpScan(temp_storage[threadIdx.x / kWarpThreads]).InclusiveSum(e, e, tile_aggregate);
    e += aggregate;
    // Skip write if data is 0
    // This actually helps quite a bit at deeper tree levels where we have a lot of empty bins
    if (thread_participates && (e.grad > 0 || e.hess > 0)) {
      vectorised_store(&histogram[{scan_node_idx, output, bin_idx}], e);
    }
    aggregate += tile_aggregate;
  }

  // This node has no sibling we are finished
  if (scan_node_idx == 0) { return; }

  int const sibling_node_idx = BinaryTree::Sibling(scan_node_idx);

  // The sibling did not compute a histogram
  // Do the subtraction trick using the histogram we just computed in the previous step
  if (!ComputeHistogramBin(sibling_node_idx, node_sums, histogram.ContainsNode(parent))) {
    // Infer right side
    for (int bin_idx = feature_begin + lane_idx; bin_idx < feature_end; bin_idx += kWarpThreads) {
      auto scanned_sum = vectorised_load(&histogram[{scan_node_idx, output, bin_idx}]);
      auto parent_sum =
        vectorised_load(&histogram[{BinaryTree::Parent(scan_node_idx), output, bin_idx}]);
      auto other_sum = parent_sum - scanned_sum;
      vectorised_store(&histogram[{sibling_node_idx, output, bin_idx}], other_sum);
    }
  }
}
// NOLINTEND(performance-unnecessary-value-param)

// Key/value pair to simplify reduction
struct GainFeaturePair {
  double gain;
  int bin_idx;
  __device__ auto operator>(const GainFeaturePair& other) const -> bool
  {
    return gain > other.gain;
  }
};

// NOLINTBEGIN(performance-unnecessary-value-param)
template <typename TYPE, int BLOCK_THREADS>
__global__ void __launch_bounds__(BLOCK_THREADS)
  perform_best_split(Histogram<IntegerGPair> histogram,
                     size_t n_outputs,
                     SparseSplitProposals<TYPE> split_proposals,
                     double eps,
                     double alpha,
                     legate::Buffer<double, 2> tree_leaf_value,
                     legate::Buffer<IntegerGPair, 2> node_sums,
                     legate::Buffer<int32_t, 1> tree_feature,
                     legate::Buffer<double, 1> tree_split_value,
                     legate::Buffer<double, 1> tree_gain,
                     NodeBatch batch,
                     GradientQuantiser quantiser)
{
  // using one block per (level) node to have blockwise reductions
  int const node_id = narrow<int>(batch.node_idx_begin + blockIdx.x);
  // Early exit if this node has no samples
  if (vectorised_load(&node_sums[{node_id, 0}]).hess <= 0) { return; }

  using BlockReduce = cub::BlockReduce<GainFeaturePair, BLOCK_THREADS>;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  __shared__ double node_best_gain;
  __shared__ int node_best_bin_idx;

  double thread_best_gain = 0;
  int thread_best_bin_idx = -1;

  for (int bin_idx = narrow_cast<int>(threadIdx.x); bin_idx < split_proposals.histogram_size;
       bin_idx += BLOCK_THREADS) {
    double gain = 0;
    for (int output = 0; output < n_outputs; ++output) {
      auto node_sum  = vectorised_load(&node_sums[{node_id, output}]);
      auto left_sum  = vectorised_load(&histogram[{node_id, output, bin_idx}]);
      auto right_sum = node_sum - left_sum;
      if (left_sum.hess <= 0 || right_sum.hess <= 0) {
        gain = 0;
        break;
      }
      double const reg = std::max(eps, alpha);  // Regularisation term
      auto [G, H]      = quantiser.Dequantise(node_sum);
      gain -= (G * G) / (H + reg);
      auto [G_L, H_L] = quantiser.Dequantise(left_sum);

      gain += (G_L * G_L) / (H_L + reg);

      auto [G_R, H_R] = quantiser.Dequantise(right_sum);
      gain += (G_R * G_R) / (H_R + reg);
    }
    gain *= 0.5;
    if (gain > thread_best_gain) {
      thread_best_gain    = gain;
      thread_best_bin_idx = bin_idx;
    }
  }

  // SYNC BEST GAIN TO FULL BLOCK/NODE
  GainFeaturePair const thread_best_pair{thread_best_gain, thread_best_bin_idx};
  GainFeaturePair const node_best_pair =
    BlockReduce(temp_storage).Reduce(thread_best_pair, cub::Max(), BLOCK_THREADS);
  if (threadIdx.x == 0) {
    node_best_gain    = node_best_pair.gain;
    node_best_bin_idx = node_best_pair.bin_idx;
  }
  __syncthreads();

  if (node_best_gain > eps) {
    int const node_best_feature = split_proposals.FindFeature(node_best_bin_idx);
    for (int output = narrow_cast<int>(threadIdx.x); output < n_outputs; output += BLOCK_THREADS) {
      auto node_sum  = vectorised_load(&node_sums[{node_id, output}]);
      auto left_sum  = vectorised_load(&histogram[{node_id, output, node_best_bin_idx}]);
      auto right_sum = node_sum - left_sum;
      node_sums[{BinaryTree::LeftChild(node_id), output}]  = left_sum;
      node_sums[{BinaryTree::RightChild(node_id), output}] = right_sum;

      auto [G_L, H_L] = quantiser.Dequantise(left_sum);
      tree_leaf_value[{BinaryTree::LeftChild(node_id), output}] =
        CalculateLeafValue(G_L, H_L, alpha);

      auto [G_R, H_R] = quantiser.Dequantise(right_sum);
      tree_leaf_value[{BinaryTree::RightChild(node_id), output}] =
        CalculateLeafValue(G_R, H_R, alpha);

      if (output == 0) {
        tree_feature[node_id]     = node_best_feature;
        tree_split_value[node_id] = split_proposals.split_proposals[node_best_bin_idx];
        tree_gain[node_id]        = node_best_gain;
      }
    }
  }
}
// NOLINTEND(performance-unnecessary-value-param)

struct Tree {
  template <typename THRUST_POLICY>
  Tree(int max_nodes, int num_outputs, cudaStream_t stream, const THRUST_POLICY& thrust_exec_policy)
    : num_outputs(num_outputs), max_nodes(max_nodes), stream(stream)
  {
    leaf_value  = legate::create_buffer<double, 2>({max_nodes, num_outputs});
    feature     = legate::create_buffer<int32_t, 1>(max_nodes);
    split_value = legate::create_buffer<double, 1>(max_nodes);
    gain        = legate::create_buffer<double, 1>(max_nodes);
    node_sums   = legate::create_buffer<IntegerGPair, 2>({max_nodes, num_outputs});
    // Legate doesn't provide a convenient way to do this other than pointers
    // NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    thrust::fill(thrust_exec_policy,
                 leaf_value.ptr({0, 0}),
                 leaf_value.ptr({0, 0}) + static_cast<ptrdiff_t>(max_nodes * num_outputs),
                 0.0);
    thrust::fill(thrust_exec_policy, feature.ptr(0), feature.ptr(0) + max_nodes, -1);
    thrust::fill(thrust_exec_policy, split_value.ptr(0), split_value.ptr(0) + max_nodes, 0.0);
    thrust::fill(thrust_exec_policy, gain.ptr(0), gain.ptr(0) + max_nodes, 0.0);
    thrust::fill(thrust_exec_policy,
                 node_sums.ptr({0, 0}),
                 node_sums.ptr({0, 0}) + static_cast<ptrdiff_t>(max_nodes * num_outputs),
                 IntegerGPair{0, 0});
    // NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  }

  template <typename T, int DIM, typename ThrustPolicyT>
  void WriteOutput(const legate::PhysicalStore& out,
                   const legate::Buffer<T, DIM>& x,
                   const ThrustPolicyT& policy)
  {
    // Write a tile of x to the output
    const legate::Rect<DIM> out_shape = out.shape<DIM>();
    auto out_acc                      = out.write_accessor<T, DIM>();
    thrust::for_each_n(policy,
                       UnravelIter(out_shape),
                       out_shape.volume(),
                       [=] __device__(const legate::Point<DIM>& p) { out_acc[p] = x[p]; });
  }

  void WriteTreeOutput(legate::TaskContext context, GradientQuantiser quantiser)
  {
    auto stream       = context.get_task_stream();
    auto thrust_alloc = ThrustAllocator(legate::Memory::GPU_FB_MEM);
    auto policy       = DEFAULT_POLICY(thrust_alloc).on(stream);

    WriteOutput(context.output(0).data(), leaf_value, policy);
    WriteOutput(context.output(1).data(), feature, policy);
    WriteOutput(context.output(2).data(), split_value, policy);
    WriteOutput(context.output(3).data(), gain, policy);

    // Dequantise and write the hessians
    auto hessian                        = context.output(4).data();
    const legate::Rect<2> hessian_shape = hessian.shape<2>();
    auto hessian_acc                    = hessian.write_accessor<double, 2>();
    auto node_sums                      = this->node_sums;  // Dont let device lambda capture this
    thrust::for_each_n(
      policy, UnravelIter(hessian_shape), hessian_shape.volume(), [=] __device__(auto p) {
        hessian_acc[p] = quantiser.Dequantise(node_sums[p]).hess;
      });
    CHECK_CUDA_STREAM(stream);
  }

  ~Tree()                              = default;
  Tree(const Tree&)                    = delete;
  Tree(Tree&&)                         = default;
  auto operator=(const Tree&) -> Tree& = delete;
  auto operator=(Tree&&) -> Tree&      = delete;

  legate::Buffer<double, 2> leaf_value;
  legate::Buffer<int32_t, 1> feature;
  legate::Buffer<double, 1> split_value;
  legate::Buffer<double, 1> gain;
  legate::Buffer<IntegerGPair, 2> node_sums;
  const int num_outputs;
  const int max_nodes;
  cudaStream_t stream;
};

// Randomly sample split_samples rows from X
// Use nccl to share the samples with all workers
// Remove any duplicates
// Return sparse matrix of split samples for each feature
template <typename T, template <typename> class XMatrix>
SparseSplitProposals<T> SelectSplitSamples(legate::TaskContext context,
                                           const XMatrix<T>& X,
                                           int split_samples,
                                           int seed,
                                           int64_t dataset_rows,
                                           cudaStream_t stream)
{
  auto thrust_alloc = ThrustAllocator(legate::Memory::GPU_FB_MEM);
  auto policy       = DEFAULT_POLICY(thrust_alloc).on(stream);
  // Randomly choose split_samples rows
  auto row_samples = legate::create_buffer<int64_t, 1>(split_samples);
  auto counting    = thrust::make_counting_iterator(0);
  thrust::transform(
    policy, counting, counting + split_samples, row_samples.ptr(0), [=] __device__(int64_t idx) {
      thrust::default_random_engine eng(seed);
      thrust::uniform_int_distribution<int64_t> dist(0, dataset_rows - 1);
      eng.discard(idx);
      return dist(eng);
    });
  auto draft_proposals = legate::create_buffer<T, 2>({X.NumFeatures(), split_samples});

  // fill with local data
  LaunchN(X.NumFeatures() * split_samples, stream, [=] __device__(auto idx) {
    auto i                  = idx / X.NumFeatures();
    auto j                  = idx % X.NumFeatures();
    auto row                = row_samples[i];
    bool has_data           = X.RowRange().contains(row);
    draft_proposals[{j, i}] = has_data ? X.Get(row, j) : T(0);
  });

  // Sum reduce over all workers
  SumAllReduce(
    context, tcb::span<T>(draft_proposals.ptr({0, 0}), X.NumFeatures() * split_samples), stream);

  CHECK_CUDA_STREAM(stream);

  // Condense split samples to unique values
  // First sort the samples
  auto keys = legate::create_buffer<int32_t, 1>(X.NumFeatures() * split_samples);
  thrust::transform(policy,
                    counting,
                    counting + X.NumFeatures() * split_samples,
                    keys.ptr(0),
                    [=] __device__(int i) { return i / split_samples; });

  // Segmented sort
  auto begin =
    thrust::make_zip_iterator(thrust::make_tuple(keys.ptr(0), draft_proposals.ptr({0, 0})));
  thrust::sort(
    policy, begin, begin + X.NumFeatures() * split_samples, [] __device__(auto a, auto b) {
      if (thrust::get<0>(a) != thrust::get<0>(b)) { return thrust::get<0>(a) < thrust::get<0>(b); }
      return thrust::get<1>(a) < thrust::get<1>(b);
    });

  // Extract the unique values
  auto out_keys        = legate::create_buffer<int32_t, 1>(X.NumFeatures() * split_samples);
  auto split_proposals = legate::create_buffer<T, 1>(X.NumFeatures() * split_samples);
  auto key_val =
    thrust::make_zip_iterator(thrust::make_tuple(keys.ptr(0), draft_proposals.ptr({0, 0})));
  auto out_iter =
    thrust::make_zip_iterator(thrust::make_tuple(out_keys.ptr(0), split_proposals.ptr(0)));
  auto result =
    thrust::unique_copy(policy, key_val, key_val + X.NumFeatures() * split_samples, out_iter);
  auto n_unique = thrust::distance(out_iter, result);
  // Count the unique values for each feature
  auto row_pointers = legate::create_buffer<int32_t, 1>(X.NumFeatures() + 1);
  CHECK_CUDA(
    cudaMemsetAsync(row_pointers.ptr(0), 0, (X.NumFeatures() + 1) * sizeof(int32_t), stream));

  tcb::span<int32_t> const out_keys_span(out_keys.ptr(0), X.NumFeatures() * split_samples);
  auto unique_keys_span = out_keys_span.subspan(0, n_unique);
  thrust::reduce_by_key(policy,
                        unique_keys_span.begin(),
                        unique_keys_span.end(),
                        thrust::make_constant_iterator(1),
                        thrust::make_discard_iterator(),
                        row_pointers.ptr(1));
  // Scan the counts to get the row pointers for a CSR matrix
  tcb::span<int32_t> const row_pointers_span(row_pointers.ptr(0), X.NumFeatures() + 1);
  thrust::inclusive_scan(policy,
                         row_pointers_span.begin() + 1,
                         row_pointers_span.begin() + 1 + X.NumFeatures(),
                         row_pointers_span.begin() + 1);

  CHECK_CUDA(cudaStreamSynchronize(stream));
  row_samples.destroy();
  draft_proposals.destroy();
  out_keys.destroy();
  return SparseSplitProposals<T>(split_proposals, row_pointers, X.NumFeatures(), n_unique);
}

// Can't put a device lambda in constructor so make this a function
void FillPositions(const legate::Buffer<cuda::std::tuple<int32_t, int32_t>>& sorted_positions,
                   std::size_t num_rows,
                   cudaStream_t stream)
{
  LaunchN(num_rows, stream, [=] __device__(int64_t idx) {
    sorted_positions[idx] = cuda::std::make_tuple(0, narrow<int32_t>(idx));
  });
}

}  // namespace

template <typename MatrixT>
struct TreeBuilder {
  using T = typename MatrixT::value_type;
  TreeBuilder(int32_t num_rows,
              int32_t num_features,
              int32_t num_outputs,
              cudaStream_t stream,
              int32_t max_nodes,
              int32_t max_depth,
              const SparseSplitProposals<T>& split_proposals,
              GradientQuantiser quantiser,
              int64_t seed)
    : num_rows(num_rows),
      num_features(num_features),
      num_outputs(num_outputs),
      stream(stream),
      max_nodes(max_nodes),
      max_depth(max_depth),
      split_proposals(split_proposals),
      quantiser(quantiser),
      histogram_kernel(split_proposals, stream),
      seed(seed)
  {
    sorted_positions = legate::create_buffer<cuda::std::tuple<int32_t, int32_t>>(num_rows);
    FillPositions(sorted_positions, num_rows, stream);

    // Calculate the number of node histograms we are willing to cache
    // User a fixed reasonable upper bound on memory usage
    // CAUTION: all workers MUST have the same max_batch_size
    // Therefore we don't try to calculate this based on available memory
    const std::size_t max_bytes      = 1000000000;  // 1 GB
    const std::size_t bytes_per_node = num_outputs * split_proposals.histogram_size * sizeof(GPair);
    const std::size_t max_histogram_nodes = std::max(1UL, max_bytes / bytes_per_node);
    int depth                             = 0;
    while (BinaryTree::LevelEnd(depth + 1) <= max_histogram_nodes && depth <= max_depth) {
      depth++;
    }
    histogram      = Histogram<IntegerGPair>(BinaryTree::LevelBegin(0),
                                        BinaryTree::LevelEnd(depth),
                                        num_outputs,
                                        split_proposals.histogram_size,
                                        stream);
    max_batch_size = max_histogram_nodes;
  }

  Tree Build(legate::TaskContext context,
             const MatrixT& X_matrix,
             legate::AccessorRO<double, 3> g_accessor,
             legate::AccessorRO<double, 3> h_accessor,
             legate::Rect<3> g_shape,
             double alpha)
  {
    auto stream             = context.get_task_stream();
    auto thrust_alloc       = ThrustAllocator(legate::Memory::GPU_FB_MEM);
    auto thrust_exec_policy = DEFAULT_POLICY(thrust_alloc).on(stream);

    Tree tree(max_nodes, num_outputs, stream, thrust_exec_policy);

    this->InitialiseRoot(context, tree, g_accessor, h_accessor, g_shape, alpha);

    for (int depth = 0; depth < max_depth; ++depth) {
      auto batches = this->PrepareBatches(depth);
      for (auto batch : batches) {
        auto histogram = this->GetHistogram(batch);

        this->ComputeHistogram(histogram, context, tree, X_matrix, g_accessor, h_accessor, batch);

        this->PerformBestSplit(tree, histogram, alpha, batch);
      }
      // Update position of entire level
      // Don't bother updating positions for the last level
      if (depth < max_depth - 1) { this->UpdatePositions(tree, X_matrix); }
    }

    return tree;
  }

  TreeBuilder(const TreeBuilder&)                    = delete;
  TreeBuilder(TreeBuilder&&)                         = delete;
  auto operator=(const TreeBuilder&) -> TreeBuilder& = delete;
  auto operator=(TreeBuilder&&) -> TreeBuilder&      = delete;
  ~TreeBuilder()                                     = default;

  void UpdatePositions(Tree& tree, const MatrixT X)
  {
    tcb::span<int32_t> const tree_feature_span(tree.feature.ptr(0), max_nodes);
    tcb::span<double> const tree_split_value_span(tree.split_value.ptr(0), max_nodes);
    auto max_nodes_ = this->max_nodes;

    LaunchN(num_rows,
            stream,
            [=, sorted_positions = this->sorted_positions] __device__(std::int64_t idx) {
              auto [pos, row] = sorted_positions[idx];

              if (pos < 0 || pos >= max_nodes_ || tree_feature_span[pos] == -1) {
                sorted_positions[idx] = cuda::std::make_tuple(-1, row);
                return;
              }

              double x_value =
                X.Get(X.RowRange().lo[0] + static_cast<int64_t>(row), tree_feature_span[pos]);
              bool left = x_value <= tree_split_value_span[pos];
              pos       = left ? BinaryTree::LeftChild(pos) : BinaryTree::RightChild(pos);
              sorted_positions[idx] = cuda::std::make_tuple(pos, row);
            });
    CHECK_CUDA_STREAM(stream);

    auto decomposer = cuda::proclaim_return_type<cuda::std::tuple<int32_t&>>(
      [] __device__(auto& a) { return cuda::std::tuple<int32_t&>{cuda::std::get<0>(a)}; });
    auto sorted_out           = legate::create_buffer<cuda::std::tuple<int32_t, int32_t>>(num_rows);
    size_t temp_storage_bytes = 0;
    const int bits_per_key    = sizeof(int32_t) * 8;
    cub::DeviceRadixSort::SortKeys(nullptr,
                                   temp_storage_bytes,
                                   sorted_positions.ptr(0),
                                   sorted_out.ptr(0),
                                   num_rows,
                                   decomposer,
                                   0,
                                   bits_per_key,
                                   stream);
    auto temp_storage = legate::create_buffer<char, 1>(narrow<int64_t>(temp_storage_bytes));
    cub::DeviceRadixSort::SortKeys(temp_storage.ptr(0),
                                   temp_storage_bytes,
                                   sorted_positions.ptr(0),
                                   sorted_out.ptr(0),
                                   num_rows,
                                   decomposer,
                                   0,
                                   bits_per_key,
                                   stream);

    CHECK_CUDA(cudaMemcpyAsync(sorted_positions.ptr(0),
                               sorted_out.ptr(0),
                               num_rows * sizeof(cuda::std::tuple<int32_t, int32_t>),
                               cudaMemcpyDeviceToDevice,
                               stream));
  }

  void ComputeHistogram(Histogram<IntegerGPair> histogram,
                        legate::TaskContext context,
                        Tree& tree,
                        const MatrixT X,
                        const legate::AccessorRO<double, 3>& g,
                        const legate::AccessorRO<double, 3>& h,
                        NodeBatch batch)
  {
    histogram_kernel.BuildHistogram(X,
                                    g,
                                    h,
                                    num_outputs,
                                    split_proposals,
                                    batch,
                                    histogram,
                                    tree.node_sums,
                                    quantiser,
                                    seed,
                                    stream);
    CHECK_CUDA_STREAM(stream);

    using ReduceT = Histogram<IntegerGPair>::value_type::value_type;
    SumAllReduce(
      context,
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
      tcb::span<ReduceT>(reinterpret_cast<ReduceT*>(histogram.Ptr(batch.node_idx_begin)),
                         batch.NodesInBatch() * num_outputs * split_proposals.histogram_size * 2),
      stream);

    const int kScanBlockThreads  = 256;
    const size_t warps_needed    = num_features * batch.NodesInBatch();
    const size_t warps_per_block = kScanBlockThreads / 32;
    const size_t blocks_needed   = (warps_needed + warps_per_block - 1) / warps_per_block;

    // Scan the histograms
    dim3 const scan_grid = dim3(blocks_needed, num_outputs);
    scan_kernel<T, kScanBlockThreads><<<scan_grid, kScanBlockThreads, 0, stream>>>(
      histogram, tree.node_sums, num_features, split_proposals, batch);
    CHECK_CUDA_STREAM(stream);
  }

  void PerformBestSplit(Tree& tree,
                        const Histogram<IntegerGPair>& histogram,
                        double alpha,
                        NodeBatch batch)
  {
    const int kBlockThreads = 512;
    perform_best_split<T, kBlockThreads>
      <<<batch.NodesInBatch(), kBlockThreads, 0, stream>>>(histogram,
                                                           num_outputs,
                                                           split_proposals,
                                                           eps,
                                                           alpha,
                                                           tree.leaf_value,
                                                           tree.node_sums,
                                                           tree.feature,
                                                           tree.split_value,
                                                           tree.gain,
                                                           batch,
                                                           quantiser);
    CHECK_CUDA_STREAM(stream);
  }
  void InitialiseRoot(legate::TaskContext context,
                      Tree& tree,
                      const legate::AccessorRO<double, 3>& g,
                      const legate::AccessorRO<double, 3>& h,
                      legate::Rect<3> g_shape,
                      double alpha)
  {
    const int kBlockThreads = 256;
    const size_t blocks     = (num_rows + kBlockThreads - 1) / kBlockThreads;
    dim3 const grid_shape   = dim3(blocks, num_outputs);
    reduce_base_sums<kBlockThreads><<<grid_shape, kBlockThreads, 0, stream>>>(
      g, h, num_rows, g_shape.lo[0], tree.node_sums, quantiser, seed);
    CHECK_CUDA_STREAM(stream);

    SumAllReduce(context,
                 // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
                 tcb::span<int64_t>(reinterpret_cast<int64_t*>(tree.node_sums.ptr({0, 0})),
                                    static_cast<size_t>(num_outputs * 2)),
                 stream);
    LaunchN(num_outputs,
            stream,
            [            =,
             num_outputs = this->num_outputs,
             leaf_value  = tree.leaf_value,
             node_sums   = tree.node_sums,
             quantiser   = this->quantiser] __device__(int output) {
              GPair const sum         = quantiser.Dequantise(node_sums[{0, output}]);
              leaf_value[{0, output}] = CalculateLeafValue(sum.grad, sum.hess, alpha);
            });
    CHECK_CUDA_STREAM(stream);
  }

  // Create a new histogram for this batch if we need to
  // Destroy the old one
  auto GetHistogram(NodeBatch batch) -> Histogram<IntegerGPair>
  {
    if (histogram.ContainsBatch(batch.node_idx_begin, batch.node_idx_end)) { return histogram; }

    CHECK_CUDA(cudaStreamSynchronize(stream));
    histogram.Destroy();
    histogram = Histogram<IntegerGPair>(batch.node_idx_begin,
                                        batch.node_idx_end,
                                        num_outputs,
                                        split_proposals.histogram_size,
                                        stream);
    return histogram;
  }

  auto PrepareBatches(int depth) -> std::vector<NodeBatch>
  {
    tcb::span<cuda::std::tuple<int32_t, int32_t>> const sorted_positions_span(
      sorted_positions.ptr(0), num_rows);
    // Shortcut if we have 1 batch
    if (BinaryTree::NodesInLevel(depth) <= max_batch_size) {
      // All instances are in batch
      return {NodeBatch{
        BinaryTree::LevelBegin(depth), BinaryTree::LevelEnd(depth), sorted_positions_span}};
    }

    // Launch a kernel where each thread computes the range of instances for a batch using binary
    // search
    const int num_batches = (BinaryTree::NodesInLevel(depth) + max_batch_size - 1) / max_batch_size;
    auto batches          = legate::create_buffer<NodeBatch, 1>(num_batches);
    auto batches_span     = tcb::span<NodeBatch>(batches.ptr(0), num_batches);
    LaunchN(num_batches,
            stream,
            [                     =,
             sorted_positions_ptr = this->sorted_positions.ptr(0),
             num_rows             = this->num_rows,
             max_batch_size       = this->max_batch_size] __device__(int batch_idx) {
              int batch_begin = BinaryTree::LevelBegin(depth) + (batch_idx * max_batch_size);
              int const batch_end =
                std::min(batch_begin + max_batch_size, BinaryTree::LevelEnd(depth));
              auto comp = [] __device__(auto a, auto b) {
                return cuda::std::get<0>(a) < cuda::std::get<0>(b);
              };  // NOLINT(readability/braces)

              auto lower = thrust::lower_bound(thrust::seq,
                                               sorted_positions_span.begin(),
                                               sorted_positions_span.end(),
                                               cuda::std::tuple(batch_begin, 0),
                                               comp);
              auto upper = thrust::upper_bound(thrust::seq,
                                               lower,
                                               sorted_positions_span.end(),
                                               cuda::std::tuple(batch_end - 1, 0),
                                               comp);
              tcb::span<cuda::std::tuple<int32_t, int32_t>> const span(lower, upper);
              batches_span[batch_idx] = {
                cuda::std::get<0>(span.front()), cuda::std::get<0>(span.back()) + 1, span};
            });

    std::vector<NodeBatch> result(num_batches);
    CHECK_CUDA(cudaMemcpyAsync(result.data(),
                               batches.ptr(0),
                               num_batches * sizeof(NodeBatch),
                               cudaMemcpyDeviceToHost,
                               stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    // Filter empty
    result.erase(
      std::remove_if(
        result.begin(), result.end(), [](const NodeBatch& b) { return b.InstancesInBatch() == 0; }),
      result.end());
    return result;
  }

  legate::Buffer<cuda::std::tuple<int32_t, int32_t>> sorted_positions;  // (node, row)
  const int32_t num_rows;
  const int32_t num_features;
  const int32_t num_outputs;
  const int32_t max_nodes;
  const int32_t max_depth;
  const int64_t seed;
  SparseSplitProposals<T> split_proposals;
  Histogram<IntegerGPair> histogram;
  int max_batch_size;
  GradientQuantiser quantiser;
  HistogramKernel<MatrixT> histogram_kernel;

  cudaStream_t stream;
};

struct build_tree_fn {
  template <typename T>
  void operator()(legate::TaskContext context)
  {
    auto [X, X_shape, X_accessor] = GetInputStore<T, 3>(context.input(0).data());
    auto [g, g_shape, g_accessor] = GetInputStore<double, 3>(context.input(1).data());
    auto [h, h_shape, h_accessor] = GetInputStore<double, 3>(context.input(2).data());

    DenseXMatrix X_matrix(X_accessor, X_shape);

    EXPECT_DENSE_ROW_MAJOR(X_accessor.accessor, X_shape);
    auto num_features = X_shape.hi[1] - X_shape.lo[1] + 1;
    auto num_rows     = std::max<int64_t>(X_shape.hi[0] - X_shape.lo[0] + 1, 0);
    auto num_outputs  = X_shape.hi[2] - X_shape.lo[2] + 1;
    EXPECT(g_shape.lo[2] == 0, "Outputs should not be split between workers.");
    EXPECT_AXIS_ALIGNED(0, X_shape, g_shape);
    EXPECT_AXIS_ALIGNED(0, g_shape, h_shape);
    EXPECT_AXIS_ALIGNED(1, g_shape, h_shape);

    // Scalars
    auto max_depth     = context.scalars().at(0).value<int>();
    auto max_nodes     = context.scalars().at(1).value<int>();
    auto alpha         = context.scalars().at(2).value<double>();
    auto split_samples = context.scalars().at(3).value<int>();
    auto seed          = context.scalars().at(4).value<int>();
    auto dataset_rows  = context.scalars().at(5).value<int64_t>();

    auto* stream            = context.get_task_stream();
    auto thrust_alloc       = ThrustAllocator(legate::Memory::GPU_FB_MEM);
    auto thrust_exec_policy = DEFAULT_POLICY(thrust_alloc).on(stream);

    const SparseSplitProposals<T> split_proposals =
      SelectSplitSamples(context, X_matrix, split_samples, seed, dataset_rows, stream);

    GradientQuantiser const quantiser(context, g_accessor, h_accessor, g_shape, stream);

    auto tree = TreeBuilder<DenseXMatrix<T>>(num_rows,
                                             num_features,
                                             num_outputs,
                                             stream,
                                             max_nodes,
                                             max_depth,
                                             split_proposals,
                                             quantiser,
                                             seed)
                  .Build(context, X_matrix, g_accessor, h_accessor, g_shape, alpha);

    tree.WriteTreeOutput(context, quantiser);

    CHECK_CUDA(cudaStreamSynchronize(stream));
    CHECK_CUDA_STREAM(stream);
  }
};

struct build_tree_csr_fn {
  template <typename T>
  void operator()(legate::TaskContext context)
  {
    auto [X_vals, X_vals_shape, X_vals_accessor] = GetInputStore<T, 1>(context.input(0).data());
    auto [X_coords, X_coords_shape, X_coords_accessor] =
      GetInputStore<int64_t, 1>(context.input(1).data());
    auto [X_offsets, X_offsets_shape, X_offsets_accessor] =
      GetInputStore<legate::Rect<1>, 1>(context.input(2).data());
    auto [g, g_shape, g_accessor] = GetInputStore<double, 3>(context.input(3).data());
    auto [h, h_shape, h_accessor] = GetInputStore<double, 3>(context.input(4).data());

    auto num_rows    = std::max<int64_t>(X_offsets_shape.hi[0] - X_offsets_shape.lo[0] + 1, 0);
    auto num_outputs = g_shape.hi[2] - g_shape.lo[2] + 1;
    EXPECT(g_shape.lo[2] == 0, "Outputs should not be split between workers.");

    // Scalars
    auto max_depth     = context.scalars().at(0).value<int>();
    auto max_nodes     = context.scalars().at(1).value<int>();
    auto alpha         = context.scalars().at(2).value<double>();
    auto split_samples = context.scalars().at(3).value<int>();
    auto seed          = context.scalars().at(4).value<int>();
    auto dataset_rows  = context.scalars().at(5).value<int64_t>();
    auto num_features  = context.scalars().at(6).value<int64_t>();

    auto* stream = context.get_task_stream();
    CSRXMatrix<T> X_matrix(
      X_vals_accessor, X_coords_accessor, X_offsets_accessor, X_offsets_shape, num_features);
    const SparseSplitProposals<T> split_proposals =
      SelectSplitSamples(context, X_matrix, split_samples, seed, dataset_rows, stream);

    GradientQuantiser quantiser(context, g_accessor, h_accessor, g_shape, stream);

    // Begin building the tree
    auto tree = TreeBuilder<CSRXMatrix<T>>(num_rows,
                                           num_features,
                                           num_outputs,
                                           stream,
                                           max_nodes,
                                           max_depth,
                                           split_proposals,
                                           quantiser,
                                           seed)
                  .Build(context, X_matrix, g_accessor, h_accessor, g_shape, alpha);

    tree.WriteTreeOutput(context, quantiser);

    CHECK_CUDA(cudaStreamSynchronize(stream));
    CHECK_CUDA_STREAM(stream);
  }
};

/*static*/ void BuildTreeDenseTask::gpu_variant(legate::TaskContext context)
{
  const auto& X = context.input(0).data();
  type_dispatch_float(X.code(), build_tree_fn(), context);
}

/*static*/ void BuildTreeCSRTask::gpu_variant(legate::TaskContext context)
{
  const auto& X = context.input(0).data();
  type_dispatch_float(X.code(), build_tree_csr_fn(), context);
}

}  // namespace legateboost
