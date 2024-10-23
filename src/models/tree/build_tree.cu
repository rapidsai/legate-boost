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
#include "legate_library.h"
#include "legateboost.h"
#include "../../cpp_utils/cpp_utils.h"
#include "../../cpp_utils/cpp_utils.cuh"
#include "legate/comm/coll.h"
#include "build_tree.h"
#include <numeric>

#include <cuda/std/tuple>
#include <thrust/execution_policy.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/sort.h>
#include <thrust/random.h>
#include <thrust/unique.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

namespace legateboost {

struct NodeBatch {
  int32_t node_idx_begin;
  int32_t node_idx_end;
  cuda::std::tuple<int32_t, int32_t>* instances_begin;
  cuda::std::tuple<int32_t, int32_t>* instances_end;
  __host__ __device__ std::size_t InstancesInBatch() const
  {
    return instances_end - instances_begin;
  }
  __host__ __device__ std::size_t NodesInBatch() const { return node_idx_end - node_idx_begin; }
};

class GradientQuantiser {
  GPair scale;
  GPair inverse_scale;

 public:
  struct GetAbsGPair {
    int num_outputs;
    legate::AccessorRO<double, 3> g;
    legate::AccessorRO<double, 3> h;
    __device__ GPair operator()(int n) const
    {
      legate::Point<3> p = {n / num_outputs, 0, n % num_outputs};
      return GPair{abs(g[p]), abs(h[p])};
    }
  };

  // Calculate scale from upper bound on data
  GradientQuantiser(legate::TaskContext context,
                    legate::AccessorRO<double, 3> g,
                    legate::AccessorRO<double, 3> h,
                    legate::Rect<3> g_shape,
                    cudaStream_t stream)
  {
    auto thrust_alloc = ThrustAllocator(legate::Memory::GPU_FB_MEM);
    auto policy       = DEFAULT_POLICY(thrust_alloc).on(stream);
    auto counting     = thrust::make_counting_iterator(0);
    int num_outputs   = g_shape.hi[2] - g_shape.lo[2] + 1;
    std::size_t n     = (g_shape.hi[0] - g_shape.lo[0] + 1) * num_outputs;
    auto zip_gpair    = thrust::make_transform_iterator(counting, GetAbsGPair{num_outputs, g, h});
    GPair local_abs_sum =
      thrust::reduce(policy, zip_gpair, zip_gpair + n, GPair{0.0, 0.0}, thrust::plus<GPair>());

    auto local_abs_sum_device = legate::create_buffer<GPair, 1>(1);
    CHECK_CUDA(cudaMemcpyAsync(
      local_abs_sum_device.ptr(0), &local_abs_sum, sizeof(GPair), cudaMemcpyHostToDevice, stream));
    // Take the max of the local sums
    AllReduce(context, reinterpret_cast<double*>(local_abs_sum_device.ptr(0)), 2, ncclMax, stream);
    CHECK_CUDA(cudaMemcpyAsync(
      &local_abs_sum, local_abs_sum_device.ptr(0), sizeof(GPair), cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));

    // We will quantise values between -max_int and max_int
    int64_t max_int    = std::numeric_limits<int32_t>::max();
    scale.grad         = local_abs_sum.grad == 0 ? 1 : max_int / local_abs_sum.grad;
    scale.hess         = local_abs_sum.hess == 0 ? 1 : max_int / local_abs_sum.hess;
    inverse_scale.grad = 1.0 / scale.grad;
    inverse_scale.hess = 1.0 / scale.hess;
  }

  // Round gradient and hessian using stochastic rounding
  // Thus the expected value of the quantised value is unbiased
  // Also the expected error grows as O(1/sqrt(n)) where n is the number of samples
  // Vs. O(1/n) for round nearest
  // The seed here should be unique for each gpair over each boosting iteration
  // Use a hash combine function to generate the seed
  __device__ IntegerGPair QuantiseStochasticRounding(GPair value, int64_t seed) const
  {
    thrust::default_random_engine eng(seed);
    thrust::uniform_real_distribution<double> dist(0.0, 1.0);
    auto scaled_grad                        = value.grad * scale.grad;
    auto scaled_hess                        = value.hess * scale.hess;
    double grad_remainder                   = scaled_grad - floor(scaled_grad);
    double hess_remainder                   = scaled_hess - floor(scaled_hess);
    IntegerGPair::value_type grad_quantised = floor(scaled_grad) + (dist(eng) < grad_remainder);
    IntegerGPair::value_type hess_quantised = floor(scaled_hess) + (dist(eng) < hess_remainder);
    return IntegerGPair{grad_quantised, hess_quantised};
  }

  __device__ GPair Dequantise(IntegerGPair value) const
  {
    GPair result;
    result.grad = value.grad * inverse_scale.grad;
    result.hess = value.hess * inverse_scale.hess;
    return result;
  }
};

// Hash function fmix64 from MurmurHash3
__device__ int64_t hash(int64_t k)
{
  k ^= k >> 33;
  k *= 0xff51afd7ed558ccd;
  k ^= k >> 33;
  k *= 0xc4ceb9fe1a85ec53;
  k ^= k >> 33;
  return k;
}

__device__ int64_t hash_combine(int64_t seed) { return seed; }

// Hash combine from boost
// This function is used to combine several random seeds e.g. a 3d index
template <typename... Rest>
__device__ int64_t hash_combine(int64_t seed, const int64_t& v, Rest... rest)
{
  seed ^= hash(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return hash_combine(seed, rest...);
}

template <int BLOCK_THREADS>
__global__ static void __launch_bounds__(BLOCK_THREADS)
  reduce_base_sums(legate::AccessorRO<double, 3> g,
                   legate::AccessorRO<double, 3> h,
                   size_t n_local_samples,
                   int64_t sample_offset,
                   legate::Buffer<IntegerGPair, 2> node_sums,
                   size_t n_outputs,
                   GradientQuantiser quantiser,
                   int64_t seed)
{
  typedef cub::BlockReduce<IntegerGPair, BLOCK_THREADS> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  int32_t output = blockIdx.y;

  int64_t sample_id = threadIdx.x + blockDim.x * blockIdx.x;

  legate::Point<3> p = {sample_id + sample_offset, 0, output};
  double grad        = sample_id < n_local_samples ? g[p] : 0.0;
  double hess        = sample_id < n_local_samples ? h[p] : 0.0;

  auto quantised =
    quantiser.QuantiseStochasticRounding({grad, hess}, hash_combine(seed, p[0], p[2]));
  IntegerGPair blocksum = BlockReduce(temp_storage).Sum(quantised);

  if (threadIdx.x == 0) {
    atomicAdd(
      reinterpret_cast<Histogram<IntegerGPair>::atomic_add_type*>(&node_sums[{0, output}].grad),
      blocksum.grad);
    atomicAdd(
      reinterpret_cast<Histogram<IntegerGPair>::atomic_add_type*>(&node_sums[{0, output}].hess),
      blocksum.hess);
  }
}

template <typename TYPE, int TPB, int FEATURES_PER_WARP>
__global__ static void __launch_bounds__(TPB, 4)
  fill_histogram_warp(legate::AccessorRO<TYPE, 3> X,
                      size_t n_features,
                      int64_t sample_offset,
                      legate::AccessorRO<double, 3> g,
                      legate::AccessorRO<double, 3> h,
                      size_t n_outputs,
                      SparseSplitProposals<TYPE> split_proposals,
                      NodeBatch batch,
                      Histogram<IntegerGPair> histogram,
                      legate::Buffer<IntegerGPair, 2> node_sums,
                      GradientQuantiser quantiser,
                      int64_t seed)
{
  constexpr int32_t WarpSize = 32;
  const int32_t warp_id      = threadIdx.x / WarpSize;
  const int32_t lane_id      = threadIdx.x % WarpSize;

  const int32_t localIdx = blockIdx.x * TPB + warp_id * WarpSize + lane_id;

  // prefetch sampleNode information for all 32 ids
  auto [sampleNode_lane, localSampleId_lane] = (localIdx < batch.InstancesInBatch())
                                                 ? batch.instances_begin[localIdx]
                                                 : cuda::std::make_tuple(-1, -1);
  const bool computeHistogram =
    localIdx < batch.InstancesInBatch() &&
    ComputeHistogramBin(
      sampleNode_lane, node_sums, histogram.ContainsNode(BinaryTree::Parent(sampleNode_lane)));

  // mask contains all sample bits of the next 32 ids that need to be bin'ed
  auto lane_mask = ballot(computeHistogram);

  // reverse to use __clz instead of __ffs
  lane_mask = __brev(lane_mask);

  while (lane_mask) {
    // look for next lane_offset / sample to process within warp-batch
    const uint32_t lane_offset  = __clz(lane_mask);
    const int32_t sampleNode    = shfl(sampleNode_lane, lane_offset);
    const int32_t localSampleId = shfl(localSampleId_lane, lane_offset);

    // remove lane_offset bit from lane_mask for next iteration
    lane_mask &= (0x7fffffff >> lane_offset);

    auto feature_begin = blockIdx.y * FEATURES_PER_WARP;
    auto feature_end   = min(n_features, (size_t)feature_begin + FEATURES_PER_WARP);
    for (int32_t feature = feature_begin + lane_id; feature < feature_end; feature += WarpSize) {
      const int32_t bin_idx =
        split_proposals.FindBin(X[{sample_offset + localSampleId, feature, 0}], feature);
      for (int32_t output = 0; output < n_outputs; output++) {
        // get same G/H from every thread in warp
        legate::Point<3> p = {sample_offset + localSampleId, feature, output};
        auto gpair_quantised =
          quantiser.QuantiseStochasticRounding({g[p], h[p]}, hash_combine(seed, p[0], p[2]));
        auto* addPosition = reinterpret_cast<typename IntegerGPair::value_type*>(
          &histogram[{sampleNode, output, bin_idx}]);

        if (bin_idx != SparseSplitProposals<TYPE>::NOT_FOUND) {
          Histogram<IntegerGPair>::atomic_add_type* addPosition =
            reinterpret_cast<Histogram<IntegerGPair>::atomic_add_type*>(
              &histogram[{sampleNode, output, bin_idx}]);
          atomicAdd(addPosition, gpair_quantised.grad);
          atomicAdd(addPosition + 1, gpair_quantised.hess);
        }
      }
    }
  }
}

__device__ IntegerGPair vectorised_load(const IntegerGPair* ptr)
{
  static_assert(sizeof(IntegerGPair) == sizeof(int4), "size inconsistent");
  auto load = *reinterpret_cast<const int4*>(ptr);
  return *reinterpret_cast<const IntegerGPair*>(&load);
}

__device__ void vectorised_store(IntegerGPair* ptr, IntegerGPair value)
{
  static_assert(sizeof(IntegerGPair) == sizeof(int4), "size inconsistent");
  auto store = reinterpret_cast<int4*>(ptr);
  *store     = *reinterpret_cast<int4*>(&value);
}

template <typename T, int BLOCK_THREADS>
__global__ static void __launch_bounds__(BLOCK_THREADS)
  scan_kernel(Histogram<IntegerGPair> histogram,
              legate::Buffer<IntegerGPair, 2> node_sums,
              int n_features,
              const SparseSplitProposals<T> split_proposals,
              NodeBatch batch)

{
  auto warp      = cg::tiled_partition<32>(cg::this_thread_block());
  int rank       = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  auto num_nodes = batch.NodesInBatch();
  int i          = rank / num_nodes;
  int j          = rank % num_nodes;
  int output     = blockIdx.y;

  // Specialize WarpScan for type int
  typedef cub::WarpScan<IntegerGPair> WarpScan;

  __shared__ typename WarpScan::TempStorage temp_storage[BLOCK_THREADS / 32];

  int scan_node_idx = batch.node_idx_begin + j;
  int parent        = BinaryTree::Parent(scan_node_idx);
  // Exit if we didn't compute this histogram
  if (!ComputeHistogramBin(scan_node_idx, node_sums, histogram.ContainsNode(parent))) return;
  if (i >= n_features || scan_node_idx >= batch.node_idx_end) return;

  const int feature_idx             = i;
  auto [feature_begin, feature_end] = split_proposals.FeatureRange(feature_idx);
  const int num_bins                = feature_end - feature_begin;
  const int num_tiles               = (num_bins + warp.num_threads() - 1) / warp.num_threads();

  IntegerGPair aggregate;
  // Scan left side
  for (int tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
    const int bin_idx        = feature_begin + tile_idx * warp.num_threads() + warp.thread_rank();
    bool thread_participates = bin_idx < feature_end;
    auto e = thread_participates ? vectorised_load(&histogram[{scan_node_idx, output, bin_idx}])
                                 : IntegerGPair{0, 0};
    IntegerGPair tile_aggregate;
    WarpScan(temp_storage[threadIdx.x / warp.num_threads()]).InclusiveSum(e, e, tile_aggregate);
    e += aggregate;
    // Skip write if data is 0
    // This actually helps quite a bit at deeper tree levels where we have a lot of empty bins
    if (thread_participates && (e.grad > 0 || e.hess > 0)) {
      vectorised_store(&histogram[{scan_node_idx, output, bin_idx}], e);
    }
    aggregate += tile_aggregate;
  }

  // This node has no sibling we are finished
  if (scan_node_idx == 0) return;

  int sibling_node_idx = BinaryTree::Sibling(scan_node_idx);

  // The sibling did not compute a histogram
  // Do the subtraction trick using the histogram we just computed in the previous step
  if (!ComputeHistogramBin(sibling_node_idx, node_sums, histogram.ContainsNode(parent))) {
    // Infer right side
    for (int bin_idx = feature_begin + warp.thread_rank(); bin_idx < feature_end;
         bin_idx += warp.num_threads()) {
      auto scanned_sum = vectorised_load(&histogram[{scan_node_idx, output, bin_idx}]);
      auto parent_sum =
        vectorised_load(&histogram[{BinaryTree::Parent(scan_node_idx), output, bin_idx}]);
      auto other_sum = parent_sum - scanned_sum;
      vectorised_store(&histogram[{sibling_node_idx, output, bin_idx}], other_sum);
    }
  }
}

// Key/value pair to simplify reduction
struct GainFeaturePair {
  double gain;
  int bin_idx;

  __device__ void operator=(const GainFeaturePair& other)
  {
    gain    = other.gain;
    bin_idx = other.bin_idx;
  }

  __device__ bool operator==(const GainFeaturePair& other) const
  {
    return gain == other.gain && bin_idx == other.bin_idx;
  }

  __device__ bool operator>(const GainFeaturePair& other) const { return gain > other.gain; }

  __device__ bool operator<(const GainFeaturePair& other) const { return gain < other.gain; }
};

template <typename TYPE, int BLOCK_THREADS>
__global__ static void __launch_bounds__(BLOCK_THREADS)
  perform_best_split(Histogram<IntegerGPair> histogram,
                     size_t n_features,
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
  int node_id = batch.node_idx_begin + blockIdx.x;

  typedef cub::BlockReduce<GainFeaturePair, BLOCK_THREADS> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  __shared__ double node_best_gain;
  __shared__ int node_best_bin_idx;

  double thread_best_gain = 0;
  int thread_best_bin_idx = -1;

  for (int bin_idx = threadIdx.x; bin_idx < split_proposals.histogram_size;
       bin_idx += BLOCK_THREADS) {
    double gain = 0;
    for (int output = 0; output < n_outputs; ++output) {
      auto node_sum  = vectorised_load(&node_sums[{node_id, output}]);
      auto left_sum  = vectorised_load(&histogram[{node_id, output, bin_idx}]);
      auto right_sum = node_sum - left_sum;
      if (left_sum.hess <= 0.0 || right_sum.hess <= 0.0) {
        gain = 0;
        break;
      }
      double reg  = std::max(eps, alpha);  // Regularisation term
      auto [G, H] = quantiser.Dequantise(node_sum);
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
  GainFeaturePair thread_best_pair{thread_best_gain, thread_best_bin_idx};
  GainFeaturePair node_best_pair =
    BlockReduce(temp_storage).Reduce(thread_best_pair, cub::Max(), BLOCK_THREADS);
  if (threadIdx.x == 0) {
    node_best_gain    = node_best_pair.gain;
    node_best_bin_idx = node_best_pair.bin_idx;
  }
  __syncthreads();

  if (node_best_gain > eps) {
    int node_best_feature = split_proposals.FindFeature(node_best_bin_idx);
    for (int output = threadIdx.x; output < n_outputs; output += BLOCK_THREADS) {
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

namespace {

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
    thrust::fill(thrust_exec_policy,
                 leaf_value.ptr({0, 0}),
                 leaf_value.ptr({0, 0}) + max_nodes * num_outputs,
                 0.0);
    thrust::fill(thrust_exec_policy, feature.ptr(0), feature.ptr(0) + max_nodes, -1);
    thrust::fill(thrust_exec_policy, split_value.ptr(0), split_value.ptr(0) + max_nodes, 0.0);
    thrust::fill(thrust_exec_policy, gain.ptr(0), gain.ptr(0) + max_nodes, 0.0);
    thrust::fill(thrust_exec_policy,
                 node_sums.ptr({0, 0}),
                 node_sums.ptr({0, 0}) + max_nodes * num_outputs,
                 IntegerGPair{0, 0});
  }

  template <typename T, int DIM, typename ThrustPolicyT>
  void WriteOutput(legate::PhysicalStore out,
                   const legate::Buffer<T, DIM> x,
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

  template <typename ThrustPolicyT>
  void WriteTreeOutput(legate::TaskContext context,
                       const ThrustPolicyT& policy,
                       GradientQuantiser quantiser)
  {
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
template <typename T>
SparseSplitProposals<T> SelectSplitSamples(legate::TaskContext context,
                                           legate::AccessorRO<T, 3> X,
                                           legate::Rect<3> X_shape,
                                           int split_samples,
                                           int seed,
                                           int64_t dataset_rows,
                                           cudaStream_t stream)
{
  auto thrust_alloc = ThrustAllocator(legate::Memory::GPU_FB_MEM);
  auto policy       = DEFAULT_POLICY(thrust_alloc).on(stream);
  int num_features  = X_shape.hi[1] - X_shape.lo[1] + 1;
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
  auto draft_proposals = legate::create_buffer<T, 2>({num_features, split_samples});

  // fill with local data
  LaunchN(num_features * split_samples, stream, [=] __device__(auto idx) {
    auto i                  = idx / num_features;
    auto j                  = idx % num_features;
    auto row                = row_samples[i];
    bool has_data           = row >= X_shape.lo[0] && row <= X_shape.hi[0];
    draft_proposals[{j, i}] = has_data ? X[{row, j, 0}] : T(0);
  });

  // Sum reduce over all workers
  SumAllReduce(context, draft_proposals.ptr({0, 0}), num_features * split_samples, stream);

  CHECK_CUDA_STREAM(stream);

  // Condense split samples to unique values
  // First sort the samples
  auto keys = legate::create_buffer<int32_t, 1>(num_features * split_samples);
  thrust::transform(
    policy, counting, counting + num_features * split_samples, keys.ptr(0), [=] __device__(int i) {
      return i / split_samples;
    });

  // Segmented sort
  auto begin =
    thrust::make_zip_iterator(thrust::make_tuple(keys.ptr(0), draft_proposals.ptr({0, 0})));
  thrust::sort(policy, begin, begin + num_features * split_samples, [] __device__(auto a, auto b) {
    if (thrust::get<0>(a) != thrust::get<0>(b)) { return thrust::get<0>(a) < thrust::get<0>(b); }
    return thrust::get<1>(a) < thrust::get<1>(b);
  });

  // Extract the unique values
  auto out_keys        = legate::create_buffer<int32_t, 1>(num_features * split_samples);
  auto split_proposals = legate::create_buffer<T, 1>(num_features * split_samples);
  auto key_val =
    thrust::make_zip_iterator(thrust::make_tuple(keys.ptr(0), draft_proposals.ptr({0, 0})));
  auto out_iter =
    thrust::make_zip_iterator(thrust::make_tuple(out_keys.ptr(0), split_proposals.ptr(0)));
  auto result =
    thrust::unique_copy(policy, key_val, key_val + num_features * split_samples, out_iter);
  auto n_unique = thrust::distance(out_iter, result);
  // Count the unique values for each feature
  auto row_pointers = legate::create_buffer<int32_t, 1>(num_features + 1);
  CHECK_CUDA(cudaMemsetAsync(row_pointers.ptr(0), 0, (num_features + 1) * sizeof(int32_t), stream));

  thrust::reduce_by_key(policy,
                        out_keys.ptr(0),
                        out_keys.ptr(0) + n_unique,
                        thrust::make_constant_iterator(1),
                        thrust::make_discard_iterator(),
                        row_pointers.ptr(1));
  // Scan the counts to get the row pointers for a CSR matrix
  thrust::inclusive_scan(
    policy, row_pointers.ptr(1), row_pointers.ptr(1) + num_features, row_pointers.ptr(1));

  CHECK_CUDA(cudaStreamSynchronize(stream));
  row_samples.destroy();
  draft_proposals.destroy();
  out_keys.destroy();
  return SparseSplitProposals<T>(split_proposals, row_pointers, num_features, n_unique);
}

// Can't put a device lambda in constructor so make this a function
void FillPositions(legate::Buffer<cuda::std::tuple<int32_t, int32_t>> sorted_positions,
                   std::size_t num_rows,
                   cudaStream_t stream)
{
  LaunchN(num_rows, stream, [=] __device__(std::size_t idx) {
    sorted_positions[idx] = cuda::std::make_tuple(0, idx);
  });
}

template <typename T>
struct TreeBuilder {
  TreeBuilder(int32_t num_rows,
              int32_t num_features,
              int32_t num_outputs,
              cudaStream_t stream,
              int32_t max_nodes,
              int32_t max_depth,
              SparseSplitProposals<T> split_proposals,
              GradientQuantiser quantiser)
    : num_rows(num_rows),
      num_features(num_features),
      num_outputs(num_outputs),
      stream(stream),
      max_nodes(max_nodes),
      split_proposals(split_proposals),
      quantiser(quantiser)
  {
    sorted_positions = legate::create_buffer<cuda::std::tuple<int32_t, int32_t>>(num_rows);
    FillPositions(sorted_positions, num_rows, stream);

    // Calculate the number of node histograms we are willing to cache
    // User a fixed reasonable upper bound on memory usage
    // CAUTION: all workers MUST have the same max_batch_size
    // Therefore we don't try to calculate this based on available memory
    const std::size_t max_bytes      = std::pow(10, 9);  // 1 GB
    const std::size_t bytes_per_node = num_outputs * split_proposals.histogram_size * sizeof(GPair);
    const std::size_t max_histogram_nodes = std::max(1ul, max_bytes / bytes_per_node);
    int depth                             = 0;
    while (BinaryTree::LevelEnd(depth + 1) <= max_histogram_nodes && depth <= max_depth) depth++;
    histogram      = Histogram<IntegerGPair>(BinaryTree::LevelBegin(0),
                                        BinaryTree::LevelEnd(depth),
                                        num_outputs,
                                        split_proposals.histogram_size,
                                        stream);
    max_batch_size = max_histogram_nodes;
  }

  template <typename TYPE>
  void UpdatePositions(Tree& tree, legate::AccessorRO<TYPE, 3> X, legate::Rect<3> X_shape)
  {
    auto tree_split_value_ptr = tree.split_value.ptr(0);
    auto tree_feature_ptr     = tree.feature.ptr(0);
    auto max_nodes_           = this->max_nodes;

    LaunchN(
      num_rows, stream, [=, sorted_positions = this->sorted_positions] __device__(size_t idx) {
        auto [pos, row] = sorted_positions[idx];

        if (pos < 0 || pos >= max_nodes_ || tree_feature_ptr[pos] == -1) {
          sorted_positions[idx] = cuda::std::make_tuple(-1, row);
          return;
        }
        double x_value        = X[{X_shape.lo[0] + (int64_t)row, tree_feature_ptr[pos], 0}];
        bool left             = x_value <= tree_split_value_ptr[pos];
        pos                   = left ? BinaryTree::LeftChild(pos) : BinaryTree::RightChild(pos);
        sorted_positions[idx] = cuda::std::make_tuple(pos, row);
      });
    CHECK_CUDA_STREAM(stream);
  }

  template <typename TYPE>
  void ComputeHistogram(Histogram<IntegerGPair> histogram,
                        legate::TaskContext context,
                        Tree& tree,
                        legate::AccessorRO<TYPE, 3> X,
                        legate::Rect<3> X_shape,
                        legate::AccessorRO<double, 3> g,
                        legate::AccessorRO<double, 3> h,
                        NodeBatch batch,
                        int64_t seed)
  {
    // warp kernel without additional caching / prefetching
    const int threads_per_block = 256;
    const size_t blocks_x = (batch.InstancesInBatch() + threads_per_block - 1) / threads_per_block;

    // splitting the features to ensure better work distribution for large numbers of features
    // while larger value also allow better caching of g & h,
    // smaller values improve access of the split_proposals
    const int features_per_warp = 64;
    const size_t blocks_y       = (num_features + features_per_warp - 1) / features_per_warp;
    dim3 grid_shape             = dim3(blocks_x, blocks_y, 1);
    fill_histogram_warp<TYPE, threads_per_block, features_per_warp>
      <<<grid_shape, threads_per_block, 0, stream>>>(X,
                                                     num_features,
                                                     X_shape.lo[0],
                                                     g,
                                                     h,
                                                     num_outputs,
                                                     split_proposals,
                                                     batch,
                                                     histogram,
                                                     tree.node_sums,
                                                     quantiser,
                                                     seed);

    CHECK_CUDA_STREAM(stream);

    SumAllReduce(context,
                 reinterpret_cast<Histogram<IntegerGPair>::value_type::value_type*>(
                   histogram.Ptr(batch.node_idx_begin)),
                 batch.NodesInBatch() * num_outputs * split_proposals.histogram_size * 2,
                 stream);

    const int kScanBlockThreads  = 256;
    const size_t warps_needed    = num_features * batch.NodesInBatch();
    const size_t warps_per_block = kScanBlockThreads / 32;
    const size_t blocks_needed   = (warps_needed + warps_per_block - 1) / warps_per_block;

    // Scan the histograms
    dim3 scan_grid = dim3(blocks_needed, num_outputs);
    scan_kernel<T, kScanBlockThreads><<<scan_grid, kScanBlockThreads, 0, stream>>>(
      histogram, tree.node_sums, num_features, split_proposals, batch);
    CHECK_CUDA_STREAM(stream);
  }

  void PerformBestSplit(Tree& tree,
                        Histogram<IntegerGPair> histogram,
                        double alpha,
                        NodeBatch batch)
  {
    const int kBlockThreads = 256;
    perform_best_split<T, kBlockThreads>
      <<<batch.NodesInBatch(), kBlockThreads, 0, stream>>>(histogram,
                                                           num_features,
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
                      legate::AccessorRO<double, 3> g,
                      legate::AccessorRO<double, 3> h,
                      legate::Rect<3> g_shape,
                      double alpha,
                      int64_t seed)
  {
    const int kBlockThreads = 256;
    const size_t blocks     = (num_rows + kBlockThreads - 1) / kBlockThreads;
    dim3 grid_shape         = dim3(blocks, num_outputs);
    reduce_base_sums<kBlockThreads><<<grid_shape, kBlockThreads, 0, stream>>>(
      g, h, num_rows, g_shape.lo[0], tree.node_sums, num_outputs, quantiser, seed);
    CHECK_CUDA_STREAM(stream);

    SumAllReduce(
      context, reinterpret_cast<int64_t*>(tree.node_sums.ptr({0, 0})), num_outputs * 2, stream);
    LaunchN(num_outputs,
            stream,
            [            =,
             num_outputs = this->num_outputs,
             leaf_value  = tree.leaf_value,
             node_sums   = tree.node_sums,
             quantiser   = this->quantiser] __device__(int output) {
              GPair sum               = quantiser.Dequantise(node_sums[{0, output}]);
              leaf_value[{0, output}] = CalculateLeafValue(sum.grad, sum.hess, alpha);
            });
    CHECK_CUDA_STREAM(stream);
  }

  // Create a new histogram for this batch if we need to
  // Destroy the old one
  Histogram<IntegerGPair> GetHistogram(NodeBatch batch)
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

  template <typename PolicyT>
  std::vector<NodeBatch> PrepareBatches(int depth, PolicyT& policy)
  {
    // Shortcut if we have 1 batch
    if (BinaryTree::NodesInLevel(depth) <= max_batch_size) {
      // All instances are in batch
      return {NodeBatch{BinaryTree::LevelBegin(depth),
                        BinaryTree::LevelEnd(depth),
                        sorted_positions.ptr(0),
                        sorted_positions.ptr(0) + num_rows}};
    }

    thrust::sort(
      policy,
      sorted_positions.ptr(0),
      sorted_positions.ptr(num_rows),
      [] __device__(auto a, auto b) { return cuda::std::get<0>(a) < cuda::std::get<0>(b); });

    // Launch a kernel where each thread computes the range of instances for a batch using binary
    // search
    const int num_batches = (BinaryTree::NodesInLevel(depth) + max_batch_size - 1) / max_batch_size;
    auto batches          = legate::create_buffer<NodeBatch, 1>(num_batches);
    LaunchN(num_batches,
            stream,
            [                     =,
             batches_ptr          = batches.ptr(0),
             sorted_positions_ptr = this->sorted_positions.ptr(0),
             num_rows             = this->num_rows,
             max_batch_size       = this->max_batch_size] __device__(int batch_idx) {
              int batch_begin = BinaryTree::LevelBegin(depth) + batch_idx * max_batch_size;
              int batch_end   = std::min(batch_begin + max_batch_size, BinaryTree::LevelEnd(depth));
              auto comp       = [] __device__(auto a, auto b) {
                return cuda::std::get<0>(a) < cuda::std::get<0>(b);
              };

              auto lower             = thrust::lower_bound(thrust::seq,
                                               sorted_positions_ptr,
                                               sorted_positions_ptr + num_rows,
                                               cuda::std::tuple(batch_begin, 0),
                                               comp);
              auto upper             = thrust::upper_bound(thrust::seq,
                                               lower,
                                               sorted_positions_ptr + num_rows,
                                               cuda::std::tuple(batch_end - 1, 0),
                                               comp);
              batches_ptr[batch_idx] = {batch_begin, batch_end, lower, upper};
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
  SparseSplitProposals<T> split_proposals;
  Histogram<IntegerGPair> histogram;
  int max_batch_size;
  GradientQuantiser quantiser;

  cudaStream_t stream;
};

struct build_tree_fn {
  template <typename T>
  void operator()(legate::TaskContext context)
  {
    auto [X, X_shape, X_accessor] = GetInputStore<T, 3>(context.input(0).data());
    auto [g, g_shape, g_accessor] = GetInputStore<double, 3>(context.input(1).data());
    auto [h, h_shape, h_accessor] = GetInputStore<double, 3>(context.input(2).data());

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

    auto stream             = context.get_task_stream();
    auto thrust_alloc       = ThrustAllocator(legate::Memory::GPU_FB_MEM);
    auto thrust_exec_policy = DEFAULT_POLICY(thrust_alloc).on(stream);

    Tree tree(max_nodes, num_outputs, stream, thrust_exec_policy);

    SparseSplitProposals<T> split_proposals =
      SelectSplitSamples(context, X_accessor, X_shape, split_samples, seed, dataset_rows, stream);

    GradientQuantiser quantiser(context, g_accessor, h_accessor, g_shape, stream);

    // Begin building the tree
    TreeBuilder<T> builder(num_rows,
                           num_features,
                           num_outputs,
                           stream,
                           tree.max_nodes,
                           max_depth,
                           split_proposals,
                           quantiser);

    builder.InitialiseRoot(context, tree, g_accessor, h_accessor, g_shape, alpha, seed);

    for (int depth = 0; depth < max_depth; ++depth) {
      auto batches = builder.PrepareBatches(depth, thrust_exec_policy);
      for (auto batch : batches) {
        auto histogram = builder.GetHistogram(batch);

        builder.ComputeHistogram(
          histogram, context, tree, X_accessor, X_shape, g_accessor, h_accessor, batch, seed);

        builder.PerformBestSplit(tree, histogram, alpha, batch);
      }
      // Update position of entire level
      // Don't bother updating positions for the last level
      if (depth < max_depth - 1) { builder.UpdatePositions(tree, X_accessor, X_shape); }
    }

    tree.WriteTreeOutput(context, thrust_exec_policy, quantiser);

    CHECK_CUDA(cudaStreamSynchronize(stream));
    CHECK_CUDA_STREAM(stream);
  }
};

}  // namespace

/*static*/ void BuildTreeTask::gpu_variant(legate::TaskContext context)
{
  const auto& X = context.input(0).data();
  type_dispatch_float(X.code(), build_tree_fn(), context);
}

}  // namespace legateboost
