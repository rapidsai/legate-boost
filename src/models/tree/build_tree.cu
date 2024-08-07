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
#include "core/comm/coll.h"
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

__global__ static void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  reduce_base_sums(legate::AccessorRO<double, 3> g,
                   legate::AccessorRO<double, 3> h,
                   size_t n_local_samples,
                   int64_t sample_offset,
                   legate::Buffer<double, 1> base_sums,
                   size_t n_outputs)
{
  typedef cub::BlockReduce<double, THREADS_PER_BLOCK> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage_g;
  __shared__ typename BlockReduce::TempStorage temp_storage_h;

  int32_t output = blockIdx.y;

  int64_t sample_id = threadIdx.x + blockDim.x * blockIdx.x;

  double G = sample_id < n_local_samples ? g[{sample_id + sample_offset, 0, output}] : 0.0;
  double H = sample_id < n_local_samples ? h[{sample_id + sample_offset, 0, output}] : 0.0;

  double blocksumG = BlockReduce(temp_storage_g).Sum(G);
  double blocksumH = BlockReduce(temp_storage_h).Sum(H);

  if (threadIdx.x == 0) {
    atomicAdd(&base_sums[output], blocksumG);
    atomicAdd(&base_sums[output + n_outputs], blocksumH);
  }
}

// Cache to utilize shared memory for some bins
// start off with level0 only (lvl 1 should also be possible as it also only has 1 sample node)
class BinCache {
 public:
  legate::Buffer<int32_t, 1> cache_position;
  legate::Buffer<int32_t, 1> bin_idx_at_position;
  legate::Buffer<int32_t, 1> row_pointers;
  int32_t num_features;
  int32_t num_outputs;
  int32_t num_cached_features;
  int32_t num_cached_bins;
  int32_t max_cached_bins;  // == cache_size_bytes // (2 * 8 * num_outputs)
  static const int NOT_FOUND = -1;
  static const int TPB       = 512;

  BinCache(legate::Buffer<int32_t, 1> row_pointers,
           legate::Buffer<int32_t, 1> cache_position,
           legate::Buffer<int32_t, 1> bin_idx_at_position,
           int32_t num_features,
           int32_t num_outputs,
           int32_t num_cached_features,
           int32_t num_cached_bins,
           int32_t max_cached_bins)
    : row_pointers(row_pointers),
      cache_position(cache_position),
      bin_idx_at_position(bin_idx_at_position),
      num_features(num_features),
      num_outputs(num_outputs),
      num_cached_features(num_cached_features),
      num_cached_bins(num_cached_bins),
      max_cached_bins(max_cached_bins)
  {
  }

  __device__ inline void addToBin(int sampleNode,
                                  int output,
                                  int bin_idx,
                                  int feature,
                                  double val1,
                                  double val2,
                                  Histogram histogram,
                                  double* smem)
  {
    auto feature_start_bin = num_cached_features > 0 ? cache_position[{feature}] : NOT_FOUND;
    if (feature_start_bin != NOT_FOUND) {
      int feature_bin = bin_idx - row_pointers[feature];
      smem += ((feature_start_bin + feature_bin) * num_outputs + output) * 2;
      atomicAdd(smem, val1);
      atomicAdd(smem + 1, val2);
    } else {
      double* addPosition = reinterpret_cast<double*>(&histogram[{sampleNode, output, bin_idx}]);
      atomicAdd(addPosition, val1);
      atomicAdd(addPosition + 1, val2);
    }
  }

  __device__ inline void flushCache(int sampleNode, Histogram histogram, double* smem)
  {
    int cache_entries = num_cached_bins * num_outputs * 2;
    __syncthreads();
    for (int pos = threadIdx.x; pos < cache_entries; pos += blockDim.x) {
      double value = smem[pos];
      if (value != 0.0) {
        int output          = (pos >> 1) % num_outputs;
        int cache_bin       = (pos >> 1) / num_outputs;
        int bin_idx         = bin_idx_at_position[cache_bin];
        double* addPosition = reinterpret_cast<double*>(&histogram[{sampleNode, output, bin_idx}]);
        atomicAdd(addPosition + (pos & 1), value);
      }
    }
  }

  __device__ inline void initCache(double* smem)
  {
    int cache_entries = num_cached_bins * num_outputs * 2;
    for (int pos = threadIdx.x; pos < cache_entries; pos += blockDim.x) { smem[pos] = 0.0; }
    __syncthreads();
  }

  size_t smemBytes() { return num_cached_bins * num_outputs * 16; }

  int32_t numCachedFeatures() { return num_cached_features; }
};

BinCache SetupBinCache(legate::Buffer<int32_t, 1> row_pointers,
                       int32_t num_features,
                       int32_t num_outputs,
                       const void* kernel,
                       cudaStream_t stream)
{
  auto thrust_alloc = ThrustAllocator(legate::Memory::GPU_FB_MEM);
  auto policy       = DEFAULT_POLICY(thrust_alloc).on(stream);

  // 0. retrieve max. occupancy and extract available shared memory
  size_t avail_cache_size_bytes;
  int occupancy;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy, kernel, BinCache::TPB, 0);
  cudaOccupancyAvailableDynamicSMemPerBlock(
    &avail_cache_size_bytes, kernel, occupancy, BinCache::TPB);

  int max_cached_bins      = avail_cache_size_bytes / (16 * num_outputs);
  auto cache_position      = legate::create_buffer<int32_t, 1>({num_features});
  auto bin_idx_at_position = legate::create_buffer<int32_t, 1>({max_cached_bins});

  // 1. compute length per feature
  auto counting = thrust::make_counting_iterator(0);
  auto num_bins = legate::create_buffer<int32_t, 1>({num_features});
  thrust::transform(
    policy, counting, counting + num_features, num_bins.ptr(0), [=] __device__(int i) {
      return row_pointers[{i + 1}] - row_pointers[{i}];
    });

  // 2. argsort with iota
  auto feature_idx = legate::create_buffer<int32_t, 1>({num_features});
  thrust::sequence(policy, feature_idx.ptr(0), feature_idx.ptr(0) + num_features);
  thrust::sort_by_key(policy, num_bins.ptr(0), num_bins.ptr(0) + num_features, feature_idx.ptr(0));
  int bin_threshold = 64;
  int features_below_threshold =
    thrust::upper_bound(policy, num_bins.ptr(0), num_bins.ptr(0) + num_features, bin_threshold) -
    num_bins.ptr(0);

  // 3. scan
  thrust::inclusive_scan(policy, num_bins.ptr(0), num_bins.ptr(0) + num_features, num_bins.ptr(0));

  // 4. select last <= max_cached_bins, also restrict to max_features_size_k
  int max_cached_features =
    thrust::upper_bound(policy, num_bins.ptr(0), num_bins.ptr(0) + num_features, max_cached_bins) -
    num_bins.ptr(0);

  int cached_features = std::min(features_below_threshold, max_cached_features);

  // extract num_cached_bins
  int num_cached_bins = 0;
  if (cached_features > 0) {
    CHECK_CUDA(cudaMemcpyAsync(&num_cached_bins,
                               num_bins.ptr(cached_features - 1),
                               1 * sizeof(int32_t),
                               cudaMemcpyDeviceToHost,
                               stream));
  }

  // 5. fill data with custom kernel
  LaunchN(num_features, stream, [=] __device__(auto idx) {
    auto feature_pos = feature_idx[idx];
    auto cache_start = idx > 0 ? num_bins[idx - 1] : 0;
    auto num_entries = num_bins[idx] - cache_start;
    if (cache_start + num_entries < num_cached_bins) {
      cache_position[feature_pos] = cache_start;
      int bin_start               = row_pointers[feature_pos];
      for (int i = 0; i < num_entries; ++i) bin_idx_at_position[cache_start + i] = bin_start + i;
    } else {
      cache_position[feature_pos] = BinCache::NOT_FOUND;
    }
  });

  num_bins.destroy();
  feature_idx.destroy();
  std::cout << "DEBUG: caching " << cached_features << "/" << num_features
            << " features with a total of " << num_cached_bins
            << " bins, max_cached_bins = " << max_cached_bins << ", occupancy = " << occupancy
            << std::endl;
  return BinCache(row_pointers,
                  cache_position,
                  bin_idx_at_position,
                  num_features,
                  num_outputs,
                  cached_features,
                  num_cached_bins,
                  max_cached_bins);
}

// kernel with smem -- utilizes 1 warp per 32 samples, processing all features/outputs at once
template <typename TYPE, int TPB, bool ENABLE_CACHE, bool USE_PREFETCH = true>
__global__ static void __launch_bounds__(TPB, MIN_CTAS_PER_SM)
  fill_histogram(legate::AccessorRO<TYPE, 3> X,
                 size_t n_local_samples,
                 size_t n_features,
                 int64_t sample_offset,
                 legate::AccessorRO<double, 3> g,
                 legate::AccessorRO<double, 3> h,
                 size_t n_outputs,
                 SparseSplitProposals<TYPE> split_proposals,
                 NodeBatch batch,
                 Histogram histogram,
                 legate::Buffer<double, 2> node_hessians,
                 BinCache bin_cache)
{
  constexpr int32_t WarpSize = 32;
  const int32_t warp_id      = threadIdx.x / WarpSize;
  const int32_t lane_id      = threadIdx.x % WarpSize;
  // const bool enable_gh_prefetch = USE_PREFETCH && n_outputs <= WarpSize;

  extern __shared__ double smem[];

  const int32_t localIdx = blockIdx.x * TPB + warp_id * WarpSize + lane_id;

  // prefetch sampleNode information for all 32 ids
  auto [sampleNode_lane, localSampleId_lane] = (localIdx < batch.InstancesInBatch())
                                                 ? batch.instances_begin[localIdx]
                                                 : cuda::std::make_tuple(-1, -1);

  // check first and last element of block
  const bool block_use_cache =
    cuda::std::get<0>(batch.instances_begin[blockIdx.x * TPB]) ==
    cuda::std::get<0>(
      batch.instances_begin[min((blockIdx.x + 1) * TPB, (uint32_t)batch.InstancesInBatch()) - 1]);

  if constexpr (ENABLE_CACHE) {
    if (block_use_cache) bin_cache.initCache(smem);
  }

  for (int32_t lane_offset = 0; lane_offset < WarpSize; ++lane_offset) {
    const int32_t localSampleId = shfl(localSampleId_lane, lane_offset);

    if (localSampleId < 0) break;

    const int32_t sampleNode = shfl(sampleNode_lane, lane_offset);

    // maybe prefetch?
    // TODO

#pragma nounroll
    for (int32_t feature0 = 0; feature0 < n_features; feature0 += WarpSize) {
      const int32_t feature = feature0 + lane_id;
      const int32_t bin_idx =
        feature < n_features
          ? split_proposals.FindBin(X[{sample_offset + localSampleId, feature, 0}], feature)
          : SparseSplitProposals<TYPE>::NOT_FOUND;
#pragma nounroll
      for (int32_t output = 0; output < n_outputs; output++) {
        const double val1 = g[{sample_offset + localSampleId, 0, output}];
        const double val2 = h[{sample_offset + localSampleId, 0, output}];
        if (bin_idx != SparseSplitProposals<TYPE>::NOT_FOUND) {
          if (ENABLE_CACHE && block_use_cache) {
            bin_cache.addToBin(sampleNode, output, bin_idx, feature, val1, val2, histogram, smem);
          } else {
            double* addPosition =
              reinterpret_cast<double*>(&histogram[{sampleNode, output, bin_idx}]);
            atomicAdd(addPosition, val1);
            atomicAdd(addPosition + 1, val2);
          }
        }
      }
    }
  }

  if constexpr (ENABLE_CACHE) {
    if (block_use_cache)
      bin_cache.flushCache(
        cuda::std::get<0>(batch.instances_begin[blockIdx.x * TPB]), histogram, smem);
  }

  /*if (shfl(localSampleId_lane, 0) >= 0) {
    // preload G,H
    // every thread in the warp holds one element each, means that we pre-load WarpSize
    // elements of bot G & H.
    int32_t prefetch_offset = 0;
    int32_t prefetch_sample =
      enable_gh_prefetch ? shfl(localSampleId_lane, lane_id / n_outputs) : -1;
    double G_lane = prefetch_sample >= 0
                      ? g[{sample_offset + prefetch_sample, 0, lane_id % (int)n_outputs}]
                      : 0.0;
    double H_lane = prefetch_sample >= 0
                      ? h[{sample_offset + prefetch_sample, 0, lane_id % (int)n_outputs}]
                      : 0.0;

    // reverse to use __clz instead of __ffs
    lane_mask = __brev(lane_mask);

    do {
      // look for next lane_offset / sample to process within warp-batch
      const uint32_t lane_offset  = __clz(lane_mask);
      const int32_t sampleNode    = shfl(sampleNode_lane, lane_offset);
      const int32_t localSampleId = shfl(localSampleId_lane, lane_offset);

      // ensure all G.H for current sample are cached
      // when cache gets updated, cache start shifts to current working sample at offset
      if (((lane_offset + 1) * n_outputs - prefetch_offset) >= WarpSize) {
        prefetch_offset = lane_offset * n_outputs;
        prefetch_sample =
          enable_gh_prefetch
            ? shfl(localSampleId_lane, min(lane_offset + lane_id / (int)n_outputs, WarpSize - 1))
            : -1;
        G_lane = prefetch_sample >= 0
                   ? g[{sample_offset + prefetch_sample, 0, lane_id % (int)n_outputs}]
                   : 0.0;
        H_lane = prefetch_sample >= 0
                   ? h[{sample_offset + prefetch_sample, 0, lane_id % (int)n_outputs}]
                   : 0.0;
      }

      // remove lane_offset bit from lane_mask for next iteration
      lane_mask &= (0x7fffffff >> lane_offset);

#pragma nounroll
      for (int32_t feature0 = 0; feature0 < n_features; feature0 += WarpSize) {
        const int32_t feature = feature0 + lane_id;
        const int32_t bin_idx =
          feature < n_features
            ? split_proposals.FindBin(X[{sample_offset + localSampleId, feature, 0}], feature)
            : SparseSplitProposals<TYPE>::NOT_FOUND;
#pragma nounroll
        for (int32_t output = 0; output < n_outputs; output++) {
          // get G/H from thread that did the pre-fetch
          const int32_t prefetch_pos = lane_offset * n_outputs + output - prefetch_offset;
          const double val1          = enable_gh_prefetch ? shfl(G_lane, prefetch_pos)
                                                          : g[{sample_offset + localSampleId, 0,
output}]; const double val2          = enable_gh_prefetch ? shfl(H_lane, prefetch_pos) :
h[{sample_offset + localSampleId, 0, output}]; if (bin_idx != SparseSplitProposals<TYPE>::NOT_FOUND)
{ if constexpr (USE_CACHE) { bin_cache.addToBin(sampleNode, output, bin_idx, feature, val1, val2,
histogram, smem); } else { double* addPosition = reinterpret_cast<double*>(&histogram[{sampleNode,
output, bin_idx}]); atomicAdd(addPosition, val1); atomicAdd(addPosition + 1, val2);
            }
          }
        }
      }
    } while (lane_mask);
  }

  if constexpr (USE_CACHE) {
    // assert(depth == 0);
    // on lvl 0 / 1 we only have a single sample
    const int32_t histogram_node = 0;
    bin_cache.flushCache(histogram_node, histogram, smem);
  }*/
}

template <typename TYPE, int ELEMENTS_PER_THREAD, int FEATURES_PER_BLOCK>
__global__ static void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  fill_histogram_old(legate::AccessorRO<TYPE, 3> X,
                     size_t n_features,
                     int64_t sample_offset,
                     legate::AccessorRO<double, 3> g,
                     legate::AccessorRO<double, 3> h,
                     size_t n_outputs,
                     SparseSplitProposals<TYPE> split_proposals,
                     NodeBatch batch,
                     Histogram histogram,
                     legate::Buffer<double, 2> node_hessians)
{
  // block dimensions are (THREADS_PER_BLOCK, 1, 1)
  // each thread processes ELEMENTS_PER_THREAD samples and FEATURES_PER_BLOCK features
  // the features to process are defined via blockIdx.y

  // further improvements:
  // * quantize values to work with int instead of double

#pragma unroll
  for (int32_t elementIdx = 0; elementIdx < ELEMENTS_PER_THREAD; ++elementIdx) {
    // within each iteration a (THREADS_PER_BLOCK, FEATURES_PER_BLOCK)-block of
    // data from X is processed.

    // check if thread has actual work to do
    int64_t idx      = (blockIdx.x + elementIdx * gridDim.x) * THREADS_PER_BLOCK + threadIdx.x;
    bool validThread = idx < batch.InstancesInBatch();
    if (!validThread) continue;
    auto [sampleNode, localSampleId] = batch.instances_begin[idx];
    int64_t globalSampleId           = localSampleId + sample_offset;

    bool computeHistogram = ComputeHistogramBin(
      sampleNode, node_hessians, histogram.ContainsNode(BinaryTree::Parent(sampleNode)));

    for (int32_t output = 0; output < n_outputs; output++) {
      double G = g[{globalSampleId, 0, output}];
      double H = h[{globalSampleId, 0, output}];
      for (int32_t featureIdx = 0; featureIdx < FEATURES_PER_BLOCK; featureIdx++) {
        int32_t feature = featureIdx + blockIdx.y * FEATURES_PER_BLOCK;
        if (computeHistogram && feature < n_features) {
          auto x_value = X[{globalSampleId, feature, 0}];
          auto bin_idx = split_proposals.FindBin(x_value, feature);

          // bin_idx is the first sample that is larger than x_value
          if (bin_idx != SparseSplitProposals<TYPE>::NOT_FOUND) {
            double* addPosition =
              reinterpret_cast<double*>(&histogram[{sampleNode, output, bin_idx}]);
            atomicAdd(addPosition, G);
            atomicAdd(addPosition + 1, H);
          }
        }
      }
    }
  }
}

template <typename T>
__global__ static void __launch_bounds__(THREADS_PER_BLOCK)
  scan_kernel(Histogram histogram,
              legate::Buffer<double, 2> node_hessians,
              int n_features,
              int n_outputs,
              const SparseSplitProposals<T> split_proposals,
              NodeBatch batch)

{
  auto warp      = cg::tiled_partition<32>(cg::this_thread_block());
  int rank       = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  auto num_nodes = batch.NodesInBatch();
  int i          = rank / num_nodes;
  int j          = rank % num_nodes;

  // Specialize WarpScan for type int
  typedef cub::WarpScan<GPair> WarpScan;

  __shared__ typename WarpScan::TempStorage temp_storage[THREADS_PER_BLOCK / 32];

  int scan_node_idx = batch.node_idx_begin + j;
  int parent        = BinaryTree::Parent(scan_node_idx);
  // Exit if we didn't compute this histogram
  if (!ComputeHistogramBin(scan_node_idx, node_hessians, histogram.ContainsNode(parent))) return;
  if (i >= n_features || scan_node_idx >= batch.node_idx_end) return;

  int feature_idx                   = i;
  auto [feature_begin, feature_end] = split_proposals.FeatureRange(feature_idx);
  int num_bins                      = feature_end - feature_begin;
  int num_tiles                     = (num_bins + warp.num_threads() - 1) / warp.num_threads();

  for (int output = 0; output < n_outputs; output++) {
    GPair aggregate;
    // Scan left side
    for (int tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
      int bin_idx              = feature_begin + tile_idx * warp.num_threads() + warp.thread_rank();
      bool thread_participates = bin_idx < feature_end;
      auto e = thread_participates ? histogram[{scan_node_idx, output, bin_idx}] : GPair{0, 0};
      GPair tile_aggregate;
      WarpScan(temp_storage[threadIdx.x / warp.num_threads()]).InclusiveSum(e, e, tile_aggregate);
      __syncwarp();
      if (thread_participates) { histogram[{scan_node_idx, output, bin_idx}] = e + aggregate; }
      aggregate += tile_aggregate;
    }
  }

  // This node has no sibling we are finished
  if (scan_node_idx == 0) return;

  int sibling_node_idx = BinaryTree::Sibling(scan_node_idx);

  // The sibling did not compute a histogram
  // Do the subtraction trick using the histogram we just computed in the previous step
  if (!ComputeHistogramBin(sibling_node_idx, node_hessians, histogram.ContainsNode(parent))) {
    for (int output = 0; output < n_outputs; output++) {
      // Infer right side
      for (int bin_idx = feature_begin + warp.thread_rank(); bin_idx < feature_end;
           bin_idx += warp.num_threads()) {
        GPair scanned_sum = histogram[{scan_node_idx, output, bin_idx}];
        GPair parent_sum  = histogram[{BinaryTree::Parent(scan_node_idx), output, bin_idx}];
        GPair other_sum   = parent_sum - scanned_sum;
        histogram[{sibling_node_idx, output, bin_idx}] = other_sum;
      }
    }
  }
}

// Key/value pair to simplify reduction
struct GainFeaturePair {
  double gain;
  int feature;
  int feature_sample_idx;

  __device__ void operator=(const GainFeaturePair& other)
  {
    gain               = other.gain;
    feature            = other.feature;
    feature_sample_idx = other.feature_sample_idx;
  }

  __device__ bool operator==(const GainFeaturePair& other) const
  {
    return gain == other.gain && feature == other.feature &&
           feature_sample_idx == other.feature_sample_idx;
  }

  __device__ bool operator>(const GainFeaturePair& other) const { return gain > other.gain; }

  __device__ bool operator<(const GainFeaturePair& other) const { return gain < other.gain; }
};

template <typename TYPE>
__global__ static void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  perform_best_split(Histogram histogram,
                     size_t n_features,
                     size_t n_outputs,
                     SparseSplitProposals<TYPE> split_proposals,
                     double eps,
                     double alpha,
                     legate::Buffer<double, 2> tree_leaf_value,
                     legate::Buffer<double, 2> tree_gradient,
                     legate::Buffer<double, 2> tree_hessian,
                     legate::Buffer<int32_t, 1> tree_feature,
                     legate::Buffer<double, 1> tree_split_value,
                     legate::Buffer<double, 1> tree_gain,
                     NodeBatch batch)
{
  // using one block per (level) node to have blockwise reductions
  int node_id = batch.node_idx_begin + blockIdx.x;

  typedef cub::BlockReduce<GainFeaturePair, THREADS_PER_BLOCK> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  __shared__ double node_best_gain;
  __shared__ int node_best_feature;
  __shared__ int node_best_bin_idx;

  double thread_best_gain = 0;
  int thread_best_feature = -1;
  int thread_best_bin_idx = -1;

  for (int feature_id = 0; feature_id < n_features; feature_id++) {
    auto [feature_start, feature_end] = split_proposals.FeatureRange(feature_id);

    for (int bin_idx = feature_start + threadIdx.x; bin_idx < feature_end; bin_idx += blockDim.x) {
      double gain = 0;
      for (int output = 0; output < n_outputs; ++output) {
        auto G          = tree_gradient[{node_id, output}];
        auto H          = tree_hessian[{node_id, output}];
        auto [G_L, H_L] = histogram[{node_id, output, bin_idx}];
        auto G_R        = G - G_L;
        auto H_R        = H - H_L;

        if (H_L <= 0.0 || H_R <= 0.0) {
          gain = 0;
          break;
        }
        double reg = std::max(eps, alpha);  // Regularisation term
        gain += 0.5 * ((G_L * G_L) / (H_L + reg) + (G_R * G_R) / (H_R + reg) - (G * G) / (H + reg));
      }
      if (gain > thread_best_gain) {
        thread_best_gain    = gain;
        thread_best_feature = feature_id;
        thread_best_bin_idx = bin_idx;
      }
    }
  }

  // SYNC BEST GAIN TO FULL BLOCK/NODE
  GainFeaturePair thread_best_pair{thread_best_gain, thread_best_feature, thread_best_bin_idx};
  GainFeaturePair node_best_pair =
    BlockReduce(temp_storage).Reduce(thread_best_pair, cub::Max(), THREADS_PER_BLOCK);
  if (threadIdx.x == 0) {
    node_best_gain    = node_best_pair.gain;
    node_best_feature = node_best_pair.feature;
    node_best_bin_idx = node_best_pair.feature_sample_idx;
  }
  __syncthreads();

  if (node_best_gain > eps) {
    for (int output = threadIdx.x; output < n_outputs; output += blockDim.x) {
      auto [G_L, H_L] = histogram[{node_id, output, node_best_bin_idx}];
      auto G_R        = tree_gradient[{node_id, output}] - G_L;
      auto H_R        = tree_hessian[{node_id, output}] - H_L;

      int left_child                         = BinaryTree::LeftChild(node_id);
      int right_child                        = BinaryTree::RightChild(node_id);
      tree_leaf_value[{left_child, output}]  = CalculateLeafValue(G_L, H_L, alpha);
      tree_leaf_value[{right_child, output}] = CalculateLeafValue(G_R, H_R, alpha);
      tree_hessian[{left_child, output}]     = H_L;
      tree_hessian[{right_child, output}]    = H_R;
      tree_gradient[{left_child, output}]    = G_L;
      tree_gradient[{right_child, output}]   = G_R;

      if (output == 0) {
        tree_feature[node_id]     = node_best_feature;
        tree_split_value[node_id] = split_proposals.split_proposals[{node_best_bin_idx}];
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
    feature     = legate::create_buffer<int32_t, 1>({max_nodes});
    split_value = legate::create_buffer<double, 1>({max_nodes});
    gain        = legate::create_buffer<double, 1>({max_nodes});
    hessian     = legate::create_buffer<double, 2>({max_nodes, num_outputs});
    gradient    = legate::create_buffer<double, 2>({max_nodes, num_outputs});
    thrust::fill(thrust_exec_policy,
                 leaf_value.ptr({0, 0}),
                 leaf_value.ptr({0, 0}) + max_nodes * num_outputs,
                 0.0);
    thrust::fill(thrust_exec_policy, feature.ptr({0}), feature.ptr({0}) + max_nodes, -1);
    thrust::fill(
      thrust_exec_policy, hessian.ptr({0, 0}), hessian.ptr({0, 0}) + max_nodes * num_outputs, 0.0);
    thrust::fill(thrust_exec_policy, split_value.ptr({0}), split_value.ptr({0}) + max_nodes, 0.0);
    thrust::fill(thrust_exec_policy, gain.ptr({0}), gain.ptr({0}) + max_nodes, 0.0);
    thrust::fill(thrust_exec_policy,
                 gradient.ptr({0, 0}),
                 gradient.ptr({0, 0}) + max_nodes * num_outputs,
                 0.0);
  }

  ~Tree()
  {
    leaf_value.destroy();
    feature.destroy();
    split_value.destroy();
    gain.destroy();
    hessian.destroy();
    gradient.destroy();
  }

  void InitializeBase(legate::Buffer<double, 1> base_sums, double alpha)
  {
    LaunchN(num_outputs,
            stream,
            [            =,
             num_outputs = this->num_outputs,
             leaf_value  = this->leaf_value,
             gradient    = this->gradient,
             hessian     = this->hessian] __device__(int output) {
              leaf_value[{0, output}] =
                CalculateLeafValue(base_sums[output], base_sums[output + num_outputs], alpha);
              gradient[{0, output}] = base_sums[output];
              hessian[{0, output}]  = base_sums[output + num_outputs];
            });
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
                       [=] __host__ __device__(const legate::Point<DIM>& p) { out_acc[p] = x[p]; });
  }

  template <typename ThrustPolicyT>
  void WriteTreeOutput(legate::TaskContext context, const ThrustPolicyT& policy)
  {
    WriteOutput(context.output(0).data(), leaf_value, policy);
    WriteOutput(context.output(1).data(), feature, policy);
    WriteOutput(context.output(2).data(), split_value, policy);
    WriteOutput(context.output(3).data(), gain, policy);
    WriteOutput(context.output(4).data(), hessian, policy);
    CHECK_CUDA_STREAM(stream);
  }

  legate::Buffer<double, 2> leaf_value;
  legate::Buffer<int32_t, 1> feature;
  legate::Buffer<double, 1> split_value;
  legate::Buffer<double, 1> gain;
  legate::Buffer<double, 2> hessian;
  legate::Buffer<double, 2> gradient;
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
              SparseSplitProposals<T> split_proposals)
    : num_rows(num_rows),
      num_features(num_features),
      num_outputs(num_outputs),
      stream(stream),
      max_nodes(max_nodes),
      split_proposals(split_proposals)
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
    histogram      = Histogram(BinaryTree::LevelBegin(0),
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
  void ComputeHistogram(Histogram histogram,
                        legate::TaskContext context,
                        Tree& tree,
                        legate::AccessorRO<TYPE, 3> X,
                        legate::Rect<3> X_shape,
                        legate::AccessorRO<double, 3> g,
                        legate::AccessorRO<double, 3> h,
                        NodeBatch batch,
                        BinCache histogram_bin_cache,
                        int depth)
  {
    // TODO adjust kernel parameters dynamically

    if (histogram_bin_cache.numCachedFeatures() > 0) {
      const int threads_per_block = BinCache::TPB;
      const size_t blocks_x =
        (batch.InstancesInBatch() + threads_per_block - 1) / threads_per_block;
      dim3 grid_shape = dim3(blocks_x, 1, 1);

      fill_histogram<TYPE, threads_per_block, true>
        <<<grid_shape, threads_per_block, histogram_bin_cache.smemBytes(), stream>>>(
          X,
          num_rows,
          num_features,
          X_shape.lo[0],
          g,
          h,
          num_outputs,
          split_proposals,
          batch,
          histogram,
          tree.hessian,
          histogram_bin_cache);
    } else {
      const int threads_per_block = 256;
      const size_t blocks_x =
        (batch.InstancesInBatch() + threads_per_block - 1) / threads_per_block;
      dim3 grid_shape = dim3(blocks_x, 1, 1);
      fill_histogram<TYPE, threads_per_block, false>
        <<<grid_shape, threads_per_block, 0, stream>>>(X,
                                                       num_rows,
                                                       num_features,
                                                       X_shape.lo[0],
                                                       g,
                                                       h,
                                                       num_outputs,
                                                       split_proposals,
                                                       batch,
                                                       histogram,
                                                       tree.hessian,
                                                       histogram_bin_cache);
    }

    /*
    constexpr size_t elements_per_thread = 8;
    constexpr size_t features_per_block  = 16;

    const size_t blocks_x =
      (batch.InstancesInBatch() + THREADS_PER_BLOCK * elements_per_thread - 1) /
      (THREADS_PER_BLOCK * elements_per_thread);
    const size_t blocks_y = (num_features + features_per_block - 1) / features_per_block;
    dim3 grid_shape       = dim3(blocks_x, blocks_y, 1);
    fill_histogram_old<TYPE, elements_per_thread, features_per_block>
      <<<grid_shape, THREADS_PER_BLOCK, 0, stream>>>(X,
                                                     num_features,
                                                     X_shape.lo[0],
                                                     g,
                                                     h,
                                                     num_outputs,
                                                     split_proposals,
                                                     batch,
                                                     histogram,
                                                     tree.hessian);*/

    CHECK_CUDA_STREAM(stream);
    static_assert(sizeof(GPair) == 2 * sizeof(double), "GPair must be 2 doubles");
    SumAllReduce(context,
                 reinterpret_cast<double*>(histogram.Ptr(batch.node_idx_begin)),
                 batch.NodesInBatch() * num_outputs * split_proposals.histogram_size * 2,
                 stream);

    const size_t warps_needed    = num_features * batch.NodesInBatch();
    const size_t warps_per_block = THREADS_PER_BLOCK / 32;
    const size_t blocks_needed   = (warps_needed + warps_per_block - 1) / warps_per_block;

    // Scan the histograms
    scan_kernel<<<blocks_needed, THREADS_PER_BLOCK, 0, stream>>>(
      histogram, tree.hessian, num_features, num_outputs, split_proposals, batch);
    CHECK_CUDA_STREAM(stream);
  }

  void PerformBestSplit(Tree& tree, Histogram histogram, double alpha, NodeBatch batch)
  {
    perform_best_split<<<batch.NodesInBatch(), THREADS_PER_BLOCK, 0, stream>>>(histogram,
                                                                               num_features,
                                                                               num_outputs,
                                                                               split_proposals,
                                                                               eps,
                                                                               alpha,
                                                                               tree.leaf_value,
                                                                               tree.gradient,
                                                                               tree.hessian,
                                                                               tree.feature,
                                                                               tree.split_value,
                                                                               tree.gain,
                                                                               batch);
    CHECK_CUDA_STREAM(stream);
  }
  void InitialiseRoot(legate::TaskContext context,
                      Tree& tree,
                      legate::AccessorRO<double, 3> g,
                      legate::AccessorRO<double, 3> h,
                      legate::Rect<3> g_shape,
                      double alpha)
  {
    auto base_sums = legate::create_buffer<double, 1>(num_outputs * 2);

    CHECK_CUDA(cudaMemsetAsync(base_sums.ptr(0), 0, num_outputs * 2 * sizeof(double), stream));
    const size_t blocks = (num_rows + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    dim3 grid_shape     = dim3(blocks, num_outputs);
    reduce_base_sums<<<grid_shape, THREADS_PER_BLOCK, 0, stream>>>(
      g, h, num_rows, g_shape.lo[0], base_sums, num_outputs);
    CHECK_CUDA_STREAM(stream);

    SumAllReduce(context, reinterpret_cast<double*>(base_sums.ptr(0)), num_outputs * 2, stream);

    // base sums contain g-sums first, h sums second
    tree.InitializeBase(base_sums, alpha);

    base_sums.destroy();
    CHECK_CUDA_STREAM(stream);
  }

  // Create a new histogram for this batch if we need to
  // Destroy the old one
  Histogram GetHistogram(NodeBatch batch)
  {
    if (histogram.ContainsBatch(batch.node_idx_begin, batch.node_idx_end)) { return histogram; }

    CHECK_CUDA(cudaStreamSynchronize(stream));
    histogram.Destroy();
    histogram = Histogram(batch.node_idx_begin,
                          batch.node_idx_end,
                          num_outputs,
                          split_proposals.histogram_size,
                          stream);
    return histogram;
  }

  template <typename PolicyT>
  std::vector<NodeBatch> PrepareBatches(int depth, Tree& tree, PolicyT& policy)
  {
    if (depth > 0) {
      if (BinaryTree::LevelEnd(depth) < max_batch_size) {
        // here we still work with the initial histogram containing parents
        // which means due to the subtraction trick not all nodes have to be computed
        auto compute_bool = legate::create_buffer<bool, 1>({BinaryTree::NodesInLevel(depth)});
        LaunchN(BinaryTree::NodesInLevel(depth),
                stream,
                [offset           = BinaryTree::LevelBegin(depth),
                 compute_bool_ptr = compute_bool.ptr(0),
                 node_hessians    = tree.hessian] __device__(int idx) {
                  compute_bool_ptr[idx] = ComputeHistogramBin(idx + offset, node_hessians, true);
                });

        /*{
          auto count_negative = thrust::count_if(policy,
                  sorted_positions.ptr(0),
                  sorted_positions.ptr(0) + num_rows,
                  [] __device__(auto a) { return cuda::std::get<0>(a) < 0; });
          std::cerr << "Before sort: negative = " << count_negative << std::endl;
        }
        {
          auto count_negative = thrust::count_if(policy,
                  sorted_positions.ptr(0),
                  sorted_positions.ptr(0) + num_rows,
                  [offset = BinaryTree::LevelBegin(depth)] __device__(auto a) { return
        cuda::std::get<0>(a) < offset; }); std::cerr << "Before sort: < begin = " << count_negative
        << std::endl;
        }
        {
          auto count_negative = thrust::count_if(policy,
                  sorted_positions.ptr(0),
                  sorted_positions.ptr(0) + num_rows,
                  [offset = BinaryTree::LevelEnd(depth)] __device__(auto a) { return
        cuda::std::get<0>(a) >= offset; }); std::cerr << "Before sort: >= end = " << count_negative
        << std::endl;
        }*/

        auto comp2 = [offset           = BinaryTree::LevelBegin(depth),
                      compute_bool_ptr = compute_bool.ptr(0)] __device__(auto a, auto b) {
          bool need_a =
            cuda::std::get<0>(a) < offset ? false : compute_bool_ptr[cuda::std::get<0>(a) - offset];
          bool need_b =
            cuda::std::get<0>(b) < offset ? false : compute_bool_ptr[cuda::std::get<0>(b) - offset];
          if (need_a != need_b)
            return need_a;  // true before false
          else
            return cuda::std::get<0>(a) < cuda::std::get<0>(b);
        };

        thrust::sort(policy, sorted_positions.ptr(0), sorted_positions.ptr(num_rows), comp2);

        auto count_total =
          thrust::count_if(policy,
                           sorted_positions.ptr(0),
                           sorted_positions.ptr(num_rows),
                           [offset           = BinaryTree::LevelBegin(depth),
                            compute_bool_ptr = compute_bool.ptr(0)] __device__(auto a) {
                             return cuda::std::get<0>(a) >= offset &&
                                    compute_bool_ptr[cuda::std::get<0>(a) - offset];
                           });
        // std::cerr << "Total " << count_total << " of " << num_rows << " weights will be added to
        // histogram." <<  std::endl;

        compute_bool.destroy();

        return {NodeBatch{BinaryTree::LevelBegin(depth),
                          BinaryTree::LevelEnd(depth),
                          sorted_positions.ptr(0),
                          sorted_positions.ptr(0) + count_total}};

      } else {
        thrust::sort(
          policy,
          sorted_positions.ptr(0),
          sorted_positions.ptr(num_rows),
          [] __device__(auto a, auto b) { return cuda::std::get<0>(a) < cuda::std::get<0>(b); });
      }
    }

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
    auto batches          = legate::create_buffer<NodeBatch, 1>({num_batches});
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
  Histogram histogram;
  int max_batch_size;

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

    auto stream             = legate::cuda::StreamPool::get_stream_pool().get_stream();
    auto thrust_alloc       = ThrustAllocator(legate::Memory::GPU_FB_MEM);
    auto thrust_exec_policy = DEFAULT_POLICY(thrust_alloc).on(stream);

    Tree tree(max_nodes, num_outputs, stream, thrust_exec_policy);

    SparseSplitProposals<T> split_proposals =
      SelectSplitSamples(context, X_accessor, X_shape, split_samples, seed, dataset_rows, stream);
    // Begin building the tree
    TreeBuilder<T> builder(
      num_rows, num_features, num_outputs, stream, tree.max_nodes, max_depth, split_proposals);

    BinCache histogram_bin_cache =
      SetupBinCache(split_proposals.row_pointers,
                    num_features,
                    num_outputs,
                    (const void*)fill_histogram<T, BinCache::TPB, true>,
                    stream);

    builder.InitialiseRoot(context, tree, g_accessor, h_accessor, g_shape, alpha);

    for (int depth = 0; depth < max_depth; ++depth) {
      auto batches = builder.PrepareBatches(depth, tree, thrust_exec_policy);
      for (auto batch : batches) {
        auto histogram = builder.GetHistogram(batch);

        builder.ComputeHistogram(histogram,
                                 context,
                                 tree,
                                 X_accessor,
                                 X_shape,
                                 g_accessor,
                                 h_accessor,
                                 batch,
                                 histogram_bin_cache,
                                 depth);

        builder.PerformBestSplit(tree, histogram, alpha, batch);
      }
      // Update position of entire level
      // Don't bother updating positions for the last level
      if (depth < max_depth - 1) { builder.UpdatePositions(tree, X_accessor, X_shape); }

      /*{
        auto count_negative = thrust::count_if(thrust_exec_policy,
                  builder.sorted_positions.ptr(0),
                  builder.sorted_positions.ptr(0) + num_rows,
                  [] __device__(auto a) { return cuda::std::get<0>(a) < 0; });
        std::cerr << "count negative = " << count_negative << std::endl;
      }*/
    }

    tree.WriteTreeOutput(context, thrust_exec_policy);

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
