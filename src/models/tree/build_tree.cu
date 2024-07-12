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
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/count.h>
#include <thrust/version.h>
#include <thrust/binary_search.h>
#include <cooperative_groups.h>
#include <cooperative_groups/scan.h>
namespace cg = cooperative_groups;

namespace legateboost {

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

template <typename TYPE, bool TRANSPOSE, int TPB, int SAMPLES_PER_BLOCK, int FEATURES_PER_BLOCK>
__global__ static void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  prepare_histogram_bins(legate::AccessorRO<TYPE, 3> X,
                         size_t n_local_samples,
                         size_t n_features,
                         int64_t sample_offset,
                         legate::AccessorRO<TYPE, 2> split_proposal,
                         int32_t samples_per_feature,
                         legate::Buffer<short, 2> bin_idx_buffer)
{
  // example :
  // TDB = 128
  // SAMPLES_PER_BLOCK = 16
  // FEATURES_PER_BLOCK = 16
  // ELEMENTS_PER_THREAD = 2
  // SMEM -> 16*17*8 = 2176
  // TRANSPOSE true  : returns features x samples
  // TRANSPOSE false : returns samples x features
  constexpr int ELEMENTS_PER_THREAD = SAMPLES_PER_BLOCK * FEATURES_PER_BLOCK / TPB;
  __shared__ double buffer_dbl[SAMPLES_PER_BLOCK][FEATURES_PER_BLOCK + 1];

  // load sample block into shared memory (row-wise strided)
  int32_t blockSample   = threadIdx.x / FEATURES_PER_BLOCK;
  int32_t blockFeature  = threadIdx.x % FEATURES_PER_BLOCK;
  int32_t localSampleId = blockIdx.x * SAMPLES_PER_BLOCK + blockSample;
  int32_t feature       = blockIdx.y * FEATURES_PER_BLOCK + blockFeature;
  int32_t stride        = TPB / FEATURES_PER_BLOCK;
  for (int32_t i = 0; i < ELEMENTS_PER_THREAD;
       i++, blockSample += stride, localSampleId += stride) {
    buffer_dbl[blockSample][blockFeature] = localSampleId < n_local_samples && feature < n_features
                                              ? X[{localSampleId + sample_offset, feature, 0}]
                                              : 0.0;  // TODO maybe inf?
  }

  __syncthreads();

  // compute bins --> now we access buffer col-wise strided
  blockSample   = threadIdx.x % SAMPLES_PER_BLOCK;
  blockFeature  = threadIdx.x / SAMPLES_PER_BLOCK;
  localSampleId = blockIdx.x * SAMPLES_PER_BLOCK + blockSample;
  feature       = blockIdx.y * FEATURES_PER_BLOCK + blockFeature;
  stride        = TPB / SAMPLES_PER_BLOCK;
  for (int32_t i = 0; i < ELEMENTS_PER_THREAD; i++, blockFeature += stride, feature += stride) {
    if (localSampleId < n_local_samples && feature < n_features) {
      int bin_idx = thrust::lower_bound(thrust::seq,
                                        split_proposal.ptr({feature, 0}),
                                        split_proposal.ptr({feature, samples_per_feature}),
                                        buffer_dbl[blockSample][blockFeature]) -
                    split_proposal.ptr({feature, 0});
      if constexpr (TRANSPOSE) {
        bin_idx_buffer[{feature, localSampleId}] = (short)bin_idx;
      } else {
        reinterpret_cast<int*>(&buffer_dbl[blockSample][blockFeature])[0] = bin_idx;
      }
    }
  }

  if constexpr (!TRANSPOSE) {
    __syncthreads();

    blockSample   = threadIdx.x / FEATURES_PER_BLOCK;
    blockFeature  = threadIdx.x % FEATURES_PER_BLOCK;
    localSampleId = blockIdx.x * SAMPLES_PER_BLOCK + blockSample;
    feature       = blockIdx.y * FEATURES_PER_BLOCK + blockFeature;
    stride        = TPB / FEATURES_PER_BLOCK;
    for (int32_t i = 0; i < ELEMENTS_PER_THREAD;
         i++, blockSample += stride, localSampleId += stride) {
      if (localSampleId < n_local_samples && feature < n_features) {
        bin_idx_buffer[{localSampleId, feature}] =
          reinterpret_cast<int*>(&buffer_dbl[blockSample][blockFeature])[0];
      }
    }
  }
}

// kernel without smem -- utilizes 1 warp per 32 samples, processing all features/outputs at once
template <typename TYPE, int TPB>
__global__ static void __launch_bounds__(TPB, MIN_CTAS_PER_SM)
  fill_histogram(legate::AccessorRO<TYPE, 3> X,
                 size_t n_local_samples,
                 size_t n_features,
                 int64_t sample_offset,
                 legate::AccessorRO<double, 3> g,
                 legate::AccessorRO<double, 3> h,
                 size_t n_outputs,
                 int32_t samples_per_feature,
                 int32_t* positions_local,
                 legate::Buffer<GPair, 4> histogram,
                 legate::Buffer<double, 2> node_hessians,
                 int32_t depth,
                 legate::Buffer<short, 2> bin_idx_buffer)
{
  constexpr int32_t WarpSize    = 32;
  const int32_t warp_id         = threadIdx.x / WarpSize;
  const int32_t lane_id         = threadIdx.x % WarpSize;
  const bool enable_gh_prefetch = n_outputs <= WarpSize;

  int32_t localSampleId0 = blockIdx.x * TPB + warp_id * WarpSize;

  // prefetch sampleNode information for all 32 ids
  const int32_t sampleNode_lane =
    (localSampleId0 + lane_id) < n_local_samples ? positions_local[(localSampleId0 + lane_id)] : 0;
  bool computeHistogram = (localSampleId0 + lane_id) < n_local_samples &&
                          ComputeHistogramBin(sampleNode_lane, depth, node_hessians);

  // mask contains all sample bits of the next 32 ids that need to be bin'ed
  auto lane_mask = ballot(computeHistogram);

  if (lane_mask == 0) return;

  // preload G,H
  // every thread in the warp holds one element each
  int32_t cache_start  = 0;
  int32_t cache_sample = localSampleId0 + lane_id / n_outputs;
  int32_t last_warp_sample =
    enable_gh_prefetch ? min(localSampleId0 + WarpSize, (int32_t)n_local_samples) : 0;
  double G_lane = cache_sample < last_warp_sample
                    ? g[{sample_offset + cache_sample, 0, lane_id % (int)n_outputs}]
                    : 0.0;
  double H_lane = cache_sample < last_warp_sample
                    ? h[{sample_offset + cache_sample, 0, lane_id % (int)n_outputs}]
                    : 0.0;

  // reverse to use __clz instead of __ffs
  lane_mask = __brev(lane_mask);

  do {
    // look for next lane_offset
    const uint32_t lane_offset = __clz(lane_mask);
    const int32_t sampleNode   = shfl(sampleNode_lane, lane_offset);

    // ensure all G.H for current sample are cached
    if (enable_gh_prefetch && (((lane_offset + 1) * n_outputs - cache_start) >= WarpSize)) {
      cache_start  = lane_offset * n_outputs;
      cache_sample = localSampleId0 + lane_offset + lane_id / n_outputs;
      G_lane       = cache_sample < last_warp_sample
                       ? g[{sample_offset + cache_sample, 0, lane_id % (int)n_outputs}]
                       : 0.0;
      H_lane       = cache_sample < last_warp_sample
                       ? h[{sample_offset + cache_sample, 0, lane_id % (int)n_outputs}]
                       : 0.0;
    }

    // remove lane_offset bit from lane_mask for next iteration
    lane_mask &= (0x7fffffff >> lane_offset);

#pragma nounroll
    for (int32_t feature0 = 0; feature0 < n_features; feature0 += WarpSize) {
      auto bin_idx = (feature0 + lane_id) < n_features
                       ? (int)bin_idx_buffer[{localSampleId0 + lane_offset, (feature0 + lane_id)}]
                       : samples_per_feature;
#pragma nounroll
      for (int32_t output = 0; output < n_outputs; output++) {
        // get G/H from cache
        int32_t cache_pos = lane_offset * n_outputs + output - cache_start;
        double val        = enable_gh_prefetch
                              ? shfl(G_lane, cache_pos)
                              : g[{sample_offset + localSampleId0 + lane_offset, 0, output}];
        double* addPosition =
          reinterpret_cast<double*>(&histogram[{sampleNode, feature0 + lane_id, output, bin_idx}]);
        if (bin_idx < samples_per_feature) {
          // add G
          atomicAdd(addPosition, val);
        }
        val = enable_gh_prefetch ? shfl(H_lane, cache_pos)
                                 : h[{sample_offset + localSampleId0 + lane_offset, 0, output}];
        if (bin_idx < samples_per_feature) {
          // add H
          atomicAdd(addPosition + 1, val);
        }
      }
    }
  } while (lane_mask);
}

__global__ static void __launch_bounds__(THREADS_PER_BLOCK)
  scan_kernel(legate::Buffer<GPair, 4> histogram,
              legate::Buffer<double, 2> node_hessians,
              int n_features,
              int n_outputs,
              int samples_per_feature,
              int depth,
              int num_nodes_to_process)

{
  auto warp = cg::tiled_partition<32>(cg::this_thread_block());
  int rank  = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  int i     = rank / num_nodes_to_process;
  int j     = rank % num_nodes_to_process;

  // Specialize WarpScan for type int
  typedef cub::WarpScan<GPair> WarpScan;

  __shared__ typename WarpScan::TempStorage temp_storage[THREADS_PER_BLOCK / 32];

  if (i >= n_features) return;

  int scan_node_idx, subtract_node_idx;
  if (depth == 0) {
    scan_node_idx     = 0;
    subtract_node_idx = -1;
  } else {
    int parent_idx    = BinaryTree::LevelBegin(depth - 1) + j;
    auto [scan, sub]  = SelectHistogramNode(parent_idx, node_hessians);
    scan_node_idx     = scan;
    subtract_node_idx = sub;
  }

  int feature_idx = i;
  int num_tiles   = (samples_per_feature + warp.num_threads() - 1) / warp.num_threads();

  for (int output = 0; output < n_outputs; output++) {
    GPair aggregate;
    // Scan left side
    for (int tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
      int sample_idx           = tile_idx * warp.num_threads() + warp.thread_rank();
      bool thread_participates = sample_idx < samples_per_feature;
      auto e = thread_participates ? histogram[{scan_node_idx, feature_idx, output, sample_idx}]
                                   : GPair{0, 0};
      GPair tile_aggregate;
      WarpScan(temp_storage[threadIdx.x / warp.num_threads()]).InclusiveSum(e, e, tile_aggregate);
      __syncwarp();
      if (thread_participates) {
        histogram[{scan_node_idx, feature_idx, output, sample_idx}] = e + aggregate;
      }
      aggregate += tile_aggregate;
    }
  }

  if (depth == 0) return;

  for (int output = 0; output < n_outputs; output++) {
    // Infer right side
    for (int sample_idx = warp.thread_rank(); sample_idx < samples_per_feature;
         sample_idx += warp.num_threads()) {
      GPair scanned_sum = histogram[{scan_node_idx, feature_idx, output, sample_idx}];
      GPair parent_sum =
        histogram[{BinaryTree::Parent(scan_node_idx), feature_idx, output, sample_idx}];
      GPair other_sum                                                 = parent_sum - scanned_sum;
      histogram[{subtract_node_idx, feature_idx, output, sample_idx}] = other_sum;
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
  perform_best_split(legate::Buffer<GPair, 4> histogram,
                     size_t n_features,
                     size_t n_outputs,
                     legate::AccessorRO<TYPE, 2> split_proposal,
                     int32_t samples_per_feature,
                     double eps,
                     double alpha,
                     legate::Buffer<double, 2> tree_leaf_value,
                     legate::Buffer<double, 2> tree_gradient,
                     legate::Buffer<double, 2> tree_hessian,
                     legate::Buffer<int32_t, 1> tree_feature,
                     legate::Buffer<double, 1> tree_split_value,
                     legate::Buffer<double, 1> tree_gain,
                     int depth)
{
  // using one block per (level) node to have blockwise reductions
  int node_id = blockIdx.x + BinaryTree::LevelBegin(depth);

  typedef cub::BlockReduce<GainFeaturePair, THREADS_PER_BLOCK> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  __shared__ double node_best_gain;
  __shared__ int node_best_feature;
  __shared__ int node_best_feature_sample;

  double thread_best_gain        = 0;
  int thread_best_feature        = -1;
  int thread_best_feature_sample = -1;

  for (int feature_id = 0; feature_id < n_features; feature_id++) {
    for (int feature_sample_idx = threadIdx.x; feature_sample_idx < samples_per_feature;
         feature_sample_idx += blockDim.x) {
      double gain = 0;
      for (int output = 0; output < n_outputs; ++output) {
        auto G          = tree_gradient[{node_id, output}];
        auto H          = tree_hessian[{node_id, output}];
        auto [G_L, H_L] = histogram[{node_id, feature_id, output, feature_sample_idx}];
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
        thread_best_gain           = gain;
        thread_best_feature        = feature_id;
        thread_best_feature_sample = feature_sample_idx;
      }
    }
  }

  // SYNC BEST GAIN TO FULL BLOCK/NODE
  GainFeaturePair thread_best_pair{
    thread_best_gain, thread_best_feature, thread_best_feature_sample};
  GainFeaturePair node_best_pair =
    BlockReduce(temp_storage).Reduce(thread_best_pair, cub::Max(), THREADS_PER_BLOCK);
  if (threadIdx.x == 0) {
    node_best_gain           = node_best_pair.gain;
    node_best_feature        = node_best_pair.feature;
    node_best_feature_sample = node_best_pair.feature_sample_idx;
  }
  __syncthreads();

  if (node_best_gain > eps) {
    for (int output = threadIdx.x; output < n_outputs; output += blockDim.x) {
      auto [G_L, H_L] = histogram[{node_id, node_best_feature, output, node_best_feature_sample}];
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
        tree_split_value[node_id] = split_proposal[{node_best_feature, node_best_feature_sample}];
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

struct TreeBuilder {
  TreeBuilder(int32_t num_rows,
              int32_t num_features,
              int32_t num_outputs,
              cudaStream_t stream,
              int32_t max_nodes,
              int32_t samples_per_feature)
    : num_rows(num_rows),
      num_features(num_features),
      num_outputs(num_outputs),
      stream(stream),
      max_nodes(max_nodes),
      samples_per_feature(samples_per_feature)
  {
    positions = legate::create_buffer<int32_t>(num_rows);
    histogram_buffer =
      legate::create_buffer<GPair, 4>({max_nodes, num_features, num_outputs, samples_per_feature});
    CHECK_CUDA(
      cudaMemsetAsync(histogram_buffer.ptr(legate::Point<4>::ZEROES()),
                      0,
                      max_nodes * num_features * samples_per_feature * num_outputs * sizeof(GPair),
                      stream));
    // some initialization on first pass
    CHECK_CUDA(cudaMemsetAsync(positions.ptr(0), 0, (size_t)num_rows * sizeof(int32_t), stream));
  }

  ~TreeBuilder()
  {
    positions.destroy();
    histogram_buffer.destroy();
    if (cub_buffer_size > 0) cub_buffer.destroy();
    bin_idx_buffer.destroy();
  }

  template <typename TYPE>
  void UpdatePositions(int depth,
                       Tree& tree,
                       legate::AccessorRO<TYPE, 3> X,
                       legate::Rect<3> X_shape)
  {
    if (depth == 0) return;
    auto tree_split_value_ptr    = tree.split_value.ptr(0);
    auto tree_feature_ptr        = tree.feature.ptr(0);
    auto positions_ptr           = positions.ptr(0);
    auto max_nodes_              = this->max_nodes;
    auto update_positions_lambda = [=] __device__(size_t idx) {
      int32_t& pos = positions_ptr[idx];
      if (pos < 0 || pos >= max_nodes_ || tree_feature_ptr[pos] == -1) {
        pos = -1;
        return;
      }
      double x_value = X[{X_shape.lo[0] + (int64_t)idx, tree_feature_ptr[pos], 0}];
      bool left      = x_value <= tree_split_value_ptr[pos];
      pos            = left ? BinaryTree::LeftChild(pos) : BinaryTree::RightChild(pos);
    };
    LaunchN(num_rows, stream, update_positions_lambda);
    CHECK_CUDA_STREAM(stream);
  }

  template <typename TYPE>
  void ComputeHistogram(int depth,
                        legate::TaskContext context,
                        Tree& tree,
                        legate::AccessorRO<TYPE, 3> X,
                        legate::Rect<3> X_shape,
                        legate::AccessorRO<TYPE, 2> split_proposal,
                        legate::AccessorRO<double, 3> g,
                        legate::AccessorRO<double, 3> h)
  {
    if (depth == 0) {
      bin_idx_buffer = legate::create_buffer<short, 2>({num_rows, num_features});
      constexpr size_t features_per_block = 16;
      constexpr size_t samples_per_block  = 16;
      const size_t blocks_x               = (num_rows + samples_per_block - 1) / samples_per_block;
      const size_t blocks_y = (num_features + features_per_block - 1) / features_per_block;
      dim3 grid_shape       = dim3(blocks_x, blocks_y, 1);
      prepare_histogram_bins<TYPE, false, THREADS_PER_BLOCK, samples_per_block, features_per_block>
        <<<grid_shape, THREADS_PER_BLOCK, 0, stream>>>(X,
                                                       num_rows,
                                                       num_features,
                                                       X_shape.lo[0],
                                                       split_proposal,
                                                       samples_per_feature,
                                                       bin_idx_buffer);
      CHECK_CUDA_STREAM(stream);
    }

    const int threads_per_block = 256;
    const size_t blocks_x       = (num_rows + threads_per_block - 1) / threads_per_block;
    dim3 grid_shape             = dim3(blocks_x, 1, 1);

    fill_histogram<TYPE, threads_per_block>
      <<<grid_shape, threads_per_block, 0, stream>>>(X,
                                                     num_rows,
                                                     num_features,
                                                     X_shape.lo[0],
                                                     g,
                                                     h,
                                                     num_outputs,
                                                     samples_per_feature,
                                                     positions.ptr(0),
                                                     histogram_buffer,
                                                     tree.hessian,
                                                     depth,
                                                     bin_idx_buffer);

    CHECK_CUDA_STREAM(stream);
    static_assert(sizeof(GPair) == 2 * sizeof(double), "GPair must be 2 doubles");
    SumAllReduce(
      context,
      reinterpret_cast<double*>(histogram_buffer.ptr({BinaryTree::LevelBegin(depth), 0, 0, 0})),
      BinaryTree::NodesInLevel(depth) * num_features * samples_per_feature * num_outputs * 2,
      stream);

    const int num_nodes_to_process = std::max(BinaryTree::NodesInLevel(depth) / 2, 1);
    const size_t warps_needed      = num_features * num_nodes_to_process;
    const size_t warps_per_block   = THREADS_PER_BLOCK / 32;
    const size_t blocks_needed     = (warps_needed + warps_per_block - 1) / warps_per_block;

    // Scan the histogram
    // Then do subtraction trick to infer right side from parent and left side
    scan_kernel<<<blocks_needed, THREADS_PER_BLOCK, 0, stream>>>(histogram_buffer,
                                                                 tree.hessian,
                                                                 num_features,
                                                                 num_outputs,
                                                                 samples_per_feature,
                                                                 depth,
                                                                 num_nodes_to_process);
    CHECK_CUDA_STREAM(stream);
  }

  template <typename TYPE>
  void PerformBestSplit(int depth,
                        Tree& tree,
                        legate::AccessorRO<TYPE, 2> split_proposal,
                        double alpha)
  {
    perform_best_split<<<BinaryTree::NodesInLevel(depth), THREADS_PER_BLOCK, 0, stream>>>(
      histogram_buffer,
      num_features,
      num_outputs,
      split_proposal,
      samples_per_feature,
      eps,
      alpha,
      tree.leaf_value,
      tree.gradient,
      tree.hessian,
      tree.feature,
      tree.split_value,
      tree.gain,
      depth);
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

  legate::Buffer<int32_t> positions;
  legate::Buffer<short, 2> bin_idx_buffer;
  const int32_t num_rows;
  const int32_t num_features;
  const int32_t num_outputs;
  const int32_t max_nodes;
  const int32_t samples_per_feature;

  legate::Buffer<unsigned char> cub_buffer;
  size_t cub_buffer_size = 0;

  legate::Buffer<GPair, 4> histogram_buffer;

  cudaStream_t stream;
};

struct build_tree_fn {
  template <typename T>
  void operator()(legate::TaskContext context)
  {
    auto [X, X_shape, X_accessor] = GetInputStore<T, 3>(context.input(0).data());
    auto [g, g_shape, g_accessor] = GetInputStore<double, 3>(context.input(1).data());
    auto [h, h_shape, h_accessor] = GetInputStore<double, 3>(context.input(2).data());
    auto [split_proposals, split_proposals_shape, split_proposals_accessor] =
      GetInputStore<T, 2>(context.input(3).data());
    EXPECT_DENSE_ROW_MAJOR(X_accessor.accessor, X_shape);
    EXPECT_DENSE_ROW_MAJOR(split_proposals_accessor.accessor, split_proposals_shape);
    auto num_features = X_shape.hi[1] - X_shape.lo[1] + 1;
    auto num_rows     = std::max<int64_t>(X_shape.hi[0] - X_shape.lo[0] + 1, 0);
    auto num_outputs  = X_shape.hi[2] - X_shape.lo[2] + 1;
    EXPECT(g_shape.lo[2] == 0, "Outputs should not be split between workers.");
    EXPECT_AXIS_ALIGNED(0, X_shape, g_shape);
    EXPECT_AXIS_ALIGNED(0, g_shape, h_shape);
    EXPECT_AXIS_ALIGNED(1, g_shape, h_shape);
    EXPECT_IS_BROADCAST(split_proposals_shape);
    auto samples_per_feature = split_proposals_shape.hi[1] - split_proposals_shape.lo[1] + 1;

    // Scalars
    auto max_depth = context.scalars().at(0).value<int>();
    auto max_nodes = context.scalars().at(1).value<int>();
    auto alpha     = context.scalars().at(2).value<double>();

    auto stream             = legate::cuda::StreamPool::get_stream_pool().get_stream();
    auto thrust_alloc       = ThrustAllocator(legate::Memory::GPU_FB_MEM);
    auto thrust_exec_policy = DEFAULT_POLICY(thrust_alloc).on(stream);

    Tree tree(max_nodes, num_outputs, stream, thrust_exec_policy);

    // Begin building the tree
    TreeBuilder builder(
      num_rows, num_features, num_outputs, stream, tree.max_nodes, samples_per_feature);

    builder.InitialiseRoot(context, tree, g_accessor, h_accessor, g_shape, alpha);

    for (int depth = 0; depth < max_depth; ++depth) {
      // update positions from previous step
      builder.UpdatePositions(depth, tree, X_accessor, X_shape);

      // actual histogram creation
      builder.ComputeHistogram(depth,
                               context,
                               tree,
                               X_accessor,
                               X_shape,
                               split_proposals_accessor,
                               g_accessor,
                               h_accessor);

      // Select the best split
      builder.PerformBestSplit(depth, tree, split_proposals_accessor, alpha);
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
