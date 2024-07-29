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

template <typename TYPE, int ELEMENTS_PER_THREAD, int FEATURES_PER_BLOCK>
__global__ static void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  fill_histogram(legate::AccessorRO<TYPE, 3> X,
                 size_t n_local_samples,
                 size_t n_features,
                 int64_t sample_offset,
                 legate::AccessorRO<double, 3> g,
                 legate::AccessorRO<double, 3> h,
                 size_t n_outputs,
                 SparseSplitProposals<TYPE> split_proposals,
                 int32_t* positions_local,
                 legate::Buffer<GPair, 3> histogram,
                 legate::Buffer<double, 2> node_hessians,
                 int depth)
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
    int32_t localSampleId = (blockIdx.x + elementIdx * gridDim.x) * THREADS_PER_BLOCK + threadIdx.x;
    int64_t globalSampleId = localSampleId + sample_offset;
    bool validThread       = localSampleId < n_local_samples;
    if (!validThread) continue;

    int32_t sampleNode    = positions_local[localSampleId];
    bool computeHistogram = ComputeHistogramBin(sampleNode, depth, node_hessians);

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
  scan_kernel(legate::Buffer<GPair, 3> histogram,
              legate::Buffer<double, 2> node_hessians,
              int n_features,
              int n_outputs,
              const SparseSplitProposals<T> split_proposals,
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

  if (depth == 0) return;

  for (int output = 0; output < n_outputs; output++) {
    // Infer right side
    for (int bin_idx = feature_begin + warp.thread_rank(); bin_idx < feature_end;
         bin_idx += warp.num_threads()) {
      GPair scanned_sum = histogram[{scan_node_idx, output, bin_idx}];
      GPair parent_sum  = histogram[{BinaryTree::Parent(scan_node_idx), output, bin_idx}];
      GPair other_sum   = parent_sum - scanned_sum;
      histogram[{subtract_node_idx, output, bin_idx}] = other_sum;
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
  perform_best_split(legate::Buffer<GPair, 3> histogram,
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
                     int depth)
{
  // using one block per (level) node to have blockwise reductions
  int node_id = blockIdx.x + BinaryTree::LevelBegin(depth);

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
template <typename T>
struct TreeBuilder {
  TreeBuilder(int32_t num_rows,
              int32_t num_features,
              int32_t num_outputs,
              cudaStream_t stream,
              int32_t max_nodes,
              SparseSplitProposals<T> split_proposals)
    : num_rows(num_rows),
      num_features(num_features),
      num_outputs(num_outputs),
      stream(stream),
      max_nodes(max_nodes),
      split_proposals(split_proposals)
  {
    positions = legate::create_buffer<int32_t>(num_rows);
    histogram_buffer =
      legate::create_buffer<GPair, 3>({max_nodes, num_outputs, split_proposals.histogram_size});
    CHECK_CUDA(
      cudaMemsetAsync(histogram_buffer.ptr(legate::Point<3>::ZEROES()),
                      0,
                      max_nodes * num_outputs * split_proposals.histogram_size * sizeof(GPair),
                      stream));
    // some initialization on first pass
    CHECK_CUDA(cudaMemsetAsync(positions.ptr(0), 0, (size_t)num_rows * sizeof(int32_t), stream));
  }

  ~TreeBuilder()
  {
    positions.destroy();
    histogram_buffer.destroy();
    if (cub_buffer_size > 0) cub_buffer.destroy();
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
                        legate::AccessorRO<double, 3> g,
                        legate::AccessorRO<double, 3> h)
  {
    // TODO adjust kernel parameters dynamically
    constexpr size_t elements_per_thread = 8;
    constexpr size_t features_per_block  = 16;
    const size_t blocks_x = (num_rows + THREADS_PER_BLOCK * elements_per_thread - 1) /
                            (THREADS_PER_BLOCK * elements_per_thread);
    const size_t blocks_y = (num_features + features_per_block - 1) / features_per_block;
    dim3 grid_shape       = dim3(blocks_x, blocks_y, 1);
    fill_histogram<TYPE, elements_per_thread, features_per_block>
      <<<grid_shape, THREADS_PER_BLOCK, 0, stream>>>(X,
                                                     num_rows,
                                                     num_features,
                                                     X_shape.lo[0],
                                                     g,
                                                     h,
                                                     num_outputs,
                                                     split_proposals,
                                                     positions.ptr(0),
                                                     histogram_buffer,
                                                     tree.hessian,
                                                     depth);
    CHECK_CUDA_STREAM(stream);
    static_assert(sizeof(GPair) == 2 * sizeof(double), "GPair must be 2 doubles");
    SumAllReduce(
      context,
      reinterpret_cast<double*>(histogram_buffer.ptr({BinaryTree::LevelBegin(depth), 0, 0})),
      BinaryTree::NodesInLevel(depth) * num_outputs * split_proposals.histogram_size * 2,
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
                                                                 split_proposals,
                                                                 depth,
                                                                 num_nodes_to_process);
    CHECK_CUDA_STREAM(stream);
  }

  void PerformBestSplit(int depth, Tree& tree, double alpha)
  {
    perform_best_split<<<BinaryTree::NodesInLevel(depth), THREADS_PER_BLOCK, 0, stream>>>(
      histogram_buffer,
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
  const int32_t num_rows;
  const int32_t num_features;
  const int32_t num_outputs;
  const int32_t max_nodes;
  SparseSplitProposals<T> split_proposals;

  legate::Buffer<unsigned char> cub_buffer;
  size_t cub_buffer_size = 0;

  legate::Buffer<GPair, 3> histogram_buffer;

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
      num_rows, num_features, num_outputs, stream, tree.max_nodes, split_proposals);

    builder.InitialiseRoot(context, tree, g_accessor, h_accessor, g_shape, alpha);

    for (int depth = 0; depth < max_depth; ++depth) {
      // update positions from previous step
      builder.UpdatePositions(depth, tree, X_accessor, X_shape);

      // actual histogram creation
      builder.ComputeHistogram(depth, context, tree, X_accessor, X_shape, g_accessor, h_accessor);

      // Select the best split
      builder.PerformBestSplit(depth, tree, alpha);
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
