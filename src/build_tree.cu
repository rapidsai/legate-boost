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
#include "utils.h"
#include "core/comm/coll.h"
#include "build_tree.h"
#include "cuda_help.h"
#include "kernel_helper.cuh"
#include <numeric>

#include <cub/device/device_radix_sort.cuh>
#include <thrust/execution_policy.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/count.h>
#include <thrust/version.h>

namespace legateboost {

__global__ static void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  reduce_base_sums(legate::AccessorRO<double, 2> g,
                   legate::AccessorRO<double, 2> h,
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

  double G = sample_id < n_local_samples ? g[{sample_id + sample_offset, output}] : 0.0;
  double H = sample_id < n_local_samples ? h[{sample_id + sample_offset, output}] : 0.0;

  double blocksumG = BlockReduce(temp_storage_g).Sum(G);
  double blocksumH = BlockReduce(temp_storage_h).Sum(H);

  if (threadIdx.x == 0) {
    atomicAdd(&base_sums[output], blocksumG);
    atomicAdd(&base_sums[output + n_outputs], blocksumH);
  }
}

template <typename TYPE, int ELEMENTS_PER_THREAD, int FEATURES_PER_BLOCK>
__global__ static void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  fill_histogram_blockreduce(legate::AccessorRO<TYPE, 2> X,
                             size_t n_local_samples,
                             size_t n_features,
                             int64_t sample_offset,
                             legate::AccessorRO<double, 2> g,
                             legate::AccessorRO<double, 2> h,
                             size_t n_outputs,
                             legate::AccessorRO<TYPE, 2> split_proposal,
                             int32_t* positions_local,
                             int32_t* sample_index_local,
                             legate::Buffer<GPair, 3> histogram,
                             int32_t max_nodes_in_level,
                             int32_t depth)
{
  // block dimensions are (THREADS_PER_BLOCK, 1, 1)
  // each thread processes ELEMENTS_PER_THREAD samples and FEATURES_PER_BLOCK features
  // the features to process are defined via blockIdx.y

  // further improvements:
  // * quantize values to work with int instead of double

  typedef cub::BlockReduce<double, THREADS_PER_BLOCK> BlockReduce;

  // alternate storages to spare syncthreads
  __shared__ typename BlockReduce::TempStorage temp_storage1;
  __shared__ typename BlockReduce::TempStorage temp_storage2;

  __shared__ bool left_shared[FEATURES_PER_BLOCK][THREADS_PER_BLOCK + 1];

  // mapping from local sampleId to global sample id (also accounting for reordering)
  __shared__ int32_t index_mapping[THREADS_PER_BLOCK];

#pragma unroll
  for (int32_t elementIdx = 0; elementIdx < ELEMENTS_PER_THREAD; ++elementIdx) {
    // within each iteration a (THREADS_PER_BLOCK, FEATURES_PER_BLOCK)-block of
    // data from X is processed.

    // check if any thread of this block has work to do
    if ((blockIdx.x + elementIdx * gridDim.x) * THREADS_PER_BLOCK >= n_local_samples) break;

    // check if thread has actual work todo (besides taking part in reductions)
    int32_t sampleId = (blockIdx.x + elementIdx * gridDim.x) * THREADS_PER_BLOCK + threadIdx.x;
    bool validThread = sampleId < n_local_samples;

    index_mapping[threadIdx.x] = validThread ? sample_index_local[sampleId] + sample_offset : -1;

    __syncthreads();

    // read input X and store split decision in shared memory
    for (int32_t featureIdx = 0; featureIdx < FEATURES_PER_BLOCK; featureIdx++) {
      // load is done with transpose access to the (THREADS_PER_BLOCK, FEATURES_PER_BLOCK)-block
      int32_t localFeatureId = (threadIdx.x + featureIdx * THREADS_PER_BLOCK) % FEATURES_PER_BLOCK;
      int32_t localSampleId  = (threadIdx.x + featureIdx * THREADS_PER_BLOCK) / FEATURES_PER_BLOCK;
      int32_t feature        = blockIdx.y * FEATURES_PER_BLOCK + localFeatureId;
      int32_t globalSampleId =
        (blockIdx.x + elementIdx * gridDim.x) * THREADS_PER_BLOCK + localSampleId;
      left_shared[localFeatureId][localSampleId] =
        (globalSampleId < n_local_samples && feature < n_features)
          ? X[{index_mapping[localSampleId], feature}] <= split_proposal[{depth, feature}]
          : false;
    }

    // loading left_shared was done in different order
    __syncthreads();

    bool useAtomicAdd = false;
    {
      int32_t firstNode =
        positions_local[(blockIdx.x + elementIdx * gridDim.x) * THREADS_PER_BLOCK];
      int32_t lastActiveThread = (blockIdx.x + elementIdx * gridDim.x + 1) * THREADS_PER_BLOCK - 1;
      if (lastActiveThread >= n_local_samples) lastActiveThread = n_local_samples - 1;
      int32_t lastNode = positions_local[lastActiveThread];
      useAtomicAdd     = (firstNode < lastNode);
    }
    int32_t sampleNode = validThread ? positions_local[sampleId] - max_nodes_in_level + 1 : -1;

    for (int32_t output = 0; output < n_outputs; output++) {
      double G = validThread ? g[{index_mapping[threadIdx.x], output}] : 0.0;
      double H = validThread ? h[{index_mapping[threadIdx.x], output}] : 0.0;
      for (int32_t featureIdx = 0; featureIdx < FEATURES_PER_BLOCK; featureIdx++) {
        int32_t feature = featureIdx + blockIdx.y * FEATURES_PER_BLOCK;
        if (feature < n_features) {
          if (useAtomicAdd) {
            if (left_shared[featureIdx][threadIdx.x] && sampleNode >= 0) {
              double* addPosition =
                reinterpret_cast<double*>(&histogram[{sampleNode, feature, output}]);
              atomicAdd(addPosition, G);
              atomicAdd(addPosition + 1, H);
            }
          } else {
            // reduce within block first
            double* addPosition =
              reinterpret_cast<double*>(&histogram[{sampleNode, feature, output}]);
            double input = left_shared[featureIdx][threadIdx.x] ? G : 0.0;
            double sum   = BlockReduce(temp_storage1).Sum(input);
            if (threadIdx.x == 0 && sum != 0) { atomicAdd(addPosition, sum); }
            input = left_shared[featureIdx][threadIdx.x] ? H : 0.0;
            sum   = BlockReduce(temp_storage2).Sum(input);
            if (threadIdx.x == 0 && sum != 0) { atomicAdd(addPosition + 1, sum); }
          }
        }
      }
    }
  }
}

// Key/value pair to simplify reduction
struct GainFeaturePair {
  double gain;
  int feature;

  __device__ void operator=(const GainFeaturePair& other)
  {
    gain    = other.gain;
    feature = other.feature;
  }

  __device__ bool operator==(const GainFeaturePair& other) const
  {
    return gain == other.gain && feature == other.feature;
  }

  __device__ bool operator>(const GainFeaturePair& other) const { return gain > other.gain; }

  __device__ bool operator<(const GainFeaturePair& other) const { return gain < other.gain; }
};

template <typename TYPE>
__global__ static void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  perform_best_split(legate::Buffer<GPair, 3> histogram,
                     size_t n_features,
                     size_t n_outputs,
                     legate::AccessorRO<TYPE, 2> split_proposal,
                     double eps,
                     legate::Buffer<double, 2> tree_leaf_value,
                     legate::Buffer<double, 2> tree_gradient,
                     legate::Buffer<double, 2> tree_hessian,
                     legate::Buffer<int32_t, 1> tree_feature,
                     legate::Buffer<double, 1> tree_split_value,
                     legate::Buffer<double, 1> tree_gain,
                     int64_t depth)
{
  // using one block per (level) node to have blockwise reductions
  int node_id        = blockIdx.x;
  int global_node_id = node_id + ((1 << depth) - 1);

  typedef cub::BlockReduce<GainFeaturePair, THREADS_PER_BLOCK> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  __shared__ double node_best_gain;
  __shared__ int node_best_feature;

  double thread_best_gain = 0;
  int thread_best_feature = -1;

  for (int feature_id = threadIdx.x; feature_id < n_features; feature_id += blockDim.x) {
    double gain = 0;
    for (int output = 0; output < n_outputs; ++output) {
      auto G          = tree_gradient[{global_node_id, output}];
      auto H          = tree_hessian[{global_node_id, output}];
      auto [G_L, H_L] = histogram[{node_id, feature_id, output}];
      auto G_R        = G - G_L;
      auto H_R        = H - H_L;

      if (H_L <= 0.0 || H_R <= 0.0) {
        gain = 0;
        break;
      }
      gain += 0.5 * ((G_L * G_L) / (H_L + eps) + (G_R * G_R) / (H_R + eps) - (G * G) / (H + eps));
    }
    if (gain > thread_best_gain) {
      thread_best_gain    = gain;
      thread_best_feature = feature_id;
    }
  }

  // SYNC BEST GAIN TO FULL BLOCK/NODE
  GainFeaturePair thread_best_pair{thread_best_gain, thread_best_feature};
  GainFeaturePair node_best_pair =
    BlockReduce(temp_storage).Reduce(thread_best_pair, cub::Max(), THREADS_PER_BLOCK);
  if (threadIdx.x == 0) {
    node_best_gain    = node_best_pair.gain;
    node_best_feature = node_best_pair.feature;
  }
  __syncthreads();

  // from here on we need the global node id
  if (node_best_gain > eps) {
    for (int output = threadIdx.x; output < n_outputs; output += blockDim.x) {
      auto [G_L, H_L] = histogram[{node_id, node_best_feature, output}];
      auto G_R        = tree_gradient[{global_node_id, output}] - G_L;
      auto H_R        = tree_hessian[{global_node_id, output}] - H_L;

      int left_child                         = global_node_id * 2 + 1;
      int right_child                        = left_child + 1;
      tree_leaf_value[{left_child, output}]  = -G_L / H_L;
      tree_leaf_value[{right_child, output}] = -G_R / H_R;
      tree_hessian[{left_child, output}]     = H_L;
      tree_hessian[{right_child, output}]    = H_R;
      tree_gradient[{left_child, output}]    = G_L;
      tree_gradient[{right_child, output}]   = G_R;

      if (output == 0) {
        tree_feature[global_node_id]     = node_best_feature;
        tree_split_value[global_node_id] = split_proposal[{depth, node_best_feature}];
        tree_gain[global_node_id]        = node_best_gain;
      }
    }
  }
}

namespace {

struct Tree {
  Tree(int max_depth, int num_outputs, cudaStream_t stream)
    : num_outputs(num_outputs), max_nodes(1 << (max_depth + 1)), stream(stream)
  {
    leaf_value  = legate::create_buffer<double, 2>({max_nodes, num_outputs});
    feature     = legate::create_buffer<int32_t, 1>({max_nodes});
    split_value = legate::create_buffer<double, 1>({max_nodes});
    gain        = legate::create_buffer<double, 1>({max_nodes});
    hessian     = legate::create_buffer<double, 2>({max_nodes, num_outputs});
    gradient    = legate::create_buffer<double, 2>({max_nodes, num_outputs});
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

  template <typename THRUST_POLICY>
  void InitializeBase(double* base_sums, const THRUST_POLICY& thrust_exec_policy)
  {
    std::vector<double> base_sums_host(2 * num_outputs);
    CHECK_CUDA(cudaMemcpyAsync(base_sums_host.data(),
                               base_sums,
                               sizeof(double) * num_outputs * 2,
                               cudaMemcpyDeviceToHost,
                               stream));

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

    CHECK_CUDA(cudaStreamSynchronize(stream));

    std::vector<double> leaf_value_init(num_outputs);
    for (auto i = 0; i < num_outputs; ++i) {
      leaf_value_init[i] = (-base_sums_host[i] / base_sums_host[i + num_outputs]);
    }
    CHECK_CUDA(cudaMemcpyAsync(leaf_value.ptr({0, 0}),
                               leaf_value_init.data(),
                               sizeof(double) * num_outputs,
                               cudaMemcpyHostToDevice,
                               stream));
    CHECK_CUDA(cudaMemcpyAsync(gradient.ptr({0, 0}),
                               base_sums,
                               sizeof(double) * num_outputs,
                               cudaMemcpyDeviceToDevice,
                               stream));
    CHECK_CUDA(cudaMemcpyAsync(hessian.ptr({0, 0}),
                               base_sums + num_outputs,
                               sizeof(double) * num_outputs,
                               cudaMemcpyDeviceToDevice,
                               stream));
  }

  template <typename T, int DIM>
  void WriteOutput(legate::PhysicalStore out, const legate::Buffer<T, DIM>& x)
  {
    // all outputs are 2D
    // for those where the internal buffer is 1D we expect the 2nd extent to be 1
    const legate::Point<DIM> zero   = legate::Point<DIM>::ZEROES();
    const legate::Point<2> zero2    = legate::Point<2>::ZEROES();
    const legate::Rect<2> out_shape = out.shape<2>();
    auto out_acc                    = out.write_accessor<T, 2>();
    EXPECT(DIM == 2 || out_shape.hi[1] == out_shape.lo[1], "Buffer is 1D but store has 2D.");
    EXPECT(out_shape.lo == zero2, "Output store shape should start at zero.");
    EXPECT(out_acc.accessor.is_dense_row_major(out_shape), "Output store is not dense row major.");
    CHECK_CUDA(cudaMemcpyAsync(out_acc.ptr(zero2),
                               x.ptr(zero),
                               out_shape.volume() * sizeof(T),
                               cudaMemcpyDeviceToDevice,
                               stream));
  }

  void WriteTreeOutput(legate::TaskContext context)
  {
    WriteOutput(context.output(0).data(), leaf_value);
    WriteOutput(context.output(1).data(), feature);
    WriteOutput(context.output(2).data(), split_value);
    WriteOutput(context.output(3).data(), gain);
    WriteOutput(context.output(4).data(), hessian);
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

struct TreeLevelInfo {
  TreeLevelInfo(int32_t num_rows,
                int32_t num_features,
                int32_t num_outputs,
                cudaStream_t stream,
                int32_t max_nodes)
    : num_rows(num_rows),
      num_features(num_features),
      num_outputs(num_outputs),
      stream(stream),
      max_nodes(max_nodes)
  {
    positions           = legate::create_buffer<int32_t>(num_rows);
    sequence            = legate::create_buffer<int32_t>(num_rows);
    indices_reordered   = legate::create_buffer<int32_t>(num_rows);
    positions_reordered = legate::create_buffer<int32_t>(num_rows);
  }

  ~TreeLevelInfo()
  {
    positions.destroy();
    sequence.destroy();
    indices_reordered.destroy();
    positions_reordered.destroy();
    if (cub_buffer_size > 0) cub_buffer.destroy();
    if (current_depth >= 0) histogram_buffer.destroy();
  }

  template <typename THRUST_POLICY>
  void InitializeHistogramForDepth(int32_t depth, const THRUST_POLICY& thrust_exec_policy)
  {
    if (current_depth < 0) {
      // some initialization on first pass
      CHECK_CUDA(cudaMemsetAsync(positions.ptr(0), 0, (size_t)num_rows * sizeof(int32_t), stream));
      thrust::sequence(thrust_exec_policy, sequence.ptr(0), sequence.ptr(0) + num_rows);
    } else {
      histogram_buffer.destroy();
    }
    current_depth    = depth;
    int max_nodes    = 1 << depth;
    histogram_buffer = legate::create_buffer<GPair, 3>({max_nodes, num_features, num_outputs});
    CHECK_CUDA(cudaMemsetAsync(histogram_buffer.ptr(legate::Point<3>::ZEROES()),
                               0,
                               max_nodes * num_features * num_outputs * sizeof(GPair),
                               stream));
  }

  template <typename TYPE>
  void UpdatePositions(Tree& tree, legate::AccessorRO<TYPE, 2> X, legate::Rect<2> X_shape)
  {
    if (current_depth > 0 && skip_rows < num_rows) {
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
        double x_value = X[{X_shape.lo[0] + (int64_t)idx, tree_feature_ptr[pos]}];
        bool left      = x_value <= tree_split_value_ptr[pos];
        pos            = left ? 2 * pos + 1 : 2 * pos + 2;
      };
      LaunchN(num_rows, stream, update_positions_lambda);
      CHECK_CUDA_STREAM(stream);
    }
  }

  template <typename THRUST_POLICY>
  void ReorderPositions(const THRUST_POLICY& thrust_exec_policy)
  {
    if (current_depth > 0 && skip_rows < num_rows) {
      size_t temp_storage_bytes = 0;
      cub::DeviceRadixSort::SortPairs(nullptr,
                                      temp_storage_bytes,
                                      positions.ptr(0),
                                      positions_reordered.ptr(0),
                                      sequence.ptr(0),
                                      indices_reordered.ptr(0),
                                      num_rows,
                                      0,
                                      sizeof(int32_t) * 8,
                                      stream);

      // use cached cub buffer for sorting -- should remain constant
      if (temp_storage_bytes > cub_buffer_size) {
        if (cub_buffer_size > 0) cub_buffer.destroy();
        cub_buffer      = legate::create_buffer<unsigned char>(temp_storage_bytes);
        cub_buffer_size = temp_storage_bytes;
      }

      cub::DeviceRadixSort::SortPairs(cub_buffer.ptr(0),
                                      temp_storage_bytes,
                                      positions.ptr(0),
                                      positions_reordered.ptr(0),
                                      sequence.ptr(0),
                                      indices_reordered.ptr(0),
                                      num_rows,
                                      0,
                                      sizeof(int32_t) * 8,
                                      stream);

      auto res = thrust::find_if(thrust_exec_policy,
                                 positions_reordered.ptr(0),
                                 positions_reordered.ptr(0) + num_rows,
                                 [=] __device__(int32_t & x) { return x >= 0; });

      skip_rows = res - positions_reordered.ptr(0);

      CHECK_CUDA_STREAM(stream);
    }
  }

  int32_t* PositionsPtr()
  {
    if (current_depth > 0)
      return positions_reordered.ptr(skip_rows);
    else
      return positions.ptr(0);
  }

  int32_t* IndicesPtr()
  {
    if (current_depth > 0)
      return indices_reordered.ptr(skip_rows);
    else
      return sequence.ptr(0);
  }

  template <typename TYPE>
  void FillHistogram(Tree& tree,
                     legate::AccessorRO<TYPE, 2> X,
                     legate::Rect<2> X_shape,
                     legate::AccessorRO<TYPE, 2> split_proposal,
                     legate::AccessorRO<double, 2> g,
                     legate::AccessorRO<double, 2> h)
  {
    if (skip_rows < num_rows) {
      // TODO adjust kernel parameters dynamically
      constexpr size_t elements_per_thread = 1;
      constexpr size_t features_per_block  = 8;
      const size_t blocks_x = (num_rows - skip_rows + THREADS_PER_BLOCK * elements_per_thread - 1) /
                              (THREADS_PER_BLOCK * elements_per_thread);
      const size_t blocks_y = (num_features + features_per_block - 1) / features_per_block;
      dim3 grid_shape       = dim3(blocks_x, blocks_y, 1);
      fill_histogram_blockreduce<TYPE, elements_per_thread, features_per_block>
        <<<grid_shape, THREADS_PER_BLOCK, 0, stream>>>(X,
                                                       num_rows - skip_rows,
                                                       num_features,
                                                       X_shape.lo[0],
                                                       g,
                                                       h,
                                                       num_outputs,
                                                       split_proposal,
                                                       PositionsPtr(),
                                                       IndicesPtr(),
                                                       histogram_buffer,
                                                       1 << current_depth,
                                                       current_depth);
      CHECK_CUDA_STREAM(stream);
    }
  }

  template <typename TYPE>
  void PerformBestSplit(Tree& tree, legate::AccessorRO<TYPE, 2> split_proposal, double eps)
  {
    perform_best_split<<<(1 << current_depth), THREADS_PER_BLOCK, 0, stream>>>(histogram_buffer,
                                                                               num_features,
                                                                               num_outputs,
                                                                               split_proposal,
                                                                               eps,
                                                                               tree.leaf_value,
                                                                               tree.gradient,
                                                                               tree.hessian,
                                                                               tree.feature,
                                                                               tree.split_value,
                                                                               tree.gain,
                                                                               current_depth);
    CHECK_CUDA_STREAM(stream);
  }

  legate::Buffer<int32_t> positions;
  legate::Buffer<int32_t> positions_reordered;
  legate::Buffer<int32_t> sequence;
  legate::Buffer<int32_t> indices_reordered;
  const int32_t num_rows;
  const int32_t num_features;
  const int32_t num_outputs;
  const int32_t max_nodes;

  legate::Buffer<unsigned char> cub_buffer;
  size_t cub_buffer_size = 0;

  int32_t skip_rows = 0;
  legate::Buffer<GPair, 3> histogram_buffer;
  int32_t current_depth = -1;

  cudaStream_t stream;
};

void ReduceBaseSums(legate::Buffer<double> base_sums,
                    int32_t num_rows,
                    int32_t num_outputs,
                    legate::AccessorRO<double, 2> g,
                    legate::AccessorRO<double, 2> h,
                    legate::Rect<2> shape,
                    cudaStream_t stream)
{
  CHECK_CUDA(cudaMemsetAsync(base_sums.ptr(0), 0, num_outputs * 2 * sizeof(double), stream));
  const size_t blocks = (num_rows + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  dim3 grid_shape     = dim3(blocks, num_outputs);
  reduce_base_sums<<<grid_shape, THREADS_PER_BLOCK, 0, stream>>>(
    g, h, num_rows, shape.lo[0], base_sums, num_outputs);
  CHECK_CUDA_STREAM(stream);
}

struct build_tree_fn {
  template <legate::Type::Code CODE>
  void operator()(legate::TaskContext context)
  {
    using T           = legate::type_of<CODE>;
    const auto& X     = context.input(0).data();
    auto X_shape      = X.shape<2>();
    auto X_accessor   = X.read_accessor<T, 2>();
    auto num_features = X_shape.hi[1] - X_shape.lo[1] + 1;
    auto num_rows     = X_shape.hi[0] - X_shape.lo[0] + 1;
    const auto& g     = context.input(1).data();
    const auto& h     = context.input(2).data();
    auto g_shape      = g.shape<2>();
    auto h_shape      = h.shape<2>();
    EXPECT_AXIS_ALIGNED(0, X_shape, g_shape);
    EXPECT_AXIS_ALIGNED(0, g_shape, h_shape);
    EXPECT_AXIS_ALIGNED(1, g_shape, h_shape);
    auto num_outputs            = g_shape.hi[1] - g_shape.lo[1] + 1;
    auto g_accessor             = g.read_accessor<double, 2>();
    auto h_accessor             = h.read_accessor<double, 2>();
    const auto& split_proposals = context.input(3).data();
    EXPECT_AXIS_ALIGNED(1, split_proposals.shape<2>(), X_shape);
    auto split_proposal_accessor = split_proposals.read_accessor<T, 2>();

    // Scalars
    auto max_depth = context.scalars().at(0).value<int>();

    auto stream             = legate::cuda::StreamPool::get_stream_pool().get_stream();
    auto thrust_alloc       = ThrustAllocator(legate::Memory::GPU_FB_MEM);
    auto thrust_exec_policy = DEFAULT_POLICY(thrust_alloc).on(stream);

    Tree tree(max_depth, num_outputs, stream);

    // Initialize the root node
    {
      auto base_sums = legate::create_buffer<double, 1>(num_outputs * 2);

      ReduceBaseSums(base_sums, num_rows, num_outputs, g_accessor, h_accessor, g_shape, stream);

      SumAllReduce(context, reinterpret_cast<double*>(base_sums.ptr(0)), num_outputs * 2, stream);

      // base sums contain g-sums first, h sums second
      tree.InitializeBase(base_sums.ptr(0), thrust_exec_policy);

      base_sums.destroy();
      CHECK_CUDA_STREAM(stream);
    }

    // Begin building the tree
    TreeLevelInfo tree_state(num_rows, num_features, num_outputs, stream, tree.max_nodes);

    for (int depth = 0; depth < max_depth; ++depth) {
      int max_nodes = 1 << depth;

      tree_state.InitializeHistogramForDepth(depth, thrust_exec_policy);

      // update positions from previous step
      tree_state.UpdatePositions(tree, X_accessor, X_shape);

      // reorder indices to sort by node id
      tree_state.ReorderPositions(thrust_exec_policy);

      // actual histogram creation
      tree_state.FillHistogram(
        tree, X_accessor, X_shape, split_proposal_accessor, g_accessor, h_accessor);

      SumAllReduce(
        context,
        reinterpret_cast<double*>(tree_state.histogram_buffer.ptr(legate::Point<3>::ZEROES())),
        max_nodes * num_features * num_outputs * sizeof(GPair),
        stream);

      // Select the best split
      double eps = 1e-5;
      tree_state.PerformBestSplit(tree, split_proposal_accessor, eps);
    }

    if (context.get_task_index()[0] == 0) { tree.WriteTreeOutput(context); }

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
