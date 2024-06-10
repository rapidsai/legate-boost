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

#include <cub/device/device_radix_sort.cuh>
#include <thrust/execution_policy.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/count.h>
#include <thrust/version.h>
#include <thrust/binary_search.h>

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
  fill_histogram_blockreduce(legate::AccessorRO<TYPE, 3> X,
                             size_t n_local_samples,
                             size_t n_features,
                             int64_t sample_offset,
                             legate::AccessorRO<double, 3> g,
                             legate::AccessorRO<double, 3> h,
                             size_t n_outputs,
                             legate::AccessorRO<TYPE, 2> split_proposal,
                             int32_t samples_per_feature,
                             int32_t* positions_local,
                             legate::Buffer<GPair, 4> histogram)
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

    // check if thread has actual work todo (besides taking part in reductions)
    int32_t localSampleId = (blockIdx.x + elementIdx * gridDim.x) * THREADS_PER_BLOCK + threadIdx.x;
    int64_t globalSampleId = localSampleId + sample_offset;
    bool validThread       = localSampleId < n_local_samples;

    int32_t sampleNode = validThread ? positions_local[localSampleId] : -1;

    for (int32_t output = 0; output < n_outputs; output++) {
      double G = validThread ? g[{globalSampleId, 0, output}] : 0.0;
      double H = validThread ? h[{globalSampleId, 0, output}] : 0.0;
      for (int32_t featureIdx = 0; featureIdx < FEATURES_PER_BLOCK; featureIdx++) {
        int32_t feature = featureIdx + blockIdx.y * FEATURES_PER_BLOCK;
        if (feature < n_features && validThread && sampleNode >= 0) {
          auto x_value = X[{globalSampleId, feature, 0}];
          int bin_idx  = thrust::lower_bound(thrust::seq,
                                            split_proposal.ptr({feature, 0}),
                                            split_proposal.ptr({feature, samples_per_feature}),
                                            x_value) -
                        split_proposal.ptr({feature, 0});

          // bin_idx is the first sample that is larger than x_value
          if (bin_idx < samples_per_feature) {
            double* addPosition =
              reinterpret_cast<double*>(&histogram[{sampleNode, feature, bin_idx, output}]);
            atomicAdd(addPosition, G);
            atomicAdd(addPosition + 1, H);
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
  int node_id = blockIdx.x + ((1 << depth) - 1);

  typedef cub::BlockReduce<GainFeaturePair, THREADS_PER_BLOCK> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  __shared__ double node_best_gain;
  __shared__ int node_best_feature;
  __shared__ int node_best_feature_sample;

  double thread_best_gain        = 0;
  int thread_best_feature        = -1;
  int thread_best_feature_sample = -1;

  for (int feature_id = threadIdx.x; feature_id < n_features; feature_id += blockDim.x) {
    for (int feature_sample_idx = 0; feature_sample_idx < samples_per_feature;
         feature_sample_idx++) {
      double gain = 0;
      for (int output = 0; output < n_outputs; ++output) {
        auto G          = tree_gradient[{node_id, output}];
        auto H          = tree_hessian[{node_id, output}];
        auto [G_L, H_L] = histogram[{node_id, feature_id, feature_sample_idx, output}];
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

  // from here on we need the global node id
  if (node_best_gain > eps) {
    for (int output = threadIdx.x; output < n_outputs; output += blockDim.x) {
      auto [G_L, H_L] = histogram[{node_id, node_best_feature, node_best_feature_sample, output}];
      auto G_R        = tree_gradient[{node_id, output}] - G_L;
      auto H_R        = tree_hessian[{node_id, output}] - H_L;

      int left_child                         = node_id * 2 + 1;
      int right_child                        = left_child + 1;
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
  Tree(int max_nodes, int num_outputs, cudaStream_t stream)
    : num_outputs(num_outputs), max_nodes(max_nodes), stream(stream)
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
  void InitializeBase(double* base_sums, const THRUST_POLICY& thrust_exec_policy, double alpha)
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
      leaf_value_init[i] =
        CalculateLeafValue(base_sums_host[i], base_sums_host[i + num_outputs], alpha);
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

struct TreeLevelInfo {
  TreeLevelInfo(int32_t num_rows,
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
      legate::create_buffer<GPair, 4>({max_nodes, num_features, samples_per_feature, num_outputs});
    CHECK_CUDA(
      cudaMemsetAsync(histogram_buffer.ptr(legate::Point<4>::ZEROES()),
                      0,
                      max_nodes * num_features * samples_per_feature * num_outputs * sizeof(GPair),
                      stream));
    // some initialization on first pass
    CHECK_CUDA(cudaMemsetAsync(positions.ptr(0), 0, (size_t)num_rows * sizeof(int32_t), stream));
  }

  ~TreeLevelInfo()
  {
    positions.destroy();
    histogram_buffer.destroy();
    if (cub_buffer_size > 0) cub_buffer.destroy();
  }

  template <typename TYPE>
  void UpdatePositions(Tree& tree, legate::AccessorRO<TYPE, 3> X, legate::Rect<3> X_shape)
  {
    if (current_depth > 0) {
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
        pos            = left ? 2 * pos + 1 : 2 * pos + 2;
      };
      LaunchN(num_rows, stream, update_positions_lambda);
      CHECK_CUDA_STREAM(stream);
    }
  }

  template <typename TYPE>
  void FillHistogram(Tree& tree,
                     legate::AccessorRO<TYPE, 3> X,
                     legate::Rect<3> X_shape,
                     legate::AccessorRO<TYPE, 2> split_proposal,
                     legate::AccessorRO<double, 3> g,
                     legate::AccessorRO<double, 3> h)
  {
    current_depth++;
    // TODO adjust kernel parameters dynamically
    constexpr size_t elements_per_thread = 1;
    constexpr size_t features_per_block  = 8;
    const size_t blocks_x = (num_rows + THREADS_PER_BLOCK * elements_per_thread - 1) /
                            (THREADS_PER_BLOCK * elements_per_thread);
    const size_t blocks_y = (num_features + features_per_block - 1) / features_per_block;
    dim3 grid_shape       = dim3(blocks_x, blocks_y, 1);
    fill_histogram_blockreduce<TYPE, elements_per_thread, features_per_block>
      <<<grid_shape, THREADS_PER_BLOCK, 0, stream>>>(X,
                                                     num_rows,
                                                     num_features,
                                                     X_shape.lo[0],
                                                     g,
                                                     h,
                                                     num_outputs,
                                                     split_proposal,
                                                     samples_per_feature,
                                                     positions.ptr(0),
                                                     histogram_buffer);
    CHECK_CUDA_STREAM(stream);

    // Scan the histogram
    // Member variables must be explicity copy captured otherwise the lambda calls the this pointer
    // (illegal access)
    int num_nodes_level = 1 << current_depth;
    LaunchN(num_features * num_nodes_level,
            stream,
            [                    =,
             num_features        = this->num_features,
             num_outputs         = this->num_outputs,
             depth               = this->current_depth,
             samples_per_feature = this->samples_per_feature,
             histogram_buffer    = this->histogram_buffer] __device__(int idx) {
              int node_idx    = idx / num_features + ((1 << depth) - 1);
              int feature_idx = idx % num_features;
              for (int output = 0; output < num_outputs; output++) {
                GPair sum = {0, 0};
                for (int sample_idx = 0; sample_idx < samples_per_feature; sample_idx++) {
                  sum += histogram_buffer[{node_idx, feature_idx, sample_idx, output}];
                  histogram_buffer[{node_idx, feature_idx, sample_idx, output}] = sum;
                }
              }
            });
  }

  template <typename TYPE>
  void PerformBestSplit(Tree& tree,
                        legate::AccessorRO<TYPE, 2> split_proposal,
                        double eps,
                        double alpha)
  {
    perform_best_split<<<(1 << current_depth), THREADS_PER_BLOCK, 0, stream>>>(histogram_buffer,
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
                                                                               current_depth);
    CHECK_CUDA_STREAM(stream);
  }

  legate::Buffer<int32_t> positions;
  legate::Buffer<int32_t> positions_reordered;
  legate::Buffer<int32_t> indices_reordered;
  const int32_t num_rows;
  const int32_t num_features;
  const int32_t num_outputs;
  const int32_t max_nodes;
  const int32_t samples_per_feature;

  legate::Buffer<unsigned char> cub_buffer;
  size_t cub_buffer_size = 0;

  legate::Buffer<GPair, 4> histogram_buffer;
  int32_t current_depth = -1;

  cudaStream_t stream;
};

void ReduceBaseSums(legate::Buffer<double> base_sums,
                    int32_t num_rows,
                    int32_t num_outputs,
                    legate::AccessorRO<double, 3> g,
                    legate::AccessorRO<double, 3> h,
                    legate::Rect<3> shape,
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
  template <typename T>
  void operator()(legate::TaskContext context)
  {
    auto [X, X_shape, X_accessor] = GetInputStore<T, 3>(context.input(0).data());
    auto [g, g_shape, g_accessor] = GetInputStore<double, 3>(context.input(1).data());
    auto [h, h_shape, h_accessor] = GetInputStore<double, 3>(context.input(2).data());
    auto [split_proposals, split_proposals_shape, split_proposals_accessor] =
      GetInputStore<T, 2>(context.input(3).data());
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

    Tree tree(max_nodes, num_outputs, stream);

    // Initialize the root node
    {
      auto base_sums = legate::create_buffer<double, 1>(num_outputs * 2);

      ReduceBaseSums(base_sums, num_rows, num_outputs, g_accessor, h_accessor, g_shape, stream);

      SumAllReduce(context, reinterpret_cast<double*>(base_sums.ptr(0)), num_outputs * 2, stream);

      // base sums contain g-sums first, h sums second
      tree.InitializeBase(base_sums.ptr(0), thrust_exec_policy, alpha);

      base_sums.destroy();
      CHECK_CUDA_STREAM(stream);
    }

    // Begin building the tree
    TreeLevelInfo tree_state(
      num_rows, num_features, num_outputs, stream, tree.max_nodes, samples_per_feature);

    for (int depth = 0; depth < max_depth; ++depth) {
      int max_nodes = 1 << depth;

      // update positions from previous step
      tree_state.UpdatePositions(tree, X_accessor, X_shape);

      // actual histogram creation
      tree_state.FillHistogram(
        tree, X_accessor, X_shape, split_proposals_accessor, g_accessor, h_accessor);

      SumAllReduce(
        context,
        reinterpret_cast<double*>(tree_state.histogram_buffer.ptr(legate::Point<4>::ZEROES())),
        max_nodes * num_features * num_outputs * sizeof(GPair),
        stream);

      // Select the best split
      double eps = 1e-5;
      tree_state.PerformBestSplit(tree, split_proposals_accessor, eps, alpha);
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
