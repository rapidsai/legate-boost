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
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/count.h>
#include <thrust/version.h>

namespace legateboost {

__global__ static void reduce_base_sums(legate::AccessorRO<double, 2> g,
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

// histogram_buffer(pairs of) [max_nodes][num_features][num_outputs][left/right]
template <typename TYPE>
__global__ static void fill_histogram1(legate::AccessorRO<TYPE, 2> X,
                                       size_t n_local_samples,
                                       size_t n_features,
                                       int64_t sample_offset,
                                       legate::AccessorRO<double, 2> g,
                                       legate::AccessorRO<double, 2> h,
                                       size_t n_outputs,
                                       legate::AccessorRO<TYPE, 2> split_proposal,
                                       legate::Buffer<int32_t, 1> positions,
                                       legate::Buffer<GPair, 4> histogram,
                                       int32_t max_nodes_in_level,
                                       int64_t depth)
{
  // we assume one block per feature*output selection
  // with each block being 1-dimensional
  int64_t feature = blockIdx.x;
  int64_t output  = blockIdx.y;

  for (int64_t sample_id = threadIdx.x; sample_id < n_local_samples; sample_id += blockDim.x) {
    int32_t sample_pos = positions[sample_id] - (max_nodes_in_level - 1);
    if (sample_pos < 0) continue;
    auto x_value = X[{sample_offset + sample_id, feature}];
    bool left    = x_value <= split_proposal[{depth, feature}];

    // this is probably very slow... we should do this in shared memory per block first maybe
    double* addPosition =
      reinterpret_cast<double*>(&histogram[{sample_pos, feature, output, left}]);
    atomicAdd(addPosition, g[{sample_offset + sample_id, output}]);
    atomicAdd(addPosition + 1, h[{sample_offset + sample_id, output}]);
  }
}

// histogram_buffer(pairs of) [max_nodes][num_features][num_outputs][left/right]
template <typename TYPE, int THREADS, int FEATURES_PER_BLOCK>
__global__ static void fill_histogram2(legate::AccessorRO<TYPE, 2> X,
                                       size_t n_local_samples,
                                       size_t n_features,
                                       int64_t sample_offset,
                                       legate::AccessorRO<double, 2> g,
                                       legate::AccessorRO<double, 2> h,
                                       size_t n_outputs,
                                       legate::AccessorRO<TYPE, 2> split_proposal,
                                       int32_t* positions_local,
                                       int32_t* sample_index_local,
                                       legate::Buffer<GPair, 4> histogram,
                                       int32_t max_nodes_in_level,
                                       int32_t depth)
{
  // block dimensions are (THREADS, 1, 1)
  // each block processes THREADS samples and FEATURES_PER_BLOCK features
  // the features to process are defined via blockIdx.y

  // further improvements:
  // * restructure surrounding histogram storage to keep parent and left child only
  // * optimize read alignment to shared memory
  // * quantize values to work with int instead of double

  typedef cub::BlockReduce<double, THREADS> BlockReduce;

  // alternate storages to spare syncthreads
  __shared__ typename BlockReduce::TempStorage temp_storage1;
  __shared__ typename BlockReduce::TempStorage temp_storage2;

  __shared__ bool left_shared[FEATURES_PER_BLOCK][THREADS + 1];

  // check if thread has actual work todo (besides taking part in reductions)
  int64_t sampleId         = blockIdx.x * THREADS + threadIdx.x;
  int64_t lastActiveThread = (blockIdx.x + 1) * THREADS - 1;
  if (lastActiveThread >= n_local_samples) lastActiveThread = n_local_samples - 1;
  bool validThread        = sampleId < n_local_samples;
  int64_t sampleId_global = validThread ? sample_index_local[sampleId] + sample_offset : -1;

  // read shared memory X -- maybe we can do this better transposed
  // TODO improve loading pattern!
  for (int32_t featureIdx = 0; featureIdx < FEATURES_PER_BLOCK; featureIdx++) {
    int feature = featureIdx + blockIdx.y * FEATURES_PER_BLOCK;
    left_shared[featureIdx][threadIdx.x] =
      (validThread && feature < n_features)
        ? X[{sampleId_global, feature}] <= split_proposal[{depth, feature}]
        : false;
  }

  // in case loading left_shared was done in different order we need syncthreads
  //__syncthreads();

  int32_t firstNode  = positions_local[blockIdx.x * THREADS] - max_nodes_in_level + 1;
  int32_t lastNode   = positions_local[lastActiveThread] - max_nodes_in_level + 1;
  int32_t sampleNode = validThread ? positions_local[sampleId] - max_nodes_in_level + 1 : -1;

  for (int32_t output = 0; output < n_outputs; output++) {
    double G = validThread ? g[{sampleId_global, output}] : 0.0;
    double H = validThread ? h[{sampleId_global, output}] : 0.0;
    for (int32_t featureIdx = 0; featureIdx < FEATURES_PER_BLOCK; featureIdx++) {
      int32_t feature = featureIdx + blockIdx.y * FEATURES_PER_BLOCK;
      if (feature < n_features) {
        for (int32_t nodeId = firstNode; nodeId <= lastNode; ++nodeId) {
          bool myNode = sampleNode == nodeId;
          bool left   = left_shared[featureIdx][threadIdx.x];
          // we assume c-order positions
          double* addPositionR =
            reinterpret_cast<double*>(&histogram[{nodeId, feature, output, false}]);
          // reduce right g
          double input = myNode && !left ? G : 0.0;
          double sum   = BlockReduce(temp_storage1).Sum(input);
          if (threadIdx.x == 0 && sum != 0) {
            if (nodeId < lastNode && nodeId > firstNode) {
              addPositionR[0] = sum;
            } else {
              atomicAdd(addPositionR, sum);
            }
          }
          // reduce right h
          input = myNode && !left ? H : 0.0;
          sum   = BlockReduce(temp_storage2).Sum(input);
          if (threadIdx.x == 0 && sum != 0) {
            if (nodeId < lastNode && nodeId > firstNode) {
              addPositionR[1] = sum;
            } else {
              atomicAdd(addPositionR + 1, sum);
            }
          }
          // reduce left g
          input = myNode && left ? G : 0.0;
          sum   = BlockReduce(temp_storage1).Sum(input);
          if (threadIdx.x == 0 && sum != 0) {
            if (nodeId < lastNode && nodeId > firstNode) {
              addPositionR[2] = sum;
            } else {
              atomicAdd(addPositionR + 2, sum);
            }
          }
          // reduce left h
          input = myNode && left ? H : 0.0;
          sum   = BlockReduce(temp_storage2).Sum(input);
          if (threadIdx.x == 0 && sum != 0) {
            if (nodeId < lastNode && nodeId > firstNode) {
              addPositionR[3] = sum;
            } else {
              atomicAdd(addPositionR + 3, sum);
            }
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
__global__ static void perform_best_split(legate::Buffer<GPair, 4> histogram,
                                          size_t n_features,
                                          size_t n_outputs,
                                          legate::AccessorRO<TYPE, 2> split_proposal,
                                          double eps,
                                          double learning_rate,
                                          legate::Buffer<double, 2> tree_leaf_value,
                                          legate::Buffer<double, 2> tree_hessian,
                                          legate::Buffer<int32_t, 1> tree_feature,
                                          legate::Buffer<double, 1> tree_split_value,
                                          legate::Buffer<double, 1> tree_gain,
                                          int64_t depth)
{
  // using one block per (level) node to have blockwise reductions
  int node_id = blockIdx.x;

  typedef cub::BlockReduce<GainFeaturePair, THREADS_PER_BLOCK> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  __shared__ double node_best_gain;
  __shared__ int node_best_feature;

  double thread_best_gain = 0;
  int thread_best_feature = -1;

  for (int feature_id = threadIdx.x; feature_id < n_features; feature_id += blockDim.x) {
    double gain = 0;
    for (int output = 0; output < n_outputs; ++output) {
      auto [G_L, H_L] = histogram[{node_id, feature_id, output, true}];
      auto [G_R, H_R] = histogram[{node_id, feature_id, output, false}];
      auto G          = G_L + G_R;
      auto H          = H_L + H_R;
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
    int global_node_id = node_id + ((1 << depth) - 1);
    for (int output = threadIdx.x; output < n_outputs; output += blockDim.x) {
      auto [G_L, H_L] = histogram[{node_id, node_best_feature, output, true}];
      auto [G_R, H_R] = histogram[{node_id, node_best_feature, output, false}];

      int left_child                         = global_node_id * 2 + 1;
      int right_child                        = left_child + 1;
      tree_leaf_value[{left_child, output}]  = -(G_L / (H_L + eps)) * learning_rate;
      tree_leaf_value[{right_child, output}] = -(G_R / (H_R + eps)) * learning_rate;
      tree_hessian[{left_child, output}]     = H_L;
      tree_hessian[{right_child, output}]    = H_R;

      if (output == 0) {
        tree_feature[global_node_id]     = node_best_feature;
        tree_split_value[global_node_id] = split_proposal[{depth, node_best_feature}];
        tree_gain[global_node_id]        = node_best_gain;
      }
    }
  }
}

namespace {

void SumAllReduce(legate::TaskContext& context, double* x, int count, cudaStream_t stream)
{
  if (context.communicators().size() == 0) return;
  auto& comm            = context.communicators().at(0);
  auto domain           = context.get_launch_domain();
  size_t num_ranks      = domain.get_volume();
  ncclComm_t* nccl_comm = comm.get<ncclComm_t*>();

  if (num_ranks > 1) {
    CHECK_NCCL(ncclAllReduce(x, x, count, ncclDouble, ncclSum, *nccl_comm, stream));
    CHECK_CUDA_STREAM(stream);
  }
}

struct Tree {
  Tree(int max_depth, int num_outputs, cudaStream_t stream)
    : num_outputs(num_outputs), max_nodes(1 << (max_depth + 1)), stream(stream)
  {
    leaf_value  = legate::create_buffer<double, 2>({max_nodes, num_outputs});
    feature     = legate::create_buffer<int32_t, 1>({max_nodes});
    split_value = legate::create_buffer<double, 1>({max_nodes});
    gain        = legate::create_buffer<double, 1>({max_nodes});
    hessian     = legate::create_buffer<double, 2>({max_nodes, num_outputs});
  }

  ~Tree()
  {
    leaf_value.destroy();
    feature.destroy();
    split_value.destroy();
    gain.destroy();
    hessian.destroy();
  }

  void InitializeBase(double* base_sums, double learning_rate)
  {
    std::vector<double> base_sums_host(2 * num_outputs);
    CHECK_CUDA(cudaMemcpyAsync(base_sums_host.data(),
                               base_sums,
                               sizeof(double) * num_outputs * 2,
                               cudaMemcpyDeviceToHost,
                               stream));

    auto alloc       = ThrustAllocator(legate::Memory::GPU_FB_MEM);
    auto exec_policy = DEFAULT_POLICY(alloc).on(stream);
    thrust::fill(
      exec_policy, leaf_value.ptr({0, 0}), leaf_value.ptr({0, 0}) + max_nodes * num_outputs, 0.0);
    thrust::fill(exec_policy, feature.ptr({0}), feature.ptr({0}) + max_nodes, -1);
    thrust::fill(
      exec_policy, hessian.ptr({0, 0}), hessian.ptr({0, 0}) + max_nodes * num_outputs, 0.0);

    CHECK_CUDA(cudaStreamSynchronize(stream));

    std::vector<double> leaf_value_init(num_outputs);
    for (auto i = 0; i < num_outputs; ++i) {
      leaf_value_init[i] = (-base_sums_host[i] / base_sums_host[i + num_outputs]) * learning_rate;
    }
    CHECK_CUDA(cudaMemcpyAsync(leaf_value.ptr({0, 0}),
                               leaf_value_init.data(),
                               sizeof(double) * num_outputs,
                               cudaMemcpyHostToDevice,
                               stream));
    CHECK_CUDA(cudaMemcpyAsync(hessian.ptr({0, 0}),
                               base_sums + num_outputs,
                               sizeof(double) * num_outputs,
                               cudaMemcpyDeviceToDevice,
                               stream));

    CHECK_CUDA(cudaStreamSynchronize(stream));
  }

  template <typename T, int DIM>
  void WriteOutput(legate::Store& out, const legate::Buffer<T, DIM>& x)
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

  void WriteTreeOutput(legate::TaskContext& context)
  {
    WriteOutput(context.outputs().at(0), leaf_value);
    WriteOutput(context.outputs().at(1), feature);
    WriteOutput(context.outputs().at(2), split_value);
    WriteOutput(context.outputs().at(3), gain);
    WriteOutput(context.outputs().at(4), hessian);
    CHECK_CUDA_STREAM(stream);
  }

  legate::Buffer<double, 2> leaf_value;
  legate::Buffer<int32_t, 1> feature;
  legate::Buffer<double, 1> split_value;
  legate::Buffer<double, 1> gain;
  legate::Buffer<double, 2> hessian;
  const int num_outputs;
  const int max_nodes;
  cudaStream_t stream;
};

struct build_tree_fn {
  template <legate::Type::Code CODE>
  void operator()(legate::TaskContext& context)
  {
    using T           = legate::legate_type_of<CODE>;
    const auto& X     = context.inputs().at(0);
    auto X_shape      = X.shape<2>();
    auto X_accessor   = X.read_accessor<T, 2>();
    auto num_features = X_shape.hi[1] - X_shape.lo[1] + 1;
    auto num_rows     = X_shape.hi[0] - X_shape.lo[0] + 1;
    const auto& g     = context.inputs().at(1);
    const auto& h     = context.inputs().at(2);
    EXPECT_AXIS_ALIGNED(0, X.shape<2>(), g.shape<2>());
    EXPECT_AXIS_ALIGNED(0, g.shape<2>(), h.shape<2>());
    EXPECT_AXIS_ALIGNED(1, g.shape<2>(), h.shape<2>());
    auto g_shape                = context.inputs().at(1).shape<2>();
    auto num_outputs            = g.shape<2>().hi[1] - g.shape<2>().lo[1] + 1;
    auto g_accessor             = g.read_accessor<double, 2>();
    auto h_accessor             = h.read_accessor<double, 2>();
    const auto& split_proposals = context.inputs().at(3);
    EXPECT_AXIS_ALIGNED(1, split_proposals.shape<2>(), X.shape<2>());
    auto split_proposal_accessor = split_proposals.read_accessor<T, 2>();

    // Scalars
    auto learning_rate = context.scalars().at(0).value<double>();
    auto max_depth     = context.scalars().at(1).value<int>();
    auto random_seed   = context.scalars().at(2).value<uint64_t>();

    auto stream             = legate::cuda::StreamPool::get_stream_pool().get_stream();
    auto thrust_alloc       = ThrustAllocator(legate::Memory::GPU_FB_MEM);
    auto thrust_exec_policy = DEFAULT_POLICY(thrust_alloc).on(stream);

    Tree tree(max_depth, num_outputs, stream);

    // Initialize the root node
    {
      // base sums contain g-sums first, h sums second [0,...,num_outputs-1, num_outputs, ...,
      // num_outputs*2 -1]
      auto base_sums = legate::create_buffer<double, 1>(num_outputs * 2);
      CHECK_CUDA(cudaMemsetAsync(base_sums.ptr(0), 0, num_outputs * 2 * sizeof(double), stream));

      const size_t blocks = (num_rows + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
      dim3 grid_shape     = dim3(blocks, num_outputs);
      reduce_base_sums<<<grid_shape, THREADS_PER_BLOCK, 0, stream>>>(
        g_accessor, h_accessor, num_rows, X_shape.lo[0], base_sums, num_outputs);
      CHECK_CUDA_STREAM(stream);

      SumAllReduce(context, reinterpret_cast<double*>(base_sums.ptr(0)), num_outputs * 2, stream);

      tree.InitializeBase(base_sums.ptr(0), learning_rate);

      base_sums.destroy();
      CHECK_CUDA_STREAM(stream);
    }

    // Begin building the tree
    auto positions = legate::create_buffer<int32_t>(num_rows);
    CHECK_CUDA(cudaMemsetAsync(positions.ptr(0), 0, (size_t)num_rows * sizeof(int32_t), stream));

    auto sequence            = legate::create_buffer<int32_t>(num_rows);
    auto indices_reordered   = legate::create_buffer<int32_t>(num_rows);
    auto positions_reordered = legate::create_buffer<int32_t>(num_rows);
    legate::Buffer<unsigned char, 1> cub_buffer;
    size_t cub_buffer_size = 0;
    thrust::sequence(thrust_exec_policy, sequence.ptr(0), sequence.ptr(0) + num_rows);

    int32_t skip_rows = 0;

    for (int depth = 0; depth < max_depth; ++depth) {
      int max_nodes = 1 << depth;

      // Dimensions[Node, Feature, Output, L/R]
      auto histogram_buffer =
        legate::create_buffer<GPair, 4>({max_nodes, num_features, num_outputs, 2});
      CHECK_CUDA(cudaMemsetAsync(histogram_buffer.ptr(legate::Point<4>::ZEROES()),
                                 0,
                                 max_nodes * num_features * num_outputs * 2 * sizeof(GPair),
                                 stream));

      // reorder indices to sort by nodes
      int32_t* position_ptr = positions.ptr(0);
      int32_t* indices_ptr  = sequence.ptr(0);
      if (depth > 0 && skip_rows < num_rows) {
        // update positions from previous step
        auto tree_split_value_ptr    = tree.split_value.ptr(0);
        auto tree_feature_ptr        = tree.feature.ptr(0);
        auto positions_ptr           = positions.ptr(0);
        auto update_positions_lambda = [=] __device__(size_t idx) {
          int32_t& pos = positions_ptr[idx];
          if (pos < 0 || tree_feature_ptr[pos] == -1) {
            pos = -1;
            return;
          }
          double x_value = X_accessor[{X_shape.lo[0] + (int64_t)idx, tree_feature_ptr[pos]}];
          bool left      = x_value <= tree_split_value_ptr[pos];
          pos            = left ? 2 * pos + 1 : 2 * pos + 2;
        };
        LaunchN(num_rows, stream, update_positions_lambda);
        CHECK_CUDA_STREAM(stream);

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

        // now use reordered indices/positions beginning from skip_rows
        position_ptr = positions_reordered.ptr(skip_rows);
        indices_ptr  = indices_reordered.ptr(skip_rows);

        CHECK_CUDA_STREAM(stream);
      }

      if (skip_rows < num_rows) {
        constexpr size_t threads_histogram  = 256;
        constexpr size_t features_per_block = 8;
        const size_t blocks_x = (num_rows - skip_rows + threads_histogram - 1) / threads_histogram;
        const size_t blocks_y = (num_features + features_per_block - 1) / features_per_block;
        dim3 grid_shape       = dim3(blocks_x, blocks_y, 1);
        fill_histogram2<T, threads_histogram, features_per_block>
          <<<grid_shape, threads_histogram, 0, stream>>>(X_accessor,
                                                         num_rows - skip_rows,
                                                         num_features,
                                                         X_shape.lo[0],
                                                         g_accessor,
                                                         h_accessor,
                                                         num_outputs,
                                                         split_proposal_accessor,
                                                         position_ptr,
                                                         indices_ptr,
                                                         histogram_buffer,
                                                         max_nodes,
                                                         depth);

        CHECK_CUDA_STREAM(stream);
      }

      SumAllReduce(context,
                   reinterpret_cast<double*>(histogram_buffer.ptr({0, 0, 0, 0})),
                   max_nodes * num_features * num_outputs * 4,
                   stream);

      // Find the best split
      double eps = 1e-5;
      perform_best_split<<<max_nodes, THREADS_PER_BLOCK, 0, stream>>>(histogram_buffer,
                                                                      num_features,
                                                                      num_outputs,
                                                                      split_proposal_accessor,
                                                                      eps,
                                                                      learning_rate,
                                                                      tree.leaf_value,
                                                                      tree.hessian,
                                                                      tree.feature,
                                                                      tree.split_value,
                                                                      tree.gain,
                                                                      depth);
      CHECK_CUDA_STREAM(stream);

      // cleanup buffer to prevent memory fragmentation
      histogram_buffer.destroy();
    }

    positions.destroy();
    positions_reordered.destroy();
    sequence.destroy();
    indices_reordered.destroy();

    if (context.get_task_index()[0] == 0) { tree.WriteTreeOutput(context); }
  }
};

}  // namespace

/*static*/ void BuildTreeTask::gpu_variant(legate::TaskContext& context)
{
  const auto& X = context.inputs().at(0);
  type_dispatch_float(X.code(), build_tree_fn(), context);
}

}  // namespace legateboost
