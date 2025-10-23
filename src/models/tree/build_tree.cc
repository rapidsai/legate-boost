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
#include "build_tree.h"
#include <legate.h>
#include <cstddef>
#include <random>
#include <limits>
#include <set>
#include <tuple>
#include <vector>
#include <algorithm>
#include "legate_library.h"
#include "legateboost.h"
#include "cpp_utils/cpp_utils.h"

namespace legateboost {
namespace {

class BinnedX {
  // These are stored as int16_t to save space
  // Indices are relative to the feature, not the entire histogram
  // The maximum number of bins in legate-boost is 2048
  legate::Buffer<int16_t> data;
  legate::Buffer<int32_t> row_pointers;
  int64_t num_features;
  int64_t num_rows;
  legate::Rect<3> shape;

 public:
  template <typename T>
  BinnedX(const legate::AccessorRO<T, 3>& X,
          legate::Rect<3> shape,
          const SparseSplitProposals<T>& split_proposals)
    : shape(shape),
      row_pointers(legate::create_buffer<int32_t, 1>(split_proposals.row_pointers.size())),
      num_features(std::max(shape.hi[1] - shape.lo[1] + 1, 0LL)),
      num_rows(std::max(shape.hi[0] - shape.lo[0] + 1, 0LL))
  {
    data = legate::create_buffer<int16_t>(num_features * num_rows);
    std::copy(split_proposals.row_pointers.begin(),
              split_proposals.row_pointers.end(),
              row_pointers.ptr(0));
    for (int i = 0; i < num_rows; i++) {
      for (int j = 0; j < num_features; j++) {
        auto bin_idx = split_proposals.FindBin(X[{shape.lo[0] + i, j, 0}], j);
        // Store the bin index relative to the feature to save space
        data[(i * num_features) + j] = bin_idx - row_pointers[j];
      }
    }
  }
  [[nodiscard]] auto Shape() const { return shape; }
  // This should use the local row index, not global
  int64_t operator[](const legate::Point<2>& p) const
  {
    return data[(p[0] * num_features) + p[1]] + row_pointers[p[1]];
  }
};

struct NodeBatch {
  int32_t node_idx_begin{};
  int32_t node_idx_end{};
  cuda::std::span<std::tuple<int32_t, int32_t>> instances;
  [[nodiscard]] auto begin() const { return instances.begin(); }
  [[nodiscard]] auto end() const { return instances.end(); }
  __host__ __device__ [[nodiscard]] auto InstancesInBatch() const -> std::size_t
  {
    return instances.size();
  }
  __host__ __device__ [[nodiscard]] auto NodesInBatch() const -> std::size_t
  {
    return node_idx_end - node_idx_begin;
  }
};

struct Tree {
  Tree(int max_nodes, int num_outputs) : num_outputs(num_outputs)
  {
    feature.resize(max_nodes, -1);
    split_value.resize(max_nodes);
    gain.resize(max_nodes);
    leaf_value = legate::create_buffer<double, 2>({max_nodes, num_outputs});
    node_sums  = legate::create_buffer<GPair, 2>({max_nodes, num_outputs});
    for (int i = 0; i < max_nodes; ++i) {
      for (int j = 0; j < num_outputs; ++j) {
        leaf_value[{i, j}] = 0.0;
        node_sums[{i, j}]  = GPair{0.0, 0.0};
      }
    }
  }
  void AddSplit(int node_id,
                int feature_id,
                double split_value,
                const std::vector<double>& left_leaf_value,
                const std::vector<double>& right_leaf_value,
                double gain,
                std::vector<GPair> left_sum,
                std::vector<GPair> right_sum)
  {
    auto num_outputs           = left_leaf_value.size();
    feature[node_id]           = feature_id;
    this->split_value[node_id] = split_value;
    this->gain[node_id]        = gain;
    for (int output = 0; output < num_outputs; output++) {
      this->node_sums[{BinaryTree::LeftChild(node_id), output}]   = left_sum[output];
      this->node_sums[{BinaryTree::RightChild(node_id), output}]  = right_sum[output];
      this->leaf_value[{BinaryTree::LeftChild(node_id), output}]  = left_leaf_value[output];
      this->leaf_value[{BinaryTree::RightChild(node_id), output}] = right_leaf_value[output];
    }
  }
  [[nodiscard]] auto IsLeaf(int node_id) const -> bool { return feature[node_id] == -1; }

  legate::Buffer<double, 2> leaf_value;
  std::vector<int32_t> feature;
  std::vector<double> split_value;
  std::vector<double> gain;
  legate::Buffer<GPair, 2> node_sums;
  int num_outputs;
};

template <typename T>
void WriteOutput(const legate::PhysicalStore& out, const std::vector<T>& x)
{
  auto shape = out.shape<1>();
  auto write = out.write_accessor<T, 1>();
  for (auto i = shape.lo[0]; i <= shape.hi[0]; ++i) { write[i] = x[i]; }
}

template <typename T>
void WriteOutput(const legate::PhysicalStore& out, const legate::Buffer<T, 2>& x)
{
  auto shape = out.shape<2>();
  auto write = out.write_accessor<T, 2>();
  for (auto i = shape.lo[0]; i <= shape.hi[0]; ++i) {
    for (auto j = shape.lo[1]; j <= shape.hi[1]; ++j) { write[{i, j}] = x[{i, j}]; }
  }
}

void WriteTreeOutput(legate::TaskContext context, const Tree& tree)
{
  WriteOutput(context.output(0).data(), tree.leaf_value);
  WriteOutput(context.output(1).data(), tree.feature);
  WriteOutput(context.output(2).data(), tree.split_value);
  WriteOutput(context.output(3).data(), tree.gain);
  auto hessian                        = context.output(4).data();
  const legate::Rect<2> hessian_shape = hessian.shape<2>();
  auto hessian_acc                    = hessian.write_accessor<double, 2>();
  for (auto i = hessian_shape.lo[0]; i <= hessian_shape.hi[0]; ++i) {
    for (auto j = hessian_shape.lo[1]; j <= hessian_shape.hi[1]; ++j) {
      hessian_acc[{i, j}] = tree.node_sums[{i, j}].hess;
    }
  }
}

// Randomly sample split_samples rows from X
// Share the samples with all workers
// Remove any duplicates
// Return sparse matrix of split samples for each feature
template <typename T>
auto SelectSplitSamples(legate::TaskContext context,
                        const legate::AccessorRO<T, 3>& X,
                        legate::Rect<3> X_shape,
                        int split_samples,
                        int seed,
                        int64_t dataset_rows) -> SparseSplitProposals<T>
{
  std::vector<int64_t> row_samples(split_samples);

  std::default_random_engine eng(seed);
  std::uniform_int_distribution<int64_t> dist(0, dataset_rows - 1);
  std::transform(row_samples.begin(), row_samples.end(), row_samples.begin(), [&dist, &eng](int) {
    return dist(eng);
  });

  auto num_features    = X_shape.hi[1] - X_shape.lo[1] + 1;
  auto draft_proposals = legate::create_buffer<T, 2>({num_features, split_samples});
  for (int i = 0; i < split_samples; i++) {
    auto row            = row_samples[i];
    bool const has_data = row >= X_shape.lo[0] && row <= X_shape.hi[0];
    for (int j = 0; j < num_features; j++) {
      draft_proposals[{j, i}] = has_data ? X[{row, j, 0}] : T(0);
    }
  }
  SumAllReduce(context,
               cuda::std::span<T>(draft_proposals.ptr({0, 0}), num_features * split_samples));

  // Sort samples
  std::vector<T> split_proposals_tmp;
  split_proposals_tmp.reserve(num_features * split_samples);
  auto row_pointers = legate::create_buffer<int32_t, 1>(num_features + 1);
  row_pointers[0]   = 0;
  for (int j = 0; j < num_features; j++) {
    auto ptr = draft_proposals.ptr({j, 0});
    cuda::std::span<T> const feature_proposals(draft_proposals.ptr({j, 0}), split_samples);
    std::set<T> const unique(feature_proposals.begin(), feature_proposals.end());
    row_pointers[j + 1] = row_pointers[j] + unique.size();
    split_proposals_tmp.insert(split_proposals_tmp.end(), unique.begin(), unique.end());
  }

  draft_proposals.destroy();

  auto split_proposals = legate::create_buffer<T, 1>(split_proposals_tmp.size());
  std::copy(split_proposals_tmp.begin(), split_proposals_tmp.end(), split_proposals.ptr(0));

  // Set the largest split sample to +inf such that an element must belong to one of the bins
  // i.e. we cannot go off the end when searching for a bin
  for (int feature = 0; feature < num_features; feature++) {
    auto end                 = row_pointers[feature + 1];
    split_proposals[end - 1] = std::numeric_limits<T>::infinity();
  }

  return SparseSplitProposals<T>({split_proposals.ptr(0), split_proposals_tmp.size()},
                                 {row_pointers.ptr(0), narrow<std::size_t>(num_features + 1)});
}

template <typename T>
struct TreeBuilder {
  TreeBuilder(int32_t num_rows,
              int32_t num_features,
              int32_t num_outputs,
              int32_t max_nodes,
              int32_t max_depth,
              const SparseSplitProposals<T>& split_proposals)
    : num_rows(num_rows),
      num_features(num_features),
      num_outputs(num_outputs),
      max_nodes(max_nodes),
      split_proposals(split_proposals)
  {
    sorted_positions = legate::create_buffer<std::tuple<int32_t, int32_t>>(num_rows);
    for (auto i = 0; i < num_rows; ++i) { sorted_positions[i] = {0, i}; }
    const std::size_t max_bytes = 1000000000;  // 1 GB
    const std::size_t bytes_per_node =
      num_outputs * split_proposals.HistogramSize() * sizeof(GPair);
    const std::size_t max_histogram_nodes = std::max(1UL, max_bytes / bytes_per_node);
    int depth                             = 0;
    while (BinaryTree::LevelEnd(depth + 1) <= max_histogram_nodes && depth <= max_depth) {
      depth++;
    }
    histogram      = Histogram<GPair>(BinaryTree::LevelBegin(0),
                                 BinaryTree::LevelEnd(depth),
                                 num_outputs,
                                 split_proposals.HistogramSize());
    max_batch_size = max_histogram_nodes;
  }
  void ComputeHistogram(Histogram<GPair> histogram,
                        legate::TaskContext context,
                        Tree& tree,
                        const BinnedX& X,
                        const legate::AccessorRO<double, 3>& g,
                        const legate::AccessorRO<double, 3>& h,
                        NodeBatch batch)
  {
    // Build the histogram
    for (auto [position, index_local] : batch) {
      auto index_global  = index_local + X.Shape().lo[0];
      bool const compute = ComputeHistogramBin(
        position, tree.node_sums, histogram.ContainsNode(BinaryTree::Parent(position)));
      if (position < 0 || !compute) { continue; }
      for (int64_t k = 0; k < num_outputs; ++k) {
        const GPair grad = {g[{index_global, 0, k}], h[{index_global, 0, k}]};
        for (int64_t j = 0; j < num_features; j++) {
          auto bin_idx = X[{index_local, j}];

          histogram[{position, k, bin_idx}] += grad;
        }
      }
    }

    // NCCL cannot allreduce custom types, need to reinterpret as double
    SumAllReduce(context,
                 cuda::std::span<double>(
                   // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
                   reinterpret_cast<double*>(histogram.Ptr(batch.node_idx_begin)),
                   batch.NodesInBatch() * num_outputs * split_proposals.HistogramSize() * 2));
    this->Scan(histogram, batch, tree);
  }

  void Scan(Histogram<GPair> histogram, NodeBatch batch, Tree& tree)
  {
    auto scan_node_histogram = [&](int node_idx) {
      for (int feature = 0; feature < num_features; feature++) {
        auto [feature_begin, feature_end] = split_proposals.FeatureRange(feature);
        for (int output = 0; output < num_outputs; output++) {
          GPair sum = {0.0, 0.0};
          for (int bin_idx = feature_begin; bin_idx < feature_end; bin_idx++) {
            sum += histogram[{node_idx, output, bin_idx}];
            histogram[{node_idx, output, bin_idx}] = sum;
          }
        }
      }
    };

    auto subtract_node_histogram =
      [&](int subtract_node_idx, int scanned_node_idx, int parent_node_idx) {
        for (int feature = 0; feature < num_features; feature++) {
          auto [feature_begin, feature_end] = split_proposals.FeatureRange(feature);
          for (int output = 0; output < num_outputs; output++) {
            for (int bin_idx = feature_begin; bin_idx < feature_end; bin_idx++) {
              auto scanned_sum = histogram[{scanned_node_idx, output, bin_idx}];
              auto parent_sum  = histogram[{parent_node_idx, output, bin_idx}];
              histogram[{subtract_node_idx, output, bin_idx}] = parent_sum - scanned_sum;
            }
          }
        }
      };

    if (batch.node_idx_begin == 0 && batch.node_idx_end == 1) {
      scan_node_histogram(0);
      return;
    }

    for (int node_idx = batch.node_idx_begin; node_idx < batch.node_idx_end; node_idx++) {
      auto parent = BinaryTree::Parent(node_idx);
      if (!ComputeHistogramBin(node_idx, tree.node_sums, histogram.ContainsNode(parent))) {
        continue;
      }
      scan_node_histogram(node_idx);
      // This node has no sibling we are finished
      if (node_idx == 0) { continue; }

      auto sibling_node_idx = BinaryTree::Sibling(node_idx);
      // The sibling did not compute a histogram
      // Do the subtraction trick using the histogram we just computed in the previous step
      if (!ComputeHistogramBin(sibling_node_idx, tree.node_sums, histogram.ContainsNode(parent))) {
        subtract_node_histogram(sibling_node_idx, node_idx, parent);
      }
    }
  }
  void PerformBestSplit(Tree& tree,
                        Histogram<GPair> histogram,
                        NodeBatch batch,
                        double l1_regularization,
                        double l2_regularization,
                        double min_split_gain,
                        std::optional<legate::AccessorRO<bool, 1>> optional_feature_set)
  {
    for (int node_id = batch.node_idx_begin; node_id < batch.node_idx_end; node_id++) {
      double best_gain = 0;
      int best_feature = -1;
      int best_bin     = -1;
      for (int feature = 0; feature < num_features; feature++) {
        if (optional_feature_set.has_value() && !optional_feature_set.value()[feature]) {
          continue;
        }
        auto [feature_begin, feature_end] = split_proposals.FeatureRange(feature);
        for (int bin_idx = feature_begin; bin_idx < feature_end; bin_idx++) {
          double gain = 0;
          for (int output = 0; output < num_outputs; ++output) {
            auto left   = histogram[{node_id, output, bin_idx}];
            auto parent = tree.node_sums[{node_id, output}];
            auto right  = parent - left;
            gain += CalculateGain(
              left, right, parent, l1_regularization, l2_regularization, min_split_gain);
          }
          if (gain > best_gain) {
            best_gain    = gain;
            best_feature = feature;
            best_bin     = bin_idx;
          }
        }
      }
      if (best_gain > eps) {
        std::vector<double> left_leaf(num_outputs);
        std::vector<double> right_leaf(num_outputs);
        std::vector<GPair> left_sum(num_outputs);
        std::vector<GPair> right_sum(num_outputs);
        for (int output = 0; output < num_outputs; ++output) {
          auto [G_L, H_L]    = histogram[{node_id, output, best_bin}];
          auto [G, H]        = tree.node_sums[{node_id, output}];
          auto G_R           = G - G_L;
          auto H_R           = H - H_L;
          left_leaf[output]  = CalculateLeafValue(G_L, H_L, l1_regularization, l2_regularization);
          right_leaf[output] = CalculateLeafValue(G_R, H_R, l1_regularization, l2_regularization);
          left_sum[output]   = {G_L, H_L};
          right_sum[output]  = {G_R, H_R};
        }
        if (left_sum[0].hess <= 0.0 || right_sum[0].hess <= 0.0) { continue; }
        tree.AddSplit(node_id,
                      best_feature,
                      split_proposals.split_proposals[legate::coord_t{best_bin}],
                      left_leaf,
                      right_leaf,
                      best_gain,
                      left_sum,
                      right_sum);
      }
    }
  }
  template <typename TYPE>
  void UpdatePositions(Tree& tree, const legate::AccessorRO<TYPE, 3>& X, legate::Rect<3> X_shape)
  {
    // Update the positions
    for (int i = 0; i < num_rows; i++) {
      auto [pos, index_local] = sorted_positions[i];
      if (pos < 0 || pos >= max_nodes || tree.IsLeaf(pos)) {
        sorted_positions[i] = {-1, index_local};
        continue;
      }
      auto x              = X[{X_shape.lo[0] + index_local, tree.feature[pos], 0}];
      bool const left     = x <= tree.split_value[pos];
      pos                 = left ? BinaryTree::LeftChild(pos) : BinaryTree::RightChild(pos);
      sorted_positions[i] = {pos, index_local};
    }
  }

  // Create a new histogram for this batch if we need to
  // Destroy the old one
  auto GetHistogram(NodeBatch batch) -> Histogram<GPair>
  {
    if (histogram.ContainsBatch(batch.node_idx_begin, batch.node_idx_end)) { return histogram; }

    histogram.Destroy();
    histogram = Histogram<GPair>(
      batch.node_idx_begin, batch.node_idx_end, num_outputs, split_proposals.HistogramSize());
    return histogram;
  }

  auto PrepareBatches(int depth) -> std::vector<NodeBatch>
  {
    cuda::std::span<std::tuple<int32_t, int32_t>> const sorted_positions_span(
      sorted_positions.ptr(0), num_rows);
    // Shortcut if we have 1 batch
    if (BinaryTree::NodesInLevel(depth) <= max_batch_size) {
      // All instances are in batch
      return {NodeBatch{
        BinaryTree::LevelBegin(depth), BinaryTree::LevelEnd(depth), sorted_positions_span}};
    }

    std::sort(
      sorted_positions_span.begin(),
      sorted_positions_span.end(),
      [] __device__(const auto& a, const auto& b) { return std::get<0>(a) < std::get<0>(b); });

    const int num_batches = (BinaryTree::NodesInLevel(depth) + max_batch_size - 1) / max_batch_size;
    std::vector<NodeBatch> batches(num_batches);

    for (auto batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
      int const batch_begin = BinaryTree::LevelBegin(depth) + (batch_idx * max_batch_size);
      int const batch_end   = std::min(batch_begin + max_batch_size, BinaryTree::LevelEnd(depth));
      auto comp             = [] __device__(auto a, auto b) {
        return std::get<0>(a) < std::get<0>(b);
      };  // NOLINT(readability/braces)

      auto lower = std::lower_bound(sorted_positions_span.begin(),
                                    sorted_positions_span.end(),
                                    std::tuple(batch_begin, 0),
                                    comp);
      auto upper =
        std::upper_bound(lower, sorted_positions_span.end(), std::tuple(batch_end - 1, 0), comp);
      batches[batch_idx] = {batch_begin, batch_end, {lower, upper}};
    }
    batches.erase(std::remove_if(batches.begin(),
                                 batches.end(),
                                 [](const NodeBatch& b) { return b.InstancesInBatch() == 0; }),
                  batches.end());
    return batches;
  }
  void InitialiseRoot(legate::TaskContext context,
                      Tree& tree,
                      const legate::AccessorRO<double, 3>& g_accessor,
                      const legate::AccessorRO<double, 3>& h_accessor,
                      const legate::Rect<3>& g_shape,
                      double l2_regularization,
                      double l1_regularization)
  {
    for (auto i = g_shape.lo[0]; i <= g_shape.hi[0]; ++i) {
      for (auto j = 0; j < num_outputs; ++j) {
        tree.node_sums[{0, j}] += {g_accessor[{i, 0, j}], h_accessor[{i, 0, j}]};
      }
    }
    // NCCL cannot allreduce custom types, need to reinterpret as double
    SumAllReduce(context,
                 // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
                 cuda::std::span<double>(reinterpret_cast<double*>(tree.node_sums.ptr({0, 0})),
                                         static_cast<size_t>(num_outputs * 2)));
    for (auto i = 0; i < num_outputs; ++i) {
      auto [G, H]             = tree.node_sums[{0, i}];
      tree.leaf_value[{0, i}] = CalculateLeafValue(G, H, l1_regularization, l2_regularization);
    }
  }

  legate::Buffer<std::tuple<int32_t, int32_t>, 1> sorted_positions;
  int32_t num_rows;
  int32_t num_features;
  int32_t num_outputs;
  int32_t max_nodes;
  int max_batch_size;
  SparseSplitProposals<T> split_proposals;
  Histogram<GPair> histogram;
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
    EXPECT_AXIS_ALIGNED(0, X_shape, g_shape);
    EXPECT_AXIS_ALIGNED(0, g_shape, h_shape);
    EXPECT_AXIS_ALIGNED(1, g_shape, h_shape);
    auto num_outputs = g.shape<3>().hi[2] - g.shape<3>().lo[2] + 1;
    EXPECT(g_shape.lo[2] == 0, "Expect all outputs to be present");

    // Scalars
    auto max_depth         = context.scalars().at(0).value<int>();
    auto max_nodes         = context.scalars().at(1).value<int>();
    auto split_samples     = context.scalars().at(2).value<int>();
    auto seed              = context.scalars().at(3).value<int>();
    auto dataset_rows      = context.scalars().at(4).value<int64_t>();
    auto l1_regularization = context.scalars().at(5).value<double>();
    auto l2_regularization = context.scalars().at(6).value<double>();
    auto min_split_gain    = context.scalars().at(7).value<double>();

    // Get feature sample if it exists
    std::optional<legate::AccessorRO<bool, 1>> optional_feature_set;
    // if (context.inputs().size() == 4) {
    //   auto [feature_set, feature_set_shape, feature_set_accessor] =
    //     GetInputStore<bool, 1>(context.input(3).data());
    //   optional_feature_set = feature_set_accessor;
    // }

    Tree tree(max_nodes, narrow<int>(num_outputs));
    SparseSplitProposals<T> const split_proposals =
      SelectSplitSamples(context, X_accessor, X_shape, split_samples, seed, dataset_rows);

    const BinnedX binned_X(X_accessor, X_shape, split_proposals);

    // Begin building the tree
    TreeBuilder<T> builder(
      num_rows, num_features, num_outputs, max_nodes, max_depth, split_proposals);

    builder.InitialiseRoot(
      context, tree, g_accessor, h_accessor, g_shape, l2_regularization, l1_regularization);
    for (int depth = 0; depth < max_depth; ++depth) {
      auto batches = builder.PrepareBatches(depth);
      for (auto batch : batches) {
        auto histogram = builder.GetHistogram(batch);

        builder.ComputeHistogram(histogram, context, tree, binned_X, g_accessor, h_accessor, batch);

        builder.PerformBestSplit(tree,
                                 histogram,
                                 batch,
                                 l1_regularization,
                                 l2_regularization,
                                 min_split_gain,
                                 optional_feature_set);
      }
      // Update position of entire level
      // Don't bother updating positions for the last level
      if (depth < max_depth - 1) { builder.UpdatePositions(tree, X_accessor, X_shape); }
    }
    WriteTreeOutput(context, tree);
  }
};

}  // namespace

/*static*/ void BuildTreeTask::cpu_variant(legate::TaskContext context)
{
  const auto& X = context.input(0).data();
  legateboost::type_dispatch_float(X.code(), build_tree_fn(), context);
}

}  // namespace legateboost

namespace  // unnamed
{
void __attribute__((constructor)) register_tasks()
{
  legateboost::BuildTreeTask::register_variants();
}
}  // namespace
