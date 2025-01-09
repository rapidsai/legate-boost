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
#include "matrix_types.h"

namespace legateboost {
namespace {

struct NodeBatch {
  int32_t node_idx_begin{};
  int32_t node_idx_end{};
  tcb::span<std::tuple<int32_t, int32_t>> instances;
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
template <typename T, template <typename> class XMatrix>
auto SelectSplitSamples(legate::TaskContext context,
                        const XMatrix<T>& X,
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

  auto draft_proposals = legate::create_buffer<T, 2>({X.NumFeatures(), split_samples});
  for (int i = 0; i < split_samples; i++) {
    auto row            = row_samples[i];
    const bool has_data = X.RowSubset().contains(row);
    for (int j = 0; j < X.NumFeatures(); j++) {
      draft_proposals[{j, i}] = has_data ? X.Get(row, j) : T(0);
    }
  }
  SumAllReduce(context, tcb::span<T>(draft_proposals.ptr({0, 0}), X.NumFeatures() * split_samples));

  // Sort samples
  std::vector<T> split_proposals_tmp;
  split_proposals_tmp.reserve(X.NumFeatures() * split_samples);
  auto row_pointers = legate::create_buffer<int32_t, 1>(X.NumFeatures() + 1);
  row_pointers[0]   = 0;
  for (int j = 0; j < X.NumFeatures(); j++) {
    auto ptr = draft_proposals.ptr({j, 0});
    tcb::span<T> const feature_proposals(draft_proposals.ptr({j, 0}), split_samples);
    std::set<T> const unique(feature_proposals.begin(), feature_proposals.end());
    row_pointers[j + 1] = row_pointers[j] + unique.size();
    split_proposals_tmp.insert(split_proposals_tmp.end(), unique.begin(), unique.end());
  }

  auto split_proposals = legate::create_buffer<T, 1>(split_proposals_tmp.size());
  std::copy(split_proposals_tmp.begin(), split_proposals_tmp.end(), split_proposals.ptr(0));

  // Set the largest split sample to +inf such that an element must belong to one of the bins
  // i.e. we cannot go off the end when searching for a bin
  for (int feature = 0; feature < num_features; feature++) {
    auto end                 = row_pointers[feature + 1];
    split_proposals[end - 1] = std::numeric_limits<T>::infinity();
  }

  return SparseSplitProposals<T>(
    split_proposals, row_pointers, X.NumFeatures(), split_proposals_tmp.size());
}

template <typename MatrixT>
struct TreeBuilder {
  using T = typename MatrixT::value_type;
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
      max_depth(max_depth),
      split_proposals(split_proposals)
  {
    sorted_positions = legate::create_buffer<std::tuple<int32_t, int32_t>>(num_rows);
    for (auto i = 0; i < num_rows; ++i) { sorted_positions[i] = {0, i}; }
    const std::size_t max_bytes      = 1000000000;  // 1 GB
    const std::size_t bytes_per_node = num_outputs * split_proposals.histogram_size * sizeof(GPair);
    const std::size_t max_histogram_nodes = std::max(1UL, max_bytes / bytes_per_node);
    int depth                             = 0;
    while (BinaryTree::LevelEnd(depth + 1) <= max_histogram_nodes && depth <= max_depth) {
      depth++;
    }
    histogram      = Histogram<GPair>(BinaryTree::LevelBegin(0),
                                 BinaryTree::LevelEnd(depth),
                                 num_outputs,
                                 split_proposals.histogram_size);
    max_batch_size = max_histogram_nodes;
  }

  Tree Build(legate::TaskContext context,
             const MatrixT& X_matrix,
             legate::AccessorRO<double, 3> g_accessor,
             legate::AccessorRO<double, 3> h_accessor,
             legate::Rect<3> g_shape,
             double alpha)
  {
    // Begin building the tree
    Tree tree(max_nodes, narrow<int>(num_outputs));
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

  void DenseHistogramKernel(const Tree& tree,
                            Histogram<GPair>& histogram,
                            const DenseXMatrix<T>& X,
                            legate::AccessorRO<double, 3> g,
                            legate::AccessorRO<double, 3> h,
                            NodeBatch batch)
  {
    // Build the histogram
    for (auto [position, index_local] : batch) {
      auto index_global  = index_local + X.RowSubset().lo[0];
      bool const compute = ComputeHistogramBin(
        position, tree.node_sums, histogram.ContainsNode(BinaryTree::Parent(position)));
      if (position < 0 || !compute) { continue; }
      for (int64_t j = 0; j < num_features; j++) {
        auto x_value      = X.Get(index_global, j);
        int const bin_idx = split_proposals.FindBin(x_value, j);

        for (int64_t k = 0; k < num_outputs; ++k) {
          histogram[{position, k, bin_idx}] +=
            GPair{g[{index_global, 0, k}], h[{index_global, 0, k}]};
        }
      }
    }
  }

  // Kernel specialised to iterate only over the non-zero elements of the sparse matrix
  void CSRHistogramKernel(const Tree& tree,
                          Histogram<GPair>& histogram,
                          const CSRXMatrix<T>& X,
                          legate::AccessorRO<double, 3> g,
                          legate::AccessorRO<double, 3> h,
                          NodeBatch batch)
  {
    // Build the histogram
    for (auto [position, index_local] : batch) {
      auto index_global  = index_local + X.RowSubset().lo[0];
      bool const compute = ComputeHistogramBin(
        position, tree.node_sums, histogram.ContainsNode(BinaryTree::Parent(position)));
      if (position < 0 || !compute) { continue; }
      auto row_range = X.row_ranges[index_global];
      for (auto element_idx = row_range.lo[0]; element_idx <= row_range.hi[0]; element_idx++) {
        auto feature      = X.column_indices[element_idx];
        auto x            = X.values[element_idx];
        int const bin_idx = split_proposals.FindBin(x, feature);
        if (bin_idx != SparseSplitProposals<T>::NOT_FOUND) {
          for (int64_t k = 0; k < num_outputs; ++k) {
            histogram[{position, k, bin_idx}] +=
              GPair{g[{index_global, 0, k}], h[{index_global, 0, k}]};
          }
        }
      }
    }
  }

  void ComputeHistogram(Histogram<GPair> histogram,
                        legate::TaskContext context,
                        const Tree& tree,
                        const MatrixT& X,
                        const legate::AccessorRO<double, 3>& g,
                        const legate::AccessorRO<double, 3>& h,
                        NodeBatch batch)
  {
    if constexpr (std::is_same_v<MatrixT, DenseXMatrix<T>>) {
      this->DenseHistogramKernel(tree, histogram, X, g, h, batch);
    } else {
      this->CSRHistogramKernel(tree, histogram, X, g, h, batch);
    }

    // NCCL cannot allreduce custom types, need to reinterpret as double
    SumAllReduce(
      context,
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
      tcb::span<double>(reinterpret_cast<double*>(histogram.Ptr(batch.node_idx_begin)),
                        batch.NodesInBatch() * num_outputs * split_proposals.histogram_size * 2));
    this->Scan(histogram, batch, tree);
  }

  void Scan(Histogram<GPair> histogram, NodeBatch batch, const Tree& tree)
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
  void PerformBestSplit(Tree& tree, Histogram<GPair> histogram, double alpha, NodeBatch batch)
  {
    const bool is_sparse_matrix = std::is_same_v<MatrixT, CSRXMatrix<T>>;
    for (int node_id = batch.node_idx_begin; node_id < batch.node_idx_end; node_id++) {
      double best_gain = 0;
      int best_feature = -1;
      int best_bin     = -1;
      for (int feature = 0; feature < num_features; feature++) {
        auto [feature_begin, feature_end] = split_proposals.FeatureRange(feature);
        for (int bin_idx = feature_begin; bin_idx < feature_end; bin_idx++) {
          double gain = 0;
          for (int output = 0; output < num_outputs; ++output) {
            auto [left_sum, right_sum] = InferSplitSums(histogram,
                                                        split_proposals,
                                                        tree.node_sums[{node_id, output}],
                                                        node_id,
                                                        output,
                                                        bin_idx,
                                                        feature,
                                                        is_sparse_matrix);
            auto [G_L, H_L]            = left_sum;
            auto [G_R, H_R]            = right_sum;
            auto [G, H]                = tree.node_sums[{node_id, output}];

            double const reg = std::max(eps, alpha);  // Regularisation term
            gain +=
              0.5 * ((G_L * G_L) / (H_L + reg) + (G_R * G_R) / (H_R + reg) - (G * G) / (H + reg));
          }
          if (gain > best_gain) {
            best_gain    = gain;
            best_feature = feature;
            best_bin     = bin_idx;
          }
        }
      }
      if (best_gain > eps) {
        std::vector<double> left_leaves(num_outputs);
        std::vector<double> right_leaves(num_outputs);
        std::vector<GPair> left_sums(num_outputs);
        std::vector<GPair> right_sums(num_outputs);
        for (int output = 0; output < num_outputs; ++output) {
          auto [left_sum, right_sum] = InferSplitSums(histogram,
                                                      split_proposals,
                                                      tree.node_sums[{node_id, output}],
                                                      node_id,
                                                      output,
                                                      best_bin,
                                                      best_feature,
                                                      is_sparse_matrix);
          left_leaves[output]        = CalculateLeafValue(left_sum.grad, left_sum.hess, alpha);
          right_leaves[output]       = CalculateLeafValue(right_sum.grad, right_sum.hess, alpha);
          left_sums[output]          = left_sum;
          right_sums[output]         = right_sum;
        }
        if (left_sums[0].hess <= 0.0 || right_sums[0].hess <= 0.0) { continue; }
        tree.AddSplit(node_id,
                      best_feature,
                      split_proposals.split_proposals[legate::coord_t{best_bin}],
                      left_leaves,
                      right_leaves,
                      best_gain,
                      left_sums,
                      right_sums);
      }
    }
  }
  void UpdatePositions(Tree& tree, const MatrixT& X)
  {
    // Update the positions
    for (int i = 0; i < num_rows; i++) {
      auto [pos, index_local] = sorted_positions[i];
      if (pos < 0 || pos >= max_nodes || tree.IsLeaf(pos)) {
        sorted_positions[i] = {-1, index_local};
        continue;
      }
      auto x              = X.Get(X.RowSubset().lo[0] + index_local, tree.feature[pos]);
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
      batch.node_idx_begin, batch.node_idx_end, num_outputs, split_proposals.histogram_size);
    return histogram;
  }

  auto PrepareBatches(int depth) -> std::vector<NodeBatch>
  {
    tcb::span<std::tuple<int32_t, int32_t>> const sorted_positions_span(sorted_positions.ptr(0),
                                                                        num_rows);
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
                      double alpha)
  {
    for (auto i = g_shape.lo[0]; i <= g_shape.hi[0]; ++i) {
      for (auto j = 0; j < num_outputs; ++j) {
        tree.node_sums[{0, j}] += {g_accessor[{i, 0, j}], h_accessor[{i, 0, j}]};
      }
    }
    // NCCL cannot allreduce custom types, need to reinterpret as double
    SumAllReduce(context,
                 // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
                 tcb::span<double>(reinterpret_cast<double*>(tree.node_sums.ptr({0, 0})),
                                   static_cast<size_t>(num_outputs * 2)));
    for (auto i = 0; i < num_outputs; ++i) {
      auto [G, H]             = tree.node_sums[{0, i}];
      tree.leaf_value[{0, i}] = CalculateLeafValue(G, H, alpha);
    }
  }

  legate::Buffer<std::tuple<int32_t, int32_t>, 1> sorted_positions;
  int32_t num_rows;
  int32_t num_features;
  int32_t num_outputs;
  int32_t max_nodes;
  int32_t max_depth;
  int max_batch_size;
  SparseSplitProposals<T> split_proposals;
  Histogram<GPair> histogram;
};

struct build_tree_dense_fn {
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
    auto max_depth     = context.scalars().at(0).value<int>();
    auto max_nodes     = context.scalars().at(1).value<int>();
    auto alpha         = context.scalars().at(2).value<double>();
    auto split_samples = context.scalars().at(3).value<int>();
    auto seed          = context.scalars().at(4).value<int>();
    auto dataset_rows  = context.scalars().at(5).value<int64_t>();

    DenseXMatrix<T> X_matrix(X_accessor, X_shape);

    SparseSplitProposals<T> const split_proposals =
      SelectSplitSamples(context, X_matrix, split_samples, seed, dataset_rows);

    // Dispatch the tree building algorithm templated on the matrix type
    auto tree = TreeBuilder<DenseXMatrix<T>>(
                  num_rows, num_features, num_outputs, max_nodes, max_depth, split_proposals)
                  .Build(context, X_matrix, g_accessor, h_accessor, g_shape, alpha);

    WriteTreeOutput(context, tree);
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

    CSRXMatrix<T> X_matrix(X_vals_accessor,
                           X_coords_accessor,
                           X_offsets_accessor,
                           X_vals_shape,
                           X_offsets_shape,
                           num_features,
                           X_vals_shape.volume());
    const SparseSplitProposals<T> split_proposals =
      SelectSplitSamples(context, X_matrix, split_samples, seed, dataset_rows);

    auto tree = TreeBuilder<CSRXMatrix<T>>(
                  num_rows, num_features, num_outputs, max_nodes, max_depth, split_proposals)
                  .Build(context, X_matrix, g_accessor, h_accessor, g_shape, alpha);

    WriteTreeOutput(context, tree);
  }
};
}  // namespace

/*static*/ void BuildTreeDenseTask::cpu_variant(legate::TaskContext context)
{
  const auto& X = context.input(0).data();
  legateboost::type_dispatch_float(X.code(), build_tree_dense_fn(), context);
}

/*static*/ void BuildTreeCSRTask::cpu_variant(legate::TaskContext context)
{
  const auto& X = context.input(0).data();
  legateboost::type_dispatch_float(X.code(), build_tree_csr_fn(), context);
}

}  // namespace legateboost

namespace  // unnamed
{
void __attribute__((constructor)) register_tasks()
{
  legateboost::BuildTreeDenseTask::register_variants();
  legateboost::BuildTreeCSRTask::register_variants();
}
}  // namespace
