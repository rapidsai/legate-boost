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
#include "legate.h"
#include "legate_library.h"
#include "legateboost.h"
#include "../../cpp_utils/cpp_utils.h"
#include "build_tree.h"

namespace legateboost {

namespace {
struct Tree {
  Tree(int max_nodes, int num_outputs) : num_outputs(num_outputs)
  {
    feature.resize(max_nodes, -1);
    split_value.resize(max_nodes);
    gain.resize(max_nodes);
    leaf_value = legate::create_buffer<double, 2>({max_nodes, num_outputs});
    hessian    = legate::create_buffer<double, 2>({max_nodes, num_outputs});
    gradient   = legate::create_buffer<double, 2>({max_nodes, num_outputs});
    for (int i = 0; i < max_nodes; ++i) {
      for (int j = 0; j < num_outputs; ++j) {
        leaf_value[{i, j}] = 0.0;
        hessian[{i, j}]    = 0.0;
        gradient[{i, j}]   = 0.0;
      }
    }
  }
  void AddSplit(int node_id,
                int feature_id,
                double split_value,
                const std::vector<double>& left_leaf_value,
                const std::vector<double>& right_leaf_value,
                double gain,
                const std::vector<double>& gradient_left,
                const std::vector<double>& gradient_right,
                const std::vector<double>& hessian_left,
                const std::vector<double>& hessian_right)
  {
    auto num_outputs           = left_leaf_value.size();
    feature[node_id]           = feature_id;
    this->split_value[node_id] = split_value;
    this->gain[node_id]        = gain;
    for (int output = 0; output < num_outputs; output++) {
      this->gradient[{LeftChild(node_id), output}]    = gradient_left[output];
      this->gradient[{RightChild(node_id), output}]   = gradient_right[output];
      this->hessian[{LeftChild(node_id), output}]     = hessian_left[output];
      this->hessian[{RightChild(node_id), output}]    = hessian_right[output];
      this->leaf_value[{LeftChild(node_id), output}]  = left_leaf_value[output];
      this->leaf_value[{RightChild(node_id), output}] = right_leaf_value[output];
    }
  }
  static int LeftChild(int id) { return id * 2 + 1; }
  static int RightChild(int id) { return id * 2 + 2; }
  static int Parent(int id) { return (id - 1) / 2; }

  bool IsLeaf(int node_id) const { return feature[node_id] == -1; }

  legate::Buffer<double, 2> leaf_value;
  std::vector<int32_t> feature;
  std::vector<double> split_value;
  std::vector<double> gain;
  legate::Buffer<double, 2> hessian;
  legate::Buffer<double, 2>
    gradient;  // This is not used in the output tree but we use it during training
  const int num_outputs;
};

template <typename T>
void WriteOutput(legate::PhysicalStore out, const std::vector<T>& x)
{
  auto shape = out.shape<1>();
  auto write = out.write_accessor<T, 1>();
  for (auto i = shape.lo[0]; i <= shape.hi[0]; ++i) { write[i] = x[i]; }
}

template <typename T>
void WriteOutput(legate::PhysicalStore out, const legate::Buffer<T, 2>& x)
{
  auto shape = out.shape<2>();
  auto write = out.write_accessor<T, 2>();
  for (auto i = shape.lo[0]; i <= shape.hi[0]; ++i) {
    for (int j = shape.lo[1]; j <= shape.hi[1]; ++j) { write[{i, j}] = x[{i, j}]; }
  }
}

void WriteTreeOutput(legate::TaskContext context, const Tree& tree)
{
  WriteOutput(context.output(0).data(), tree.leaf_value);
  WriteOutput(context.output(1).data(), tree.feature);
  WriteOutput(context.output(2).data(), tree.split_value);
  WriteOutput(context.output(3).data(), tree.gain);
  WriteOutput(context.output(4).data(), tree.hessian);
}

struct GradientHistogram {
  // Dimensions
  // 0. Depth
  // 1. Feature
  // 2. Bin
  // 3. Output
  legate::Buffer<GPair, 4> gradient_sums;
  int size;
  int num_features;
  int depth;
  int num_outputs;
  int samples_per_feature;

  GradientHistogram(int num_features, int depth, int num_outputs, int samples_per_feature)
    : num_features(num_features),
      depth(depth),
      samples_per_feature(samples_per_feature),
      num_outputs(num_outputs),
      size((1 << depth) * num_features * num_outputs * samples_per_feature),
      gradient_sums(legate::create_buffer<GPair, 4>(
        {1 << depth, num_features, samples_per_feature, num_outputs}))
  {
    auto ptr = gradient_sums.ptr({0, 0, 0, 0});
    std::fill(ptr, ptr + size, GPair{0.0, 0.0});
  }
  ~GradientHistogram() { gradient_sums.destroy(); }
  void Add(int position_in_level, int feature, int bin_idx, int output, GPair g)
  {
    gradient_sums[{position_in_level, feature, bin_idx, output}] += g;
  }
  GPair Get(int feature, int position, int output, int bin_idx)
  {
    int position_in_level = position - ((1 << depth) - 1);
    return gradient_sums[{position_in_level, feature, bin_idx, output}];
  }
  void Scan()
  {
    for (int feature = 0; feature < num_features; feature++) {
      for (int node_id = 0; node_id < 1 << depth; node_id++) {
        for (int output = 0; output < num_outputs; output++) {
          GPair sum = {0.0, 0.0};
          for (int bin_idx = 0; bin_idx < samples_per_feature; bin_idx++) {
            sum += gradient_sums[{node_id, feature, bin_idx, output}];
            gradient_sums[{node_id, feature, bin_idx, output}] = sum;
          }
        }
      }
    }
  }
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
    EXPECT_AXIS_ALIGNED(0, X_shape, g_shape);
    EXPECT_AXIS_ALIGNED(0, g_shape, h_shape);
    EXPECT_AXIS_ALIGNED(1, g_shape, h_shape);
    auto num_outputs = g.shape<3>().hi[2] - g.shape<3>().lo[2] + 1;
    EXPECT_IS_BROADCAST(split_proposals_shape);
    auto samples_per_feature = split_proposals_shape.hi[1] - split_proposals_shape.lo[1] + 1;
    EXPECT(g_shape.lo[2] == 0, "Expect all outputs to be present");

    // Scalars
    auto max_depth = context.scalars().at(0).value<int>();
    auto max_nodes = context.scalars().at(1).value<int>();
    auto alpha     = context.scalars().at(2).value<double>();

    Tree tree(max_nodes, num_outputs);

    // Initialize the root node
    std::vector<GPair> base_sums(num_outputs);
    for (auto i = g_shape.lo[0]; i <= g_shape.hi[0]; ++i) {
      for (auto j = 0; j < num_outputs; ++j) {
        base_sums[j] += {g_accessor[{i, 0, j}], h_accessor[{i, 0, j}]};
      }
    }
    SumAllReduce(context, reinterpret_cast<double*>(base_sums.data()), num_outputs * 2);
    for (auto i = 0; i < num_outputs; ++i) {
      auto [G, H]             = base_sums[i];
      tree.leaf_value[{0, i}] = CalculateLeafValue(G, H, alpha);
      tree.gradient[{0, i}]   = G;
      tree.hessian[{0, i}]    = H;
    }

    // Begin building the tree
    std::vector<int32_t> positions(num_rows);
    for (int64_t depth = 0; depth < max_depth; ++depth) {
      // Build the histogram
      GradientHistogram histogram(num_features, depth, num_outputs, samples_per_feature);
      for (int64_t i = X_shape.lo[0]; i <= X_shape.hi[0]; i++) {
        auto index_local = i - X_shape.lo[0];
        auto position    = positions[index_local];
        if (position < 0) continue;
        auto position_in_level = position - ((1 << depth) - 1);
        for (int64_t j = 0; j < num_features; j++) {
          auto x_value = X_accessor[{i, j, 0}];
          int bin_idx  = std::lower_bound(split_proposals_accessor.ptr({j, 0}),
                                         split_proposals_accessor.ptr({j, samples_per_feature}),
                                         x_value) -
                        split_proposals_accessor.ptr({j, 0});

          if (bin_idx < samples_per_feature) {
            for (int64_t k = 0; k < num_outputs; ++k) {
              histogram.Add(position_in_level,
                            j,
                            bin_idx,
                            k,
                            GPair{g_accessor[{i, 0, k}], h_accessor[{i, 0, k}]});
            }
          }
        }
      }

      histogram.Scan();
      SumAllReduce(context,
                   reinterpret_cast<double*>(histogram.gradient_sums.ptr({0, 0, 0, 0})),
                   histogram.size * 2);
      // Find the best split
      double eps = 1e-5;
      for (int node_id = (1 << depth) - 1; node_id < (1 << (depth + 1)) - 1; node_id++) {
        double best_gain = 0;
        int best_feature = -1;
        int best_bin     = -1;
        for (int feature = 0; feature < num_features; feature++) {
          for (int bin_idx = 0; bin_idx < samples_per_feature; bin_idx++) {
            double gain = 0;
            for (int output = 0; output < num_outputs; ++output) {
              auto [G_L, H_L] = histogram.Get(feature, node_id, output, bin_idx);  // TODO: bin idx
              auto G          = tree.gradient[{node_id, output}];
              auto H          = tree.hessian[{node_id, output}];
              auto G_R        = G - G_L;
              auto H_R        = H - H_L;
              double reg      = std::max(eps, alpha);  // Regularisation term
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
          std::vector<double> left_leaf(num_outputs);
          std::vector<double> right_leaf(num_outputs);
          std::vector<double> gradient_left(num_outputs);
          std::vector<double> gradient_right(num_outputs);
          std::vector<double> hessian_left(num_outputs);
          std::vector<double> hessian_right(num_outputs);
          for (int output = 0; output < num_outputs; ++output) {
            auto [G_L, H_L] =
              histogram.Get(best_feature, node_id, output, best_bin);  // TODO: bin idx
            auto G                 = tree.gradient[{node_id, output}];
            auto H                 = tree.hessian[{node_id, output}];
            auto G_R               = G - G_L;
            auto H_R               = H - H_L;
            left_leaf[output]      = CalculateLeafValue(G_L, H_L, alpha);
            right_leaf[output]     = CalculateLeafValue(G_R, H_R, alpha);
            gradient_left[output]  = G_L;
            gradient_right[output] = G_R;
            hessian_left[output]   = H_L;
            hessian_right[output]  = H_R;
          }
          if (hessian_left[0] <= 0.0 || hessian_right[0] <= 0.0) continue;
          tree.AddSplit(node_id,
                        best_feature,
                        split_proposals_accessor[{best_feature, best_bin}],
                        left_leaf,
                        right_leaf,
                        best_gain,
                        gradient_left,
                        gradient_right,
                        hessian_left,
                        hessian_right);
        }
      }

      // Update the positions
      for (int64_t i = X_shape.lo[0]; i <= X_shape.hi[0]; i++) {
        auto index_local = i - X_shape.lo[0];
        int& pos         = positions[index_local];
        if (pos < 0 || tree.IsLeaf(pos)) {
          pos = -1;
          continue;
        }
        auto x    = X_accessor[{i, tree.feature[pos], 0}];
        bool left = x <= tree.split_value[pos];
        pos       = left ? Tree::LeftChild(pos) : Tree::RightChild(pos);
      }
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
static void __attribute__((constructor)) register_tasks(void)
{
  legateboost::BuildTreeTask::register_variants();
}
}  // namespace
