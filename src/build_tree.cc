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
#include "build_tree.h"

namespace legateboost {

namespace {
struct Tree {
  Tree(int max_depth, int num_outputs) : num_outputs(num_outputs)
  {
    int max_nodes = 1 << (max_depth + 1);
    leaf_value    = legate::create_buffer<double, 2>({max_nodes, num_outputs});
    std::fill(leaf_value.ptr({0, 0}), leaf_value.ptr({max_nodes, num_outputs}), 0.0);
    feature.resize(max_nodes, -1);
    split_value.resize(max_nodes);
    gain.resize(max_nodes);
    hessian = legate::create_buffer<double, 2>({max_nodes, num_outputs});
    std::fill(hessian.ptr({0, 0}), hessian.ptr({max_nodes, num_outputs}), 0.0);
    gradient = legate::create_buffer<double, 2>({max_nodes, num_outputs});
    std::fill(gradient.ptr({0, 0}), gradient.ptr({max_nodes, num_outputs}), 0.0);
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
void WriteOutput(legate::Store& out, const std::vector<T>& x)
{
  EXPECT(out.shape<2>().volume() >= x.size(), "Output not large enough.");
  std::copy(x.begin(), x.end(), out.write_accessor<T, 2>().ptr({0, 0}));
}
void WriteOutput(legate::Store& out, const legate::Buffer<double, 2>& x)
{
  std::copy(x.ptr({0, 0}),
            x.ptr({0, 0}) + out.shape<2>().volume(),
            out.write_accessor<double, 2>().ptr({0, 0}));
}

void WriteTreeOutput(legate::TaskContext& context, const Tree& tree)
{
  WriteOutput(context.outputs().at(0), tree.leaf_value);
  WriteOutput(context.outputs().at(1), tree.feature);
  WriteOutput(context.outputs().at(2), tree.split_value);
  WriteOutput(context.outputs().at(3), tree.gain);
  WriteOutput(context.outputs().at(4), tree.hessian);
}

struct GradientHistogram {
  // Dimensions
  // 0. Depth
  // 1. Feature
  // 2. Output
  legate::Buffer<GPair, 3> gradient_sums;
  int size;
  int num_features;
  int depth;
  int num_outputs;

  GradientHistogram(int num_features, int depth, int num_outputs)
    : num_features(num_features),
      depth(depth),
      num_outputs(num_outputs),
      size((1 << depth) * num_features * num_outputs),
      gradient_sums(legate::create_buffer<GPair, 3>({1 << depth, num_features, num_outputs}))
  {
    auto ptr = gradient_sums.ptr({0, 0, 0});
    std::fill(ptr, ptr + size, GPair{0.0, 0.0});
  }
  void Add(int feature, int position_in_level, int output, GPair g)
  {
    gradient_sums[{position_in_level, feature, output}] += g;
  }
  GPair Get(int feature, int position, int output)
  {
    int position_in_level = position - ((1 << depth) - 1);
    return gradient_sums[{position_in_level, feature, output}];
  }
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
    auto max_depth   = context.scalars().at(0).value<int>();
    auto random_seed = context.scalars().at(1).value<uint64_t>();

    Tree tree(max_depth, num_outputs);

    // Initialize the root node
    std::vector<GPair> base_sums(num_outputs);
    for (auto i = g_shape.lo[0]; i <= g_shape.hi[0]; ++i) {
      for (auto j = 0; j < num_outputs; ++j) {
        base_sums[j] += {g_accessor[{i, j}], h_accessor[{i, j}]};
      }
    }
    SumAllReduce(context, reinterpret_cast<double*>(base_sums.data()), num_outputs * 2);
    for (auto i = 0; i < num_outputs; ++i) {
      auto [G, H]             = base_sums[i];
      tree.leaf_value[{0, i}] = -G / H;
      tree.gradient[{0, i}]   = G;
      tree.hessian[{0, i}]    = H;
    }

    // Begin building the tree
    std::vector<int32_t> positions(num_rows);
    for (int64_t depth = 0; depth < max_depth; ++depth) {
      GradientHistogram histogram(num_features, depth, num_outputs);
      for (int64_t i = X_shape.lo[0]; i <= X_shape.hi[0]; i++) {
        auto index_local = i - X_shape.lo[0];
        auto position    = positions[index_local];
        if (position < 0) continue;
        auto position_in_level = position - ((1 << depth) - 1);
        for (int64_t j = 0; j < num_features; j++) {
          if (X_accessor[{i, j}] <= split_proposal_accessor[{depth, j}]) {
            for (int64_t k = 0; k < num_outputs; ++k) {
              histogram.Add(j, position_in_level, k, GPair{g_accessor[{i, k}], h_accessor[{i, k}]});
            }
          }
        }
      }
      SumAllReduce(context,
                   reinterpret_cast<double*>(histogram.gradient_sums.ptr({0, 0, 0})),
                   histogram.size * 2);
      // Find the best split
      double eps = 1e-5;
      for (int node_id = (1 << depth) - 1; node_id < (1 << (depth + 1)) - 1; node_id++) {
        double best_gain = 0;
        int best_feature = -1;
        for (int feature = 0; feature < num_features; feature++) {
          double gain = 0;
          for (int output = 0; output < num_outputs; ++output) {
            auto [G_L, H_L] = histogram.Get(feature, node_id, output);
            auto G          = tree.gradient[{node_id, output}];
            auto H          = tree.hessian[{node_id, output}];
            auto G_R        = G - G_L;
            auto H_R        = H - H_L;
            gain +=
              0.5 * ((G_L * G_L) / (H_L + eps) + (G_R * G_R) / (H_R + eps) - (G * G) / (H + eps));
          }
          if (gain > best_gain) {
            best_gain    = gain;
            best_feature = feature;
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
            auto [G_L, H_L]        = histogram.Get(best_feature, node_id, output);
            auto G                 = tree.gradient[{node_id, output}];
            auto H                 = tree.hessian[{node_id, output}];
            auto G_R               = G - G_L;
            auto H_R               = H - H_L;
            left_leaf[output]      = -G_L / H_L;
            right_leaf[output]     = -G_R / H_R;
            gradient_left[output]  = G_L;
            gradient_right[output] = G_R;
            hessian_left[output]   = H_L;
            hessian_right[output]  = H_R;
          }
          if (hessian_left[0] <= 0.0 || hessian_right[0] <= 0.0) continue;
          tree.AddSplit(node_id,
                        best_feature,
                        split_proposal_accessor[{depth, best_feature}],
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
        auto x    = X_accessor[{i, tree.feature[pos]}];
        bool left = x <= tree.split_value[pos];
        pos       = left ? Tree::LeftChild(pos) : Tree::RightChild(pos);
      }
    }

    if (context.get_task_index()[0] == 0) { WriteTreeOutput(context, tree); }
  }
};

}  // namespace

/*static*/ void BuildTreeTask::cpu_variant(legate::TaskContext& context)
{
  const auto& X = context.inputs().at(0);
  type_dispatch_float(X.code(), build_tree_fn(), context);
}

}  // namespace legateboost

namespace  // unnamed
{
static void __attribute__((constructor)) register_tasks(void)
{
  legateboost::BuildTreeTask::register_variants();
}
}  // namespace
