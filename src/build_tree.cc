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

namespace legateboost {

void SumAllReduce(legate::TaskContext& context, double* x, int count)
{
  if (context.communicators().size() == 0) return;
  auto& comm       = context.communicators().at(0);
  auto domain      = context.get_launch_domain();
  size_t num_ranks = domain.get_volume();
  std::vector<double> gather_result(num_ranks * count);
  legate::comm::coll::collAllgather(x,
                                    gather_result.data(),
                                    count,
                                    legate::comm::coll::CollDataType::CollDouble,
                                    comm.get<legate::comm::coll::CollComm>());
  for (std::size_t j = 0; j < count; j++) { x[j] = 0.0; }
  for (std::size_t i = 0; i < num_ranks; i++) {
    for (std::size_t j = 0; j < count; j++) { x[j] += gather_result[i * count + j]; }
  }
}

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
  }
  void AddSplit(int node_id,
                int feature_id,
                double split_value,
                const std::vector<double>& left_leaf_value,
                const std::vector<double>& right_leaf_value,
                double gain,
                const std::vector<double>& hessian_left,
                const std::vector<double>& hessian_right)
  {
    auto num_outputs           = left_leaf_value.size();
    feature[node_id]           = feature_id;
    this->split_value[node_id] = split_value;
    this->gain[node_id]        = gain;
    for (int output = 0; output < num_outputs; output++) {
      this->hessian[{LeftChild(node_id), output}]     = hessian_left[output];
      this->hessian[{RightChild(node_id), output}]    = hessian_right[output];
      this->leaf_value[{LeftChild(node_id), output}]  = left_leaf_value[output];
      this->leaf_value[{RightChild(node_id), output}] = right_leaf_value[output];
    }
  }
  static int LeftChild(int id) { return id * 2 + 1; }
  static int RightChild(int id) { return id * 2 + 2; }

  bool IsLeaf(int node_id) const { return feature[node_id] == -1; }

  legate::Buffer<double, 2> leaf_value;
  std::vector<int32_t> feature;
  std::vector<double> split_value;
  std::vector<double> gain;
  legate::Buffer<double, 2> hessian;
  const int num_outputs;
};

template <typename T>
void WriteOutput(legate::Store& out, const std::vector<T>& x)
{
  EXPECT(out.shape<1>().volume() >= x.size(), "Output not large enough.");
  std::copy(x.begin(), x.end(), out.write_accessor<T, 1>().ptr(0));
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

struct GPair {
  double grad = 0.0;
  double hess = 0.0;

  GPair& operator+=(const GPair& b)
  {
    this->grad += b.grad;
    this->hess += b.hess;
    return *this;
  }
};

struct GradientHistogram {
  // Dimensions
  // 0. Depth
  // 1. Feature
  // 2. Output
  // 3. Left/Right child
  legate::Buffer<GPair, 4> gradient_sums;
  int size;
  int num_features;
  int depth;
  int num_outputs;

  GradientHistogram(int num_features, int depth, int num_outputs)
    : num_features(num_features),
      depth(depth),
      num_outputs(num_outputs),
      size((1 << depth) * num_features * num_outputs * 2),
      gradient_sums(legate::create_buffer<GPair, 4>({1 << depth, num_features, num_outputs, 2}))
  {
    auto ptr = gradient_sums.ptr({0, 0, 0, 0});
    std::fill(ptr, ptr + size, GPair{0.0, 0.0});
  }
  void Add(int feature, int position, int output, GPair g, bool left)
  {
    if (position < 0) return;
    int position_in_level = position - ((1 << depth) - 1);
    gradient_sums[{position_in_level, feature, output, left}] += g;
  }
  GPair Get(int feature, int position, int output, bool left)
  {
    int position_in_level = position - ((1 << depth) - 1);
    return gradient_sums[{position_in_level, feature, output, left}];
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
    auto g_shape      = context.inputs().at(1).shape<2>();
    auto h_shape      = context.inputs().at(2).shape<2>();
    EXPECT(g_shape == h_shape, "Expected same shape for g and h.");
    EXPECT(g_shape.hi[0] - g_shape.lo[0] + 1 == num_rows,
           "Expected same number of rows for X and g/h.");

    auto num_outputs             = g.shape<2>().hi[1] - g.shape<2>().lo[1] + 1;
    auto g_accessor              = g.read_accessor<double, 2>();
    auto h_accessor              = h.read_accessor<double, 2>();
    const auto& split_proposals  = context.inputs().at(3);
    auto split_proposals_shape   = split_proposals.shape<2>();
    auto split_proposal_accessor = split_proposals.read_accessor<T, 2>();
    EXPECT(X.code() == split_proposals.code(), "Expected same type for X and split proposals.");

    // Scalars
    auto learning_rate = context.scalars().at(0).value<double>();
    auto max_depth     = context.scalars().at(1).value<int>();
    auto random_seed   = context.scalars().at(2).value<uint64_t>();

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
      tree.leaf_value[{0, i}] = (-G / H) * learning_rate;
      tree.hessian[{0, i}]    = H;
    }

    // Begin building the tree
    std::vector<int32_t> positions(num_rows);
    for (int64_t depth = 0; depth < max_depth; ++depth) {
      GradientHistogram histogram(num_features, depth, num_outputs);
      for (int64_t i = 0; i < num_rows; i++) {
        for (int64_t j = 0; j < num_features; j++) {
          for (int64_t k = 0; k < num_outputs; ++k) {
            auto x    = X_accessor[{i, j}];
            bool left = x <= split_proposal_accessor[{depth, j}];
            histogram.Add(j, positions[i], k, GPair{g_accessor[{i, k}], h_accessor[{i, k}]}, left);
          }
        }
      }
      SumAllReduce(context,
                   reinterpret_cast<double*>(histogram.gradient_sums.ptr({0, 0, 0, 0})),
                   histogram.size * 2);
      // Find the best split
      double eps = 1e-5;
      for (int node_id = (1 << depth) - 1; node_id < (1 << (depth + 1)) - 1; node_id++) {
        double best_gain = 0;
        int best_feature = -1;
        for (int feature = 0; feature < num_features; feature++) {
          double gain = 0;
          for (int output = 0; output < num_outputs; ++output) {
            auto [G_L, H_L] = histogram.Get(feature, node_id, output, true);
            auto [G_R, H_R] = histogram.Get(feature, node_id, output, false);
            auto G          = G_L + G_R;
            auto H          = H_L + H_R;
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
          std::vector<double> hessian_left(num_outputs);
          std::vector<double> hessian_right(num_outputs);
          for (int output = 0; output < num_outputs; ++output) {
            auto [G_L, H_L]       = histogram.Get(best_feature, node_id, output, true);
            auto [G_R, H_R]       = histogram.Get(best_feature, node_id, output, false);
            left_leaf[output]     = -(G_L / (H_L + eps)) * learning_rate;
            right_leaf[output]    = -(G_R / (H_R + eps)) * learning_rate;
            hessian_left[output]  = H_L;
            hessian_right[output] = H_R;
          }
          if (hessian_left[0] <= 0.0 || hessian_right[0] <= 0.0) continue;
          tree.AddSplit(node_id,
                        best_feature,
                        split_proposal_accessor[{depth, best_feature}],
                        left_leaf,
                        right_leaf,
                        best_gain,
                        hessian_left,
                        hessian_right);
        }
      }

      // Update the positions
      for (int64_t i = 0; i < num_rows; i++) {
        int& pos = positions[i];
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

class BuildTreeTask : public Task<BuildTreeTask, BUILD_TREE> {
 public:
  static void cpu_variant(legate::TaskContext& context)
  {
    const auto& X = context.inputs().at(0);
    type_dispatch_float(X.code(), build_tree_fn{}, context);
  };
};
}  // namespace legateboost

namespace  // unnamed
{
static void __attribute__((constructor)) register_tasks(void)
{
  legateboost::BuildTreeTask::register_variants();
}
}  // namespace
