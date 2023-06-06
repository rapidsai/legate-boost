#include "legate_library.h"
#include "legateboost.h"
#include "core/comm/coll.h"

namespace legateboost {

void expect(bool condition, std::string message, std::string file, int line)
{
  if (!condition) { throw std::runtime_error(file + "(" + std::to_string(line) + "): " + message); }
}
#define EXPECT(condition, message) (expect(condition, message, __FILE__, __LINE__))

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
  Tree(int max_depth)
  {
    int max_nodes = 1 << (max_depth + 1);
    leaf_value.resize(max_nodes);
    feature.resize(max_nodes, -1);
    split_value.resize(max_nodes);
  }
  void AddSplit(int node_id,
                int feature_id,
                double split_value,
                double left_leaf_value,
                double right_leaf_value)
  {
    feature[node_id]                      = feature_id;
    this->split_value[node_id]            = split_value;
    this->leaf_value[LeftChild(node_id)]  = left_leaf_value;
    this->leaf_value[RightChild(node_id)] = right_leaf_value;
  }
  static int LeftChild(int id) { return id * 2 + 1; }
  static int RightChild(int id) { return id * 2 + 2; }

  bool IsLeaf(int node_id) const { return feature[node_id] == -1; }

  std::vector<double> leaf_value;
  std::vector<int32_t> feature;
  std::vector<double> split_value;
};

template <typename T>
void WriteOutput(legate::Store& out, const std::vector<T>& x)
{
  EXPECT(out.shape<1>().volume() >= x.size(), "Output not large enough.");
  std::copy(x.begin(), x.end(), out.write_accessor<T, 1>().ptr(0));
}

void WriteTreeOutput(legate::TaskContext& context, const Tree& tree)
{
  WriteOutput(context.outputs().at(0), tree.leaf_value);
  WriteOutput(context.outputs().at(1), tree.feature);
  WriteOutput(context.outputs().at(2), tree.split_value);
}

struct GradientHistogram {
  legate::Buffer<double, 4> gradient_sums;
  int size;
  int num_features;
  int depth;
  GradientHistogram(int num_features, int depth)
    : num_features(num_features),
      depth(depth),
      size((1 << depth) * num_features * 2 * 2),
      gradient_sums(legate::create_buffer<double, 4>({1 << depth, num_features, 2, 2}))
  {
  }
  void Add(int feature, int position, double g, double h, bool left)
  {
    if (position < 0) return;
    int position_in_level = position - ((1 << depth) - 1);
    gradient_sums[{position_in_level, feature, left, 0}] += g;
    gradient_sums[{position_in_level, feature, left, 1}] += h;
  }
  std::pair<double, double> Get(int feature, int position, bool left)
  {
    int position_in_level = position - ((1 << depth) - 1);
    auto g                = gradient_sums[{position_in_level, feature, left, 0}];
    auto h                = gradient_sums[{position_in_level, feature, left, 1}];
    return {g, h};
  }
};

class BuildTreeTask : public Task<BuildTreeTask, BUILD_TREE> {
 public:
  static void cpu_variant(legate::TaskContext& context)
  {
    auto X_shape                 = context.inputs().at(0).shape<2>();
    auto X_accessor              = context.inputs().at(0).read_accessor<double, 2>();
    auto num_features            = X_shape.hi[1] - X_shape.lo[1] + 1;
    auto num_rows                = X_shape.hi[0] - X_shape.lo[0] + 1;
    auto g_ptr                   = context.inputs().at(1).read_accessor<double, 1>().ptr(0);
    auto h_ptr                   = context.inputs().at(2).read_accessor<double, 1>().ptr(0);
    auto split_proposals_shape   = context.inputs().at(3).shape<2>();
    auto split_proposal_accessor = context.inputs().at(3).read_accessor<double, 2>();

    // Scalars
    auto learning_rate = context.scalars().at(0).value<double>();
    auto max_depth     = context.scalars().at(1).value<int>();
    auto random_seed   = context.scalars().at(2).value<uint64_t>();

    Tree tree(max_depth);

    // Write the base score
    double G[2]  = {0};
    auto g_shape = context.inputs().at(1).shape<1>();
    auto h_shape = context.inputs().at(2).shape<1>();
    EXPECT(g_shape == h_shape, "");
    EXPECT(g_shape.volume() == num_rows, "Expected one value per row.");
    for (std::size_t i = g_shape.lo; i <= g_shape.hi; ++i) {
      G[0] += g_ptr[i];
      G[1] += h_ptr[i];
    }
    SumAllReduce(context, G, 2);
    tree.leaf_value[0] = -(G[0] / G[1]) * learning_rate;

    // Begin building the tree
    std::vector<int32_t> positions(num_rows);
    for (int64_t depth = 0; depth < max_depth; ++depth) {
      GradientHistogram histogram(num_features, depth);
      for (int64_t i = 0; i < num_rows; i++) {
        for (int64_t j = 0; j < num_features; j++) {
          auto x    = X_accessor[{i, j}];
          bool left = x <= split_proposal_accessor[{depth, j}];
          histogram.Add(j, positions[i], g_ptr[i], h_ptr[i], left);
        }
      }
      SumAllReduce(context, histogram.gradient_sums.ptr({0, 0, 0, 0}), histogram.size);

      // Find the best split
      double eps = 1e-5;
      for (int node_id = (1 << depth) - 1; node_id < (1 << (depth + 1)) - 1; node_id++) {
        double best_gain = 0;
        int best_feature = -1;
        for (int feature = 0; feature < num_features; feature++) {
          auto [G_L, H_L] = histogram.Get(feature, node_id, true);
          auto [G_R, H_R] = histogram.Get(feature, node_id, false);
          auto G          = G_L + G_R;
          auto H          = H_L + H_R;
          auto gain =
            0.5 * ((G_L * G_L) / (H_L + eps) + (G_R * G_R) / (H_R + eps) - (G * G) / (H + eps));
          if (gain > best_gain) {
            best_gain    = gain;
            best_feature = feature;
          }
        }
        if (best_gain > eps) {
          auto [G_L, H_L] = histogram.Get(best_feature, node_id, true);
          auto [G_R, H_R] = histogram.Get(best_feature, node_id, false);
          if (H_L <= 0.0 || H_R <= 0.0) continue;
          auto left_leaf  = -(G_L / H_L) * learning_rate;
          auto right_leaf = -(G_R / H_R) * learning_rate;
          tree.AddSplit(node_id,
                        best_feature,
                        split_proposal_accessor[{depth, best_feature}],
                        left_leaf,
                        right_leaf);
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
