#include "legate_library.h"
#include "legateboost.h"

namespace legateboost {

class BuildTreeTask : public Task<BuildTreeTask, BUILD_TREE> {
 public:
  static void cpu_variant(legate::TaskContext& context)
  {
    auto n             = context.inputs().at(1).shape<1>().volume();
    auto g_ptr         = context.inputs().at(1).read_accessor<double, 1>().ptr(0);
    auto h_ptr         = context.inputs().at(2).read_accessor<double, 1>().ptr(0);
    auto learning_rate = context.scalars().at(0).value<double>();
    auto max_depth     = context.scalars().at(1).value<int>();
    auto random_seed   = context.scalars().at(2).value<uint64_t>();

    auto left_child =
      context.outputs().at(0).create_output_buffer<int32_t, 1>(legate::Point<1>(1), true);
    left_child.ptr(0)[0] = -1;
    auto right_child =
      context.outputs().at(1).create_output_buffer<int32_t, 1>(legate::Point<1>(1), true);
    right_child.ptr(0)[0] = -1;
    auto leaf_value =
      context.outputs().at(2).create_output_buffer<double, 1>(legate::Point<1>(1), true);
    leaf_value.ptr(0)[0] = 5.0;
    auto feature =
      context.outputs().at(3).create_output_buffer<int32_t, 1>(legate::Point<1>(1), true);
    feature.ptr(0)[0] = -1;
    auto split_value =
      context.outputs().at(4).create_output_buffer<float, 1>(legate::Point<1>(1), true);
    split_value.ptr(0)[0] = 0.0;
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
