#include "legate_library.h"
#include "legateboost.h"

namespace legateboost {

class PredictTask : public Task<PredictTask, PREDICT> {
 public:
  static void cpu_variant(legate::TaskContext& context)
  {
    auto X_shape     = context.inputs().at(0).shape<2>();
    auto X_accessor  = context.inputs().at(0).read_accessor<double, 2>();
    auto leaf_value  = context.inputs().at(1).read_accessor<double, 1>();
    auto feature     = context.inputs().at(2).read_accessor<int32_t, 1>();
    auto split_value = context.inputs().at(3).read_accessor<double, 1>();

    auto pred = context.outputs().at(0).write_accessor<double, 1>();

    for (int64_t i = X_shape.lo[0]; i <= X_shape.hi[0]; i++) {
      int pos = 0;
      while (feature[pos] != -1) {
        auto x = X_accessor[{i, feature[pos]}];
        pos    = x <= split_value[pos] ? pos * 2 + 1 : pos * 2 + 2;
      }
      auto leaf = leaf_value[pos];
      pred[i]   = leaf;
    }
  }
};
}  // namespace legateboost

namespace  // unnamed
{
static void __attribute__((constructor)) register_tasks(void)
{
  legateboost::PredictTask::register_variants();
}
}  // namespace
