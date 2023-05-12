#include "legate_library.h"
#include "legateboost.h"

namespace legateboost {

class BuildTreeTask : public Task<BuildTreeTask, BUILD_TREE> {
 public:
  static void cpu_variant(legate::TaskContext& context)
  {
    auto& input     = context.inputs().at(0);
    auto in         = input.read_accessor<float, 2>();
    auto shape      = input.shape<2>();
    auto n_rows     = shape.hi[0] - shape.lo[0] + 1;
    auto n_features = context.scalars().at(0).value<std::size_t>();
    auto& output    = context.outputs().at(0);
  };

}  // namespace legateboost

namespace  // unnamed
{
  static void __attribute__((constructor)) register_tasks(void)
  {
    legateboost::BuildTreeTask::register_variants();
  }
}  // namespace
