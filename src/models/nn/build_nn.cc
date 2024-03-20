#include "build_nn.h"
namespace legateboost {
/*static*/ void BuildNNTask::cpu_variant(legate::TaskContext context)
{
  std::cout << "BuildNNTask::cpu_variant" << std::endl;
}
}  // namespace legateboost
namespace  // unnamed
{
static void __attribute__((constructor)) register_tasks(void)
{
  legateboost::BuildNNTask::register_variants();
}
}  // namespace
