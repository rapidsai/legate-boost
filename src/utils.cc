#include "utils.h"
#include "core/comm/coll.h"

namespace legateboost {
Legion::Logger logger("legateboost");

void SumAllReduce(legate::TaskContext context, double* x, int count)
{
  auto domain      = context.get_launch_domain();
  size_t num_ranks = domain.get_volume();
  EXPECT(num_ranks == 1 || context.num_communicators() > 0,
         "Expected a CPU communicator for multi-rank task.");
  if (context.num_communicators() == 0) return;
  auto comm = context.communicator(0);
  std::vector<double> gather_result(num_ranks * count);
  auto result = legate::comm::coll::collAllgather(x,
                                                  gather_result.data(),
                                                  count,
                                                  legate::comm::coll::CollDataType::CollDouble,
                                                  comm.get<legate::comm::coll::CollComm>());
  EXPECT(result == legate::comm::coll::CollSuccess, "CPU communicator failed.");
  for (std::size_t j = 0; j < count; j++) { x[j] = 0.0; }
  for (std::size_t i = 0; i < num_ranks; i++) {
    for (std::size_t j = 0; j < count; j++) { x[j] += gather_result[i * count + j]; }
  }
}

}  // namespace legateboost
