#include <limits>
#include "legate_library.h"
#include "legategbm.h"
#include <kll_sketch.hpp>

namespace legategbm {

const int kSketchK = 1000;

class QuantileTask : public Task<QuantileTask, QUANTILE> {
 public:
  static void cpu_variant(legate::TaskContext& context)
  {
    auto& input      = context.inputs().at(0);
    auto in          = input.read_accessor<float, 1>();
    auto n      = input.shape<1>().volume();
    auto& output      = context.outputs().at(0);
    datasketches::kll_sketch<float> sketch(kSketchK);
    for (size_t i = 0; i < n; ++i) { sketch.update(in[i]);}
    auto data = sketch.serialize();
    auto result = output.create_output_buffer<uint8_t, 1>(legate::Point<1>(data.size()), true);
    std::copy(data.begin(),data.end(),result.ptr(0));
  }
};

class QuantileReduceTask : public Task<QuantileReduceTask, QUANTILE_REDUCE> {
 public:
  static void cpu_variant(legate::TaskContext& context)
  {
    auto& inputs = context.inputs();
    auto& output      = context.outputs().at(0);
    datasketches::kll_sketch<float> sketch(kSketchK);
    for (int i = 0; i < inputs.size(); i++) {
      auto& input = inputs.at(i);
      auto shape  = input.shape<1>();
      auto in     = input.read_accessor<float, 1>();
      sketch.merge(datasketches::kll_sketch<float>::deserialize(in.ptr(0), shape.volume()));
    }
    auto data   = sketch.serialize();
    auto result = output.create_output_buffer<int8_t, 1>(legate::Point<1>(data.size()), true);
    std::copy(data.begin(), data.end(), result.ptr(0));
  }
};

class QuantileOutputTask : public Task<QuantileOutputTask, QUANTILE_OUTPUT> {
 public:
  static void cpu_variant(legate::TaskContext& context)
  {
    auto& inputs                           = context.inputs();
    auto& quantile_output                  = context.outputs().at(0);
    auto& serialised_sketch                = inputs.at(0);
    datasketches::kll_sketch<float> sketch = datasketches::kll_sketch<float>::deserialize(
      serialised_sketch.read_accessor<float, 1>().ptr(0), serialised_sketch.shape<1>().volume());
    std::size_t n_out = std::max(context.scalars().at(0).value<std::size_t>() + 1, std::size_t(2));
    std::vector<float> temp;
    temp.reserve(n_out);
    temp.push_back(-std::numeric_limits<float>::max());
    // Deduplicate
    for (int i = 1; i < n_out - 1; i++) {
      auto q = sketch.get_quantile(double(i) / (n_out - 1), false);
      if (q != temp.back()) temp.push_back(q);
    }
    temp.push_back(std::numeric_limits<float>::max());
    
    // Return quantiles
    auto result = quantile_output.create_output_buffer<float, 1>(legate::Point<1>(temp.size()), true);
    std::copy(temp.begin(), temp.end(), result.ptr(0));
  }
};

class QuantiseDataTask : public Task<QuantiseDataTask, QUANTISE_DATA> {
 public:
  static void cpu_variant(legate::TaskContext& context)
  {
    auto& quantised_output = context.outputs().at(0);
    auto out               = quantised_output.write_accessor<uint16_t, 1>();
    auto n                 = quantised_output.shape<1>().volume();

    auto& quantiles     = context.inputs().at(0);
    auto quantiles_iter = quantiles.read_accessor<float, 1>();
    auto n_quantiles    = quantiles.shape<1>().volume();

    auto& input_data = context.inputs().at(1);
    auto in          = input_data.read_accessor<float, 1>();

    for (std::size_t i = 0; i < n; i++) {
      out[i] = (std::lower_bound(quantiles_iter.ptr(0), quantiles_iter.ptr(n_quantiles), in[i]) -
                quantiles_iter.ptr(0)) -
               1;
    }
  }
};
}  // namespace hello

namespace  // unnamed
{
static void __attribute__((constructor)) register_tasks(void)
{
  legategbm::QuantileTask::register_variants();
  legategbm::QuantileReduceTask::register_variants();
  legategbm::QuantileOutputTask::register_variants();
  legategbm::QuantiseDataTask::register_variants();
}
}  // namespace
