#include <limits>
#include "legate_library.h"
#include "legateboost.h"
#include <kll_sketch.hpp>

namespace legateboost {

const int kSketchK = 1000;

void expect(bool condition, std::string message, std::string file, int line)
{
  if (!condition) { throw std::runtime_error(file + "(" + std::to_string(line) + "): " + message); }
}
#define EXPECT(condition, message) (expect(condition, message, __FILE__, __LINE__))

// Write a 4 byte header before each sketch with the length of the sketch
// Note: little/big endianess is not handled. Not suitable for serialising to disk.
void serialise_sketches(std::vector<datasketches::kll_sketch<float>>& sketches,
                        legate::Store& output)
{
  std::size_t size = 0;
  for (auto& sketch : sketches) {
    size += sketch.get_serialized_size_bytes();
    size += sizeof(uint32_t);
  }
  std::cout << "Writing " << size << "\n";
  auto result        = output.create_output_buffer<uint8_t, 1>(legate::Point<1>(size), true);
  auto ptr           = result.ptr(0);
  std::size_t offset = 0;
  for (auto& sketch : sketches) {
    auto serialised = sketch.serialize();
    uint32_t header = serialised.size();
    for (int i = 0; i < sizeof(uint32_t); i++) {
      ptr[offset + i] = reinterpret_cast<uint8_t*>(&header)[i];
    }
    offset += sizeof(uint32_t);
    std::copy(serialised.begin(), serialised.end(), ptr + offset);
    offset += serialised.size();
  }
  EXPECT(offset == size, "Serialisation error.");
}

std::vector<datasketches::kll_sketch<float>> deserialise_sketches(const uint8_t* in,
                                                                  std::size_t size)
{
  std::size_t offset = 0;
  std::vector<datasketches::kll_sketch<float>> sketches;
  std::cout << "Reading " << size << "\n";
  while (offset < size) {
    uint32_t header = 0;
    for (int i = 0; i < sizeof(uint32_t); i++) {
      reinterpret_cast<uint8_t*>(&header)[i] = in[offset + i];
    }
    offset += sizeof(uint32_t);
    sketches.emplace_back(datasketches::kll_sketch<float>::deserialize(in + offset, header));
    offset += header;
  }

  std::string error = "Deserialisation error. Expected " + std::to_string(size) + " bytes, got " +
                      std::to_string(offset) + ".";
  EXPECT(offset == size, error);
  return sketches;
}

class QuantileTask : public Task<QuantileTask, QUANTILE> {
 public:
  static void cpu_variant(legate::TaskContext& context)
  {
    auto& input     = context.inputs().at(0);
    auto in         = input.read_accessor<float, 2>();
    auto shape      = input.shape<2>();
    auto n_rows     = shape.hi[0] - shape.lo[0] + 1;
    auto n_features = context.scalars().at(0).value<std::size_t>();
    auto& output    = context.outputs().at(0);
    std::vector<datasketches::kll_sketch<float>> sketches(
      n_features, datasketches::kll_sketch<float>(kSketchK));
    for (size_t j = shape.lo[1]; j <= shape.hi[1]; ++j) {
      for (size_t i = shape.lo[0]; i <= shape.hi[0]; ++i) {
        sketches[j - shape.lo[1]].update(in[i][j]);
      }
    }
    serialise_sketches(sketches, output);
  }
};

class QuantileReduceTask : public Task<QuantileReduceTask, QUANTILE_REDUCE> {
 public:
  static void cpu_variant(legate::TaskContext& context)
  {
    auto& inputs = context.inputs();
    auto& output = context.outputs().at(0);
    std::vector<datasketches::kll_sketch<float>> sketches;
    for (int i = 0; i < inputs.size(); i++) {
      auto& input = inputs.at(i);
      auto shape  = input.shape<1>();
      auto in     = input.read_accessor<uint8_t, 1>();
      // auto in_sketches = deserialise_sketches(in.ptr(0), shape.volume());
      std::cout << "Partition i " << i << " size " << shape.volume() << "\n";
      // sketches.resize(in_sketches.size(), datasketches::kll_sketch<float>(kSketchK));
      // for (int j = 0; j < sketches.size(); j++) { sketches[j].merge(in_sketches[j]); }
    }
    serialise_sketches(sketches, output);
  }
};

std::vector<float> get_deduplicated_cuts(const datasketches::kll_sketch<float>& sketch,
                                         std::size_t n_bins)
{
  EXPECT(n_bins >= 2, "Need at least 2 bins.");
  EXPECT(n_bins < std::numeric_limits<uint16_t>::max(), "n_bins bigger than 2^16");
  std::size_t n = n_bins + 1;
  std::vector<float> cuts;
  cuts.reserve(n);
  cuts.push_back(-std::numeric_limits<float>::max());
  std::set<float> unique_values;
  for (int i = 1; i < n; i++) {
    unique_values.insert(sketch.get_quantile(double(i) / (n - 1), false));
  }
  // Throw away last value, use max instead
  unique_values.erase(std::prev(unique_values.end()));
  cuts.insert(cuts.end(), unique_values.begin(), unique_values.end());
  cuts.push_back(std::numeric_limits<float>::max());
  return cuts;
}

class QuantileOutputTask : public Task<QuantileOutputTask, QUANTILE_OUTPUT> {
 public:
  static void cpu_variant(legate::TaskContext& context)
  {
    auto& inputs            = context.inputs();
    auto& cuts_output       = context.outputs().at(0);
    auto& ptr_output        = context.outputs().at(1);
    auto& serialised_sketch = inputs.at(0);
    auto sketches = deserialise_sketches(serialised_sketch.read_accessor<uint8_t, 1>().ptr(0),
                                         serialised_sketch.shape<1>().volume());
    auto n_bins   = context.scalars().at(0).value<std::size_t>();
    // Return sparse matrix of cuts
    std::vector<float> cuts;
    std::vector<uint64_t> ptr(1, 0);
    for (const auto& sketch : sketches) {
      auto feature_cuts = get_deduplicated_cuts(sketch, n_bins);
      cuts.insert(cuts.end(), feature_cuts.begin(), feature_cuts.end());
      ptr.push_back(cuts.size());
    }

    // Return quantiles
    auto cuts_result =
      cuts_output.create_output_buffer<float, 1>(legate::Point<1>(cuts.size()), true);
    std::copy(cuts.begin(), cuts.end(), cuts_result.ptr(0));
    auto ptr_result =
      ptr_output.create_output_buffer<uint64_t, 1>(legate::Point<1>(ptr.size()), true);
    std::copy(ptr.begin(), ptr.end(), ptr_result.ptr(0));
  }
};

class QuantiseDataTask : public Task<QuantiseDataTask, QUANTISE_DATA> {
 public:
  static void cpu_variant(legate::TaskContext& context)
  {
    auto& quantised_output = context.outputs().at(0);
    auto out               = quantised_output.write_accessor<uint16_t, 2>();
    auto shape             = quantised_output.shape<2>();
    auto n_rows            = shape.hi[0] - shape.lo[0] + 1;
    auto n_features        = shape.hi[1] - shape.lo[1] + 1;

    auto& quantiles     = context.inputs().at(0);
    auto quantiles_iter = quantiles.read_accessor<float, 1>();

    auto& ptr     = context.inputs().at(1);
    auto ptr_iter = ptr.read_accessor<uint64_t, 1>();

    auto& input_data = context.inputs().at(2);
    auto in          = input_data.read_accessor<float, 2>();
    auto in_shape    = input_data.shape<2>();

    for (std::size_t i = 0; i < n_rows; i++) {
      for (std::size_t j = 0; j < n_features; j++) {
        out[legate::Point<2>(i, j)] = (std::lower_bound(quantiles_iter.ptr(ptr_iter[j]),
                                                        quantiles_iter.ptr(ptr_iter[j + 1]),
                                                        in[legate::Point<2>(i, j)]) -
                                       quantiles_iter.ptr(ptr_iter[j])) -
                                      1;
      }
    }
  }
};

}  // namespace legateboost

namespace  // unnamed
{
static void __attribute__((constructor)) register_tasks(void)
{
  legateboost::QuantileTask::register_variants();
  legateboost::QuantileReduceTask::register_variants();
  legateboost::QuantileOutputTask::register_variants();
  legateboost::QuantiseDataTask::register_variants();
}
}  // namespace
