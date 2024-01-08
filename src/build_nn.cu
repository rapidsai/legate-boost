#include "build_nn.h"
#include "utils.h"
#include <cuda/std/mdspan>
#include <thrust/device_vector.h>
namespace legateboost {
struct build_nn_fn {
  template <legate::Type::Code CODE>
  void operator()(legate::TaskContext context)
  {
    using T           = legate::type_of<CODE>;
    const auto& X     = context.input(0).data();
    auto X_shape      = X.shape<3>();
    auto X_accessor   = X.read_accessor<T, 3>();
    auto num_features = X_shape.hi[1] - X_shape.lo[1] + 1;
    auto num_rows     = X_shape.hi[0] - X_shape.lo[0] + 1;
    const auto& g     = context.input(1).data();
    const auto& h     = context.input(2).data();
    auto g_shape      = g.shape<3>();
    auto h_shape      = h.shape<3>();
    EXPECT_AXIS_ALIGNED(0, X_shape, g_shape);
    EXPECT_AXIS_ALIGNED(0, g_shape, h_shape);
    EXPECT_AXIS_ALIGNED(1, g_shape, h_shape);
    auto num_outputs = g_shape.hi[2] - g_shape.lo[2] + 1;
    auto g_accessor  = g.read_accessor<double, 3>();
    auto h_accessor  = h.read_accessor<double, 3>();
    std::vector<legate::Rect<2>> coefficient_shapes;
    std::vector<legate::AccessorRO<T, 2>> coefficient_accessors;
    std::vector<legate::Rect<1>> bias_shapes;
    std::vector<legate::AccessorRO<T, 1>> bias_accessors;

    constexpr auto dyn           = cuda::std::dynamic_extent;
    std::size_t activations_size = 0;
    for (int i = 3; i < context.num_inputs(); i += 2) {
      coefficient_shapes.push_back(context.input(i).shape<2>());
      coefficient_accessors.push_back(context.input(i).data().read_accessor<T, 2>());
      activations_size += num_rows * (coefficient_shapes.back().hi[1] + 1);
      bias_shapes.push_back(context.input(i + 1).shape<1>());
      bias_accessors.push_back(context.input(i + 1).data().read_accessor<T, 1>());
    }

    thrust::device_vector<legate::Rect<2>> coefficient_shapes_d(coefficient_shapes);
    thrust::device_vector<legate::Rect<1>> bias_shapes_d(bias_shapes);
    thrust::device_vector<legate::AccessorRO<T, 2>> coefficient_accessors_d(coefficient_accessors);
    thrust::device_vector<legate::AccessorRO<T, 1>> bias_accessors_d(bias_accessors);

    thrust::device_vector<T> activations(activations_size);
    using ext_t = cuda::std::extents<int, dyn, dyn>;
    std::vector<cuda::std::mdspan<T, ext_t>> activations_spans;
    std::size_t offset = 0;
    for (auto shape : coefficient_shapes) {
      activations_spans.push_back(
        cuda::std::mdspan<T, ext_t>(activations.data().get() + offset, num_rows, shape.hi[1] + 1));
      offset += shape.volume();
    }
  }
};

/*static*/ void BuildNNTask::gpu_variant(legate::TaskContext context)
{
  std::cout << "BuildNNTask::gpu_variant" << std::endl;
  const auto& X = context.input(0).data();
  type_dispatch_float(X.code(), build_nn_fn(), context);
}
}  // namespace legateboost
