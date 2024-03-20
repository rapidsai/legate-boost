#include "build_nn.h"
#include "../../cpp_utils/cpp_utils.h"
#include <thrust/device_vector.h>
#include <cutensor.h>

namespace legateboost {

// Handle cuTENSOR errors
#define HANDLE_ERROR(x)                                                        \
  {                                                                            \
    const auto err = x;                                                        \
    if (err != CUTENSOR_STATUS_SUCCESS) {                                      \
      printf("Error: %s in line %d\n", cutensorGetErrorString(err), __LINE__); \
      exit(-1);                                                                \
    }                                                                          \
  }

class CutensorContext {
 public:
  cutensorHandle_t* handle;
  legate::Buffer<int8_t, 1> workspace;
  std::size_t workspace_size = 0;
  cudaStream_t stream;
  CutensorContext(cudaStream_t stream) : stream(stream) { HANDLE_ERROR(cutensorCreate(&handle)); }
  ~CutensorContext()
  {
    HANDLE_ERROR(cutensorDestroy(handle));
    if (workspace_size > 0) { workspace.destroy(); }
  }
  void* AllocateWorkspace(size_t size)
  {
    if (size > workspace_size) {
      if (workspace_size == 0) { workspace.destroy(); }
      workspace      = legate::create_buffer<int8_t, 1>(legate::Point<1>{size});
      workspace_size = size;
    }
    return workspace.ptr({0});
  }
};

template <typename T>
struct Tensor {
  T* data;
  std::optional<legate::Buffer<T, 2>> buffer;
  cutensorTensorDescriptor_t desc;
  uint32_t alignment_requirement;
  std::vector<int64_t> extent;

  Tensor(CutensorContext* context, T* data, const std::vector<int64_t>& extent)
    : data(data), extent(extent)
  {
    // Create Tensor Descriptors
    cudaDataType_t type         = (std::is_same<T, float>::value) ? CUDA_R_32F : CUDA_R_64F;
    std::vector<int64_t> stride = {extent[1], 1};

    HANDLE_ERROR(cutensorInitTensorDescriptor(
      context->handle, &desc, 2, extent.data(), stride.data(), type, CUTENSOR_OP_IDENTITY));
    HANDLE_ERROR(
      cutensorGetAlignmentRequirement(context->handle, data, &desc, &alignment_requirement));
  }

  static Tensor<T> From1dStore(CutensorContext* context, legate::PhysicalStore store)
  {
    auto shape                  = store.shape<1>();
    T* data                     = store.read_accessor<T, 1>().ptr(shape.lo);
    std::vector<int64_t> extent = {shape.hi[0] - shape.lo[0] + 1};
    return Tensor<T>(context, data, extent);
  }

  static Tensor<T> From2dStore(CutensorContext* context, legate::PhysicalStore store)
  {
    auto shape                  = store.shape<2>();
    T* data                     = store.read_accessor<T, 2>().ptr(shape.lo);
    std::vector<int64_t> extent = {shape.hi[0] - shape.lo[0] + 1, shape.hi[1] - shape.lo[1] + 1};
    return Tensor<T>(context, data, extent);
  }

  // Take a 3d store, remove a broadcast dimension and return a 2d tensor
  static Tensor<T> Project3dStore(CutensorContext* context,
                                  legate::PhysicalStore store,
                                  int broadcast_dimension)
  {
    auto shape = store.shape<3>();
    auto data  = store.read_accessor<T, 3>().ptr(shape.lo);
    std::vector<int64_t> extent;
    for (int i = 0; i < 3; i++) {
      if (i != broadcast_dimension) { extent.emplace_back(shape.hi[i] - shape.lo[i] + 1); }
    }
    return Tensor<T>(context, data, extent);
  }

  static Tensor<T> Create(CutensorContext* context, const std::vector<int64_t>& extent)
  {
    auto buffer = legate::create_buffer<T>(legate::Point<2>{extent[0], extent[1]});
    auto t      = Tensor<T>(context, buffer.ptr({0, 0}), extent);
    t.buffer    = buffer;
    return t;
  }

  ~Tensor()
  {
    if (buffer) { buffer->destroy(); }
  }
};

template <typename T>
void dot(CutensorContext* context, const Tensor<const T>& A, const Tensor<const T>& B, Tensor<T>& C)
{
  // Create Contraction Descriptor
  std::vector<int> modeA{'m', 'h', 'k', 'n'};
  std::vector<int> modeB{'u', 'k', 'v', 'h'};
  std::vector<int> modeC{'m', 'u', 'n', 'v'};
  cutensorContractionDescriptor_t desc;
  cutensorComputeType_t typeCompute =
    (std::is_same<T, float>::value) ? CUTENSOR_COMPUTE_32F : CUTENSOR_COMPUTE_64F;
  HANDLE_ERROR(cutensorInitContractionDescriptor(context->handle,
                                                 &desc,
                                                 &A.desc,
                                                 modeA.data(),
                                                 A.alignment_requirement,
                                                 &B.desc,
                                                 modeB.data(),
                                                 B.alignment_requirement,
                                                 &C.desc,
                                                 modeC.data(),
                                                 C.alignment_requirement,
                                                 &C.desc,
                                                 modeC.data(),
                                                 C.alignment_requirement,
                                                 typeCompute));

  // Set the algorithm to use
  cutensorContractionFind_t find;
  HANDLE_ERROR(cutensorInitContractionFind(context->handle, &find, CUTENSOR_ALGO_DEFAULT));

  // Query workspace
  size_t worksize = 0;
  HANDLE_ERROR(cutensorContractionGetWorkspaceSize(
    context->handle, &desc, &find, CUTENSOR_WORKSPACE_RECOMMENDED, &worksize));

  // Allocate workspace
  void* work = context->AllocateWorkspace(worksize);

  // Create Contraction Plan
  cutensorContractionPlan_t plan;
  HANDLE_ERROR(cutensorInitContractionPlan(context->handle, &plan, &desc, &find, worksize));

  // Execute the tensor contraction
  T alpha = 1.0;
  T beta  = 0.0;
  HANDLE_ERROR(cutensorContraction(context->handle,
                                   &plan,
                                   (void*)&alpha,
                                   A.data,
                                   B.data,
                                   (void*)&beta,
                                   C.data,
                                   C.data,
                                   work,
                                   worksize,
                                   context->stream));
}

struct build_nn_fn {
  template <typename T>
  void operator()(legate::TaskContext context)
  {
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

    auto stream = legate::cuda::StreamPool::get_stream_pool().get_stream();
    CutensorContext ct_context(stream);
    std::vector<Tensor<const T>> coefficients;
    std::vector<Tensor<const T>> bias;

    for (int i = 3; i < context.num_inputs(); i += 2) {
      coefficients.push_back(Tensor<const T>::From2dStore(&ct_context, context.input(i).data()));
      // bias.push_back(Tensor<const T>::From1dStore(&ct_context, context.input(i + 1).data()));
    }

    /*
    auto X_tensor = Tensor<const T>::Project3dStore(&ct_context, X, 2);
    std::vector<Tensor<T>> activations;
    for(const auto &c: coefficients) {
      activations.push_back(Tensor<T>::Create(&ct_context, {num_rows, c.extent[1]}));
    }


    dot(&ct_context, X_tensor, coefficients.at(0), activations.at(0));
    for (int i = 1; i < coefficient_shapes.size(); i++) {
      dot(activations[i - 1].ptr(), activations_shapes[i - 1], coefficient_accessors[i].ptr(),
    coefficient_shapes[i], activations[i].ptr(), activations_shapes[i]);
      // TODO: Bias

      if (i < coefficient_shapes.size() - 1) {
        tanh(activations[i].ptr(), activations_shapes[i]);
      }

    }
    */
  }
};

/*static*/ void BuildNNTask::gpu_variant(legate::TaskContext context)
{
  std::cout << "BuildNNTask::gpu_variant" << std::endl;
  const auto& X = context.input(0).data();
  type_dispatch_float(X.code(), build_nn_fn(), context);
}
}  // namespace legateboost
