#include "build_nn.h"
#include "../../cpp_utils/cpp_utils.h"
#include <thrust/device_vector.h>
#include "cublas_v2.h"

namespace legateboost {

#define CUBLAS_ERROR(x)                                                            \
  do {                                                                             \
    if ((x) != CUBLAS_STATUS_SUCCESS) {                                            \
      printf("Error %s at %s:%d\n", cublasGetStatusString(x), __FILE__, __LINE__); \
      exit(EXIT_FAILURE);                                                          \
    }                                                                              \
  } while (0)

class NNContext {
 public:
  cublasHandle_t handle;
  cudaStream_t stream;
  NNContext(cudaStream_t stream) : stream(stream)
  {
    CUBLAS_ERROR(cublasCreate(&handle));
    CUBLAS_ERROR(cublasSetStream(handle, stream));
  }
  ~NNContext() { CUBLAS_ERROR(cublasDestroy(handle)); }
};

template <typename T>
struct Matrix {
  T* data;
  std::shared_ptr<legate::Buffer<T, 2>> buffer;
  std::array<int64_t, 2> extent;

  Matrix(T* data, std::array<int64_t, 2> extent) : data(data), extent(extent) {}

  std::size_t size() const { return extent[0] * extent[1]; }

  void Print(std::size_t n) const
  {
    CHECK_CUDA(cudaDeviceSynchronize());
    std::vector<T> host_data(size());
    cudaMemcpy(host_data.data(), data, size() * sizeof(T), cudaMemcpyDeviceToHost);
    for (int i = 0; i < std::min(n, this->size()); i++) { std::cout << host_data[i] << " "; }
    std::cout << std::endl;
  }

  static Matrix<T> From1dStore(legate::PhysicalStore store)
  {
    auto shape = store.shape<1>();
    T* data    = store.read_accessor<T, 1>().ptr(shape.lo);
    return Matrix<T>(data, {shape.hi[0] - shape.lo[0] + 1, 1});
  }
  static Matrix<T> From1dOutputStore(legate::PhysicalStore store)
  {
    auto shape = store.shape<1>();
    T* data    = store.read_write_accessor<T, 1>().ptr(shape.lo);
    return Matrix<T>(data, {shape.hi[0] - shape.lo[0] + 1, 1});
  }

  static Matrix<T> From2dStore(legate::PhysicalStore store)
  {
    auto shape = store.shape<2>();
    T* data    = store.read_accessor<T, 2>().ptr(shape.lo);
    return Matrix<T>(data, {shape.hi[0] - shape.lo[0] + 1, shape.hi[1] - shape.lo[1] + 1});
  }
  static Matrix<T> From2dOutputStore(legate::PhysicalStore store)
  {
    auto shape = store.shape<2>();
    T* data    = store.read_write_accessor<T, 2>().ptr(shape.lo);
    return Matrix<T>(data, {shape.hi[0] - shape.lo[0] + 1, shape.hi[1] - shape.lo[1] + 1});
  }

  // Take a 3d store, remove a broadcast dimension and return a 2d Matrix
  static Matrix<T> Project3dStore(legate::PhysicalStore store, int broadcast_dimension)
  {
    auto shape = store.shape<3>();
    auto data  = store.read_accessor<T, 3>().ptr(shape.lo);
    std::array<int64_t, 2> extent;
    if (broadcast_dimension == 0) {
      extent = {shape.hi[1] - shape.lo[1] + 1, shape.hi[2] - shape.lo[2] + 1};
    } else if (broadcast_dimension == 1) {
      extent = {shape.hi[0] - shape.lo[0] + 1, shape.hi[2] - shape.lo[2] + 1};
    } else {
      extent = {shape.hi[0] - shape.lo[0] + 1, shape.hi[1] - shape.lo[1] + 1};
    }
    return Matrix<T>(const_cast<T*>(data), extent);
  }

  static Matrix<T> Create(std::array<int64_t, 2> extent)
  {
    auto deleter = [](legate::Buffer<T, 2>* ptr) {
      ptr->destroy();
      delete ptr;
    };
    std::shared_ptr<legate::Buffer<T, 2>> buffer(
      new legate::Buffer<T, 2>(legate::create_buffer<T>(legate::Point<2>{extent[0], extent[1]})),
      deleter);
    auto t   = Matrix<T>(buffer->ptr({0, 0}), extent);
    t.buffer = buffer;
    return t;
  }

  static Matrix<T> Copy(const Matrix<T>& X)
  {
    auto Y = Create(X.extent);
    cudaMemcpy(Y.data, X.data, X.size() * sizeof(T), cudaMemcpyDeviceToDevice);
    return Y;
  }
};

template <bool transpose_A = false, bool transpose_B = false, typename T1, typename T2, typename T3>
void dot(NNContext* context, Matrix<T1>& A, Matrix<T2>& B, Matrix<T3>& C)
{
  using T = typename std::remove_const<T1>::type;
  static_assert(std::is_same<T, typename std::remove_const<T2>::type>::value,
                "T1 and T2 must be the same type");
  static_assert(std::is_same<T, typename std::remove_const<T3>::type>::value,
                "T1 and T3 must be the same type");

  // Arguments rearranged because data is row major and cublas expects column major
  // https://stackoverflow.com/questions/56043539/cublassgemm-row-major-multiplication

  int m = transpose_B ? B.extent[0] : B.extent[1];
  int n = transpose_A ? A.extent[1] : A.extent[0];
  int k = transpose_A ? A.extent[0] : A.extent[1];

  T alpha = 1.0;
  T beta  = 0.0;

  auto op_A = transpose_A ? CUBLAS_OP_T : CUBLAS_OP_N;
  auto op_B = transpose_B ? CUBLAS_OP_T : CUBLAS_OP_N;

  int lda_B = transpose_B ? k : m;
  int lda_A = transpose_A ? n : k;
  int lda_C = m;

  if constexpr (std::is_same<T, double>::value) {
    CUBLAS_ERROR(cublasDgemm(context->handle,
                             op_B,
                             op_A,
                             m,
                             n,
                             k,
                             &alpha,
                             B.data,
                             lda_B,
                             A.data,
                             lda_A,
                             &beta,
                             C.data,
                             lda_C));
  } else {
    CUBLAS_ERROR(cublasSgemm(context->handle,
                             op_B,
                             op_A,
                             m,
                             n,
                             k,
                             &alpha,
                             B.data,
                             lda_B,
                             A.data,
                             lda_A,
                             &beta,
                             C.data,
                             lda_C));
  }
}

template <typename T>
T vector_norm(NNContext* context, Matrix<T>& A)
{
  T result = 0.0;
  if constexpr (std::is_same<T, double>::value) {
    CUBLAS_ERROR(cublasDnrm2(context->handle, A.size(), A.data, 1, &result));
  } else {
    CUBLAS_ERROR(cublasSnrm2(context->handle, A.size(), A.data, 1, &result));
  }
  CHECK_CUDA(cudaStreamSynchronize(context->stream));
  return result;
}

// bias vector is added to each row of matrix A
template <typename T>
void add_bias(NNContext* context, Matrix<T>& A, Matrix<T>& bias)
{
  LaunchN(A.extent[0] * A.extent[1], context->stream, [=] __device__(int64_t idx) {
    int64_t j = idx % A.extent[1];
    A.data[idx] += bias.data[j];
  });
}

template <typename T>
void tanh(NNContext* context, Matrix<T>& A)
{
  LaunchN(A.size(), context->stream, [=] __device__(int64_t idx) {
    A.data[idx] = std::tanh(A.data[idx]);
  });
}

template <typename T>
void tanh_prime(NNContext* context, Matrix<T>& H, Matrix<T>& delta)
{
  LaunchN(H.size(), context->stream, [=] __device__(int64_t idx) {
    T val = H.data[idx];
    delta.data[idx] *= 1 - val * val;
  });
}

template <typename T>
T eval_cost(legate::TaskContext legate_context,
            NNContext* context,
            Matrix<T>& pred,
            Matrix<T>& g,
            Matrix<T>& h,
            int64_t total_rows)
{
  Matrix<T> cost_array = Matrix<T>::Create({pred.extent[0], pred.extent[1]});
  EXPECT(pred.extent == g.extent, "Preds not equal to gradient size");
  EXPECT(pred.extent == h.extent, "Preds not equal to gradient size");

  LaunchN(pred.size(), context->stream, [=] __device__(int64_t idx) {
    T p                  = pred.data[idx];
    T g_val              = g.data[idx];
    T h_val              = h.data[idx];
    cost_array.data[idx] = (p * (g_val + 0.5 * h_val * p) / total_rows);
  });

  auto result               = legate::create_buffer<T>({1});
  size_t temp_storage_bytes = 0;
  cub::DeviceReduce::Sum(nullptr,
                         temp_storage_bytes,
                         cost_array.data,
                         result.ptr({0}),
                         cost_array.size(),
                         context->stream);
  auto temp_storage = legate::create_buffer<int8_t>({temp_storage_bytes});
  cub::DeviceReduce::Sum(temp_storage.ptr({0}),
                         temp_storage_bytes,
                         cost_array.data,
                         result.ptr({0}),
                         cost_array.size(),
                         context->stream);
  SumAllReduce(legate_context, result.ptr({0}), 1, context->stream);

  T cost;
  cudaMemcpyAsync(&cost, result.ptr({0}), sizeof(T), cudaMemcpyDeviceToHost, context->stream);
  CHECK_CUDA(cudaStreamSynchronize(context->stream));
  return cost;
}

template <typename T>
Matrix<T> eval_cost_prime(legate::TaskContext legate_context,
                          NNContext* context,
                          Matrix<T>& pred,
                          Matrix<T>& g,
                          Matrix<T>& h)
{
  Matrix<T> cost_prime = Matrix<T>::Create({pred.extent[0], pred.extent[1]});
  EXPECT(pred.extent == g.extent, "Preds not equal to gradient size");
  EXPECT(pred.extent == h.extent, "Preds not equal to gradient size");

  LaunchN(pred.extent[0] * pred.extent[1], context->stream, [=] __device__(int64_t idx) {
    T p                  = pred.data[idx];
    T g_val              = g.data[idx];
    T h_val              = h.data[idx];
    cost_prime.data[idx] = g_val + h_val * p;
  });
  return cost_prime;
}

template <typename T>
void bias_grad(NNContext* context, Matrix<T>& delta, Matrix<T>& bias_grad, int64_t total_rows)
{
  cudaMemset(bias_grad.data, 0, bias_grad.extent[0] * sizeof(T));
  // TODO: make this some kind of reduction
  LaunchN(delta.extent[0] * delta.extent[1], context->stream, [=] __device__(int64_t idx) {
    int64_t j = idx % delta.extent[1];
    atomicAdd(&bias_grad.data[j], delta.data[idx] / total_rows);
  });
}

template <typename T>
void forward(NNContext* nn_context,
             std::vector<Matrix<T>>& coefficients,
             std::vector<Matrix<T>>& bias,
             std::vector<Matrix<T>>& activations)
{
  for (int i = 0; i < coefficients.size(); i++) {
    dot(nn_context, activations.at(i), coefficients.at(i), activations.at(i + 1));
    add_bias(nn_context, activations.at(i + 1), bias.at(i));
    if (i < coefficients.size() - 1) tanh(nn_context, activations.at(i + 1));
  }
}

template <typename T>
T backward(legate::TaskContext context,
           NNContext* nn_context,
           std::vector<Matrix<T>>& coefficients,
           std::vector<Matrix<T>>& bias,
           std::vector<Matrix<T>>& coefficient_grads,
           std::vector<Matrix<T>>& bias_grads,
           Matrix<T>& all_grads,
           std::vector<Matrix<T>>& activations,
           std::vector<Matrix<T>>& deltas,
           Matrix<T>& g,
           Matrix<T>& h,
           std::size_t total_rows)
{
  forward(nn_context, coefficients, bias, activations);

  T cost        = eval_cost(context, nn_context, activations.back(), g, h, total_rows);
  deltas.back() = eval_cost_prime(context, nn_context, activations.back(), g, h);
  dot<true, false>(
    nn_context, activations.at(activations.size() - 2), deltas.back(), coefficient_grads.back());
  bias_grad(nn_context, deltas.back(), bias_grads.back(), total_rows);

  for (int i = coefficients.size() - 1; i > 0; i--) {
    dot<false, true>(nn_context, deltas.at(i), coefficients.at(i), deltas.at(i - 1));
    tanh_prime(nn_context, activations.at(i), deltas.at(i - 1));
    dot<true, false>(
      nn_context, activations.at(i - 1), deltas.at(i - 1), coefficient_grads.at(i - 1));
    bias_grad(nn_context, deltas.at(i - 1), bias_grads.at(i - 1), total_rows);
  }

  // Scale and allreduce gradients
  SumAllReduce(context, all_grads.data, all_grads.size(), nn_context->stream);
  LaunchN(all_grads.size(), nn_context->stream, [=] __device__(int64_t idx) {
    all_grads.data[idx] /= total_rows;
  });
  return cost;
}

template <typename T>
void update_coefficients(NNContext* nn_context,
                         std::vector<Matrix<T>>& coefficients,
                         std::vector<Matrix<T>>& coefficient_grads,
                         std::vector<Matrix<T>>& coefficient_proposals,
                         std::vector<Matrix<T>>& bias,
                         std::vector<Matrix<T>>& bias_grads,
                         std::vector<Matrix<T>>& bias_proposals,
                         T alpha)
{
  for (int i = 0; i < coefficients.size(); i++) {
    auto coeff          = coefficients.at(i).data;
    auto grad           = coefficient_grads.at(i).data;
    auto coeff_proposal = coefficient_proposals.at(i).data;
    LaunchN(coefficients.at(i).size(), nn_context->stream, [=] __device__(int64_t idx) {
      coeff_proposal[idx] = coeff[idx] - alpha * grad[idx];
    });

    auto bias_data          = bias.at(i).data;
    auto bias_grad_data     = bias_grads.at(i).data;
    auto bias_proposal_data = bias_proposals.at(i).data;
    LaunchN(bias.at(i).size(), nn_context->stream, [=] __device__(int64_t idx) {
      bias_proposal_data[idx] = bias_data[idx] - alpha * bias_grad_data[idx];
    });
  }
}

template <typename T>
T line_search(legate::TaskContext context,
              NNContext* nn_context,
              std::vector<Matrix<T>>& coefficients,
              std::vector<Matrix<T>>& bias,
              std::vector<Matrix<T>>& coefficient_proposals,
              std::vector<Matrix<T>>& bias_proposals,
              std::vector<Matrix<T>>& coefficient_grads,
              std::vector<Matrix<T>>& bias_grads,
              Matrix<T>& all_grads,
              std::vector<Matrix<T>>& activations,
              std::vector<Matrix<T>>& deltas,
              Matrix<T>& g,
              Matrix<T>& h,
              std::size_t total_rows,
              T cost)
{
  T alpha = 1.0;
  T rho   = 0.5;
  T c     = 1e-4;
  // Direction here is negative gradient
  T norm = vector_norm(nn_context, all_grads);
  T t    = -c * norm * norm;

  update_coefficients(nn_context,
                      coefficients,
                      coefficient_grads,
                      coefficient_proposals,
                      bias,
                      bias_grads,
                      bias_proposals,
                      alpha);
  forward(nn_context, coefficient_proposals, bias_proposals, activations);
  T new_cost = eval_cost(context, nn_context, activations.back(), g, h, total_rows);

  while (cost - new_cost < alpha * t && alpha * rho > 1e-15) {
    alpha *= rho;
    update_coefficients(nn_context,
                        coefficients,
                        coefficient_grads,
                        coefficient_proposals,
                        bias,
                        bias_grads,
                        bias_proposals,
                        alpha);
    forward(nn_context, coefficient_proposals, bias_proposals, activations);
    new_cost = eval_cost(context, nn_context, activations.back(), g, h, total_rows);
  }
  update_coefficients(
    nn_context, coefficients, coefficient_grads, coefficients, bias, bias_grads, bias, alpha);
  return new_cost;
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
    auto total_rows  = context.scalar(0).value<int64_t>();
    double gtol      = context.scalar(1).value<double>();
    int32_t verbose  = context.scalar(2).value<int32_t>();

    auto stream = legate::cuda::StreamPool::get_stream_pool().get_stream();
    NNContext nn_context(stream);
    std::vector<Matrix<T>> coefficients;
    std::vector<Matrix<T>> bias;

    std::size_t buffer_size = 0;
    for (int i = 0; i < context.num_outputs(); i += 2) {
      coefficients.push_back(Matrix<T>::From2dOutputStore(context.output(i).data()));
      buffer_size += coefficients.back().size();
      bias.push_back(Matrix<T>::From1dOutputStore(context.output(i + 1).data()));
      buffer_size += bias.back().size();
    }

    auto grad_storage           = Matrix<T>::Create({int64_t(buffer_size), 1});
    auto coeff_proposal_storage = Matrix<T>::Create({int64_t(buffer_size), 1});
    std::vector<Matrix<T>> coefficient_grads;
    std::vector<Matrix<T>> bias_grads;
    std::vector<Matrix<T>> coefficient_proposals;
    std::vector<Matrix<T>> bias_proposals;
    std::size_t offset = 0;
    for (int i = 0; i < context.num_outputs(); i += 2) {
      coefficient_grads.push_back(
        Matrix<T>(grad_storage.data + offset, coefficients.at(i / 2).extent));
      coefficient_proposals.push_back(
        Matrix<T>(coeff_proposal_storage.data + offset, coefficients.at(i / 2).extent));
      offset += coefficients.at(i / 2).size();
      bias_grads.push_back(Matrix<T>(grad_storage.data + offset, bias.at(i / 2).extent));
      bias_proposals.push_back(
        Matrix<T>(coeff_proposal_storage.data + offset, bias.at(i / 2).extent));
      offset += bias.at(i / 2).size();
    }

    auto X_Matrix = Matrix<T>::Project3dStore(X, 2);
    auto g_Matrix = Matrix<T>::Project3dStore(g, 1);
    auto h_Matrix = Matrix<T>::Project3dStore(h, 1);

    std::vector<Matrix<T>> activations({X_Matrix});
    std::vector<Matrix<T>> deltas;
    for (const auto& c : coefficients) {
      activations.push_back(Matrix<T>::Create({num_rows, c.extent[1]}));
      deltas.push_back(Matrix<T>::Create({num_rows, c.extent[1]}));
    }

    int max_iters = 100;
    T cost        = backward(context,
                      &nn_context,
                      coefficients,
                      bias,
                      coefficient_grads,
                      bias_grads,
                      grad_storage,
                      activations,
                      deltas,
                      g_Matrix,
                      h_Matrix,
                      total_rows);
    T grad_norm   = vector_norm(&nn_context, grad_storage);
    for (int i = 0; i < max_iters; i++) {
      cost = line_search(context,
                         &nn_context,
                         coefficients,
                         bias,
                         coefficient_proposals,
                         bias_proposals,
                         coefficient_grads,
                         bias_grads,
                         grad_storage,
                         activations,
                         deltas,
                         g_Matrix,
                         h_Matrix,
                         total_rows,
                         cost);
      backward(context,
               &nn_context,
               coefficients,
               bias,
               coefficient_grads,
               bias_grads,
               grad_storage,
               activations,
               deltas,
               g_Matrix,
               h_Matrix,
               total_rows);
      T grad_norm = vector_norm(&nn_context, grad_storage);
      if (verbose && i % verbose == 0)
        std::cout << "L-BFGS Iteration: " << i << " Cost: " << cost << " Grad Norm: " << grad_norm
                  << std::endl;
      if (grad_norm < gtol) break;
    }
  }
};

/*static*/ void BuildNNTask::gpu_variant(legate::TaskContext context)
{
  const auto& X = context.input(0).data();
  type_dispatch_float(X.code(), build_nn_fn(), context);
}
}  // namespace legateboost
