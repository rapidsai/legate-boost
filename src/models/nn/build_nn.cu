#include "build_nn.h"
#include "../../cpp_utils/cpp_utils.h"
#include <thrust/device_vector.h>
#include "cublas_v2.h"
#include <deque>

namespace legateboost {

#define CUBLAS_ERROR(x)                                                            \
  do {                                                                             \
    if ((x) != CUBLAS_STATUS_SUCCESS) {                                            \
      printf("Error %s at %s:%d\n", cublasGetStatusString(x), __FILE__, __LINE__); \
      exit(EXIT_FAILURE);                                                          \
    }                                                                              \
  } while (0)

template <typename T>
struct Matrix {
  T* data;
  std::shared_ptr<legate::Buffer<T, 2>> buffer;
  std::array<int64_t, 2> extent;

  Matrix(T* data, std::array<int64_t, 2> extent) : data(data)
  {
    this->extent[0] = std::max(extent[0], 0L);
    this->extent[1] = std::max(extent[1], 0L);
  }

  __host__ __device__ std::int64_t size() const { return extent[0] * extent[1]; }

  __host__ __device__ T& operator[](std::array<int64_t, 2> idx) const
  {
    return data[idx[0] * extent[1] + idx[1]];
  }

  void Print(int64_t n) const
  {
    CHECK_CUDA(cudaDeviceSynchronize());
    std::vector<T> host_data(size());
    cudaMemcpy(host_data.data(), data, size() * sizeof(T), cudaMemcpyDeviceToHost);
    for (int i = 0; i < std::min(n, this->size()); i++) { std::cout << host_data[i] << " "; }
    std::cout << std::endl;
  }

  Matrix<T> operator*(T scalar) const
  {
    auto result = Matrix<T>::Create(extent);
    auto stream = legate::cuda::StreamPool::get_stream_pool().get_stream();
    auto out    = result.data;
    auto in     = data;
    LaunchN(size(), stream, [=] __device__(int64_t idx) { out[idx] = in[idx] * scalar; });

    return result;
  }

  Matrix<T> operator-(const Matrix<T>& other) const
  {
    EXPECT(extent == other.extent, "Matrix dimensions must match");
    auto result     = Matrix<T>::Create(extent);
    auto stream     = legate::cuda::StreamPool::get_stream_pool().get_stream();
    auto in         = data;
    auto out        = result.data;
    auto other_data = other.data;
    LaunchN(size(), stream, [=] __device__(int64_t idx) { out[idx] = in[idx] - other_data[idx]; });
    return result;
  }

  static Matrix<T> From1dStore(legate::PhysicalStore store)
  {
    auto shape = store.shape<1>();
    T* data    = store.read_accessor<T, 1, true>().ptr(shape.lo);
    return Matrix<T>(data, {shape.hi[0] - shape.lo[0] + 1, 1});
  }
  static Matrix<T> From1dOutputStore(legate::PhysicalStore store)
  {
    auto shape = store.shape<1>();
    T* data    = store.read_write_accessor<T, 1, true>().ptr(shape.lo);
    return Matrix<T>(data, {shape.hi[0] - shape.lo[0] + 1, 1});
  }

  static Matrix<T> From2dStore(legate::PhysicalStore store)
  {
    auto shape = store.shape<2>();
    T* data    = store.read_accessor<T, 2, true>().ptr(shape.lo);
    return Matrix<T>(data, {shape.hi[0] - shape.lo[0] + 1, shape.hi[1] - shape.lo[1] + 1});
  }
  static Matrix<T> From2dOutputStore(legate::PhysicalStore store)
  {
    auto shape = store.shape<2>();
    T* data    = store.read_write_accessor<T, 2, true>().ptr(shape.lo);
    return Matrix<T>(data, {shape.hi[0] - shape.lo[0] + 1, shape.hi[1] - shape.lo[1] + 1});
  }

  // Take a 3d store, remove a broadcast dimension and return a 2d Matrix
  static Matrix<T> Project3dStore(legate::PhysicalStore store, int broadcast_dimension)
  {
    auto shape = store.shape<3>();
    auto data  = store.read_accessor<T, 3, true>().ptr(shape.lo);
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
};

class NNContext {
 public:
  cublasHandle_t handle;
  cudaStream_t stream;
  std::vector<std::array<int64_t, 2>> coefficient_extents;
  std::vector<std::array<int64_t, 2>> bias_extents;
  int64_t num_parameters;
  template <typename T>
  NNContext(legate::TaskContext context,
            const std::vector<Matrix<T>>& coefficients,
            const std::vector<Matrix<T>>& bias,
            cudaStream_t stream)
    : stream(stream)
  {
    CUBLAS_ERROR(cublasCreate(&handle));
    // Without syncronising, cublas creation hangs
    auto temp = legate::create_buffer<float>({1});
    SumAllReduce(context, temp.ptr({0}), 1, stream);
    CHECK_CUDA(cudaStreamSynchronize(stream));
    CUBLAS_ERROR(cublasSetStream(handle, stream));
    num_parameters = 0;
    for (const auto& c : coefficients) {
      coefficient_extents.push_back(c.extent);
      num_parameters += c.size();
    }
    for (const auto& b : bias) {
      bias_extents.push_back(b.extent);
      num_parameters += b.size();
    }
  }
  ~NNContext() { CUBLAS_ERROR(cublasDestroy(handle)); }
  template <typename T>
  std::tuple<std::vector<Matrix<T>>, std::vector<Matrix<T>>> Unpack(Matrix<T>& x)
  {
    std::vector<Matrix<T>> coefficients;
    std::vector<Matrix<T>> biases;
    std::size_t offset = 0;
    for (int i = 0; i < coefficient_extents.size(); i++) {
      coefficients.push_back(Matrix<T>(x.data + offset, coefficient_extents.at(i)));
      offset += coefficients.back().size();
    }
    for (int i = 0; i < bias_extents.size(); i++) {
      biases.push_back(Matrix<T>(x.data + offset, bias_extents.at(i)));
      offset += biases.back().size();
    }
    return std::make_tuple(coefficients, biases);
  }
};

template <bool transpose_A = false, bool transpose_B = false, typename T1, typename T2, typename T3>
void dot(NNContext* context, Matrix<T1>& A, Matrix<T2>& B, Matrix<T3>& C)
{
  if (A.size() == 0 || B.size() == 0) return;
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
  if (A.size() == 0) return result;
  if constexpr (std::is_same<T, double>::value) {
    CUBLAS_ERROR(cublasDnrm2(context->handle, A.size(), A.data, 1, &result));
  } else {
    CUBLAS_ERROR(cublasSnrm2(context->handle, A.size(), A.data, 1, &result));
  }
  CHECK_CUDA(cudaStreamSynchronize(context->stream));
  return result;
}

template <typename T>
T vector_dot(NNContext* context, Matrix<T>& A, Matrix<T>& B)
{
  T result = 0.0;
  if (A.size() == 0) return result;
  if constexpr (std::is_same<T, double>::value) {
    CUBLAS_ERROR(cublasDdot(context->handle, A.size(), A.data, 1, B.data, 1, &result));
  } else {
    CUBLAS_ERROR(cublasSdot(context->handle, A.size(), A.data, 1, B.data, 1, &result));
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
void apply_alpha(NNContext* context, Matrix<T>& grad, Matrix<T>& coeff, double alpha)
{
  LaunchN(grad.size(), context->stream, [=] __device__(int64_t idx) {
    grad.data[idx] += alpha * coeff.data[idx];
  });
}

template <typename T>
T eval_cost(legate::TaskContext legate_context,
            NNContext* context,
            Matrix<T>& pred,
            Matrix<double>& g,
            Matrix<double>& h,
            std::vector<Matrix<T>>& coefficients,
            int64_t total_rows,
            double alpha)
{
  Matrix<T> cost_array = Matrix<T>::Create({pred.extent[0], pred.extent[1]});
  EXPECT(pred.extent == g.extent, "Preds not equal to gradient size");
  EXPECT(pred.extent == h.extent, "Preds not equal to gradient size");

  LaunchN(pred.size(), context->stream, [=] __device__(int64_t idx) {
    T p                  = pred.data[idx];
    T g_val              = g.data[idx];
    T h_val              = h.data[idx];
    cost_array.data[idx] = (p * (g_val + 0.5 * h_val * p) / (total_rows * pred.extent[1]));
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
  if (alpha > 0.0) {
    T L2 = 0.0;
    for (auto& c : coefficients) { L2 += vector_dot(context, c, c); }
    L2 = (0.5 * alpha) * L2 / total_rows;
    cost += L2;
  }
  return cost;
}

template <typename T>
Matrix<T> eval_cost_prime(legate::TaskContext legate_context,
                          NNContext* context,
                          Matrix<T>& pred,
                          Matrix<double>& g,
                          Matrix<double>& h)
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
void bias_grad(NNContext* context, Matrix<T>& delta, Matrix<T>& bias_grad)
{
  auto ones = Matrix<T>::Create({1, delta.extent[0]});
  LaunchN(ones.size(), context->stream, [=] __device__(int64_t idx) { ones.data[idx] = 1.0; });
  dot<false, false>(context, ones, delta, bias_grad);
}

template <typename T>
void forward(NNContext* nn_context,
             std::vector<Matrix<T>>& coefficients,
             std::vector<Matrix<T>>& biases,
             std::vector<Matrix<T>>& activations)
{
  for (int i = 0; i < coefficients.size(); i++) {
    dot(nn_context, activations.at(i), coefficients.at(i), activations.at(i + 1));
    add_bias(nn_context, activations.at(i + 1), biases.at(i));
    if (i < coefficients.size() - 1) tanh(nn_context, activations.at(i + 1));
  }
}

template <typename T>
Matrix<T> backward(legate::TaskContext context,
                   NNContext* nn_context,
                   std::vector<Matrix<T>>& coefficients,
                   std::vector<Matrix<T>>& bias,
                   std::vector<Matrix<T>>& activations,
                   std::vector<Matrix<T>>& deltas,
                   Matrix<double>& g,
                   Matrix<double>& h,
                   std::size_t total_rows,
                   double alpha)
{
  auto grads                           = Matrix<T>::Create({nn_context->num_parameters, 1});
  auto [coefficient_grads, bias_grads] = nn_context->Unpack(grads);
  forward(nn_context, coefficients, bias, activations);

  deltas.back() = eval_cost_prime(context, nn_context, activations.back(), g, h);
  dot<true, false>(
    nn_context, activations.at(activations.size() - 2), deltas.back(), coefficient_grads.back());
  bias_grad(nn_context, deltas.back(), bias_grads.back());

  for (int i = coefficients.size() - 1; i > 0; i--) {
    dot<false, true>(nn_context, deltas.at(i), coefficients.at(i), deltas.at(i - 1));
    tanh_prime(nn_context, activations.at(i), deltas.at(i - 1));
    dot<true, false>(
      nn_context, activations.at(i - 1), deltas.at(i - 1), coefficient_grads.at(i - 1));

    bias_grad(nn_context, deltas.at(i - 1), bias_grads.at(i - 1));
  }

  if (alpha > 0.0) {
    for (int i = 0; i < coefficients.size(); i++) {
      apply_alpha(nn_context, coefficient_grads.at(i), coefficients.at(i), alpha);
    }
  }

  // Scale and allreduce gradients
  SumAllReduce(context, grads.data, grads.size(), nn_context->stream);
  LaunchN(grads.size(), nn_context->stream, [=] __device__(int64_t idx) {
    grads.data[idx] /= total_rows;
  });
  return grads;
}

template <typename T>
void update_coefficients(NNContext* nn_context,
                         std::vector<Matrix<T>>& coefficients,
                         std::vector<Matrix<T>>& coefficients_out,
                         std::vector<Matrix<T>>& bias,
                         std::vector<Matrix<T>>& bias_out,
                         Matrix<T>& direction,
                         T lr)
{
  auto [coefficient_direction, bias_direction] = nn_context->Unpack(direction);

  for (int i = 0; i < coefficients.size(); i++) {
    auto coeff           = coefficients.at(i).data;
    auto coeff_direction = coefficient_direction.at(i).data;
    auto coeff_out       = coefficients_out.at(i).data;
    LaunchN(coefficients.at(i).size(), nn_context->stream, [=] __device__(int64_t idx) {
      coeff_out[idx] = coeff[idx] + lr * coeff_direction[idx];
    });

    auto bias_data           = bias.at(i).data;
    auto bias_direction_data = bias_direction.at(i).data;
    auto bias_out_data       = bias_out.at(i).data;
    LaunchN(bias.at(i).size(), nn_context->stream, [=] __device__(int64_t idx) {
      bias_out_data[idx] = bias_data[idx] + lr * bias_direction_data[idx];
    });
  }
}

template <typename T>
std::tuple<T, T> line_search(legate::TaskContext context,
                             NNContext* nn_context,
                             std::vector<Matrix<T>>& coefficients,
                             std::vector<Matrix<T>>& bias,
                             Matrix<T>& direction,
                             Matrix<T>& grad,
                             std::vector<Matrix<T>>& activations,
                             std::vector<Matrix<T>>& deltas,
                             Matrix<double>& g,
                             Matrix<double>& h,
                             std::size_t total_rows,
                             T cost,
                             double alpha)
{
  T lr  = 1.0;
  T rho = 0.5;
  T c   = 1e-4;
  T t   = -c * vector_dot(nn_context, grad, direction);
  EXPECT(t >= 0, "Search direction is not a descent direction");
  auto coeff_proposal_storage                  = Matrix<T>::Create({nn_context->num_parameters, 1});
  auto [coefficient_proposals, bias_proposals] = nn_context->Unpack(coeff_proposal_storage);
  update_coefficients(
    nn_context, coefficients, coefficient_proposals, bias, bias_proposals, direction, lr);
  forward(nn_context, coefficient_proposals, bias_proposals, activations);
  T new_cost = eval_cost(
    context, nn_context, activations.back(), g, h, coefficient_proposals, total_rows, alpha);

  while (cost - new_cost < lr * t && lr * rho > 1e-15) {
    lr *= rho;
    update_coefficients(
      nn_context, coefficients, coefficient_proposals, bias, bias_proposals, direction, lr);
    forward(nn_context, coefficient_proposals, bias_proposals, activations);
    new_cost = eval_cost(
      context, nn_context, activations.back(), g, h, coefficient_proposals, total_rows, alpha);
  }
  return std::make_tuple(lr, new_cost);
}

template <typename T>
class LBfgs {
  int m;
  std::deque<Matrix<T>> s;
  std::deque<Matrix<T>> y;
  bool verbose = false;

 public:
  LBfgs(int m, bool verbose) : m(m), verbose(verbose) {}
  void Add(Matrix<T>&& x_diff, Matrix<T>&& grad_diff)
  {
    if (s.size() >= m) {
      s.pop_front();
      y.pop_front();
    }
    s.push_back(x_diff);
    y.push_back(grad_diff);
  }
  Matrix<T> GetDirection(NNContext* context, Matrix<T>& grad)
  {
    /*
    Chen, Weizhu, Zhenghao Wang, and Jingren Zhou. "Large-scale L-BFGS using MapReduce." Advances in
    neural information processing systems 27 (2014).
    */
    if (s.size() == 0) { return grad * -1.0; }
    // Form a matrix
    auto b             = Matrix<T>::Create({int64_t(s.size() + y.size() + 1), grad.size()});
    std::size_t offset = 0;
    for (int i = 0; i < s.size(); i++) {
      auto s_i = s.at(i);
      EXPECT(b.extent[1] == s_i.size(), "s_i size does not match");
      CHECK_CUDA(cudaMemcpyAsync(b.data + offset,
                                 s_i.data,
                                 s_i.size() * sizeof(T),
                                 cudaMemcpyDeviceToDevice,
                                 context->stream));
      offset += b.extent[1];
    }
    for (int i = 0; i < y.size(); i++) {
      auto y_i = y.at(i);
      EXPECT(b.extent[1] == y_i.size(), "y_i size does not match");
      CHECK_CUDA(cudaMemcpyAsync(b.data + offset,
                                 y_i.data,
                                 y_i.size() * sizeof(T),
                                 cudaMemcpyDeviceToDevice,
                                 context->stream));
      offset += b.extent[1];
    }
    CHECK_CUDA(cudaMemcpyAsync(b.data + offset,
                               grad.data,
                               grad.size() * sizeof(T),
                               cudaMemcpyDeviceToDevice,
                               context->stream));

    auto B = Matrix<T>::Create({b.extent[0], b.extent[0]});
    dot<false, true>(context, b, b, B);

    // Clip values away from 0
    LaunchN(B.size(), context->stream, [=] __device__(int64_t idx) {
      auto& val = B.data[idx];
      if (val >= 0.0 && val < 1e-15) { val = 1e-15; }
      if (val < 0.0 && val > -1e-15) { val = -1e-15; }
    });

    auto delta = Matrix<T>::Create({B.extent[0], 1});
    auto alpha = Matrix<T>::Create({B.extent[0], 1});
    int l      = s.size();
    LaunchN(1, context->stream, [=] __device__(int64_t _) {
      for (int i = 0; i < delta.size() - 1; i++) { delta.data[i] = 0.0; }
      delta.data[delta.size() - 1] = -1.0;

      for (int i = l - 1; i >= 0; i--) {
        T sum = 0.0;
        for (int j = 0; j < delta.size(); j++) { sum += B[{j, i}] * delta.data[j]; }
        alpha.data[i] = sum / B[{i, i + l}];
        delta.data[l + i] -= alpha.data[i];
      }

      T scalar = B[{l - 1, 2 * l - 1}] / B[{2 * l - 1, 2 * l - 1}];
      for (int i = 0; i < delta.size(); i++) { delta.data[i] *= scalar; }

      for (int i = 0; i < l; i++) {
        T sum = 0.0;
        for (int j = 0; j < delta.size(); j++) { sum += B[{j, i + l}] * delta.data[j]; }
        T beta = sum / B[{i, i + l}];
        delta.data[i] += alpha.data[i] - beta;
      }
    });

    auto direction = Matrix<T>::Create(grad.extent);
    dot<true, false>(context, delta, b, direction);

    T t = vector_dot(context, grad, direction);
    if (t >= 0) {
      if (verbose)
        std::cout << "Search direction is not a descent direction. Resetting LBFGS search."
                  << std::endl;
      s.clear();
      y.clear();
      return grad * -1.0;
    }
    return direction;
  }
};

struct build_nn_fn {
  template <typename T>
  void operator()(legate::TaskContext context)
  {
    const auto& X     = context.input(0).data();
    auto X_shape      = X.shape<3>();
    auto X_accessor   = X.read_accessor<T, 3, true>();
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
    auto g_accessor  = g.read_accessor<double, 3, true>();
    auto h_accessor  = h.read_accessor<double, 3, true>();
    auto total_rows  = context.scalar(0).value<int64_t>();
    double gtol      = context.scalar(1).value<double>();
    int32_t verbose  = context.scalar(2).value<int32_t>();
    int32_t m        = context.scalar(3).value<int32_t>();
    int32_t max_iter = context.scalar(4).value<int32_t>();
    double alpha     = context.scalar(5).value<double>();

    auto stream = legate::cuda::StreamPool::get_stream_pool().get_stream();
    std::vector<Matrix<T>> coefficients;
    std::vector<Matrix<T>> bias;

    for (int i = 0; i < context.num_outputs(); i += 2) {
      coefficients.push_back(Matrix<T>::From2dOutputStore(context.output(i).data()));
      bias.push_back(Matrix<T>::From1dOutputStore(context.output(i + 1).data()));
    }

    NNContext nn_context(context, coefficients, bias, stream);

    auto X_Matrix           = Matrix<T>::Project3dStore(X, 2);
    Matrix<double> g_Matrix = Matrix<double>::Project3dStore(g, 1);
    Matrix<double> h_Matrix = Matrix<double>::Project3dStore(h, 1);

    std::vector<Matrix<T>> activations({X_Matrix});
    std::vector<Matrix<T>> deltas;
    for (const auto& c : coefficients) {
      activations.push_back(Matrix<T>::Create({num_rows, c.extent[1]}));
      deltas.push_back(Matrix<T>::Create({num_rows, c.extent[1]}));
    }

    LBfgs<T> lbfgs(m, verbose);
    auto grad = backward(context,
                         &nn_context,
                         coefficients,
                         bias,
                         activations,
                         deltas,
                         g_Matrix,
                         h_Matrix,
                         total_rows,
                         alpha);

    T cost = eval_cost(context,
                       &nn_context,
                       activations.back(),
                       g_Matrix,
                       h_Matrix,
                       coefficients,
                       total_rows,
                       alpha);

    for (int i = 0; i < max_iter; i++) {
      auto direction      = lbfgs.GetDirection(&nn_context, grad);
      auto [lr, new_cost] = line_search(context,
                                        &nn_context,
                                        coefficients,
                                        bias,
                                        direction,
                                        grad,
                                        activations,
                                        deltas,
                                        g_Matrix,
                                        h_Matrix,
                                        total_rows,
                                        cost,
                                        alpha);

      update_coefficients(&nn_context, coefficients, coefficients, bias, bias, direction, lr);

      auto new_grad = backward(context,
                               &nn_context,
                               coefficients,
                               bias,
                               activations,
                               deltas,
                               g_Matrix,
                               h_Matrix,
                               total_rows,
                               alpha);

      lbfgs.Add(direction * lr, new_grad - grad);
      grad        = new_grad;
      cost        = new_cost;
      T grad_norm = vector_norm(&nn_context, grad);
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
