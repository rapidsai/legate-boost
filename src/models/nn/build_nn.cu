#include "build_nn.h"
#include <thrust/device_vector.h>
#include "cublas_v2.h"
#include <deque>

namespace legateboost {

namespace {
#define CUBLAS_ERROR(x)                                                            \
  do {                                                                             \
    if ((x) != CUBLAS_STATUS_SUCCESS) {                                            \
      printf("Error %s at %s:%d\n", cublasGetStatusString(x), __FILE__, __LINE__); \
      exit(EXIT_FAILURE);                                                          \
    }                                                                              \
  } while (0)

void SyncCPU(legate::TaskContext context)
{
  auto domain      = context.get_launch_domain();
  size_t num_ranks = domain.get_volume();
  if (num_ranks == 1) return;
  auto comm = context.communicator(1);
  std::vector<float> gather_result(num_ranks);
  auto comm_ptr = comm.get<legate::comm::coll::CollComm>();
  EXPECT(comm_ptr != nullptr, "CPU communicator is null.");
  float tmp;
  auto result = legate::comm::coll::collAllgather(
    &tmp, gather_result.data(), 1, legate::comm::coll::CollDataType::CollFloat, comm_ptr);
  EXPECT(result == legate::comm::coll::CollSuccess, "CPU communicator failed.");
}

// Store handles to legate and cublas
// Store information about the coefficient and bias sizes
class NNContext {
 public:
  cublasHandle_t handle;
  cudaStream_t stream;
  std::vector<std::array<int64_t, 2>> coefficient_extents;
  std::vector<std::array<int64_t, 2>> bias_extents;
  int64_t num_parameters;
  legate::TaskContext legate_context;
  template <typename T>
  NNContext(legate::TaskContext context,
            const std::vector<Matrix<T>>& coefficients,
            const std::vector<Matrix<T>>& bias,
            cudaStream_t stream)
    : stream(stream), legate_context(context)
  {
    CUBLAS_ERROR(cublasCreate(&handle));
    // Without syncronising, cublas creation can hang
    SyncCPU(context);
    CUBLAS_ERROR(cublasSetStream(handle, stream));
    cublasStatus_t status = cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);
    if (status != CUBLAS_STATUS_SUCCESS) {
      logger.print() << "WARNING: cuBLAS does not support Tensor cores!";
    }
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
void print(Matrix<T>& A, int64_t n)
{
  CHECK_CUDA(cudaDeviceSynchronize());
  std::vector<T> host_data(A.size());
  cudaMemcpy(host_data.data(), A.data, A.size() * sizeof(T), cudaMemcpyDeviceToHost);
  for (int i = 0; i < std::min(n, A.size()); i++) { std::cout << host_data[i] << " "; }
  std::cout << std::endl;
}

template <typename T>
Matrix<T> multiply(Matrix<T>& A, T scalar)
{
  auto result = Matrix<T>::Create(A.extent);
  auto stream = legate::cuda::StreamPool::get_stream_pool().get_stream();
  auto out    = result.data;
  auto in     = A.data;
  LaunchN(A.size(), stream, [=] __device__(int64_t idx) { out[idx] = in[idx] * scalar; });
  return result;
}

template <typename T>
Matrix<T> subtract(const Matrix<T>& A, const Matrix<T>& B)
{
  EXPECT(A.extent == B.extent, "Matrix dimensions must match");
  auto result     = Matrix<T>::Create(A.extent);
  auto stream     = legate::cuda::StreamPool::get_stream_pool().get_stream();
  auto in         = A.data;
  auto out        = result.data;
  auto other_data = B.data;
  LaunchN(A.size(), stream, [=] __device__(int64_t idx) { out[idx] = in[idx] - other_data[idx]; });
  return result;
}

template <typename T, typename T2>
void fill(Matrix<T>& A, T2 val)
{
  auto stream = legate::cuda::StreamPool::get_stream_pool().get_stream();
  LaunchN(A.size(), stream, [=] __device__(int64_t idx) { A.data[idx] = val; });
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
T eval_cost(NNContext* context,
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

  auto result               = legate::create_buffer<T>(1);
  size_t temp_storage_bytes = 0;
  cub::DeviceReduce::Sum(nullptr,
                         temp_storage_bytes,
                         cost_array.data,
                         result.ptr(0),
                         cost_array.size(),
                         context->stream);
  auto temp_storage = legate::create_buffer<int8_t>({temp_storage_bytes});
  cub::DeviceReduce::Sum(temp_storage.ptr(0),
                         temp_storage_bytes,
                         cost_array.data,
                         result.ptr(0),
                         cost_array.size(),
                         context->stream);
  SumAllReduce(context->legate_context, result.ptr(0), 1, context->stream);

  T cost;
  cudaMemcpyAsync(&cost, result.ptr(0), sizeof(T), cudaMemcpyDeviceToHost, context->stream);
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
Matrix<T> eval_cost_prime(NNContext* context, Matrix<T>& pred, Matrix<double>& g, Matrix<double>& h)
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
  fill(ones, 1.0);
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
Matrix<T> backward(NNContext* nn_context,
                   std::vector<Matrix<T>>& coefficients,
                   std::vector<Matrix<T>>& bias,
                   std::vector<Matrix<T>>& activations,
                   std::vector<Matrix<T>>& deltas,
                   Matrix<double>& g,
                   Matrix<double>& h,
                   std::size_t total_rows,
                   double alpha)
{
  auto grads = Matrix<T>::Create({nn_context->num_parameters, 1});
  fill(grads, 0.0);
  auto [coefficient_grads, bias_grads] = nn_context->Unpack(grads);
  forward(nn_context, coefficients, bias, activations);

  deltas.back() = eval_cost_prime(nn_context, activations.back(), g, h);
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
  SumAllReduce(nn_context->legate_context, grads.data, grads.size(), nn_context->stream);
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
std::tuple<T, T> line_search(NNContext* nn_context,
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
  T rho = 0.1;
  T c   = 1e-4;
  T t   = -c * vector_dot(nn_context, grad, direction);
  EXPECT(t >= 0, "Search direction is not a descent direction");
  auto coeff_proposal_storage = Matrix<T>::Create({nn_context->num_parameters, 1});
  fill(coeff_proposal_storage, 0.0);
  auto [coefficient_proposals, bias_proposals] = nn_context->Unpack(coeff_proposal_storage);
  update_coefficients(
    nn_context, coefficients, coefficient_proposals, bias, bias_proposals, direction, lr);
  forward(nn_context, coefficient_proposals, bias_proposals, activations);
  T new_cost =
    eval_cost(nn_context, activations.back(), g, h, coefficient_proposals, total_rows, alpha);

  while (cost - new_cost < lr * t && lr * rho > 1e-15) {
    lr *= rho;
    update_coefficients(
      nn_context, coefficients, coefficient_proposals, bias, bias_proposals, direction, lr);
    forward(nn_context, coefficient_proposals, bias_proposals, activations);
    new_cost =
      eval_cost(nn_context, activations.back(), g, h, coefficient_proposals, total_rows, alpha);
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
    if (s.size() == 0) { return multiply(grad, T(-1.0)); }
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
      return multiply(grad, T(-1.0));
    }
    return direction;
  }
};

struct build_nn_fn {
  template <typename T>
  void operator()(legate::TaskContext context)
  {
    auto [X_store, X_shape, X_accessor] = GetInputStore<T, 3>(context.input(0).data());
    auto [g_store, g_shape, g_accessor] = GetInputStore<double, 3>(context.input(1).data());
    auto [h_store, h_shape, h_accessor] = GetInputStore<double, 3>(context.input(2).data());
    EXPECT_AXIS_ALIGNED(0, X_shape, g_shape);
    EXPECT_AXIS_ALIGNED(0, g_shape, h_shape);
    EXPECT_AXIS_ALIGNED(1, g_shape, h_shape);

    auto total_rows  = context.scalar(0).value<int64_t>();
    double gtol      = context.scalar(1).value<double>();
    int32_t verbose  = context.scalar(2).value<int32_t>();
    int32_t m        = context.scalar(3).value<int32_t>();
    int32_t max_iter = context.scalar(4).value<int32_t>();
    double alpha     = context.scalar(5).value<double>();

    std::vector<Matrix<T>> coefficients;
    std::vector<Matrix<T>> bias;

    for (int i = 0; i < context.num_outputs(); i += 2) {
      coefficients.push_back(Matrix<T>::From2dOutputStore(context.output(i).data()));
      bias.push_back(Matrix<T>::From1dOutputStore(context.output(i + 1).data()));
    }

    NNContext nn_context(
      context, coefficients, bias, legate::cuda::StreamPool::get_stream_pool().get_stream());

    auto X           = Matrix<T>::Project3dStore(X_store, 2);
    Matrix<double> g = Matrix<double>::Project3dStore(g_store, 1);
    Matrix<double> h = Matrix<double>::Project3dStore(h_store, 1);

    std::vector<Matrix<T>> activations({X});
    std::vector<Matrix<T>> deltas;
    for (const auto& c : coefficients) {
      activations.push_back(Matrix<T>::Create({X.extent[0], c.extent[1]}));
      deltas.push_back(Matrix<T>::Create({X.extent[0], c.extent[1]}));
    }

    LBfgs<T> lbfgs(m, verbose);
    auto grad =
      backward(&nn_context, coefficients, bias, activations, deltas, g, h, total_rows, alpha);
    T grad_norm = vector_norm(&nn_context, grad);
    T cost      = eval_cost(&nn_context, activations.back(), g, h, coefficients, total_rows, alpha);

    LearningMonitor monitor(max_iter, verbose, gtol);

    while (!monitor.IsConverged(cost, grad_norm)) {
      auto direction      = lbfgs.GetDirection(&nn_context, grad);
      auto [lr, new_cost] = line_search(&nn_context,
                                        coefficients,
                                        bias,
                                        direction,
                                        grad,
                                        activations,
                                        deltas,
                                        g,
                                        h,
                                        total_rows,
                                        cost,
                                        alpha);
      cost                = new_cost;

      update_coefficients(&nn_context, coefficients, coefficients, bias, bias, direction, lr);

      auto new_grad =
        backward(&nn_context, coefficients, bias, activations, deltas, g, h, total_rows, alpha);

      lbfgs.Add(multiply(direction, lr), subtract(new_grad, grad));
      grad      = new_grad;
      grad_norm = vector_norm(&nn_context, grad);
    }
  }
};

}  // namespace

/*static*/ void BuildNNTask::gpu_variant(legate::TaskContext context)
{
  const auto& X = context.input(0).data();
  type_dispatch_float(X.code(), build_nn_fn(), context);
}
}  // namespace legateboost
