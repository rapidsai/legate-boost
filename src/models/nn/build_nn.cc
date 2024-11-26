/* Copyright 2024 NVIDIA Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */
#include "build_nn.h"
#include <cblas.h>
#include <algorithm>
#include <tuple>
#include <vector>
#include <iostream>
#include <deque>
#include <utility>

namespace legateboost {

namespace {

// Store information about the coefficient and bias sizes
class NNContext {
 public:
  std::vector<std::array<int64_t, 2>> coefficient_extents;
  std::vector<std::array<int64_t, 2>> bias_extents;
  int64_t num_parameters{};
  legate::TaskContext legate_context;
  template <typename T>
  NNContext(legate::TaskContext context,
            const std::vector<Matrix<T>>& coefficients,
            const std::vector<Matrix<T>>& bias)
    : legate_context(context)
  {
    for (const auto& c : coefficients) {
      coefficient_extents.push_back(c.extent);
      num_parameters += c.size();
    }
    for (const auto& b : bias) {
      bias_extents.push_back(b.extent);
      num_parameters += b.size();
    }
  }
  template <typename T>
  std::tuple<std::vector<Matrix<T>>, std::vector<Matrix<T>>> Unpack(Matrix<T>& x)
  {
    std::vector<Matrix<T>> coefficients;
    std::vector<Matrix<T>> biases;
    std::size_t offset = 0;
    for (int i = 0; i < coefficient_extents.size(); i++) {
      auto size = narrow<std::size_t>(coefficient_extents.at(i)[0] * coefficient_extents.at(i)[1]);
      coefficients.push_back(Matrix<T>(x.data.subspan(offset, size), coefficient_extents.at(i)));
      offset += size;
    }
    for (int i = 0; i < bias_extents.size(); i++) {
      auto size = narrow<std::size_t>(bias_extents.at(i)[0] * bias_extents.at(i)[1]);
      biases.push_back(Matrix<T>(x.data.subspan(offset, size), bias_extents.at(i)));
      offset += biases.back().size();
    }
    return std::make_tuple(coefficients, biases);
  }
};

template <bool transpose_A = false, bool transpose_B = false, typename T1, typename T2, typename T3>
void dot(Matrix<T1>& A, Matrix<T2>& B, Matrix<T3>& C)
{
  if (A.size() == 0 || B.size() == 0) return;
  using T = typename std::remove_const<T1>::type;
  static_assert(std::is_same<T, typename std::remove_const<T2>::type>::value,
                "T1 and T2 must be the same type");
  static_assert(std::is_same<T, typename std::remove_const<T3>::type>::value,
                "T1 and T3 must be the same type");

  int m = transpose_B ? B.extent[0] : B.extent[1];
  int n = transpose_A ? A.extent[1] : A.extent[0];
  int k = transpose_A ? A.extent[0] : A.extent[1];

  T alpha = 1.0;
  T beta  = 0.0;

  auto op_A = transpose_A ? CblasTrans : CblasNoTrans;
  auto op_B = transpose_B ? CblasTrans : CblasNoTrans;

  int lda_B = transpose_B ? k : m;
  int lda_A = transpose_A ? n : k;
  int lda_C = m;

  if constexpr (std::is_same<T, double>::value) {
    cblas_dgemm(CblasColMajor,
                op_B,
                op_A,
                m,
                n,
                k,
                alpha,
                B.data.data(),
                lda_B,
                A.data.data(),
                lda_A,
                beta,
                C.data.data(),
                lda_C);
  } else {
    cblas_sgemm(CblasColMajor,
                op_B,
                op_A,
                m,
                n,
                k,
                alpha,
                B.data.data(),
                lda_B,
                A.data.data(),
                lda_A,
                beta,
                C.data.data(),
                lda_C);
  }
}

template <typename T>
void print(Matrix<T>& A, int64_t n)
{
  for (int i = 0; i < std::min(n, A.size()); i++) { std::cout << A.data[i] << " "; }
  std::cout << '\n';
}

template <typename T>
Matrix<T> multiply(Matrix<T>& A, T scalar)
{
  auto result = Matrix<T>::Create(A.extent);
  for (int i = 0; i < A.size(); i++) result.data[i] = A.data[i] * scalar;
  return result;
}

template <typename T>
Matrix<T> subtract(const Matrix<T>& A, const Matrix<T>& B)
{
  EXPECT(A.extent == B.extent, "Matrix dimensions must match");
  auto result = Matrix<T>::Create(A.extent);
  for (int i = 0; i < A.size(); i++) result.data[i] = A.data[i] - B.data[i];
  return result;
}

template <typename T, typename T2>
void fill(Matrix<T>& A, T2 val)
{
  for (auto& a : A.data) a = val;
}

template <typename T>
T vector_norm(Matrix<T>& A)
{
  T result = 0.0;
  if (A.size() == 0) return result;
  if constexpr (std::is_same<T, double>::value) {
    result = cblas_dnrm2(A.size(), A.data.data(), 1);
  } else {
    result = cblas_snrm2(A.size(), A.data.data(), 1);
  }
  return result;
}

template <typename T>
T vector_dot(Matrix<T>& A, Matrix<T>& B)
{
  T result = 0.0;
  if (A.size() == 0) return result;
  if constexpr (std::is_same<T, double>::value) {
    result = cblas_ddot(A.size(), A.data.data(), 1, B.data.data(), 1);
  } else {
    result = cblas_sdot(A.size(), A.data.data(), 1, B.data.data(), 1);
  }
  return result;
}

// bias vector is added to each row of matrix A
template <typename T>
void add_bias(Matrix<T>& A, Matrix<T>& bias)
{
  for (int i = 0; i < A.extent[0]; i++) {
    for (int j = 0; j < A.extent[1]; j++) { A[{i, j}] += bias.data[j]; }
  }
}

template <typename T>
void tanh(Matrix<T>& A)
{
  for (int i = 0; i < A.size(); i++) { A.data[i] = std::tanh(A.data[i]); }
}

template <typename T>
void tanh_prime(Matrix<T>& H, Matrix<T>& delta)
{
  for (int i = 0; i < H.size(); i++) { delta.data[i] *= 1 - H.data[i] * H.data[i]; }
}

template <typename T>
void apply_alpha(Matrix<T>& grad, Matrix<T>& coeff, double alpha)
{
  for (int i = 0; i < grad.size(); i++) { grad.data[i] += alpha * coeff.data[i]; }
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
  EXPECT(pred.extent == g.extent, "Preds not equal to gradient size");
  EXPECT(pred.extent == h.extent, "Preds not equal to gradient size");

  double sum = 0.0;
  for (int i = 0; i < pred.size(); i++) {
    T p     = pred.data[i];
    T g_val = g.data[i];
    T h_val = h.data[i];
    sum += p * (g_val + 0.5 * h_val * p);  // NOLINT(cppcoreguidelines-avoid-magic-numbers)
  }

  sum /= total_rows * pred.extent[1];

  SumAllReduce(context->legate_context, tcb::span<double>(&sum, 1));

  if (alpha > 0.0) {
    T L2 = 0.0;
    for (auto& c : coefficients) { L2 += vector_dot(c, c); }
    L2 = (0.5 * alpha) * L2 / total_rows;  // NOLINT(cppcoreguidelines-avoid-magic-numbers)

    sum += L2;
  }

  return sum;
}

template <typename T>
Matrix<T> eval_cost_prime(Matrix<T>& pred, Matrix<double>& g, Matrix<double>& h)
{
  Matrix<T> cost_prime = Matrix<T>::Create({pred.extent[0], pred.extent[1]});
  EXPECT(pred.extent == g.extent, "Preds not equal to gradient size");
  EXPECT(pred.extent == h.extent, "Preds not equal to gradient size");
  for (int i = 0; i < pred.size(); i++) {
    T p                = pred.data[i];
    T g_val            = g.data[i];
    T h_val            = h.data[i];
    cost_prime.data[i] = g_val + h_val * p;
  }
  return cost_prime;
}

template <typename T>
void bias_grad(Matrix<T>& delta, Matrix<T>& bias_grad)
{
  auto ones = Matrix<T>::Create({1, delta.extent[0]});
  fill(ones, 1.0);
  dot<false, false>(ones, delta, bias_grad);
}

template <typename T>
void forward(std::vector<Matrix<T>>& coefficients,
             std::vector<Matrix<T>>& biases,
             std::vector<Matrix<T>>& activations)
{
  for (int i = 0; i < coefficients.size(); i++) {
    dot(activations.at(i), coefficients.at(i), activations.at(i + 1));
    add_bias(activations.at(i + 1), biases.at(i));
    if (i < coefficients.size() - 1) tanh(activations.at(i + 1));
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
  forward(coefficients, bias, activations);

  deltas.back() = eval_cost_prime(activations.back(), g, h);
  dot<true, false>(activations.at(activations.size() - 2), deltas.back(), coefficient_grads.back());
  bias_grad(deltas.back(), bias_grads.back());

  for (int i = coefficients.size() - 1; i > 0; i--) {
    dot<false, true>(deltas.at(i), coefficients.at(i), deltas.at(i - 1));
    tanh_prime(activations.at(i), deltas.at(i - 1));
    dot<true, false>(activations.at(i - 1), deltas.at(i - 1), coefficient_grads.at(i - 1));
    bias_grad(deltas.at(i - 1), bias_grads.at(i - 1));
  }

  if (alpha > 0.0) {
    for (int i = 0; i < coefficients.size(); i++) {
      apply_alpha(coefficient_grads.at(i), coefficients.at(i), alpha);
    }
  }

  // Scale and allreduce gradients
  SumAllReduce(nn_context->legate_context, grads.data);
  for (auto& grad : grads.data) grad /= total_rows;
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
    for (int j = 0; j < coefficients.at(i).size(); j++) {
      coeff_out[j] = coeff[j] + lr * coeff_direction[j];
    }

    auto bias_data           = bias.at(i).data;
    auto bias_direction_data = bias_direction.at(i).data;
    auto bias_out_data       = bias_out.at(i).data;
    for (int j = 0; j < bias.at(i).size(); j++) {
      bias_out_data[j] = bias_data[j] + lr * bias_direction_data[j];
    }
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
  T lr        = 1.0;  // NOLINT(cppcoreguidelines-avoid-magic-numbers)
  const T rho = 0.1;
  const T c   = 1e-4;
  const T t   = -c * vector_dot(grad, direction);
  EXPECT(t >= 0, "Search direction is not a descent direction");
  auto coeff_proposal_storage = Matrix<T>::Create({nn_context->num_parameters, 1});
  fill(coeff_proposal_storage, 0.0);
  auto [coefficient_proposals, bias_proposals] = nn_context->Unpack(coeff_proposal_storage);
  update_coefficients(
    nn_context, coefficients, coefficient_proposals, bias, bias_proposals, direction, lr);
  forward(coefficient_proposals, bias_proposals, activations);
  T new_cost =
    eval_cost(nn_context, activations.back(), g, h, coefficient_proposals, total_rows, alpha);

  const double eps = 1e-15;
  while (cost - new_cost < lr * t && lr * rho > eps) {
    lr *= rho;
    update_coefficients(
      nn_context, coefficients, coefficient_proposals, bias, bias_proposals, direction, lr);
    forward(coefficient_proposals, bias_proposals, activations);
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
    s.push_back(std::move(x_diff));
    y.push_back(std::move(grad_diff));
  }
  Matrix<T> GetDirection(Matrix<T>& grad)
  {
    /*
    Chen, Weizhu, Zhenghao Wang, and Jingren Zhou. "Large-scale L-BFGS using MapReduce." Advances in
    neural information processing systems 27 (2014).
    */
    if (s.size() == 0) { return multiply(grad, T(-1.0)); }
    // Form a matrix
    auto b = Matrix<T>::Create({static_cast<int64_t>(s.size() + y.size() + 1), grad.size()});
    std::size_t offset = 0;
    for (int i = 0; i < s.size(); i++) {
      auto s_i = s.at(i);
      EXPECT(b.extent[1] == s_i.size(), "s_i size does not match");
      for (int j = 0; j < s_i.size(); j++) b.data[offset + j] = s_i.data[j];
      offset += b.extent[1];
    }
    for (int i = 0; i < y.size(); i++) {
      auto y_i = y.at(i);
      EXPECT(b.extent[1] == y_i.size(), "y_i size does not match");
      for (int j = 0; j < y_i.size(); j++) b.data[offset + j] = y_i.data[j];
      offset += b.extent[1];
    }
    for (int i = 0; i < grad.size(); i++) b.data[offset + i] = grad.data[i];

    auto B = Matrix<T>::Create({b.extent[0], b.extent[0]});
    dot<false, true>(b, b, B);

    // Clip values away from 0
    const double eps = 1e-15;
    for (int i = 0; i < B.size(); i++) {
      auto& val = B.data[i];
      if (val >= 0.0 && val < eps) { val = eps; }
      if (val < 0.0 && val > -eps) { val = -eps; }
    }

    auto delta = Matrix<T>::Create({B.extent[0], 1});
    auto alpha = Matrix<T>::Create({B.extent[0], 1});
    int l      = s.size();
    fill(delta, 0.0);
    delta.data[delta.size() - 1] = -1.0;
    for (int i = l - 1; i >= 0; i--) {
      T sum = 0.0;
      for (int j = 0; j < delta.size(); j++) { sum += B[{j, i}] * delta.data[j]; }
      alpha.data[i] = sum / B[{i, i + l}];
      delta.data[l + i] -= alpha.data[i];
    }

    T scalar = B[{l - 1, 2 * l - 1}] / B[{2 * l - 1, 2 * l - 1}];
    multiply(delta, scalar);

    for (int i = 0; i < l; i++) {
      T sum = 0.0;
      for (int j = 0; j < delta.size(); j++) { sum += B[{j, i + l}] * delta.data[j]; }
      T beta = sum / B[{i, i + l}];
      delta.data[i] += alpha.data[i] - beta;
    }

    auto direction = Matrix<T>::Create(grad.extent);
    dot<true, false>(delta, b, direction);

    T t = vector_dot(grad, direction);
    if (t >= 0) {
      if (verbose)
        std::cout << "Search direction is not a descent direction. Resetting LBFGS search." << '\n';
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
    // Openblas will try to multithread by default
    openblas_set_num_threads(1);

    auto [X_store, X_shape, X_accessor] = GetInputStore<T, 3>(context.input(0).data());
    auto [g_store, g_shape, g_accessor] = GetInputStore<double, 3>(context.input(1).data());
    auto [h_store, h_shape, h_accessor] = GetInputStore<double, 3>(context.input(2).data());
    EXPECT_AXIS_ALIGNED(0, X_shape, g_shape);
    EXPECT_AXIS_ALIGNED(0, g_shape, h_shape);
    EXPECT_AXIS_ALIGNED(1, g_shape, h_shape);

    // NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers)
    auto total_rows  = context.scalar(0).value<int64_t>();
    double gtol      = context.scalar(1).value<double>();
    int32_t verbose  = context.scalar(2).value<int32_t>();
    int32_t m        = context.scalar(3).value<int32_t>();
    int32_t max_iter = context.scalar(4).value<int32_t>();
    double alpha     = context.scalar(5).value<double>();
    // NOLINTEND(cppcoreguidelines-avoid-magic-numbers)

    std::vector<Matrix<T>> coefficients;
    std::vector<Matrix<T>> bias;

    for (int i = 0; i < context.num_outputs(); i += 2) {
      coefficients.push_back(Matrix<T>::From2dOutputStore(context.output(i).data()));
      bias.push_back(Matrix<T>::From1dOutputStore(context.output(i + 1).data()));
    }

    NNContext nn_context(context, coefficients, bias);

    Matrix<T> X      = Matrix<T>::Project3dStore(X_store, 2);
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
    T grad_norm = vector_norm(grad);
    T cost      = eval_cost(&nn_context, activations.back(), g, h, coefficients, total_rows, alpha);

    LearningMonitor monitor(max_iter, verbose, gtol);

    while (!monitor.IsConverged(cost, grad_norm)) {
      auto direction      = lbfgs.GetDirection(grad);
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
      grad_norm = vector_norm(grad);
    }
  }
};

}  // namespace
/*static*/ void BuildNNTask::cpu_variant(legate::TaskContext context)
{
  const auto& X = context.input(0).data();
  type_dispatch_float(X.code(), build_nn_fn(), context);
}
}  // namespace legateboost
namespace  // unnamed
{
static void __attribute__((constructor)) register_tasks(void)
{
  legateboost::BuildNNTask::register_variants();
}
}  // namespace
