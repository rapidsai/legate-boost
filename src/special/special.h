/* Copyright 2024, NVIDIA Corporation
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
 */
#pragma once

#include <math.h>
#include <thrust/iterator/counting_iterator.h>  // for make_counting_iterator
#include <thrust/for_each.h>                    // for for_each_n
#include <legate/task/task_context.h>           // for TaskContext
#include <legate/task/exception.h>              // for TaskException
#include <legate/data/physical_array.h>         // for PhysicalArray
#include <cstdint>                              // for int32_t
#include <cmath>                                // for lgamma, erf
#include <type_traits>                          // for is_same_v
#include <limits>
#include <tuple>
#include "legateboost.h"     // for ERF
#include "legate_library.h"  // for Task
#include "../cpp_utils/cpp_utils.h"

namespace legateboost {
/*
 * Code from PyTorch, which comes from cephes. See thirdparty/LICENSES/LICENSE.pytorch
 */
// M_PI macro in glibc, translated into C++ template variable.
inline constexpr double m_PI = 3.14159265358979323846;

/*
 * This function is derived from the implementation of the digamma function in the Cephes Math
 * Library. See note [3-Clause BSD License for the Cephes Math Library].
 *
 * Evaluates polynomial of degree N:
 *
 *                     2          N
 * y  =  C  + C x + C x  +...+ C x
 *        0    1     2          N
 *
 * Coefficients are stored in reverse order:
 *
 * coef[0] = C  , ..., coef[N] = C  .
 *            N                   0
 */
template <typename T>
__host__ __device__ inline T polevl(const T x, const T A[], size_t len)
{
  T result = 0;
  for (size_t i = 0; i <= len; i++) { result = result * x + A[i]; }
  return result;
}

/*
 * This function is derived from the implementation of the digamma function in the Cephes Math
 * Library. See note [3-Clause BSD License for the Cephes Math Library].
 */
__host__ __device__ inline double calc_digamma(double x)
{
  constexpr double PSI_10 = 2.25175258906672110764;
  if (x == 0) {
    // As per C++ standard for gamma related functions and SciPy,
    // If the argument is ±0, ±∞ is returned
    return std::copysign(INFINITY, -x);
  }

  bool x_is_integer = x == trunc(x);
  if (x < 0) {
    if (x_is_integer) {
      // As per C++ standard for gamma related functions and SciPy,
      // If the argument is a negative integer, NaN is returned
      return std::numeric_limits<double>::quiet_NaN();
    }
    // Extracts the fractional part of x as r, since tan(pi * r) is more numerically
    // accurate than tan(pi * x). While these operations are mathematically equivalent
    // since both x and r are in radians and tan() has a periodicity of pi, in practice
    // the computation of pi * x is a source of error (when |x| > 1).
    double q = NAN, r = NAN;
    r = std::modf(x, &q);
    return calc_digamma(1 - x) - m_PI / std::tan(m_PI * r);
  }

  // Push x to be >= 10
  double result = 0;
  while (x < 10) {
    result -= 1 / x;
    x += 1;
  }
  if (x == 10) { return result + PSI_10; }

  // Compute asymptotic digamma
  constexpr double A[] = {
    8.33333333333333333333E-2,
    -2.10927960927960927961E-2,
    7.57575757575757575758E-3,
    -4.16666666666666666667E-3,
    3.96825396825396825397E-3,
    -8.33333333333333333333E-3,
    8.33333333333333333333E-2,
  };

  double y = 0;
  if (x < 1.0e17) {
    double z = 1.0 / (x * x);
    y        = z * polevl(z, A, 6);
  }
  return result + log(x) - (0.5 / x) - y;
}

/*
 * This function is derived from the implementation of the digamma function in the Cephes Math
 * Library. See note [3-Clause BSD License for the Cephes Math Library].
 */
__host__ __device__ inline float calc_digamma(float x)
{
  // See [C++ Standard Reference: Gamma Function]
  constexpr float PSI_10 = 2.25175258906672110764f;
  if (x == 0) {
    // As per C++ standard for gamma related functions and SciPy,
    // If the argument is ±0, ±∞ is returned
    return std::copysign(INFINITY, -x);
  }

  bool x_is_integer = x == truncf(x);
  if (x < 0) {
    if (x_is_integer) {
      // As per C++ standard for gamma related functions and SciPy,
      // If the argument is a negative integer, NaN is returned
      return std::numeric_limits<float>::quiet_NaN();
    }
    // Extracts the fractional part of x as r, since tan(pi * r) is more numerically
    // accurate than tan(pi * x). While these operations are mathematically equivalent
    // since both x and r are in radians and tan() has a periodicity of pi, in practice
    // the computation of pi * x is a source of error (when |x| > 1).
    double q = NAN, r = NAN;
    r                      = std::modf(x, &q);
    float pi_over_tan_pi_x = static_cast<float>(m_PI / std::tan(m_PI * r));
    return calc_digamma(1 - x) - pi_over_tan_pi_x;
  }

  // Push x to be >= 10
  float result = 0;
  while (x < 10) {
    result -= 1 / x;
    x += 1;
  }
  if (x == 10) { return result + PSI_10; }

  // Compute asymptotic digamma
  constexpr float A[] = {
    8.33333333333333333333E-2f,
    -2.10927960927960927961E-2f,
    7.57575757575757575758E-3f,
    -4.16666666666666666667E-3f,
    3.96825396825396825397E-3f,
    -8.33333333333333333333E-3f,
    8.33333333333333333333E-2f,
  };

  float y = 0;
  if (x < 1.0e17f) {
    float z = 1 / (x * x);
    y       = z * polevl(z, A, 6);
  }
  return result + logf(x) - (0.5f / x) - y;
}

/*
 * This function is from the implementation of the zeta function in the Cephes Math
 * Library. See note [3-Clause BSD License for the Cephes Math Library].
 */
/* 30 Nov 86 -- error in third coefficient fixed */
__host__ __device__ inline double zeta(double x, double q)
{
  /* Expansion coefficients
   * for Euler-Maclaurin summation formula
   * (2k)! / B2k
   * where B2k are Bernoulli numbers
   */
  static double A[] = {
    12.0,
    -720.0,
    30240.0,
    -1209600.0,
    47900160.0,
    -1.8924375803183791606e9, /*1.307674368e12/691 */
    7.47242496e10,
    -2.950130727918164224e12,  /*1.067062284288e16/3617 */
    1.1646782814350067249e14,  /*5.109094217170944e18/43867 */
    -4.5979787224074726105e15, /*8.028576626982912e20/174611 */
    1.8152105401943546773e17,  /*1.5511210043330985984e23/854513 */
    -7.1661652561756670113e18  /*1.6938241367317436694528e27/236364091 */
  };

  int i    = 0;
  double a = NAN, b = NAN, k = NAN, s = NAN, t = NAN, w = NAN;

  if (x == 1.0) { return std::numeric_limits<double>::infinity(); }

  if (x < 1.0) { return std::numeric_limits<double>::quiet_NaN(); }
  double MACHEP = 1.11022302462515654042E-16;
  if (q <= 0.0) {
    if (q == floor(q)) { return std::numeric_limits<double>::infinity(); }
    if (x != floor(x)) {
      /* because q^-x not defined */
      return std::numeric_limits<double>::quiet_NaN();
    }
  }

  /* Asymptotic expansion
   * https://dlmf.nist.gov/25.11#E43
   */
  if (q > 1e8) { return (1 / (x - 1) + 1 / (2 * q)) * pow(q, 1 - x); }

  /* Euler-Maclaurin summation formula */

  /* Permit negative q but continue sum until n+q > +9 .
   * This case should be handled by a reflection formula.
   * If q<0 and x is an integer, there is a relation to
   * the polyGamma function.
   */
  s = pow(q, -x);
  a = q;
  i = 0;
  b = 0.0;
  while ((i < 9) || (a <= 9.0)) {
    i += 1;
    a += 1.0;
    b = pow(a, -x);
    s += b;
    if (fabs(b / s) < MACHEP) { return s; }
  }

  w = a;
  s += b * w / (x - 1.0);
  s -= 0.5 * b;
  a = 1.0;
  k = 0.0;
  for (i = 0; i < 12; i++) {
    a *= x + k;
    b /= w;
    t = a * b / A[i];
    s = s + t;
    t = fabs(t / s);
    if (t < MACHEP) { return s; }
    k += 1.0;
    a *= x + k;
    b /= w;
    k += 1.0;
  }
  return s;
}

struct ErfOp {
  using ArgsT = std::tuple<>;
  template <typename T>
  __host__ __device__ T operator()(T const& v) const
  {
    return std::erf(v);
  }
};

using ErfTask = UnaryOpTask<ErfOp, ERF>;

struct LgammaOp {
  using ArgsT = std::tuple<>;
  template <typename T>
  __host__ __device__ T operator()(T const& v) const
  {
    return std::lgamma(v);
  }
};

using LgammaTask = UnaryOpTask<LgammaOp, LGAMMA>;

struct TgammaOp {
  using ArgsT = std::tuple<>;
  template <typename T>
  __host__ __device__ T operator()(T const& v) const
  {
    return std::tgamma(v);
  }
};

using TgammaTask = UnaryOpTask<TgammaOp, TGAMMA>;

struct DigammaOp {
  using ArgsT = std::tuple<>;
  template <typename T>
  __host__ __device__ T operator()(T const& v) const
  {
    return calc_digamma(v);
  }
};

using DigammaTask = UnaryOpTask<DigammaOp, DIGAMMA>;

struct ZetaOp {
  using ArgsT = std::tuple<double>;
  double x;

  explicit ZetaOp(double x) : x{x} {}

  template <typename T>
  __host__ __device__ T operator()(T const& q) const
  {
    return zeta(x, q);
  }
};

using ZetaTask = UnaryOpTask<ZetaOp, ZETA>;

}  // namespace legateboost
