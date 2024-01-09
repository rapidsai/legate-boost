/*
 * Code from PyTorch, which comes from cephes. See thirdparty/LICENSES/LICENSE.pytorch
 */
#pragma once

#include <cmath>                   // for copysign, modf, trunc, log, tan, sinf
#include <limits>                  // for numeric_limits
#include <thrust/detail/config.h>  // for __host__ __device__

namespace legateboost {
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
  static double PSI_10 = 2.25175258906672110764;
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
    double q, r;
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
  static const double A[] = {
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
  static float PSI_10 = 2.25175258906672110764f;
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
    double q, r;
    r                      = std::modf(x, &q);
    float pi_over_tan_pi_x = (float)(m_PI / std::tan(m_PI * r));
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
  static const float A[] = {
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
}  // namespace legateboost
