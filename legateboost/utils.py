from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

import cunumeric as cn
from legate.core import Store, get_legate_runtime


class PickleCunumericMixin:
    """When reading back from pickle, convert numpy arrays to cunumeric
    arrays."""

    def __getstate__(self) -> dict[str, Any]:
        return self.__dict__.copy()

    def __setstate__(self, state: dict[str, Any]) -> None:
        def replace(data: Any) -> None:
            if isinstance(data, (dict, list)):
                for k, v in data.items() if isinstance(data, dict) else enumerate(data):
                    if isinstance(v, np.ndarray):
                        data[k] = cn.asarray(v)
                    replace(v)

        replace(state)
        self.__dict__.update(state)


def pick_col_by_idx(a: cn.ndarray, b: cn.ndarray) -> cn.ndarray:
    """Alternative implementation for a[cn.arange(b.size), b]"""

    assert a.ndim == 2
    assert b.ndim == 1
    assert a.shape[0] == b.shape[0]

    range = cn.arange(a.shape[1])
    bools = b[:, cn.newaxis] == range[cn.newaxis, :]
    result = a * bools
    return result.sum(axis=1)


def set_col_by_idx(a: cn.ndarray, b: cn.ndarray, delta: float) -> None:
    """Alternative implementation for a[cn.arange(b.size), b] = delta."""

    assert a.ndim == 2
    assert b.ndim == 1
    assert a.shape[0] == b.shape[0]

    range = cn.arange(a.shape[1])
    bools = b[:, cn.newaxis] == range[cn.newaxis, :]
    a -= a * bools
    a += delta * bools
    return


def mod_col_by_idx(a: cn.ndarray, b: cn.ndarray, delta: float) -> None:
    """Alternative implementation for a[cn.arange(b.size), b] += delta."""

    assert a.ndim == 2
    assert b.ndim == 1
    assert a.shape[0] == b.shape[0]

    range = cn.arange(a.shape[1])
    bools = b[:, cn.newaxis] == range[cn.newaxis, :]
    a += delta * bools
    return


def preround(x: cn.ndarray) -> cn.ndarray:
    """Apply this function to grad/hess ensure reproducible floating point
    summation.

    Algorithm 5: Reproducible Sequential Sum in 'Fast Reproducible
    Floating-Point Summation' by Demmel and Nguyen.

    Instead of using max(abs(x)) * n as an upper bound we use sum(abs(x))
    """
    assert x.dtype == cn.float32 or x.dtype == cn.float64
    m = cn.sum(cn.abs(x))
    n = x.size
    delta = cn.floor(m / (1 - 2 * n * cn.finfo(x.dtype).eps))
    M = 2 ** cn.ceil(cn.log2(delta))
    return (x + M) - M


def get_store(input: Any) -> Store:
    """Extracts a Legate store from any object implementing the legete data
    interface.

    Args:
        input (Any): The input object

    Returns:
        Store: The extracted Legate store
    """
    if isinstance(input, Store):
        return input
    data = input.__legate_data_interface__["data"]
    field = next(iter(data))
    array = data[field]
    _, store = array.stores()
    return store


def solve_singular(a, b):
    """Solve a singular linear system Ax = b for x. The same as
    np.linalg.solve, but if A is singular, then we use Algorithm 3.3 from:

    Nocedal, Jorge, and Stephen J. Wright, eds. Numerical optimization.
    New York, NY: Springer New York, 1999.

    This progressively adds to the diagonal of the matrix until it is
    non-singular.
    """
    # ensure we are doing all calculations in float 64 for stability
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    # try first without modification
    try:
        res = cn.linalg.solve(a, b)
        get_legate_runtime().raise_exceptions()
        if np.isnan(res).any():
            raise np.linalg.LinAlgError
        return res
    except (np.linalg.LinAlgError, cn.linalg.LinAlgError):
        pass

    # if that fails, try adding to the diagonal
    eps = 1e-3
    min_diag = a[::].min()
    if min_diag > 0:
        tau = eps
    else:
        tau = -min_diag + eps
    while True:
        try:
            res = cn.linalg.solve(a + cn.eye(a.shape[0]) * tau, b)
            get_legate_runtime().raise_exceptions()
            if np.isnan(res).any():
                raise np.linalg.LinAlgError
            return res
        except (np.linalg.LinAlgError, cn.linalg.LinAlgError):
            tau = max(tau * 2, eps)
        if tau > 1e10:
            raise ValueError(
                "Numerical instability in linear model solve. "
                "Consider normalising your data."
            )


def sample_average(
    y: cn.ndarray, sample_weight: Optional[cn.ndarray] = None
) -> cn.ndarray:
    """Compute weighted average on the first axis (usually the sample
    dimension).

    Returns 0 if sum weight is zero or if the input is empty.
    """
    if y.ndim > 2:
        raise ValueError("Expecting a 1-dim or 2-dim input.")
    if y.shape[0] == 0:
        return cn.zeros(shape=(1,))
    if sample_weight is None:
        n_columns = y.shape[1:] if y.ndim > 1 else 1
        return cn.sum(y, axis=0) / cn.full(shape=n_columns, value=float(y.shape[0]))
    if sample_weight.ndim > 1:
        raise ValueError("Expecting 1-dim sample weight")
    sum_w = sample_weight.sum()
    if y.ndim == 2:
        sample_weight = sample_weight[:, cn.newaxis]
    if cn.isclose(sum_w, cn.zeros(shape=(1,))):
        return 0.0
    return (y * sample_weight).sum(axis=0) / sum_w


def __line_search(f, eval, g, x, d, args=()):
    alpha = 1.0
    c = 1e-4
    rho = 0.5
    new_eval, new_g = f(x + alpha * d, *args)
    beta = c * np.dot(g, d)
    while new_eval > eval + alpha * beta:
        alpha *= rho
        new_eval, new_g = f(x + alpha * d, *args)
    return alpha, new_eval, new_g


def __lbfgs_recursion(g, s, y):
    q = g.copy()
    m = len(s)
    alpha = cn.zeros(m)
    rho = [1 / cn.dot(y[i], s[i]) for i in range(m)]
    for i in reversed(range(m)):
        alpha[i] = rho[i] * cn.dot(s[i], q)
        q -= alpha[i] * y[i]
    if s == []:
        H_k0 = 1
    else:
        H_k0 = np.dot(s[-1], y[-1]) / np.dot(y[-1], y[-1])
    r = H_k0 * q
    for i in range(m):
        beta = rho[i] * cn.dot(y[i], r)
        r += s[i] * (alpha[i] - beta)
    return r


@dataclass
class LbfgsResult:
    """Result of L-BFGS optimization.

    Attributes:
        x: The solution.
        eval: The final value of the objective function.
        norm: The norm of the gradient.
        num_iter: The number of iterations.
        feval: The number of function evaluations.
    """

    x: cn.ndarray
    eval: float
    norm: float
    num_iter: int
    feval: int

    def __str__(self):
        return (
            "L-BFGS Result:\n\teval: {}\n\tnorm: {}\n\tnum_iter:"
            " {}\n\tfeval: {}".format(self.eval, self.norm, self.num_iter, self.feval)
        )


def lbfgs(x0, f, max_iter=100, m=10, gtol=1e-5, args=(), verbose=False):
    """Minimize a function using the L-BFGS algorithm.

    Parameters
    ----------
    x0 : array_like
        Initial guess for the minimum point.
    f : callable
        Objective function to minimize. The function must return the value of
        the objective function and its gradient at a given point x, i.e.,
        `f(x, *args) -> (float, array_like)`.
    max_iter : int, optional
        Maximum number of iterations.
    m : int, optional
        Number of previous iterations to use in the L-BFGS recursion.
    gtol : float, optional
        Tolerance for the norm of the gradient.
    args : tuple, optional
        Extra arguments to pass to the objective function.
    verbose : bool, optional
        Whether to print information about the optimization process.

    Returns
    -------
    result : LbfgsResult
        The optimization result represented as a named tuple with fields:
        `x` (the minimum point), `eval` (the minimum value of the objective
        function), `norm` (the norm of the gradient at the minimum point),
        `num_iter` (the number of iterations performed), and `feval` (the
        number of function evaluations performed).
    """
    x = x0.copy()

    def count_f(x, *args):
        count_f.count += 1
        return f(x, *args)

    count_f.count = 0

    eval, g = count_f(x, *args)
    s = []
    y = []
    for k in range(max_iter):
        r = __lbfgs_recursion(g, s, y)
        lr, eval, new_g = __line_search(count_f, eval, g, x, -r, args=args)
        if lr < 1e-10:
            if verbose:
                print("L-BFGS: lr too small")
            break
        s.append(-lr * r)
        x = x + s[-1]
        y.append(new_g - g)
        g = new_g
        norm = np.linalg.norm(g)
        if verbose and k % verbose == 0:
            print("L-BFGS:\tk={}\tfeval:{:8.5}\tnorm:{:8.5f}".format(k, eval, norm))
        if norm < gtol:
            break
        if len(s) > m:
            s.pop(0)
            y.pop(0)

    return LbfgsResult(x, eval, norm, k + 1, count_f.count)
