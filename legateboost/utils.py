import copy
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, List, Optional, Tuple

import numpy as np

import cunumeric as cn
from legate.core import (
    LogicalArray,
    LogicalStore,
    TaskTarget,
    get_legate_runtime,
    types,
)

from .library import user_context, user_lib


class AddMember(Enum):
    ADD = 1
    PREFER_A = 2
    ASSERT_SAME = 3


class AddableMixin:
    _add_behaviour: dict[str, AddMember] = {}

    def __add__(self, other: Any) -> Any:
        # check is same subclass
        if not isinstance(other, self.__class__):
            raise TypeError("Can only add two instances of the same class")

        new = copy.deepcopy(self)
        diff = set(vars(self)) - set(vars(other))
        if diff:
            raise ValueError(
                "The following attributes are not present in both models: {}".format(
                    diff
                )
            )

        for v in vars(self):
            try:
                case = self._add_behaviour[v]
                if case == AddMember.ADD:
                    new.__dict__[v] = self.__dict__[v] + other.__dict__[v]
                elif case == AddMember.PREFER_A:
                    new.__dict__[v] = self.__dict__[v]
                elif case == AddMember.ASSERT_SAME:
                    if not np.array_equal(self.__dict__[v], other.__dict__[v]):
                        raise ValueError(
                            (
                                "{} for operand a has value {}, and operand b has"
                                " value {}. They must be equal."
                            ).format(v, self.__dict__[v], other.__dict__[v])
                        )
                    new.__dict__[v] = self.__dict__[v]
                else:
                    raise ValueError("Unknown AddMember case")

            except KeyError:
                raise ValueError(
                    "Attribute {} has no defined behaviour for addition".format(v)
                )

        return new


class PickleCunumericMixin:
    """When reading back from pickle, convert numpy arrays to cunumeric
    arrays."""

    def __getstate__(self) -> dict[str, Any]:
        return self.__dict__.copy()

    def __setstate__(self, state: dict[str, Any]) -> None:
        def replace(data: Any) -> None:
            if isinstance(data, (dict, list)):
                # Note: mypy 1.12.0 required iter() around data.items() to pass
                items = (
                    iter(data.items()) if isinstance(data, dict) else enumerate(data)
                )
                for k, v in items:
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


def get_store(input: Any) -> LogicalStore:
    """Extracts a Legate store from any object implementing the legate data
    interface.

    Args:
        input (Any): The input object

    Returns:
        LogicalStore: The extracted Legate store
    """
    if isinstance(input, LogicalStore):
        return input
    if isinstance(input, LogicalArray):
        assert not (input.nullable or input.nested)
        return input.data
    data = input.__legate_data_interface__["data"]
    field = next(iter(data))
    array = data[field]
    assert not (array.nullable or array.nested)
    store = array.data
    return store


def solve_singular(a: cn.ndarray, b: cn.ndarray) -> cn.ndarray:
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


def __line_search(
    f: Callable[..., Tuple[float, Any]],
    eval: float,
    g: cn.ndarray,
    x: cn.ndarray,
    d: cn.ndarray,
    args: Tuple[Any, ...] = (),
) -> Tuple[float, float, cn.ndarray]:
    alpha = 1.0
    c = 1e-4
    rho = 0.5
    new_eval, new_g = f(x + alpha * d, *args)
    beta = c * cn.dot(g, d)
    while new_eval > eval + alpha * beta and alpha * rho > 1e-15:
        alpha *= rho
        new_eval, new_g = f(x + alpha * d, *args)
    return alpha, new_eval, new_g


def __vlbfgs_recursion(
    g: cn.ndarray, s: List[cn.ndarray], y: List[cn.ndarray]
) -> cn.ndarray:
    m = len(s)
    if m == 0:
        return -g
    b = cn.array(s + y + [g])

    # Perform this computation using numpy to avoid Legate overhead
    # B matrix is small
    B = b.dot(b.T).__array__()
    # elements of B are not allowed to be near 0
    B[(B >= 0.0) & (B < 1e-15)] = 1e-15
    B[(B < 0.0) & (B > -1e-15)] = -1e-15

    delta = np.zeros(len(b))
    alpha = np.zeros(len(b))
    delta[-1] = -1.0
    for i in reversed(range(m)):
        alpha[i] = delta.dot(B[:, i]) / B[i, i + m]
        delta[m + i] = delta[m + i] - alpha[i]

    delta = delta * B[m - 1, 2 * m - 1] / B[2 * m - 1, 2 * m - 1]

    for i in range(m):
        beta = delta.dot(B[:, i + m]) / B[i, i + m]
        delta[i] = delta[i] + (alpha[i] - beta)
    # Convert back to cunumeric
    return cn.dot(delta, b)


def __lbfgs_recursion(
    g: cn.ndarray, s: List[cn.ndarray], y: List[cn.ndarray]
) -> cn.ndarray:
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
        H_k0 = cn.dot(s[-1], y[-1]) / cn.dot(y[-1], y[-1])
    r = H_k0 * q
    for i in range(m):
        beta = rho[i] * cn.dot(y[i], r)
        r += s[i] * (alpha[i] - beta)
    return -r


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

    def __str__(self) -> str:
        return (
            "L-BFGS Result:\n\teval: {}\n\tnorm: {}\n\tnum_iter:"
            " {}\n\tfeval: {}".format(self.eval, self.norm, self.num_iter, self.feval)
        )


def lbfgs(
    x0: cn.array,
    f: Callable[..., Tuple[float, Any]],
    max_iter: int = 100,
    m: int = 10,
    gtol: float = 1e-5,
    args: Tuple[Any, ...] = (),
    verbose: int = 0,
) -> LbfgsResult:
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
    assert x0.ndim == 1
    x = x0.copy()

    class CountF:
        def __init__(self, func: Callable[..., Tuple[float, Any]]):
            self.count = 0
            self.func = func

        def __call__(self, *args: Any, **kwargs: Any) -> Tuple[float, Any]:
            self.count += 1
            return self.func(*args, **kwargs)

    count_f = CountF(f)

    eval, g = count_f(x, *args)
    s: List[cn.ndarray] = []
    y: List[cn.ndarray] = []
    norm = 0.0
    # allow limited restarts so the algorithm doesn't get stuck
    max_restarts = 10
    restarts = 0
    for k in range(max_iter):
        r = __vlbfgs_recursion(g, s, y)
        lr, eval, new_g = __line_search(count_f, eval, g, x, r, args=args)
        norm = cn.linalg.norm(new_g)
        s.append(lr * r)
        x = x + s[-1]
        y.append(new_g - g)
        g = new_g
        if lr < 1e-10:
            if restarts >= max_restarts:
                break
            if verbose:
                print("L-BFGS: lr too small, restarting iteration.")
            s = []
            y = []
            restarts += 1
        if verbose and k % verbose == 0:
            print(
                "L-BFGS:\tk={}\tfeval:{:8.5}\tnorm:{:8.5f}".format(k, float(eval), norm)
            )
        if norm < gtol:
            break
        if len(s) > m:
            s.pop(0)
            y.pop(0)

    assert x.ndim == 1
    return LbfgsResult(x, eval, norm, k + 1, count_f.count)


def gather_from_array(X: cn.array, samples: cn.array) -> cn.array:
    samples = samples.astype(cn.int64)
    if samples.shape[0] == 0:
        return cn.empty(shape=(0, X.shape[1]), dtype=X.dtype)
    if samples.size == 1:
        return X[samples[0]].reshape(1, -1)
    task = get_legate_runtime().create_auto_task(
        user_context,
        user_lib.cffi.GATHER,
    )

    output = cn.zeros(shape=(samples.shape[0], X.shape[1]), dtype=X.dtype)
    task.add_input(get_store(X))
    task.add_input(get_store(samples))
    task.add_broadcast(get_store(samples))
    task.add_output(get_store(output))
    task.add_broadcast(get_store(output))

    if get_legate_runtime().machine.count(TaskTarget.GPU) > 1:
        task.add_nccl_communicator()
    elif get_legate_runtime().machine.count() > 1:
        task.add_cpu_communicator()

    task.execute()
    return output


def gather(X: cn.array, samples: Tuple[int, ...]) -> cn.array:
    num_samples = len(samples)
    if num_samples == 0:
        return cn.empty(shape=(0, X.shape[1]), dtype=X.dtype)
    if num_samples == 1:
        return X[samples[0]].reshape(1, -1)
    task = get_legate_runtime().create_auto_task(
        user_context,
        user_lib.cffi.GATHER,
    )

    output = cn.zeros(shape=(num_samples, X.shape[1]), dtype=X.dtype)
    task.add_input(get_store(X))
    task.add_scalar_arg(samples, (types.int64,))
    task.add_output(get_store(output))
    task.add_broadcast(get_store(output))

    if get_legate_runtime().machine.count(TaskTarget.GPU) > 1:
        task.add_nccl_communicator()
    elif get_legate_runtime().machine.count() > 1:
        task.add_cpu_communicator()

    task.execute()
    return output
