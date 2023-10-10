from typing import Any

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
    """Alternative implementation for a[cn.arange(b.size), b] = delta"""

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
    """Solve a singular linear system Ax = b for x.
    The same as np.linalg.solve, but if A is singular,
    then we use Algorithm 3.3 from:

    Nocedal, Jorge, and Stephen J. Wright, eds.
    Numerical optimization. New York, NY: Springer New York, 1999.

    This progressively adds to the diagonal of the matrix until it is non-singular.
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
