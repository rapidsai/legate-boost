from typing import Any

import numpy as np

import cunumeric as cn
from legate.core import Store


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
