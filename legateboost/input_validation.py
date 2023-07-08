from typing import Any

import numpy as np
import scipy.sparse as sp

import cunumeric as cn


def check_sample_weight(sample_weight: Any, n: int) -> cn.ndarray:
    if sample_weight is None:
        sample_weight = cn.ones(n)
    elif cn.isscalar(sample_weight):
        sample_weight = cn.full(n, sample_weight)
    elif not isinstance(sample_weight, cn.ndarray):
        sample_weight = cn.array(sample_weight)
    if sample_weight.shape != (n,):
        raise ValueError(
            "Incorrect sample weight shape: "
            + str(sample_weight.shape)
            + ", expected: ("
            + str(n)
            + ",)"
        )
    assert sample_weight.min() >= 0, "Negative weights are not supported."
    return sample_weight.astype(cn.float64)


def check_array(x: Any) -> cn.ndarray:
    if sp.issparse(x):
        raise ValueError("Sparse matrix not allowed.")

    if not hasattr(x, "__legate_data_interface__"):
        x = cn.array(np.require(x, requirements=["C", "A"]))
    if hasattr(x, "__array_interface__"):
        shape = x.__array_interface__["shape"]
        if shape[0] <= 0:
            raise ValueError(
                "Found array with %d sample(s) (shape=%s) while a"
                " minimum of %d is required." % (shape[0], shape, 1)
            )
        if len(shape) >= 2 and 0 in shape:
            raise ValueError(
                "Found array with %d feature(s) (shape=%s) while"
                " a minimum of %d is required." % (shape[1], shape, 1)
            )

    if cn.iscomplexobj(x):
        raise ValueError("Complex data not supported.")

    if np.issubdtype(x.dtype, np.floating) and not cn.isfinite(x).all():
        raise ValueError("Input contains NaN or inf")

    x = cn.array(x, copy=False)

    return x


def check_X_y(X: Any, y: Any = None) -> Any:
    X = check_array(X)
    if len(X.shape) != 2:
        raise ValueError("X must be 2-dimensional. Reshape your data.")
    if X.shape[0] == 0:
        raise ValueError("Empty input")
    if X.shape[1] == 0:
        raise ValueError(
            "0 feature(s) (shape=({}, 0)) while a minimum of 1 is required.".format(
                X.shape[0]
            )
        )

    if y is not None:
        y = check_array(y)
        y = y.astype(cn.float64)
        y = cn.atleast_1d(y)

        if y.ndim == 1:
            y = y[:, cn.newaxis]
        if y.shape[0] != X.shape[0]:
            raise ValueError("Number of labels does not match number of samples.")

    if np.issubdtype(X.dtype, np.integer):
        X = X.astype(cn.float32)

    if y is not None:
        return X, y
    return X
