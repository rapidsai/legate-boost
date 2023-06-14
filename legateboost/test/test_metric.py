import numpy as np
from sklearn.metrics import log_loss, mean_squared_error

import cunumeric as cn
import legateboost as lb


def test_mse():
    def compare_to_sklearn(y, p, w):
        a = lb.MSEMetric().metric(cn.array(y), cn.array(p), cn.array(w))
        b = mean_squared_error(y, p, sample_weight=w)
        assert np.isclose(float(a), float(b))

    compare_to_sklearn([1, 2, 3], [1, 2, 3], [1, 1, 1])
    compare_to_sklearn([1, 2, 3], [1, 3, 4], [1, 1, 1])
    compare_to_sklearn(
        [[1, 4], [2, 5], [3, 6]],
        [
            [1, 7],
            [2, 5],
            [3, 6],
        ],
        [1, 1, 1],
    )
    compare_to_sklearn([1, 2, 3], [1, 8, 4], [1, 0.2, 3.0])

    rng = np.random.RandomState(0)
    n = 15000
    y = rng.normal(size=(n, 3))
    pred = rng.normal(size=(n, 3))
    w = np.abs(rng.normal(size=n))
    compare_to_sklearn(y, pred, w)


def test_log_loss():
    def compare_to_sklearn(y, p, w):
        a = lb.LogLossMetric().metric(cn.array(y), cn.array(p), cn.array(w))
        b = log_loss(y, p, sample_weight=w)
        assert np.isclose(a, b)

    # binary
    compare_to_sklearn([1.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0])
    compare_to_sklearn([1.0, 0.0, 1.0], [0.6, 0.1, 0.3], [1.0, 1.0, 1.0])
    compare_to_sklearn([1.0, 0.0, 1.0], [0.6, 0.1, 0.3], [0.3, 1.0, 0.7])
    compare_to_sklearn([1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.3, 1.0, 0.7])

    rng = np.random.RandomState(0)
    n = 15000
    y = rng.randint(0, 2, size=n)
    pred = rng.uniform(size=n)
    w = np.abs(rng.normal(size=n))
    compare_to_sklearn(y, pred, w)

    # multi-class
    compare_to_sklearn(
        [2.0, 0.0, 1.0],
        [[0.2, 0.3, 0.5], [0.9, 0.05, 0.05], [0.2, 0.3, 0.5]],
        [1.0, 1.0, 1.0],
    )
    compare_to_sklearn(
        [2.0, 0.0, 1.0],
        [[0.2, 0.3, 0.5], [0.9, 0.05, 0.05], [0.2, 0.3, 0.5]],
        [0.3, 1.0, 0.7],
    )

    y = rng.randint(0, 5, size=n)
    pred = rng.uniform(size=(n, 5))
    pred = pred / pred.sum(axis=1)[:, np.newaxis]
    compare_to_sklearn(y, pred, w)
