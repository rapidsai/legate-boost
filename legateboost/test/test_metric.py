import numpy as np
from sklearn.metrics import mean_squared_error

import cunumeric as cn
import legateboost as lb


def test_mse():
    def compare_to_sklearn(y, p, w):
        a = lb.MSEMetric().metric(cn.array(y), cn.array(p), cn.array(w))
        b = mean_squared_error(y, p, sample_weight=w)
        assert np.isclose(float(a), float(b))

    compare_to_sklearn(np.array([1, 2, 3]), np.array([1, 2, 3]), np.array([1, 1, 1]))
    compare_to_sklearn(np.array([1, 2, 3]), np.array([1, 3, 4]), np.array([1, 1, 1]))
    compare_to_sklearn(
        np.array(
            [[1, 4], [2, 5], [3, 6]],
        ),
        np.array(
            [
                [1, 7],
                [2, 5],
                [3, 6],
            ]
        ),
        np.array([1, 1, 1]),
    )
    compare_to_sklearn(
        np.array([1, 2, 3]), np.array([1, 8, 4]), np.array([1, 0.2, 3.0])
    )

    rng = np.random.RandomState(0)
    n = 15000
    y = rng.normal(size=(n, 3))
    pred = rng.normal(size=(n, 3))
    w = np.abs(rng.normal(size=n))
    compare_to_sklearn(y, pred, w)
