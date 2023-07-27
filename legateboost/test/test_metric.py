import numpy as np
from sklearn.metrics import log_loss, mean_squared_error

import cunumeric as cn
import legateboost as lb


def test_multiple_metrics():
    X = cn.random.random((10, 1))
    y = cn.random.randint(0, 2, size=X.shape[0])
    X_eval = cn.random.random((5, 1))
    y_eval = cn.random.randint(0, 2, size=X_eval.shape[0])

    eval_result = {}
    lb.LBClassifier(n_estimators=2, metric=["log_loss", "exp"]).fit(
        X, y, eval_set=[(X_eval, y_eval)], eval_result=eval_result
    )
    assert "log_loss" in eval_result["train"]
    assert "log_loss" in eval_result["eval-0"]
    assert len(eval_result["train"]["log_loss"]) == 2
    assert "exp" in eval_result["train"]
    assert "exp" in eval_result["eval-0"]

    lb.LBClassifier(n_estimators=2, metric=[lb.ExponentialMetric()]).fit(
        X, y, eval_result=eval_result
    )
    assert "exp" in eval_result["train"]
    lb.LBClassifier(n_estimators=2, metric=lb.ExponentialMetric()).fit(
        X, y, eval_result=eval_result
    )
    assert "exp" in eval_result["train"]


def test_eval_tuple():
    # check weights get registered
    X = cn.random.random((10, 1))
    y = cn.random.randint(0, 2, size=X.shape[0])
    X_eval = cn.random.random((10, 1))
    y_eval = cn.random.randint(0, 2, size=X.shape[0])
    w_eval = cn.zeros(X.shape[0])

    eval_result = {}
    lb.LBClassifier(n_estimators=2).fit(
        X, y, eval_set=[(X_eval, y_eval, w_eval)], eval_result=eval_result
    )
    assert eval_result["eval-0"]["log_loss"][-1] == 0.0


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
