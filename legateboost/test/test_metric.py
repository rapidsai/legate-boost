import numpy as np
from sklearn.metrics import (
    log_loss,
    mean_gamma_deviance as skl_gamma_deviance,
    mean_pinball_loss,
    mean_squared_error,
)

import cunumeric as cn
import legateboost as lb
from legateboost.metrics import GammaDevianceMetric, erf


def test_multiple_metrics() -> None:
    np.random.seed(0)
    X = np.random.random((10, 1))
    y = np.random.randint(0, 2, size=X.shape[0])
    X_eval = np.random.random((5, 1))
    y_eval = np.random.randint(0, 2, size=X_eval.shape[0])

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
    np.random.seed(0)
    X = np.random.random((10, 1))
    y = np.random.randint(0, 2, size=X.shape[0])
    X_eval = np.random.random((10, 1))
    y_eval = np.random.randint(0, 2, size=X.shape[0])
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


def test_exp():
    # standard exp metric that accepts raw function output
    def exp_metric(y, f):
        if f.shape[1] == 1:
            f = f.squeeze()
            y_adjusted = y * 2 - 1
            return np.mean(np.exp(-y_adjusted * f))
        K = y.max() + 1
        y_k = cn.full((y.size, K), -1.0 / (K - 1.0))
        y_k[cn.arange(y.size), y.astype(cn.int32)] = 1.0
        return cn.exp(-1 / K * cn.sum(y_k * f, axis=1)).mean()

    # compare against our version that accepts probabilities

    # binary
    obj = lb.ExponentialObjective()
    y = cn.array([1.0, 0.0, 1.0, 0.0])
    raw_pred = cn.array([[-1.5], [3.0], [10.0], [-0.3]])
    reference = exp_metric(y, raw_pred)
    metric = lb.ExponentialMetric()
    assert cn.allclose(
        reference, metric.metric(y, obj.transform(raw_pred), cn.ones(y.shape))
    )

    # multi-class
    y = cn.array([0, 1, 2, 0])
    raw_pred = cn.array(
        [[10, 0.3, 0.5], [1.2, 0.05, 0.5], [0.5, 3.0, 7.0], [2.2, 0.3, 0.5]]
    )
    reference = exp_metric(y, raw_pred)
    assert cn.allclose(
        reference, metric.metric(y, obj.transform(raw_pred), cn.ones(y.shape))
    )


def test_normal_neg_ll():
    metric = lb.NormalLLMetric()

    def neg_ll(y, p):
        from scipy.stats import norm

        return float(-norm.logpdf(y.squeeze(), loc=p[:, 0], scale=p[:, 1]).mean())

    y = cn.array([1.0, 0.0, 1.0]).reshape(-1, 1)
    sigma_pred = cn.array([[0.0, 1.0], [0.0, 2.0], [0.0, 1.0]])
    log_sigma_pred = sigma_pred.copy()
    log_sigma_pred[:, 1] = cn.log(log_sigma_pred[:, 1])
    assert cn.allclose(
        metric.metric(y, log_sigma_pred, cn.ones(y.shape)), neg_ll(y, sigma_pred)
    )

    # multi_output
    y_1 = cn.array([0.0, 1.0, 2.0]).reshape(-1, 1)
    sigma_pred_1 = cn.array([[0.0, 1.0], [0.0, 2.0], [2.0, 0.5]])
    log_sigma_pred_1 = sigma_pred_1.copy()
    log_sigma_pred_1[:, 1] = cn.log(log_sigma_pred_1[:, 1])

    our_metric = metric.metric(
        cn.hstack((y, y_1)),
        cn.hstack(
            (
                log_sigma_pred_1.reshape(log_sigma_pred_1.shape[0], 1, -1),
                log_sigma_pred_1.reshape(log_sigma_pred_1.shape[0], 1, -1),
            )
        ),
        cn.ones(y_1.shape),
    )
    ref_metric = cn.mean([neg_ll(y, sigma_pred_1), neg_ll(y_1, sigma_pred_1)])
    assert cn.allclose(our_metric, ref_metric)


def test_normal_crps() -> None:
    """Tests for the `NormalCRPSMetric`."""
    cprs = lb.NormalCRPSMetric()
    y = cn.array([1.0, 0.0, 1.0]).T
    p = cn.array([[1.0, 1.0], [0.0, 1.0], [1.0, 1.0]])
    score = cprs.metric(y, p, cn.ones(y.shape))
    assert np.isclose(score, 0.233695)

    y = cn.array([12.0, 13.0, 14.0]).T
    p = cn.array([[4.0, 8.0], [5.0, 9.0], [6.0, 10.0]])
    score = cprs.metric(y, p, cn.ones(y.shape))
    assert np.isclose(score, 6.316697)


def test_quantile_metric() -> None:
    quantiles = cn.array([0.1, 0.5, 0.9])
    metric = lb.QuantileMetric(quantiles)

    def sklearn_loss(y, p, w=None):
        return cn.mean(
            [
                mean_pinball_loss(y, p[:, i], sample_weight=w, alpha=q)
                for i, q in enumerate(quantiles)
            ]
        )

    np.random.seed(5)
    y = np.random.normal(size=(100, 1))
    pred = np.random.normal(size=(100, 3))
    assert cn.allclose(
        metric.metric(y, pred, cn.ones(y.shape[0])), sklearn_loss(y, pred)
    )

    w = np.random.normal(size=(100))
    assert cn.allclose(metric.metric(y, pred, w), sklearn_loss(y, pred, w))


def test_gamma_deviance() -> None:
    rng = cn.random.default_rng(0)

    X = rng.normal(size=(100, 10))
    y = rng.gamma(3.0, 1.0, size=100)
    w = rng.uniform(0.0, 1.0, size=y.shape[0])

    reg = lb.LBRegressor()
    reg.fit(X, y, sample_weight=w)
    p = reg.predict(X)

    m = GammaDevianceMetric()
    d0 = m.metric(y, p, w=w)

    d1 = skl_gamma_deviance(y, p, sample_weight=w)

    assert cn.allclose(d0, d1, rtol=1e-3)
