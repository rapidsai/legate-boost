import numpy as np
import pytest
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error

import cupynumeric as cn
import legateboost as lb
from legateboost.testing.utils import non_increasing


@pytest.mark.parametrize("weights", [True, False])
@pytest.mark.parametrize("solver", ["direct", "lbfgs"])
def test_against_sklearn(weights, solver):
    X = np.linspace(0, 1, 100)[:, np.newaxis]
    y = np.sin(X[:, 0] * 8 * np.pi)
    w = (
        np.random.RandomState(0).uniform(0.1, 100.0, size=y.shape) * np.maximum(0.0, y)
        if weights
        else None
    )
    alpha = 0.00001
    sigma = 0.1
    gamma = 1 / (2 * sigma**2)
    model = lb.LBRegressor(
        n_estimators=1,
        learning_rate=1.0,
        base_models=(
            lb.models.KRR(
                n_components=X.shape[0], alpha=alpha, sigma=sigma, solver=solver
            ),
        ),
    ).fit(X, y, sample_weight=w)
    skl = KernelRidge(kernel="rbf", alpha=alpha, gamma=gamma).fit(X, y, sample_weight=w)
    skl_mse = mean_squared_error(y, skl.predict(X), sample_weight=w)
    lb_mse = mean_squared_error(y, model.predict(X), sample_weight=w)
    assert np.allclose(skl_mse, lb_mse, atol=1e-3)


@pytest.mark.parametrize("num_outputs", [1, 5])
@pytest.mark.parametrize("solver", ["direct", "lbfgs"])
def test_improving_with_components(num_outputs, solver):
    rs = cn.random.RandomState(0)
    X = rs.random((100, 10))
    g = rs.normal(size=(X.shape[0], num_outputs))
    h = rs.random(g.shape) + 0.1
    X, g, h = cn.array(X), cn.array(g), cn.array(h)
    y = -g / h
    metrics = []
    for n_components in range(2, 15):
        model = (
            lb.models.KRR(n_components=n_components, solver=solver)
            .set_random_state(np.random.RandomState(2))
            .fit(X, g, h)
        )
        predict = model.predict(X)
        loss = ((predict - y) ** 2 * h).sum(axis=0) / h.sum(axis=0)
        loss = loss.mean()
        metrics.append(loss)

    assert non_increasing(metrics)


@pytest.mark.parametrize("num_outputs", [1, 5])
@pytest.mark.parametrize("solver", ["direct", "lbfgs"])
def test_alpha(num_outputs, solver):
    # higher alpha hyperparameter should lead to smaller coefficients
    rs = cn.random.RandomState(0)
    X = rs.random((100, 10))
    g = rs.normal(size=(X.shape[0], num_outputs))
    h = rs.random(g.shape) + 0.1
    X, g, h = cn.array(X), cn.array(g), cn.array(h)
    norms = []
    for alpha in np.linspace(0.0, 2.5, 5):
        model = (
            lb.models.KRR(alpha=alpha, solver=solver)
            .set_random_state(np.random.RandomState(2))
            .fit(X, g, h)
        )
        norms.append(np.linalg.norm(model.betas_))
    assert non_increasing(norms)
