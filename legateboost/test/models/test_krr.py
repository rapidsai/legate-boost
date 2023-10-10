import numpy as np
import pytest

import cunumeric as cn
import legateboost as lb

from ..utils import non_increasing


@pytest.mark.parametrize("num_outputs", [1, 5])
def test_improving_with_components(num_outputs):
    rs = cn.random.RandomState(0)
    X = rs.random((100, 10))
    g = rs.normal(size=(X.shape[0], num_outputs))
    h = rs.random(g.shape) + 0.1
    X, g, h = cn.array(X), cn.array(g), cn.array(h)
    y = -g / h
    metrics = []
    for n_components in range(1, 15):
        model = (
            lb.models.KRR(n_components=n_components)
            .set_random_state(np.random.RandomState(2))
            .fit(X, g, h)
        )
        predict = model.predict(X)
        loss = ((predict - y) ** 2 * h).sum(axis=0) / h.sum(axis=0)
        loss = loss.mean()
        metrics.append(loss)

    assert non_increasing(metrics)


@pytest.mark.parametrize("num_outputs", [1, 5])
def test_alpha(num_outputs):
    # higher alpha hyperparameter should lead to smaller coefficients
    rs = cn.random.RandomState(0)
    X = rs.random((100, 10))
    g = rs.normal(size=(X.shape[0], num_outputs))
    h = rs.random(g.shape) + 0.1
    X, g, h = cn.array(X), cn.array(g), cn.array(h)
    norms = []
    for alpha in np.linspace(0.0, 2.5, 5):
        model = (
            lb.models.KRR(alpha=alpha)
            .set_random_state(np.random.RandomState(2))
            .fit(X, g, h)
        )
        norms.append(np.linalg.norm(model.betas_))
    assert non_increasing(norms)
