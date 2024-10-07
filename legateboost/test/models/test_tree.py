import numpy as np
import pytest

import cunumeric as cn
import legateboost as lb
from legateboost.testing.utils import non_increasing


def test_basic():
    # tree loss can go to zero
    X = cn.array([[0.0], [1.0]])
    g = cn.array([[0.0], [-1.0]])
    h = cn.array([[1.0], [1.0]])
    model = (
        lb.models.Tree(max_depth=1, alpha=0.0)
        .set_random_state(np.random.RandomState(2))
        .fit(X, g, h)
    )
    assert np.allclose(model.predict(X), np.array([[0.0], [1.0]]))


@pytest.mark.parametrize("num_outputs", [1, 5])
def test_improving_with_depth(num_outputs):
    rs = cn.random.RandomState(0)
    X = rs.random((10000, 100))
    g = rs.normal(size=(X.shape[0], num_outputs))
    h = rs.random(g.shape) + 0.1
    X, g, h = cn.array(X), cn.array(g), cn.array(h)
    y = -g / h
    metrics = []
    for max_depth in range(0, 12):
        model = (
            lb.models.Tree(max_depth=max_depth)
            .set_random_state(np.random.RandomState(2))
            .fit(X, g, h)
        )
        predict = model.predict(X)
        loss = ((predict - y) ** 2 * h).sum(axis=0) / h.sum(axis=0)
        loss = loss.mean()
        metrics.append(loss)

    assert non_increasing(metrics)
    assert metrics[-1] < metrics[0]


def test_max_depth():
    # we should be able to run deep trees with OOM
    max_depth = 20
    X = cn.random.random((2, 1))
    y = cn.array([500.0, 500.0])
    model = lb.LBRegressor(
        init=None,
        base_models=(lb.models.Tree(max_depth=max_depth),),
        learning_rate=1.0,
        n_estimators=1,
        random_state=0,
    )

    model.fit(X, y)


def test_alpha():
    X = cn.random.random((2, 1))
    y = cn.array([500.0, 500.0])
    alpha = 10.0
    model = lb.LBRegressor(
        init=None,
        base_models=(lb.models.Tree(alpha=alpha, max_depth=0),),
        learning_rate=1.0,
        n_estimators=1,
        random_state=0,
    )
    model.fit(X, y)
    assert np.isclose(model.predict(X)[0], y.sum() / (y.size + alpha))
