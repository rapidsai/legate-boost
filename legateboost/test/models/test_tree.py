import numpy as np
import pytest

import cupynumeric as cn
import legateboost as lb
from legateboost.testing.utils import non_increasing


def test_basic():
    # tree loss can go to zero
    X = cn.array([[0.0], [1.0]])
    g = cn.array([[0.0], [-1.0]])
    h = cn.array([[1.0], [1.0]])
    model = (
        lb.models.Tree(max_depth=1, l2_regularization=0.0)
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


def test_l2_regularization():
    X = cn.array([[0.0], [0.0]])
    y = cn.array([500.0, 500.0])
    l2_regularization = 10.0
    model = lb.LBRegressor(
        init=None,
        base_models=(lb.models.Tree(l2_regularization=l2_regularization, max_depth=0),),
        learning_rate=1.0,
        n_estimators=1,
        random_state=0,
    )
    model.fit(X, y)
    assert np.isclose(model.predict(X)[0], y.sum() / (y.size + l2_regularization))
    model.update(X, y)
    assert np.isclose(model.predict(X)[0], y.sum() / (y.size + l2_regularization))


def test_l1_regularization():
    X = cn.array([[0.0], [0.0]])
    y = cn.array([500.0, 500.0])
    l1_regularization = 1.5
    model = lb.LBRegressor(
        init=None,
        base_models=(
            lb.models.Tree(
                l2_regularization=0.0, l1_regularization=l1_regularization, max_depth=0
            ),
        ),
        learning_rate=1.0,
        n_estimators=1,
        random_state=0,
    )
    model.fit(X, y)
    assert np.isclose(model.predict(X)[0], (y.sum() - l1_regularization) / (y.size))
    model.update(X, y)
    assert np.isclose(model.predict(X)[0], (y.sum() - l1_regularization) / (y.size))

    y = cn.array([0.5, 0.5])
    model.fit(X, y)
    assert np.isclose(model.predict(X)[0], 0.0)
    model.update(X, y)
    assert np.isclose(model.predict(X)[0], 0.0)


def test_min_split_gain():
    # grow two trees with the same data, but one with min_split_gain=10.0
    # we should observe a smaller tree with higher min_split_gain
    rng = np.random.RandomState(0)
    X = rng.random((100, 1))
    y = rng.random(100)
    model = lb.LBRegressor(
        init=None,
        base_models=(lb.models.Tree(min_split_gain=0.0, max_depth=12),),
        learning_rate=1.0,
        n_estimators=1,
        random_state=0,
    )
    model.fit(X, y)
    num_leaves_a = cn.sum(model[0].hessian > 0.0)
    model = lb.LBRegressor(
        init=None,
        base_models=(lb.models.Tree(min_split_gain=0.01, max_depth=12),),
        learning_rate=1.0,
        n_estimators=1,
        random_state=0,
    )
    model.fit(X, y)
    num_leaves_b = cn.sum(model[0].hessian > 0.0)
    assert num_leaves_b < num_leaves_a
