import numpy as np
import pytest

import cunumeric as cn
import legateboost as lb

from ..utils import non_increasing
from .utils import check_determinism


def test_determinism():
    check_determinism(lb.models.Tree(max_depth=0))
    check_determinism(lb.models.Tree(max_depth=12))


@pytest.mark.parametrize("num_outputs", [1, 5])
def test_improving_with_depth(num_outputs):
    rs = cn.random.RandomState(0)
    X = rs.random((100, 10))
    g = rs.normal(size=(X.shape[0], num_outputs))
    h = rs.random(g.shape) + 0.1
    X, g, h = cn.array(X), cn.array(g), cn.array(h)
    y = -g / h
    metrics = []
    for max_depth in range(0, 16):
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
