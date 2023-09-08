import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

import cunumeric as cn
import legateboost as lb

from .utils import check_determinism


def test_determinism():
    check_determinism(lb.models.Linear())


@pytest.mark.parametrize("num_outputs", [1, 5])
def test_linear(num_outputs):
    # compare against an sklearn model
    rs = cn.random.RandomState(0)
    X = rs.random((100, 10))
    g = rs.normal(size=(X.shape[0], num_outputs))
    h = rs.random(g.shape) + 0.1
    h = cn.ones_like(h)
    X, g, h = cn.array(X), cn.array(g), cn.array(h)
    model = lb.models.Linear().set_random_state(np.random.RandomState(2)).fit(X, g, h)
    for k in range(0, num_outputs):
        sklearn_model = LinearRegression().fit(X, -g[:, k] / h[:, k])
        assert np.allclose(model.bias_[k], sklearn_model.intercept_)
        assert np.allclose(model.betas_[:, k], sklearn_model.coef_)
        assert np.allclose(model.predict(X)[:, k], sklearn_model.predict(X))
