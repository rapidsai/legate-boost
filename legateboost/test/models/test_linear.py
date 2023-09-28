import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import log_loss

import cunumeric as cn
import legateboost as lb

from .utils import check_determinism


@pytest.mark.xfail
def test_determinism():
    check_determinism(lb.models.Linear())


@pytest.mark.parametrize("num_outputs", [1, 5])
@pytest.mark.parametrize("alpha", [0.0, 0.1, 1.0])
def test_linear(num_outputs, alpha):
    # compare against an sklearn model
    rs = cn.random.RandomState(0)
    X = rs.random((1000, 10))
    g = rs.normal(size=(X.shape[0], num_outputs))
    h = rs.random(g.shape) + 0.1
    X, g, h = cn.array(X), cn.array(g), cn.array(h)
    model = (
        lb.models.Linear(alpha=alpha)
        .set_random_state(np.random.RandomState(2))
        .fit(X, g, h)
    )
    for k in range(0, num_outputs):
        sklearn_model = Ridge(alpha=alpha).fit(
            X, -g[:, k] / h[:, k], sample_weight=h[:, k]
        )
        assert np.allclose(model.bias_[k], sklearn_model.intercept_, atol=1e-5)
        assert np.allclose(model.betas_[:, k], sklearn_model.coef_, atol=1e-5)
        assert np.allclose(model.predict(X)[:, k], sklearn_model.predict(X), atol=1e-5)


@pytest.mark.parametrize("num_outputs", [3])
@pytest.mark.parametrize("alpha", [0.00000001])
def test_logistic_regression(num_outputs, alpha):
    # fit a normal logistic regression model
    # see if it compares to sklearn
    # use a very small alpha because lb doesnt regularise the same way
    X, y = make_classification(
        n_samples=100,
        n_features=10,
        n_informative=5,
        random_state=0,
        n_classes=num_outputs,
    )
    sklearn_model = LogisticRegression(
        penalty="l2", C=1.0 / alpha, fit_intercept=True, solver="lbfgs", max_iter=100
    ).fit(X, y)
    X, y = cn.array(X), cn.array(y)
    model = lb.LBClassifier(
        init=None,
        n_estimators=20,
        learning_rate=0.5,
        base_models=(lb.models.Linear(alpha=alpha),),
        verbose=True,
    ).fit(X, y)
    lb_coefficients = cn.zeros(model.models_[0].betas_.shape)
    lb_bias = cn.zeros(model.models_[0].bias_.shape)
    for m in model.models_:
        lb_coefficients += m.betas_
        lb_bias += m.bias_

    assert cn.allclose(
        log_loss(y, model.predict_proba(X)),
        log_loss(y, sklearn_model.predict_proba(X)),
        atol=1e-3,
    )
    assert cn.allclose(
        model.predict_proba(X), sklearn_model.predict_proba(X), atol=1e-3
    )
