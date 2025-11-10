import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import log_loss

import cupynumeric as cn
import legateboost as lb


@pytest.mark.parametrize(
    "solver", ["direct"]
)  # "lbfgs" is too slow to test at the moment
class TestLinear:
    def test_bias(self, solver):
        X = cn.zeros((10, 5))
        rs = cn.random.RandomState(0)
        y = rs.normal(size=(10, 2), loc=10.0, scale=1.0)
        # l2_regularization should be ignored for bias
        model = lb.LBRegressor(
            n_estimators=1,
            base_models=(lb.models.Linear(l2_regularization=2.0, solver=solver),),
            init=None,
            learning_rate=1.0,
        ).fit(X, y)
        assert cn.allclose(model.models_[0].betas_[0], y.mean(axis=0), atol=1e-5)

        # with weights
        w = rs.random(size=10) + 0.1
        model = lb.LBRegressor(
            n_estimators=1,
            base_models=(lb.models.Linear(),),
            init=None,
            learning_rate=1.0,
        ).fit(X, y, sample_weight=w)
        assert cn.allclose(
            model.models_[0].betas_[0], np.average(y, axis=0, weights=w), atol=1e-5
        )

    @pytest.mark.parametrize("num_outputs", [1, 5])
    @pytest.mark.parametrize("l2_regularization", [0.0, 0.1, 1.0])
    def test_linear(self, num_outputs, l2_regularization, solver):
        # compare against an sklearn model
        rs = cn.random.RandomState(0)
        X = rs.random((1000, 10))
        g = rs.normal(size=(X.shape[0], num_outputs))
        h = rs.random(g.shape) + 0.1
        X, g, h = cn.array(X), cn.array(g), cn.array(h)
        model = (
            lb.models.Linear(l2_regularization=l2_regularization, solver=solver)
            .set_random_state(np.random.RandomState(2))
            .fit(X, g, h)
        )
        for k in range(0, num_outputs):
            sklearn_model = Ridge(alpha=l2_regularization).fit(
                X, -g[:, k] / h[:, k], sample_weight=h[:, k]
            )
            assert np.allclose(model.betas_[0, k], sklearn_model.intercept_, atol=1e-2)
            assert np.allclose(model.betas_[1:, k], sklearn_model.coef_, atol=1e-2)
            assert np.allclose(
                model.predict(X)[:, k], sklearn_model.predict(X), atol=1e-2
            ), (np.abs(model.predict(X)[:, k] - sklearn_model.predict(X)).max(),)

    @pytest.mark.parametrize("num_outputs", [3])
    @pytest.mark.parametrize("l2_regularization", [0.00000001])
    def test_logistic_regression(self, num_outputs, l2_regularization, solver):
        # fit a normal logistic regression model
        # see if it compares to sklearn
        # use a very small l2_regularization because lb doesnt regularise the same way
        X, y = make_classification(
            n_samples=100,
            n_features=10,
            n_informative=5,
            random_state=0,
            n_classes=num_outputs,
        )
        sklearn_model = LogisticRegression(
            penalty="l2",
            C=1.0 / l2_regularization,
            fit_intercept=True,
            solver="lbfgs",
            max_iter=100,
        ).fit(X, y)
        X, y = cn.array(X), cn.array(y)
        model = lb.LBClassifier(
            init=None,
            n_estimators=20,
            learning_rate=0.5,
            base_models=(
                lb.models.Linear(l2_regularization=l2_regularization, solver=solver),
            ),
            verbose=True,
        ).fit(X, y)
        assert cn.allclose(
            log_loss(y, model.predict_proba(X)),
            log_loss(y, sklearn_model.predict_proba(X)),
            atol=1e-3,
        )
        assert cn.allclose(
            model.predict_proba(X), sklearn_model.predict_proba(X), atol=1e-3
        )

    def test_singular_matrix(self, solver):
        # check that we can solve an underdetermined system
        rs = cn.random.RandomState(0)
        X = cn.random.random((2, 10))
        g = rs.normal(size=(X.shape[0], 1))
        h = cn.ones(g.shape)
        model = (
            lb.models.Linear(solver=solver)
            .set_random_state(np.random.RandomState(2))
            .fit(X, g, h)
        )
        assert not cn.any(cn.isnan(model.predict(X)))
