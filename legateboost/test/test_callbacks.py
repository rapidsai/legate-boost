import numpy as np
import pytest
from sklearn.datasets import make_regression

import cupynumeric as cn
import legateboost as lb


@pytest.fixture
def regression_dataset():
    rs = np.random.RandomState(0)
    X, y = make_regression(n_samples=100, random_state=rs)
    X, y = cn.array(X), cn.array(y)
    validation_size = 0.2
    samples = rs.permutation(X.shape[0])
    train = samples[: int(X.shape[0] * (1 - validation_size))]
    valid = samples[int(X.shape[0] * (1 - validation_size)) :]
    X_train, X_valid = X[train], X[valid]
    y_train, y_valid = y[train], y[valid]
    return X_train, X_valid, y_train, y_valid


def test_early_stopping(regression_dataset):
    X_train, X_valid, y_train, y_valid = regression_dataset
    m = lb.metrics.MSEMetric()
    n_estimators = 100
    cb = lb.callbacks.EarlyStopping(5, verbose=True)
    model = lb.LBRegressor(
        verbose=True,
        n_estimators=n_estimators,
        metric=m,
        learning_rate=0.5,
        base_models=(lb.models.Tree(max_depth=12),),
        callbacks=[cb],
        random_state=1,
    )
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)])
    assert len(model.models_) == cb.best_score[0] + 1
    assert n_estimators > cb.best_score[0]
    assert (
        m.metric(y_valid, model.predict(X_valid), cn.ones_like(y_valid))
        == cb.best_score[1]
    )

    # training continuation
    model.partial_fit(X_train, y_train, eval_set=[(X_valid, y_valid)])
    assert len(model.models_) == cb.best_score[0] + 1
    assert n_estimators > cb.best_score[0]
    assert (
        m.metric(y_valid, model.predict(X_valid), cn.ones_like(y_valid))
        == cb.best_score[1]
    )

    with pytest.raises(
        ValueError, match="Must have at least 1 validation dataset for early stopping."
    ):
        model.partial_fit(X_train, y_train, eval_set=[])
