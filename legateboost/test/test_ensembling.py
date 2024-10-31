import numpy as np
import pytest as pt

import legateboost as lb
from legateboost.testing.utils import all_base_models


@pt.fixture
def X_y_regression():
    rng = np.random.RandomState(2)
    X = rng.random((100, 10))
    y = rng.random(X.shape[0])
    return X, y


@pt.fixture
def X_y_classification():
    rng = np.random.RandomState(2)
    X = rng.random((100, 10))
    y = rng.randint(0, 2, X.shape[0])
    return X, y


def test_add(X_y_regression, X_y_classification):
    X, y = X_y_regression
    model_a = lb.LBRegressor(
        n_estimators=5, learning_rate=0.1, base_models=all_base_models()
    ).fit(X, y)
    model_b = lb.LBRegressor(
        n_estimators=2, learning_rate=0.2, base_models=all_base_models()
    ).fit(X, y)
    model_c = model_a + model_b
    assert not np.allclose(model_a.predict(X), model_b.predict(X))
    assert np.allclose(model_a.predict(X) + model_b.predict(X), model_c.predict(X))
    assert len(model_c) == 7
    assert ((model_a.model_init_ + model_b.model_init_) == model_c.model_init_).all()
    assert set(model_a) < set(model_c)
    assert set(model_b) < set(model_c)
    assert (set(model_a) | set(model_b)) == set(model_c)

    X, y = X_y_classification
    model_a = lb.LBClassifier(
        n_estimators=5, learning_rate=0.1, base_models=all_base_models()
    ).fit(X, y)
    model_b = lb.LBClassifier(
        n_estimators=2, learning_rate=0.2, base_models=all_base_models()
    ).fit(X, y)
    model_c = model_a + model_b
    assert not np.allclose(model_a.predict(X), model_b.predict(X))
    assert np.allclose(
        model_a.predict_raw(X) + model_b.predict_raw(X), model_c.predict_raw(X)
    )


def test_add_different_n_features(X_y_regression):
    X, y = X_y_regression
    model_a = lb.LBRegressor(n_estimators=1).fit(X, y)
    X = np.random.random((100, 5))
    model_b = lb.LBRegressor(n_estimators=1).fit(X, y)
    with pt.raises(ValueError, match="n_features_in_ for operand a has value"):
        model_a + model_b


def test_add_different_n_classes(X_y_classification):
    X, y = X_y_classification
    model_a = lb.LBClassifier(n_estimators=1).fit(X, y)
    y = np.random.randint(0, 3, X.shape[0])
    model_b = lb.LBClassifier(n_estimators=1).fit(X, y)
    with pt.raises(ValueError, match="classes_ for operand a has value"):
        model_a + model_b


def test_multiply(X_y_regression, X_y_classification):
    X, y = X_y_regression
    model_a = lb.LBRegressor(
        n_estimators=2, learning_rate=0.1, base_models=all_base_models()
    ).fit(X, y)
    model_b = model_a * 2
    assert np.allclose(model_a.predict(X) * 2, model_b.predict(X))
    model_c = model_a * 0.5
    assert np.allclose(model_a.predict(X) * 0.5, model_c.predict(X))
    assert np.allclose((model_a * 0.5 + model_a * 0.5).predict(X), model_a.predict(X))

    X, y = X_y_classification
    model_a = lb.LBClassifier(n_estimators=2, base_models=all_base_models()).fit(X, y)
    model_b = model_a * 2.0
    assert np.allclose(model_a.predict_raw(X) * 2, model_b.predict_raw(X))


def test_regression_and_classification(X_y_regression, X_y_classification):
    X_r, y_r = X_y_regression
    X_c, y_c = X_y_classification
    model_r = lb.LBRegressor(
        n_estimators=2, learning_rate=0.1, base_models=all_base_models()
    ).fit(X_r, y_r)
    model_c = lb.LBClassifier(
        n_estimators=2, learning_rate=0.1, base_models=all_base_models()
    ).fit(X_c, y_c)
    with pt.raises(TypeError, match="Can only add two instances of the same class"):
        model_r + model_c


def test_new_attribute(X_y_regression):
    X, y = X_y_regression
    model = lb.LBRegressor(
        n_estimators=2, learning_rate=0.1, base_models=all_base_models()
    ).fit(X, y)
    model.new_attribute = 1
    with pt.raises(
        ValueError,
        match="Attribute new_attribute has no defined behaviour for addition",
    ):
        model + model
