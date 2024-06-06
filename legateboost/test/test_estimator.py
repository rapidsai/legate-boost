import numpy as np
import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks

import cunumeric as cn
import legateboost as lb

from .utils import non_increasing, sanity_check_models


def test_init():
    np.random.seed(2)
    X = np.random.random((100, 10))
    y = np.random.random((X.shape[0], 2))
    model = lb.LBRegressor(n_estimators=0, init="average").fit(X, y)
    assert cn.allclose(model.model_init_, y.mean(axis=0))
    # weights
    w = cn.ones(X.shape[0])
    w[50:100] = 0.0
    model = model.fit(X, y, sample_weight=w)
    assert cn.allclose(model.model_init_, y[0:50].mean(axis=0))


@pytest.mark.parametrize("init", [None, "average"])
def test_update(init):
    np.random.seed(2)
    X = np.random.random((1000, 10))
    y = np.random.random((X.shape[0], 2))
    # shift the distribution of the first dataset half
    y[0 : X.shape[0] // 2] += 3.0

    eval_result = {}
    model = lb.LBRegressor(
        init=init,
        n_estimators=20,
        random_state=2,
        learning_rate=0.1,
        base_models=(lb.models.Tree(alpha=2.0),),
    )
    # fit the model on a half dataset
    metric = lb.MSEMetric()
    model.fit(X[0 : X.shape[0] // 2], y[0 : X.shape[0] // 2], eval_result=eval_result)
    half_data_train_loss = metric.metric(y, model.predict(X), cn.ones(y.shape[0]))
    # update the model on the full dataset
    model.update(X, y, eval_result=eval_result)
    update_train_loss = metric.metric(y, model.predict(X), cn.ones(y.shape[0]))
    assert update_train_loss < half_data_train_loss

    # check that updating with same dataset results in exact same model
    model.fit(X, y)
    pred = model.predict(X)
    model.update(X, y)
    updated_pred = model.predict(X)
    assert (pred == updated_pred).all()


@pytest.mark.parametrize("num_outputs", [1, 5])
@pytest.mark.parametrize("objective", ["squared_error", "normal", "quantile"])
@pytest.mark.parametrize(
    "base_models",
    [
        (lb.models.Tree(max_depth=5),),
        (lb.models.Linear(),),
        (lb.models.Tree(max_depth=1), lb.models.Linear()),
        (lb.models.KRR(),),
    ],
)
def test_regressor(num_outputs, objective, base_models):
    if objective == "quantile" and num_outputs > 1:
        pytest.skip("Quantile objective not implemented for multi-output")

    np.random.seed(2)
    X = np.random.random((100, 10))
    y = np.random.random((X.shape[0], num_outputs))
    eval_result = {}
    model = lb.LBRegressor(
        n_estimators=20,
        objective=objective,
        random_state=2,
        learning_rate=0.1,
        base_models=base_models,
    ).fit(X, y, eval_result=eval_result)
    loss_recomputed = model._metrics[0].metric(y, model.predict(X), cn.ones(y.shape[0]))
    loss = next(iter(eval_result["train"].values()))
    assert np.isclose(loss[-1], loss_recomputed)
    assert non_increasing(loss)
    sanity_check_models(model)


@pytest.fixture
def test_name(request):
    return request.node.name


@parametrize_with_checks([lb.LBRegressor(), lb.LBClassifier()])
def test_sklearn_compatible_estimator(estimator, check, test_name):
    if "check_classifiers_classes" in test_name:
        pytest.skip("Legateboost cannot handle string class labels.")
    check(estimator)


@pytest.mark.parametrize("num_class", [2, 5])
@pytest.mark.parametrize("objective", ["log_loss", "exp"])
@pytest.mark.parametrize(
    "base_models",
    [
        (lb.models.Tree(max_depth=5),),
        (lb.models.Linear(),),
        (lb.models.Tree(max_depth=1), lb.models.Linear()),
        (lb.models.KRR(),),
    ],
)
def test_classifier(num_class, objective, base_models):
    np.random.seed(3)
    X = np.random.random((100, 10))
    y = np.random.randint(0, num_class, X.shape[0])
    eval_result = {}
    model = lb.LBClassifier(
        n_estimators=10, objective=objective, base_models=base_models
    ).fit(X, y, eval_result=eval_result)
    metric = model._metrics[0]
    proba = model.predict_proba(X)
    assert cn.all(proba >= 0) and cn.all(proba <= 1)
    assert cn.all(cn.argmax(proba, axis=1) == model.predict(X))

    loss = metric.metric(y, proba, cn.ones(y.shape[0]))
    train_loss = next(iter(eval_result["train"].values()))
    assert np.isclose(train_loss[-1], loss)
    assert non_increasing(train_loss)
    # better than random guessing accuracy
    assert model.score(X, y) > 1 / num_class
    sanity_check_models(model)


def test_normal_distribution():
    # check sd converges as expected
    np.random.seed(2)
    X = np.random.random((100, 2))
    y = np.random.normal(5, 2, X.shape[0])
    model = lb.LBRegressor(
        n_estimators=50,
        objective="normal",
        base_models=(lb.models.Tree(max_depth=0),),
        random_state=2,
        learning_rate=0.3,
        init=None,
    ).fit(X, y)

    pred = model.predict(X)[0]
    assert cn.allclose(pred[0], y.mean(), atol=1e-2)
    assert cn.allclose(pred[1], cn.log(y.var()) / 2, atol=1e-2)

    # check we don't get numerical errors on 0 variance
    X = cn.array([[0.0], [0.0]])
    y = cn.array([1.0, 1.0])
    model = lb.LBRegressor(
        n_estimators=50,
        objective="normal",
        base_models=(lb.models.Tree(max_depth=0),),
        random_state=2,
        learning_rate=0.5,
        init=None,
    ).fit(X, y)
    pred = model.predict(X)[0]
    assert cn.allclose(pred[0], y.mean(), atol=1e-2)
    assert cn.all(pred[1] == -5)
