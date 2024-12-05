import numpy as np
import pytest
import scipy
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.utils.estimator_checks import parametrize_with_checks

import cupynumeric as cn
import legateboost as lb
from legateboost.testing.utils import (
    all_base_models,
    non_increasing,
    sanity_check_models,
)


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

    # check that updating with same dataset results in same model
    model.fit(X, y)
    pred = model.predict(X)
    model.update(X, y)
    updated_pred = model.predict(X)
    assert np.allclose(pred, updated_pred)


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


def test_subsample():
    subsample_test_mse = []
    full_test_mse = []
    for i in range(5):
        X, y = make_regression(
            n_samples=1000, n_features=10, noise=10.0, random_state=i
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.5, random_state=i
        )
        params = {
            "n_estimators": 20,
            "random_state": i,
            "learning_rate": 0.5,
            "base_models": (lb.models.Tree(max_depth=12, alpha=0.0),),
        }
        # Overfit the data and check if subsample improves the model
        subsample_eval_result = {}
        lb.LBRegressor(
            **params,
            subsample=0.5,
        ).fit(
            X_train,
            y_train,
            eval_result=subsample_eval_result,
            eval_set=[(X_test, y_test)],
        )
        full_eval_result = {}
        lb.LBRegressor(
            **params,
        ).fit(
            X_train, y_train, eval_result=full_eval_result, eval_set=[(X_test, y_test)]
        )
        full_test_mse.append(full_eval_result["eval-0"]["mse"][-1])
        subsample_test_mse.append(subsample_eval_result["eval-0"]["mse"][-1])

    assert np.mean(subsample_test_mse) < np.mean(full_test_mse)


def test_iterator_methods():
    np.random.seed(2)
    X = np.random.random((100, 10))
    y = np.random.random((X.shape[0], 2))
    model = lb.LBRegressor(n_estimators=5).fit(X, y)
    assert len(model) == 5
    assert list(model) == list(model.models_)
    for i, est in enumerate(model):
        assert est == model[i]


@pytest.mark.parametrize(
    "base_model", filter(lambda m: m.supports_csr(), all_base_models()), ids=type
)
def test_csr_input(base_model):
    csr_matrix = pytest.importorskip("legate_sparse").csr_matrix
    X_scipy = scipy.sparse.csr_matrix([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]])
    X_legate_sparse = csr_matrix(X_scipy)
    y = cn.array([1.0, 2.0])
    model = lb.LBRegressor(
        n_estimators=1,
        base_models=(base_model,),
    )
    model.fit(X_scipy, y)
    model.fit(X_legate_sparse, y)
