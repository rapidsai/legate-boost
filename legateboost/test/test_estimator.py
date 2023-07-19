import numpy as np
import pytest
import utils
from sklearn.utils.estimator_checks import parametrize_with_checks

import cunumeric as cn
import legateboost as lb


@pytest.mark.parametrize("num_outputs", [1, 5])
def test_regressor(num_outputs):
    np.random.seed(2)
    X = cn.random.random((100, 10))
    y = cn.random.random((X.shape[0], num_outputs))
    model = lb.LBRegressor(
        n_estimators=20, max_depth=3, random_state=2, learning_rate=0.5
    ).fit(X, y)
    mse = lb.MSEMetric().metric(y, model.predict(X), cn.ones(y.shape[0]))
    loss = next(iter(model.train_metric_.values()))
    assert np.isclose(loss[-1], mse)
    assert utils.non_increasing(loss)

    # test print
    model.dump_trees()
    utils.sanity_check_tree_stats(model.models_)


@pytest.mark.parametrize("num_outputs", [1, 5])
def test_regressor_improving_with_depth(num_outputs):
    np.random.seed(3)
    X = cn.random.random((100, 10))
    y = cn.random.random((X.shape[0], num_outputs))
    metrics = []
    for max_depth in range(0, 10):
        model = lb.LBRegressor(n_estimators=2, random_state=0, max_depth=max_depth).fit(
            X, y
        )

        loss = next(iter(model.train_metric_.values()))
        metrics.append(loss[-1])
    assert utils.non_increasing(metrics)


@pytest.mark.parametrize("num_outputs", [1, 5])
def test_regressor_weights(num_outputs):
    """We expect that a tree with high enough depth/learning rate will reach 0
    training loss.

    Check this happens with weights.
    """
    np.random.seed(4)
    X = cn.random.random((100, 10))
    y = cn.random.random((X.shape[0], num_outputs))
    w = cn.random.random(X.shape[0])
    model = lb.LBRegressor(
        n_estimators=5, random_state=0, max_depth=10, learning_rate=1.0
    ).fit(X, y, sample_weight=w)
    loss = next(iter(model.train_metric_.values()))
    assert loss[-1] < 1e-5


def test_regressor_determinism():
    X = cn.random.random((100, 10))
    y = cn.random.random(X.shape[0])
    preds = []
    for _ in range(0, 10):
        model = lb.LBRegressor(n_estimators=2, random_state=83).fit(X, y)
        p = model.predict(X)
        if preds:
            assert cn.all(p == preds[-1])
        preds.append(model.predict(X))


def test_regressor_vs_sklearn():
    # build a single randomised tree with learning rate 1
    # compare its average accuracy against sklearn
    from scipy import stats
    from sklearn.datasets import make_regression
    from sklearn.ensemble import ExtraTreesRegressor

    X, y = make_regression(n_samples=100, n_features=10, random_state=42)
    max_depth = 12
    lb_score = []
    skl_score = []
    for i in range(20):
        model = lb.LBRegressor(
            n_estimators=1,
            random_state=i,
            learning_rate=1.0,
            max_depth=max_depth,
            init=None,
        ).fit(X, y)
        skl_model = ExtraTreesRegressor(
            n_estimators=1, random_state=i, max_depth=max_depth
        ).fit(X, y)
        skl_score.append(skl_model.score(X, y))
        lb_score.append(model.score(X, y))

    # two sample t-test
    _, p = stats.ttest_ind(skl_score, lb_score)
    assert p > 0.05


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
def test_classifier(num_class, objective):
    np.random.seed(3)
    X = cn.random.random((100, 10))
    y = cn.random.randint(0, num_class, X.shape[0])
    model = lb.LBClassifier(n_estimators=10, objective=objective).fit(X, y)
    metric = model._metrics[0]
    pred = (
        model.predict_proba(X)
        if metric.requires_probability()
        else model.predict_raw(X)
    )
    loss = metric.metric(y, pred, cn.ones(y.shape[0]))
    train_loss = next(iter(model.train_metric_.values()))
    assert np.isclose(train_loss[-1], loss)
    assert utils.non_increasing(train_loss)
    assert model.score(X, y) > 0.7
    utils.sanity_check_tree_stats(model.models_)


@pytest.mark.parametrize("num_class", [2, 5])
@pytest.mark.parametrize("objective", ["log_loss", "exp"])
def test_classifier_weights(num_class, objective):
    np.random.seed(7)
    X = cn.random.random((100, 10))
    y = cn.random.randint(0, num_class, X.shape[0])
    w = cn.random.random(X.shape[0])
    model = lb.LBClassifier(
        n_estimators=10, learning_rate=1.0, max_depth=10, objective=objective
    ).fit(X, y, w)
    train_loss = next(iter(model.train_metric_.values()))
    assert np.isclose(train_loss[-1], 0.0, atol=1e-3)


@pytest.mark.parametrize("num_class", [2, 5])
@pytest.mark.parametrize("objective", ["log_loss", "exp"])
def test_classifier_improving_with_depth(num_class, objective):
    np.random.seed(3)
    X = cn.random.random((100, 10))
    y = cn.random.randint(0, num_class, X.shape[0])
    metrics = []
    for max_depth in range(0, 5):
        model = lb.LBClassifier(
            n_estimators=2, random_state=0, max_depth=max_depth, objective=objective
        ).fit(X, y)
        train_loss = next(iter(model.train_metric_.values()))
        metrics.append(train_loss[-1])
    assert utils.non_increasing(metrics)
