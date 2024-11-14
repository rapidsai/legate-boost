import pytest
from sklearn.datasets import make_classification, make_regression

import cunumeric as cn
import legateboost as lb


@pytest.mark.parametrize("random_state", range(2))
@pytest.mark.parametrize("metric", [None, lb.metrics.MSEMetric()])
@pytest.mark.parametrize("num_outputs", [1, 2])
def test_regressor_global_shapley_attributions(random_state, metric, num_outputs):
    X, y = make_regression(random_state=10, n_features=10, n_targets=num_outputs)
    model = lb.LBRegressor(n_estimators=5).fit(X, y)
    shapley, se = model.global_attributions(
        X,
        y,
        metric=metric,
        n_samples=20,
        random_state=random_state,
        check_efficiency=True,
    )
    assert cn.isfinite(shapley).all()
    assert cn.isfinite(se).all()
    assert (se >= 0).all()


@pytest.mark.parametrize("metric", [None, lb.metrics.ExponentialMetric()])
@pytest.mark.parametrize("num_classes", [2, 3])
def test_classifier_global_shapley_attributions(metric, num_classes):
    X, y = make_classification(
        random_state=10, n_features=10, n_classes=num_classes, n_clusters_per_class=1
    )
    model = lb.LBClassifier(n_estimators=5, random_state=9).fit(X, y)
    shapley, se = model.global_attributions(
        X,
        y,
        metric=metric,
        random_state=9,
        check_efficiency=True,
    )
    assert cn.isfinite(shapley).all()
    assert cn.isfinite(se).all()
    assert (se >= 0).all()


@pytest.mark.parametrize("random_state", range(2))
@pytest.mark.parametrize("num_outputs", [1, 2])
def test_regressor_local_shapley_attributions(random_state, num_outputs):
    X, y = make_regression(random_state=10, n_features=10, n_targets=num_outputs)
    model = lb.LBRegressor(n_estimators=5, random_state=random_state).fit(X, y)
    X_background = X[:10]
    shapley, se = model.local_attributions(
        X,
        X_background,
        random_state=random_state,
        check_efficiency=True,
    )
    if num_outputs > 1:
        assert shapley.shape == (X.shape[0], X.shape[1] + 1, num_outputs)
    else:
        assert shapley.shape == (X.shape[0], X.shape[1] + 1)
    assert cn.isfinite(shapley).all()
    assert cn.isfinite(se).all()
    assert (se >= 0).all()


@pytest.mark.parametrize("random_state", range(2))
@pytest.mark.parametrize("num_classes", [2, 3])
def test_classifier_local_shapley_attributions(random_state, num_classes):
    X, y = make_classification(
        random_state=10, n_features=10, n_classes=num_classes, n_clusters_per_class=1
    )
    model = lb.LBClassifier(n_estimators=5, random_state=random_state).fit(X, y)
    X_background = X[:10]
    shapley, se = model.local_attributions(
        X,
        X_background,
        random_state=random_state,
        check_efficiency=True,
    )
    assert shapley.shape == (X.shape[0], X.shape[1] + 1, num_classes)
    assert cn.isfinite(shapley).all()
    assert cn.isfinite(se).all()
    assert (se >= 0).all()

    # Do a single row
    shapley, se = model.local_attributions(
        X[:1],
        X_background,
        random_state=random_state,
        check_efficiency=True,
    )
