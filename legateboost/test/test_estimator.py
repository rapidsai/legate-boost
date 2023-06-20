import numpy as np
import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks

import cunumeric as cn
import legateboost as lb


def non_increasing(x):
    return all(x >= y for x, y in zip(x, x[1:]))


def non_decreasing(x):
    return all(x <= y for x, y in zip(x, x[1:]))


def sanity_check_tree_stats(trees):
    for tree in trees:
        # Check that we have no 0 hessian splits
        split_nodes = tree.feature != -1
        assert cn.all(tree.hessian[split_nodes] > 0.0)

        # Check gain is positive
        assert cn.all(tree.gain[split_nodes] > 0.0)

        # Check that hessians of leaves add up to root.
        leaves = (tree.feature == -1) & (tree.hessian[:, 0] > 0.0)
        leaf_sum = tree.hessian[leaves].sum(axis=0)
        assert np.isclose(leaf_sum, tree.hessian[0]).all()


@pytest.mark.parametrize("num_outputs", [1, 5])
def test_regressor(num_outputs):
    np.random.seed(2)
    X = cn.random.random((100, 10))
    y = cn.random.random((X.shape[0], num_outputs))
    model = lb.LBRegressor(
        n_estimators=20, max_depth=3, random_state=2, learning_rate=0.5
    ).fit(X, y)
    mse = lb.MSEMetric().metric(y, model.predict(X), cn.ones(y.shape[0]))
    assert np.isclose(model.train_metric_[-1], mse)
    assert non_increasing(model.train_metric_)

    # test print
    model.dump_trees()
    sanity_check_tree_stats(model.models_)


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
        metrics.append(model.train_metric_[-1])
    assert non_increasing(metrics)


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
    assert cn.isclose(model.train_metric_[-1], 0.0, atol=1e-5)


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
def test_classifier(num_class):
    np.random.seed(3)
    X = cn.random.random((100, 10))
    y = cn.random.randint(0, num_class, X.shape[0])
    model = lb.LBClassifier(n_estimators=10).fit(X, y)
    loss = lb.LogLossMetric().metric(y, model.predict_proba(X), cn.ones(y.shape[0]))
    assert np.isclose(model.train_metric_[-1], loss)
    assert non_increasing(model.train_metric_)
    assert model.score(X, y) > 0.7
    sanity_check_tree_stats(model.models_)


@pytest.mark.parametrize("num_class", [2, 5])
def test_classifier_weights(num_class):
    np.random.seed(7)
    X = cn.random.random((100, 10))
    y = cn.random.randint(0, num_class, X.shape[0])
    w = cn.random.random(X.shape[0])
    model = lb.LBClassifier(n_estimators=10, learning_rate=1.0, max_depth=10).fit(
        X, y, w
    )
    assert np.isclose(model.train_metric_[-1], 0.0, atol=1e-3)


@pytest.mark.parametrize("num_class", [2, 5])
def test_classifier_improving_with_depth(num_class):
    np.random.seed(3)
    X = cn.random.random((100, 10))
    y = cn.random.randint(0, num_class, X.shape[0])
    metrics = []
    for max_depth in range(0, 5):
        model = lb.LBClassifier(
            n_estimators=2, random_state=0, max_depth=max_depth
        ).fit(X, y)
        metrics.append(model.train_metric_[-1])
    assert non_increasing(metrics)


def test_prediction():
    tree = lb.TreeStructure(
        cn.array(
            [0.0, 0.0, -0.04619769, 0.01845179, -0.01151532, 0.0, 0.0, 0.0, 0.0, 0.0]
        ).reshape(10, 1),
        cn.array([0, 1, -1, -1, -1, -1, -1, -1, -1, -1]).astype(cn.int32),
        cn.array([0.79172504, 0.71518937, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        cn.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        cn.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).reshape(
            10,
            1,
        ),
    )
    """0:[f0<=0.79172504] yes=1 no=2 1:[f1<=0.71518937] yes=3 no=4
    3:leaf=0.01845179 4:leaf=-0.01151532 2:leaf=-0.04619769."""
    pred = tree.predict(cn.array([[1.0, 0.0], [0.5, 0.5]]))
    assert pred[0] == -0.04619769
    assert pred[1] == 0.01845179
    assert tree.predict(cn.array([[0.5, 1.0]])) == -0.01151532
    assert tree.predict(cn.array([[0.79172504, 0.71518937]])) == 0.01845179
