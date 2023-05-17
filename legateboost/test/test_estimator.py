import warnings

import numpy as np
from sklearn.utils.estimator_checks import parametrize_with_checks

import cunumeric as cn
import legateboost as lb


def non_increasing(x):
    return all(x >= y for x, y in zip(x, x[1:]))


def non_decreasing(x):
    return all(x <= y for x, y in zip(x, x[1:]))


def test_regressor():
    X = cn.random.random((100, 10))
    y = cn.random.random(X.shape[0])
    model = lb.LBRegressor().fit(X, y)
    assert non_increasing(model.train_metric_)

    # test print
    model.dump_trees()


def test_regressor_improving_with_depth():
    X = cn.random.random((100, 10))
    y = cn.random.random(X.shape[0])
    metrics = []
    for max_depth in range(0, 10):
        model = lb.LBRegressor(n_estimators=2, random_state=0, max_depth=max_depth).fit(
            X, y
        )
        metrics.append(model.train_metric_[-1])
    assert non_increasing(metrics)


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


@parametrize_with_checks([lb.LBRegressor(), lb.LBClassifier()])
def test_sklearn_compatible_estimator(estimator, check):
    check(estimator)


def test_classifier():
    np.random.seed(3)
    X = cn.random.random((100, 10))
    y = cn.random.randint(0, 1, X.shape[0])
    model = lb.LBClassifier(n_estimators=5).fit(X, y)
    assert non_increasing(model.train_metric_)
    assert model.score(X, y) > 0.7


def test_classifier_improving_with_depth():
    np.random.seed(3)
    X = cn.random.random((100, 10))
    y = cn.random.randint(0, 1, X.shape[0])
    metrics = []
    for max_depth in range(0, 10):
        model = lb.LBClassifier(
            n_estimators=2, random_state=0, max_depth=max_depth
        ).fit(X, y)
        metrics.append(model.train_metric_[-1])
    print(metrics)
    assert non_increasing(metrics)


def test_prediction():
    tree = lb.TreeStructure.from_arrays(
        cn.array([1, 3, -1, -1, -1, -1, -1, -1, -1, -1]),
        cn.array([2, 4, -1, -1, -1, -1, -1, -1, -1, -1]),
        cn.array(
            [0.0, 0.0, -0.04619769, 0.01845179, -0.01151532, 0.0, 0.0, 0.0, 0.0, 0.0]
        ),
        cn.array([0, 1, -1, -1, -1, -1, -1, -1, -1, -1]),
        cn.array([0.79172504, 0.71518937, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    )
    """0:[f0<=0.79172504] yes=1 no=2 1:[f1<=0.71518937] yes=3 no=4
    3:leaf=0.01845179 4:leaf=-0.01151532 2:leaf=-0.04619769."""
    pred = tree.predict(cn.array([[1.0, 0.0], [0.5, 0.5]]))
    assert pred[0] == -0.04619769
    assert pred[1] == 0.01845179
    assert tree.predict(cn.array([[0.5, 1.0]])) == -0.01151532
    assert tree.predict(cn.array([[0.79172504, 0.71518937]])) == 0.01845179


def test_np_input_warning():
    with warnings.catch_warnings(record=True) as w:
        np.random.seed(3)
        X = np.random.random((100, 10))
        y = np.random.randint(0, 2, X.shape[0])
        lb.LBClassifier(n_estimators=1).fit(X, y)
        assert (
            "Input of type <class 'numpy.ndarray'>"
            + " does not implement __legate_data_interface__."
            + " Performance may be affected."
            in str(w[0].message)
        )
