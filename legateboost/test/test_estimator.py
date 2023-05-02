from sklearn.utils.estimator_checks import parametrize_with_checks

import cunumeric as cn
import legateboost as lbst


def non_increasing(x):
    return all(x >= y for x, y in zip(x, x[1:]))


def non_decreasing(x):
    return all(x <= y for x, y in zip(x, x[1:]))


def test_regressor():
    X = cn.random.random((100, 10))
    y = cn.random.random(X.shape[0])
    model = lbst.LBRegressor().fit(X, y)
    assert non_increasing(model.train_score_)


def test_regressor_improving_with_depth():
    X = cn.random.random((100, 10))
    y = cn.random.random(X.shape[0])
    scores = []
    for max_depth in range(0, 10):
        model = lbst.LBRegressor(
            n_estimators=10, random_state=0, max_depth=max_depth
        ).fit(X, y)
        scores.append(model.score(X, y))
    assert non_decreasing(scores)


def test_regressor_determinism():
    X = cn.random.random((100, 10))
    y = cn.random.random(X.shape[0])
    preds = []
    for _ in range(0, 10):
        model = lbst.LBRegressor(n_estimators=10, random_state=83).fit(X, y)
        p = model.predict(X)
        if preds:
            assert cn.all(p == preds[-1])
        preds.append(model.predict(X))


@parametrize_with_checks([lbst.LBRegressor()])
def test_sklearn_compatible_estimator(estimator, check):
    check(estimator)
