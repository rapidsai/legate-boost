import numpy as np
import pytest
import scipy.stats as stats
from sklearn.datasets import make_regression

import cupynumeric as cn
import legateboost as lb
from legateboost.testing.utils import non_increasing


def test_basic():
    # tree loss can go to zero
    X = cn.array([[0.0], [1.0]])
    g = cn.array([[0.0], [-1.0]])
    h = cn.array([[1.0], [1.0]])
    model = (
        lb.models.Tree(max_depth=1, alpha=0.0)
        .set_random_state(np.random.RandomState(2))
        .fit(X, g, h)
    )
    assert np.allclose(model.predict(X), np.array([[0.0], [1.0]]))


@pytest.mark.parametrize("num_outputs", [1, 5])
def test_improving_with_depth(num_outputs):
    rs = cn.random.RandomState(0)
    X = rs.random((10000, 100))
    g = rs.normal(size=(X.shape[0], num_outputs))
    h = rs.random(g.shape) + 0.1
    X, g, h = cn.array(X), cn.array(g), cn.array(h)
    y = -g / h
    metrics = []
    for max_depth in range(0, 12):
        model = (
            lb.models.Tree(max_depth=max_depth)
            .set_random_state(np.random.RandomState(2))
            .fit(X, g, h)
        )
        predict = model.predict(X)
        loss = ((predict - y) ** 2 * h).sum(axis=0) / h.sum(axis=0)
        loss = loss.mean()
        metrics.append(loss)

    assert non_increasing(metrics)
    assert metrics[-1] < metrics[0]


def test_max_depth():
    # we should be able to run deep trees with OOM
    max_depth = 20
    X = cn.random.random((2, 1))
    y = cn.array([500.0, 500.0])
    model = lb.LBRegressor(
        init=None,
        base_models=(lb.models.Tree(max_depth=max_depth),),
        learning_rate=1.0,
        n_estimators=1,
        random_state=0,
    )

    model.fit(X, y)


def test_alpha():
    X = cn.random.random((2, 1))
    y = cn.array([500.0, 500.0])
    alpha = 10.0
    model = lb.LBRegressor(
        init=None,
        base_models=(lb.models.Tree(alpha=alpha, max_depth=0),),
        learning_rate=1.0,
        n_estimators=1,
        random_state=0,
    )
    model.fit(X, y)
    assert np.isclose(model.predict(X)[0], y.sum() / (y.size + alpha))


def get_feature_distribution(model):
    histogram = cn.zeros(model.n_features_in_)
    for m in model:
        histogram += cn.histogram(
            m.feature, bins=model.n_features_in_, range=(0, model.n_features_in_)
        )[0]
    return histogram / histogram.sum()


def test_feature_sample():
    X, y = make_regression(
        n_samples=100, n_features=10, n_informative=2, random_state=0
    )

    # We have a distribution of how often each feature is used in the model
    # Hypothesis: the baseline model should use the best features more often and the
    # sampled model should use other features more often as it won't always see the
    # best features. So we expect the entropy of the baseline model feature
    # disribution to be lower than the sampled model
    # i.e. the sampled model should be closer to uniform distribution
    baseline_samples = []
    sampled_samples = []
    for trial in range(5):
        baseline_model = lb.LBRegressor(
            base_models=(lb.models.Tree(feature_fraction=1.0),), random_state=trial
        ).fit(X, y)
        sampled_model = lb.LBRegressor(
            base_models=(lb.models.Tree(feature_fraction=0.5),), random_state=trial
        ).fit(X, y)
        baseline_samples.append(stats.entropy(get_feature_distribution(baseline_model)))
        sampled_samples.append(stats.entropy(get_feature_distribution(sampled_model)))

    _, p = stats.mannwhitneyu(baseline_samples, sampled_samples, alternative="less")
    assert p < 0.05

    # the no features model contains only the bias term - no splits
    no_features_model = lb.LBRegressor(
        base_models=(lb.models.Tree(feature_fraction=0.0),), random_state=0
    ).fit(X, y)
    for m in no_features_model:
        assert m.num_nodes() == 1


def test_callable_feature_sample():
    def feature_fraction():
        return cn.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=bool)

    rng = np.random.RandomState(0)
    X = rng.randn(100, 10)
    y = rng.randn(100)
    model = lb.LBRegressor(
        base_models=(lb.models.Tree(feature_fraction=feature_fraction),),
        random_state=0,
    ).fit(X, y)

    assert get_feature_distribution(model)[1] == 1.0

    def feature_fraction_int():
        return cn.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0])

    with pytest.raises(
        ValueError,
        match=r"feature_fraction must return a boolean array of shape \(n_features,\)",
    ):
        lb.LBRegressor(
            base_models=(lb.models.Tree(feature_fraction=feature_fraction_int),),
            random_state=0,
        ).fit(X, y)

    def feature_fraction_wrong_shape():
        return cn.array([0, 1, 0, 0, 0, 0, 0, 0, 0], dtype=bool)

    with pytest.raises(
        ValueError,
        match=r"feature_fraction must return a boolean array of shape \(n_features,\)",
    ):
        lb.LBRegressor(
            base_models=(
                lb.models.Tree(feature_fraction=feature_fraction_wrong_shape),
            ),
            random_state=0,
        ).fit(X, y)
