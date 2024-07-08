import pytest
from sklearn.datasets import make_classification

import legateboost as lb


@pytest.mark.parametrize("random_state", range(2))
@pytest.mark.parametrize(
    "metric", [None, lb.metrics.MSEMetric(), lb.metrics.ExponentialMetric()]
)
def test_global_shapley_attributions(random_state, metric):
    X, y = make_classification(random_state=10, n_features=10)
    model = lb.LBRegressor(n_estimators=5).fit(X, y)
    null_loss, shapley, std = model.global_attributions(
        X,
        y,
        metric,
        n_background_samples=10,
        random_state=random_state,
        assert_efficiency=True,
    )
