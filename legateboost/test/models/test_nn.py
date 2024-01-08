import numpy as np
import pytest
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

import legateboost as lb


@pytest.mark.parametrize("random_state", [0, 1, 2])
@pytest.mark.parametrize("hidden_layer_sizes", [(), (100,), (100, 100), (10, 10, 10)])
def test_nn(random_state, hidden_layer_sizes):
    X, y = fetch_california_housing(return_X_y=True)
    X = X[:1000]
    y = y[:1000]
    # add some extra outputs to make sure we can handle multiple outputs
    y = np.tile(y.reshape((-1, 1)), (1, 3))
    X = StandardScaler().fit_transform(X)
    max_iter = 200
    nn = lb.LBRegressor(
        n_estimators=1,
        init=None,
        learning_rate=1.0,
        base_models=(
            lb.models.NN(
                max_iter=max_iter,
                verbose=1,
                hidden_layer_sizes=hidden_layer_sizes,
                m=10,
            ),
        ),
        random_state=random_state,
    ).fit(X, y)
    sklearn = MLPRegressor(
        alpha=0.0,
        solver="lbfgs",
        activation="tanh",
        verbose=0,
        hidden_layer_sizes=hidden_layer_sizes,
        max_iter=max_iter,
        random_state=random_state,
    ).fit(X, y)
    pred = nn.predict(X)
    lb_mse = mean_squared_error(y, pred)
    sklearn_mse = mean_squared_error(y, sklearn.predict(X))
    baseline = mean_squared_error(y, np.full_like(y, y.mean()))
    # check we are doing better than average
    assert lb_mse < baseline
    # check we are in the ballpark of sklearn
    assert lb_mse < sklearn_mse * 1.5
