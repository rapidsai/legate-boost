import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

import legateboost as lb


def test_nn():
    X, y = fetch_california_housing(return_X_y=True)
    X = X[:1000]
    y = y[:1000]
    X = StandardScaler().fit_transform(X)
    # np.random.seed(6)
    # n = 100
    # X = np.linspace(0, 2*np.pi, n).reshape((n, 1))
    # y = np.sin(X[:,0]) + np.random.randn(n) * 0.1 + 3
    hidden_layer_sizes = (100, 100, 100)
    max_iter = 100
    random_state = 7
    nn = lb.LBRegressor(
        n_estimators=1,
        init=None,
        learning_rate=1.0,
        base_models=(
            lb.models.NN(
                max_iter=max_iter, verbose=1, hidden_layer_sizes=hidden_layer_sizes
            ),
        ),
        random_state=random_state,
    ).fit(X, y)
    sklearn = MLPRegressor(
        solver="lbfgs",
        verbose=10,
        hidden_layer_sizes=hidden_layer_sizes,
        max_iter=max_iter,
        random_state=random_state,
    ).fit(X, y)
    pred = nn.predict(X)
    # check we are doing better than average
    assert mean_squared_error(y, pred) < mean_squared_error(
        y, np.full_like(y, y.mean())
    )
    print("MSE: ", mean_squared_error(y, pred))
    print("baseline MSE", mean_squared_error(y, np.full_like(y, y.mean())))
    print("sklearn MSE", mean_squared_error(y, sklearn.predict(X)))
