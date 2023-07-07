import numpy as np
import utils
from hypothesis import Verbosity, given, settings, strategies as st

import cunumeric as cn
import legateboost as lb

settings.register_profile(
    "local", max_examples=10, deadline=None, verbosity=Verbosity.verbose
)

settings.load_profile("local")


general_model_param_strategy = st.fixed_dictionaries(
    {
        "n_estimators": st.integers(1, 20),
        "max_depth": st.integers(1, 18),
        "learning_rate": st.floats(0.01, 1.0),
        "init": st.sampled_from([None, "average"]),
        "random_state": st.integers(0, 10000),
    }
)


@given(general_model_param_strategy)
def test_regressor(model_params):
    num_outputs = 1
    np.random.seed(2)
    X = cn.random.random((100, 10))
    y = cn.random.random((X.shape[0], num_outputs))
    model = lb.LBRegressor(**model_params).fit(X, y)
    mse = lb.MSEMetric().metric(y, model.predict(X), cn.ones(y.shape[0]))
    assert np.isclose(model.train_metric_[-1], mse)
    assert utils.non_increasing(model.train_metric_)

    # test print
    model.dump_trees()
    utils.sanity_check_tree_stats(model.models_)
