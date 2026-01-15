import numpy as np
import pytest
from hypothesis import HealthCheck, Verbosity, assume, given, settings, strategies as st
from sklearn.preprocessing import StandardScaler

import cupynumeric as cn
import legateboost as lb
from legate.core import TaskTarget, get_legate_runtime
from legateboost.testing.utils import non_increasing, sanity_check_models

np.set_printoptions(threshold=10, edgeitems=1)

# adjust max_examples to control runtime
settings.register_profile(
    "local",
    max_examples=50,
    deadline=20_000,
    verbosity=Verbosity.verbose,
    suppress_health_check=(HealthCheck.too_slow,),
    print_blob=True,
)

settings.load_profile("local")


@st.composite
def tree_strategy(draw):
    if get_legate_runtime().machine.count(TaskTarget.GPU) > 0:
        max_depth = draw(st.integers(1, 8))
    else:
        max_depth = draw(st.integers(1, 6))
    alpha = draw(st.floats(0.0, 1.0))
    split_samples = draw(st.integers(1, 500))
    feature_fraction = draw(st.sampled_from([0.5, 1.0]))
    return lb.models.Tree(
        max_depth=max_depth,
        alpha=alpha,
        split_samples=split_samples,
        feature_fraction=feature_fraction,
    )


@st.composite
def nn_strategy(draw):
    alpha = draw(st.floats(0.0, 1.0))
    hidden_layer_sizes = draw(st.sampled_from([(), (100,), (100, 100), (10, 10, 10)]))
    # max iter needs to be sufficiently large, otherwise the models can make the loss
    # worse (from a bad initialization)
    max_iter = 200
    return lb.models.NN(
        alpha=alpha, hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter
    )


@st.composite
def linear_strategy(draw):
    alpha = draw(st.floats(0.0, 1.0))
    return lb.models.Linear(alpha=alpha)


@st.composite
def krr_strategy(draw):
    if draw(st.booleans()):
        sigma = draw(st.floats(0.1, 1.0))
    else:
        sigma = None
    alpha = draw(st.floats(0.0, 1.0))
    components = draw(st.integers(2, 10))
    return lb.models.KRR(n_components=components, alpha=alpha, sigma=sigma)


@st.composite
def base_model_strategy(draw):
    available_strategies = [tree_strategy(), linear_strategy(), krr_strategy()]
    # NN is disabled for CPU as it takes way too long
    if get_legate_runtime().machine.count(TaskTarget.GPU) > 0:
        available_strategies.append(nn_strategy())

    n = draw(st.integers(1, 5))
    base_models = ()
    for _ in range(n):
        base_models += (draw(st.one_of(available_strategies)),)
    return base_models


general_model_param_strategy = st.fixed_dictionaries(
    {
        "n_estimators": st.integers(1, 10),
        "base_models": base_model_strategy(),
        "init": st.sampled_from([None, "average"]),
        "random_state": st.integers(0, 10000),
    }
)

regression_param_strategy = st.fixed_dictionaries(
    {
        "objective": st.sampled_from(["squared_error", "normal", "quantile"]),
        "learning_rate": st.floats(0.01, 0.1),
    }
)


@st.composite
def regression_real_dataset_strategy(draw):
    from sklearn.datasets import fetch_california_housing, load_diabetes
    from sklearn.preprocessing import normalize

    name = draw(st.sampled_from(["california_housing", "diabetes"]))
    if name == "california_housing":
        return fetch_california_housing(return_X_y=True)
    elif name == "diabetes":
        X, y = load_diabetes(return_X_y=True)
        return X, normalize(y.reshape(-1, 1), axis=0).reshape(-1)


@st.composite
def regression_generated_dataset_strategy(draw):
    num_outputs = draw(st.integers(1, 5))
    num_features = draw(st.integers(1, 25))
    num_rows = draw(st.integers(10, 5000))
    np.random.seed(2)
    X = np.random.random((num_rows, num_features))
    y = np.random.random((X.shape[0], num_outputs))

    dtype = draw(st.sampled_from([np.float32, np.float64]))
    return X.astype(dtype), y.astype(dtype)


@st.composite
def regression_dataset_strategy(draw):
    X, y = draw(
        st.one_of(
            [
                regression_generated_dataset_strategy(),
                regression_real_dataset_strategy(),
            ]
        )
    )
    if draw(st.booleans()):
        w = np.random.random(y.shape[0])
    else:
        w = None

    X = StandardScaler().fit_transform(X)
    return X, y, w


@given(
    general_model_param_strategy,
    regression_param_strategy,
    regression_dataset_strategy(),
)
@cn.errstate(divide="raise", invalid="raise")
def test_regressor(model_params, regression_params, regression_dataset):
    X, y, w = regression_dataset
    eval_result = {}
    assume(regression_params["objective"] != "quantile" or y.ndim == 1)
    # training can diverge with normal objective and no init
    assume(
        regression_params["objective"] != "normal" or model_params["init"] == "average"
    )
    model = lb.LBRegressor(**model_params, **regression_params, verbose=True).fit(
        X, y, sample_weight=w, eval_result=eval_result
    )
    model.predict(X)
    loss = next(iter(eval_result["train"].values()))
    assert non_increasing(loss, tol=1e-1)
    sanity_check_models(model)


classification_param_strategy = st.fixed_dictionaries(
    {
        "objective": st.sampled_from(["log_loss", "exp"]),
        # we can technically have up to learning rate 1.0, however
        #  some problems may not converge (e.g. multiclass classification
        #  with many classes) unless the learning rate is sufficiently small
        "learning_rate": st.floats(0.01, 0.1),
    }
)


@st.composite
def classification_real_dataset_strategy(draw):
    from sklearn.datasets import fetch_covtype, load_breast_cancer
    from sklearn.preprocessing import normalize

    name = draw(st.sampled_from(["covtype", "breast_cancer"]))
    if name == "covtype":
        X, y = fetch_covtype(return_X_y=True, as_frame=False)
        # using the full dataset is somewhat slow
        X = X[0:5000]
        y = y[0:5000]
        return (normalize(X), y - 1, name)
    elif name == "breast_cancer":
        return (*load_breast_cancer(return_X_y=True, as_frame=False), name)


@st.composite
def classification_generated_dataset_strategy(draw):
    num_classes = draw(st.integers(2, 5))
    num_features = draw(st.integers(1, 25))
    num_rows = draw(st.integers(num_classes, 10000))
    np.random.seed(3)
    X = np.random.random((num_rows, num_features))
    y = np.random.randint(0, num_classes, size=X.shape[0])

    # ensure we have at least one of each class
    y[:num_classes] = np.arange(num_classes)

    X_dtype = draw(st.sampled_from([np.float32, np.float64]))
    y_dtype = draw(
        st.sampled_from(
            [np.int8, np.uint16, np.int32, np.int64, np.float32, np.float64]
        )
    )

    return (
        X.astype(X_dtype),
        y.astype(y_dtype),
        "Generated: num_classes: {}, num_features: {}, num_rows: {}".format(
            num_classes, num_features, num_rows
        ),
    )


@st.composite
def classification_dataset_strategy(draw):
    X, y, name = draw(
        st.one_of(
            [
                classification_generated_dataset_strategy(),
                classification_real_dataset_strategy(),
            ]
        )
    )
    if draw(st.booleans()):
        w = np.random.random(y.shape[0])
    else:
        w = None

    return X, y, w, name


@given(
    general_model_param_strategy,
    classification_param_strategy,
    classification_dataset_strategy(),
)
@cn.errstate(divide="raise", invalid="raise")
@pytest.mark.skip
def test_classifier(
    model_params: dict, classification_params: dict, classification_dataset: tuple
) -> None:
    X, y, w, name = classification_dataset
    eval_result = {}
    model = lb.LBClassifier(**model_params, **classification_params).fit(
        X, y, sample_weight=w, eval_result=eval_result
    )
    model.predict(X)
    model.predict_proba(X)
    model.predict_raw(X)
    loss = next(iter(eval_result["train"].values()))
    # multiclass models with higher learning rates don't always converge
    if len(model.classes_) == 2:
        assert non_increasing(loss, 1e-1)
