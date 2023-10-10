import numpy as np
from hypothesis import HealthCheck, Verbosity, assume, given, settings, strategies as st

import legateboost as lb

from .utils import non_increasing, sanity_check_models

np.set_printoptions(threshold=10, edgeitems=1)

# adjust max_examples to control runtime
settings.register_profile(
    "local",
    max_examples=50,
    deadline=None,
    verbosity=Verbosity.verbose,
    suppress_health_check=(HealthCheck.too_slow,),
    print_blob=True,
)

settings.load_profile("local")


@st.composite
def tree_strategy(draw):
    max_depth = draw(st.integers(1, 12))
    return lb.models.Tree(max_depth=max_depth)


@st.composite
def linear_strategy(draw):
    alpha = draw(st.floats(0.0, 1.0))
    return lb.models.Linear(alpha=alpha)


@st.composite
def krr_strategy(draw):
    alpha = draw(st.floats(0.0, 1.0))
    components = draw(st.integers(1, 10))
    return lb.models.KRR(n_components=components, alpha=alpha)


@st.composite
def base_model_strategy(draw):
    n = draw(st.integers(1, 5))
    base_models = ()
    for _ in range(n):
        base_models += (
            draw(st.one_of([tree_strategy(), linear_strategy(), krr_strategy()])),
        )
    return base_models


general_model_param_strategy = st.fixed_dictionaries(
    {
        "n_estimators": st.integers(1, 20),
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
    from sklearn.datasets import fetch_california_housing, fetch_openml, load_diabetes
    from sklearn.preprocessing import normalize

    name = draw(st.sampled_from(["california_housing", "million_songs", "diabetes"]))
    if name == "california_housing":
        return fetch_california_housing(return_X_y=True)
    elif name == "million_songs":
        X, y = fetch_openml(name="year", version=1, return_X_y=True, as_frame=False)
        return X, y
    elif name == "diabetes":
        X, y = load_diabetes(return_X_y=True)
        return X, normalize(y.reshape(-1, 1), axis=0).reshape(-1)


@st.composite
def regression_generated_dataset_strategy(draw):
    num_outputs = draw(st.integers(1, 5))
    num_features = draw(st.integers(1, 50))
    num_rows = draw(st.integers(10, 10000))
    np.random.seed(2)
    X = np.random.random((num_rows, num_features))
    y = np.random.random((X.shape[0], num_outputs))

    dtype = draw(st.sampled_from([np.float16, np.float32, np.float64]))
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

    return X, y, w


@given(
    general_model_param_strategy,
    regression_param_strategy,
    regression_dataset_strategy(),
)
def test_regressor(model_params, regression_params, regression_dataset):
    X, y, w = regression_dataset
    eval_result = {}
    assume(regression_params["objective"] != "quantile" or y.ndim == 1)
    model = lb.LBRegressor(**model_params, **regression_params).fit(
        X, y, sample_weight=w, eval_result=eval_result
    )
    model.predict(X)
    loss = next(iter(eval_result["train"].values()))
    assert non_increasing(loss)
    sanity_check_models(model)


classification_param_strategy = st.fixed_dictionaries(
    {
        "objective": st.sampled_from(["log_loss"]),
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
        X = X[0:50000]
        y = y[0:50000]
        return (normalize(X), y - 1, name)
    elif name == "breast_cancer":
        return (*load_breast_cancer(return_X_y=True, as_frame=False), name)


@st.composite
def classification_generated_dataset_strategy(draw):
    num_classes = draw(st.integers(2, 5))
    num_features = draw(st.integers(1, 50))
    num_rows = draw(st.integers(num_classes, 10000))
    np.random.seed(3)
    X = np.random.random((num_rows, num_features))
    y = np.random.randint(0, num_classes, size=X.shape[0])

    # ensure we have at least one of each class
    y[:num_classes] = np.arange(num_classes)

    X_dtype = draw(st.sampled_from([np.float16, np.float32, np.float64]))
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
def test_classifier(model_params, classification_params, classification_dataset):
    X, y, w, name = classification_dataset
    eval_result = {}
    model_params["n_estimators"] = 3
    model = lb.LBClassifier(**model_params, **classification_params).fit(
        X, y, sample_weight=w, eval_result=eval_result
    )
    model.predict(X)
    model.predict_proba(X)
    model.predict_raw(X)
    loss = next(iter(eval_result["train"].values()))
    # multiclass models with higher learning rates don't always converge
    if len(model.classes_) == 2 or classification_params["learning_rate"] < 0.1:
        assert non_increasing(loss)
