import numpy as np
import utils
from hypothesis import HealthCheck, Verbosity, assume, given, settings, strategies as st

import cunumeric as cn
import legateboost as lb

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


general_model_param_strategy = st.fixed_dictionaries(
    {
        "n_estimators": st.integers(1, 20),
        "max_depth": st.integers(1, 12),
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

    name = draw(st.sampled_from(["california_housing", "million_songs", "diabetes"]))
    if name == "california_housing":
        return fetch_california_housing(return_X_y=True)
    elif name == "million_songs":
        return fetch_openml(name="year", version=1, return_X_y=True, as_frame=False)
    elif name == "diabetes":
        return load_diabetes(return_X_y=True)


@st.composite
def regression_generated_dataset_strategy(draw):
    num_outputs = draw(st.integers(1, 5))
    num_features = draw(st.integers(1, 150))
    num_rows = draw(st.integers(10, 10000))
    np.random.seed(2)
    X = cn.random.random((num_rows, num_features))
    y = cn.random.random((X.shape[0], num_outputs))

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
        w = cn.random.random(y.shape[0])
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
    pred = model.predict(X)
    loss = next(iter(eval_result["train"].values()))
    assert utils.non_increasing(loss)
    utils.sanity_check_tree_stats(model.models_)

    # test partial fit
    n_estimators_first = model_params["n_estimators"] // 2
    partial_model = lb.LBRegressor(**model_params, **regression_params)
    partial_model.set_params(n_estimators=n_estimators_first)
    partial_model.partial_fit(X, y, sample_weight=w)
    partial_model.set_params(
        n_estimators=model_params["n_estimators"] - n_estimators_first
    )
    partial_model.partial_fit(X, y, sample_weight=w)

    assert cn.allclose(pred, partial_model.predict(X), atol=1e-3)


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

    name = draw(st.sampled_from(["covtype", "breast_cancer"]))
    if name == "covtype":
        X, y = fetch_covtype(return_X_y=True, as_frame=False)
        return (X, y - 1, name)
    elif name == "breast_cancer":
        return (*load_breast_cancer(return_X_y=True, as_frame=False), name)


@st.composite
def classification_generated_dataset_strategy(draw):
    num_classes = draw(st.integers(2, 5))
    num_features = draw(st.integers(1, 150))
    num_rows = draw(st.integers(num_classes, 10000))
    np.random.seed(3)
    X = cn.random.random((num_rows, num_features))
    y = cn.random.randint(0, num_classes, size=X.shape[0])

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
        w = cn.random.random(y.shape[0])
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
    model = lb.LBClassifier(**model_params, **classification_params).fit(
        X, y, sample_weight=w, eval_result=eval_result
    )
    pred = model.predict(X)
    pred_proba = model.predict_proba(X)
    model.predict_raw(X)
    loss = next(iter(eval_result["train"].values()))
    # multiclass models with higher learning rates don't always converge
    if len(model.classes_) == 2 or classification_params["learning_rate"] < 0.1:
        assert utils.non_increasing(loss)

    utils.sanity_check_tree_stats(model.models_)

    # test partial fit
    n_estimators_first = model_params["n_estimators"] // 2
    partial_model = lb.LBClassifier(**model_params, **classification_params)
    partial_model.set_params(n_estimators=n_estimators_first)
    classes = cn.unique(y)
    partial_model.partial_fit(X, y, sample_weight=w, classes=classes)
    partial_model.set_params(
        n_estimators=model_params["n_estimators"] - n_estimators_first
    )
    partial_model.partial_fit(X, y, sample_weight=w, classes=classes)

    assert cn.all(pred == partial_model.predict(X))
    assert cn.allclose(pred_proba, partial_model.predict_proba(X), atol=1e-3)
