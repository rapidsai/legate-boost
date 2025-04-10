import numpy as np
import onnxruntime as ort
import pytest

import cupynumeric as cn
import legateboost as lb


def compare_model_predictions(model, X):
    sess = ort.InferenceSession(model.to_onnx(X).SerializeToString())
    feeds = {
        "X_in": X,
    }
    pred = model.predict(cn.array(X))
    feeds["predictions_in"] = np.zeros((X.shape[0], pred.shape[1]))
    onnx_pred = sess.run(None, feeds)[1]
    onnx_pred = onnx_pred.squeeze()
    assert onnx_pred.dtype == np.float64
    pred = pred.squeeze()
    assert pred.shape == onnx_pred.shape
    assert np.allclose(
        onnx_pred, pred, atol=1e-3 if X.dtype == np.float32 else 1e-6
    ), np.linalg.norm(pred - onnx_pred)


def compare_estimator_predictions(estimator, X, predict_function):
    sess = ort.InferenceSession(
        estimator.to_onnx(X, predict_function).SerializeToString()
    )
    feeds = {
        "X_in": X,
    }
    pred_method = getattr(estimator, predict_function)
    pred = pred_method(cn.array(X))
    onnx_pred = sess.run(None, feeds)[0]

    assert onnx_pred.dtype == np.float64
    assert pred.shape == onnx_pred.shape
    assert np.allclose(
        onnx_pred, pred, atol=1e-2 if X.dtype == np.float32 else 1e-6
    ), np.linalg.norm(pred - onnx_pred)


@pytest.fixture
def model_dataset(dtype, n_outputs):
    rs = np.random.RandomState(0)
    X = rs.random((1000, 10)).astype(dtype)
    g = rs.normal(size=(X.shape[0], n_outputs))
    h = rs.random(g.shape) + 0.1
    return X, g, h


@pytest.mark.parametrize("Model", [M for M in lb.models.BaseModel.__subclasses__()])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("n_outputs", [1, 5])
def test_models(Model, model_dataset):
    X, g, h = model_dataset
    model = (
        Model()
        .set_random_state(np.random.RandomState(2))
        .fit(cn.array(X), cn.array(g), cn.array(h))
    )

    compare_model_predictions(model, X)


@pytest.mark.parametrize("n_outputs", [1, 5])
def test_init(n_outputs):
    # ONNX correctly outputs model init
    X = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    y = np.full((3, n_outputs), 5.0, dtype=np.float32)
    estimator = lb.LBRegressor(n_estimators=0, random_state=0).fit(X, y)
    assert np.all(estimator.model_init_ == 5.0)
    compare_estimator_predictions(estimator, X, "predict_raw")


@pytest.fixture
def regression_dataset(dtype, n_outputs):
    from sklearn.datasets import make_regression

    X, y = make_regression(
        n_samples=1000,
        n_features=10,
        n_informative=5,
        n_targets=n_outputs,
        random_state=0,
    )
    # make labels strictly positive for certain objectives
    return X.astype(dtype), np.abs(y.astype(dtype))


@pytest.mark.parametrize("Model", [M for M in lb.models.BaseModel.__subclasses__()])
@pytest.mark.parametrize("objective", lb.objectives.REGRESSION_OBJECTIVES)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("n_outputs", [1, 5])
def test_regressor(Model, objective, regression_dataset):
    X, y = regression_dataset
    if objective in [
        "quantile",
        "gamma_deviance",
        "gamma",
    ] and (y.ndim > 1 and y.shape[1] > 1):
        pytest.skip("skipping quantile, gamma and gamma_deviance for multiple outputs")
    model = lb.LBRegressor(
        n_estimators=2,
        objective=objective,
        base_models=(Model(),),
        random_state=0,
    ).fit(X, y)

    compare_estimator_predictions(model, X, "predict_raw")
    compare_estimator_predictions(model, X, "predict")


@pytest.fixture
def classification_dataset(dtype, n_outputs):
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=5,
        n_classes=n_outputs,
        random_state=0,
    )
    return X.astype(dtype), np.abs(y.astype(dtype))


@pytest.mark.parametrize("Model", [M for M in lb.models.BaseModel.__subclasses__()])
@pytest.mark.parametrize("objective", lb.objectives.CLASSIFICATION_OBJECTIVES)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("n_outputs", [2, 5])
def test_classifier(Model, objective, classification_dataset):
    X, y = classification_dataset
    if objective == "multi_label":
        # encode labels as one-hot
        encoded = np.zeros((y.shape[0], int(y.max() + 1)))
        encoded[np.arange(y.shape[0]), y.astype(int)] = 1
        y = encoded
    model = lb.LBClassifier(
        n_estimators=2,
        objective=objective,
        base_models=(Model(),),
        random_state=0,
    ).fit(X, y)

    compare_estimator_predictions(model, X, "predict_raw")


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("n_outputs", [1, 5])
@pytest.mark.parametrize("max_depth", list(range(0, 12, 3)))
def test_tree(regression_dataset, max_depth):
    # test tree depths more exhaustively
    # some edge cases e.g. max_depth=0
    X, y = regression_dataset
    model = lb.LBRegressor(
        init=None,
        n_estimators=2,
        base_models=(lb.models.Tree(max_depth=max_depth),),
        random_state=0,
    ).fit(X, y)

    compare_estimator_predictions(model, X, "predict_raw")
