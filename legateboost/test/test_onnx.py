import numpy as np
import onnxruntime as ort
import pytest

import cupynumeric as cn
import legateboost as lb


def pred_onnx_estimator(onnx, X, n_outputs):
    sess = ort.InferenceSession(onnx.SerializeToString())
    feeds = {"X_in": X}
    return sess.run(None, feeds)[1]


def pred_onnx_model(onnx, X, n_outputs):
    sess = ort.InferenceSession(onnx.SerializeToString())
    feeds = {
        "X_in": X,
        "predictions_in": np.zeros((X.shape[0], n_outputs), dtype=X.dtype),
    }
    return sess.run(None, feeds)[1]


@pytest.mark.parametrize("Model", [M for M in lb.models.BaseModel.__subclasses__()])
@pytest.mark.parametrize("n_outputs", [1, 5])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_models(Model, n_outputs, dtype):
    rs = np.random.RandomState(0)
    X = rs.random((1000, 10)).astype(dtype)
    g = rs.normal(size=(X.shape[0], n_outputs))
    h = rs.random(g.shape) + 0.1
    model = (
        Model()
        .set_random_state(np.random.RandomState(2))
        .fit(cn.array(X), cn.array(g), cn.array(h))
    )

    onnx_pred = pred_onnx_model(model.to_onnx(X.dtype), X, n_outputs)
    lb_pred = model.predict(cn.array(X))
    assert onnx_pred.shape == lb_pred.shape
    assert np.allclose(onnx_pred, lb_pred, atol=1e-3 if dtype == np.float32 else 1e-6)


@pytest.mark.parametrize("n_outputs", [1, 5])
def test_init(n_outputs):
    # ONNX correctly outputs model init
    X = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    y = np.full((3, n_outputs), 5.0, dtype=np.float32)
    estimator = lb.LBRegressor(n_estimators=0, random_state=0).fit(X, y)
    assert np.all(estimator.model_init_ == 5.0)
    assert np.all(estimator.predict(X) == 5.0)
    assert np.all(
        pred_onnx_estimator(estimator.to_onnx(X.dtype), X.__array__(), 1) == 5.0
    )


@pytest.mark.parametrize("Model", [M for M in lb.models.BaseModel.__subclasses__()])
@pytest.mark.parametrize("n_outputs", [1, 5])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_estimator(Model, n_outputs, dtype):
    rs = np.random.RandomState(0)
    X = rs.random((1000, 10)).astype(dtype)
    y = rs.random((1000, n_outputs)).astype(dtype)
    model = lb.LBRegressor(
        n_estimators=10,
        base_models=(Model(),),
        random_state=0,
    ).fit(X, y)

    assert np.allclose(
        model.predict(X),
        pred_onnx_estimator(model.to_onnx(X.dtype), X.__array__(), 1).squeeze(),
        atol=1e-3,
    )
