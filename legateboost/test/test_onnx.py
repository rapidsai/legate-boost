import numpy as np
import pytest
from onnx.reference import ReferenceEvaluator

import cupynumeric as cn
import legateboost as lb


@pytest.mark.parametrize(
    "Model", [M for M in lb.models.BaseModel.__subclasses__() if hasattr(M, "to_onnx")]
)
@pytest.mark.parametrize("n_outputs", [1, 5])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_onnx(Model, n_outputs, dtype):
    rs = np.random.RandomState(0)
    X = rs.random((1000, 10)).astype(dtype)
    g = rs.normal(size=(X.shape[0], n_outputs))
    h = rs.random(g.shape) + 0.1
    model = (
        Model()
        .set_random_state(np.random.RandomState(2))
        .fit(cn.array(X), cn.array(g), cn.array(h))
    )

    def pred_onnx(onnx, X):
        sess = ReferenceEvaluator(onnx)
        pred = np.empty(X.shape[0], dtype=dtype)
        feeds = {"X": X, "pred": pred}
        return sess.run(None, feeds)

    assert np.allclose(
        model.predict(cn.array(X)),
        pred_onnx(model.to_onnx(), X)[0],
        atol=1e-3 if dtype == np.float32 else 1e-6,
    )
