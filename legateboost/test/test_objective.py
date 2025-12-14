import numpy as np
import onnxruntime as ort
import pytest

import cupynumeric as cn
import legateboost as lb
from legateboost.testing.utils import non_increasing


def compare_onnx_transform(obj, pred):
    sess = ort.InferenceSession(obj.onnx_transform(pred).SerializeToString())
    feeds = {
        "predictions_in": pred,
    }
    onnx_transform = sess.run(None, feeds)[0]
    assert onnx_transform.dtype == np.float64
    transform = obj.transform(cn.array(pred))
    assert transform.shape == onnx_transform.shape
    assert np.allclose(onnx_transform, transform, atol=1e-6)


def test_normal() -> None:
    obj = lb.NormalObjective()
    y = cn.array([[1.0], [2.0], [3.0]])
    init = obj.initialise_prediction(y, cn.array([1.0, 1.0, 1.0]), True)
    assert cn.allclose(init, cn.array([y.mean(), cn.log(y.std())]))

    compare_onnx_transform(obj, np.arange(12).reshape(2, 6).astype(np.float64))


def test_gamma_deviance() -> None:
    obj = lb.GammaDevianceObjective()
    n_samples = 8196
    with pytest.raises(ValueError, match="positive"):
        y = cn.empty(shape=(n_samples,))
        y[:] = -1
        obj.initialise_prediction(y, None, True)

    rng = cn.random.default_rng(1)
    X = rng.normal(size=(n_samples, 32))
    y = rng.gamma(shape=2.0, size=n_samples)

    eval_result: dict[str, dict[str, list[float]]] = {}
    reg = lb.LBRegressor(objective=obj, init=None, n_estimators=10)
    reg.fit(X, y, eval_set=[(X, y)], eval_result=eval_result)
    assert non_increasing(eval_result["train"]["deviance_gamma"])

    eval_result.clear()
    y1 = rng.gamma(size=(n_samples, 2), shape=2.0)
    reg = lb.LBRegressor(objective=obj, init=None, n_estimators=10)
    reg.fit(X, y1, eval_set=[(X, y1)], eval_result=eval_result)
    assert non_increasing(eval_result["train"]["deviance_gamma"])

    compare_onnx_transform(obj, np.arange(12).reshape(2, 6).astype(np.float64))


def test_gamma() -> None:
    import numpy as np

    n_samples = 8196
    np.random.seed(1)
    rng = cn.random.default_rng(1)
    X = rng.normal(size=(n_samples, 32))
    y = rng.gamma(shape=2.0, size=n_samples)

    obj = lb.GammaObjective()
    eval_result: dict[str, dict[str, list[float]]] = {}
    reg = lb.LBRegressor(objective=obj, n_estimators=64, init="average")
    reg.fit(X, y, eval_set=[(X, y)], eval_result=eval_result)
    assert non_increasing(eval_result["train"]["gamma_neg_ll"])

    compare_onnx_transform(obj, np.arange(12).reshape(2, 6).astype(np.float64))


def test_log_loss() -> None:
    obj = lb.LogLossObjective()
    # binary
    g, h = obj.gradient(cn.array([[1], [0]]), cn.array([[0.5], [0.5]]))
    assert cn.allclose(g, cn.array([[-0.5], [0.5]]))
    assert cn.allclose(h, cn.array([[0.25], [0.25]]))

    # multi
    g, h = obj.gradient(
        cn.array([[2], [1], [0]]),
        cn.array([[0.3, 0.3, 0.4], [0.3, 0.4, 0.3], [0.4, 0.3, 0.3]]),
    )
    assert cn.allclose(
        g, cn.array([[0.3, 0.3, -0.6], [0.3, -0.6, 0.3], [-0.6, 0.3, 0.3]])
    )
    assert cn.allclose(
        h, cn.array([[0.21, 0.21, 0.24], [0.21, 0.24, 0.21], [0.24, 0.21, 0.21]])
    )

    with pytest.raises(
        ValueError, match="Expected labels to be non-zero whole numbers"
    ):
        obj.initialise_prediction(
            cn.array([[1], [2.5]]), cn.array([[1.0], [1.0]]), False
        )

    with pytest.raises(
        ValueError, match="Expected labels to be non-zero whole numbers"
    ):
        obj.initialise_prediction(
            cn.array(
                [
                    [-1],
                ]
            ),
            cn.array(
                [
                    [1.0],
                ]
            ),
            False,
        )

    compare_onnx_transform(obj, np.arange(12).reshape(2, 6).astype(np.float64))
    compare_onnx_transform(obj, np.arange(4).reshape(4, 1).astype(np.float64))


def test_exp():
    obj = lb.ExponentialObjective()
    # binary
    g, h = obj.gradient(cn.array([[1], [0]]), cn.array([[0.5], [0.5]]))
    assert cn.allclose(g, cn.array([[-1.0], [1.0]]))
    assert cn.allclose(h, cn.array([[1.0], [1.0]]))

    with pytest.raises(
        ValueError, match="Expected labels to be non-zero whole numbers"
    ):
        obj.initialise_prediction(
            cn.array([[1], [2.5]]), cn.array([[1.0], [1.0]]), False
        )

    with pytest.raises(
        ValueError, match="Expected labels to be non-zero whole numbers"
    ):
        obj.initialise_prediction(
            cn.array(
                [
                    [-1],
                ]
            ),
            cn.array(
                [
                    [1.0],
                ]
            ),
            False,
        )

    compare_onnx_transform(obj, np.arange(12).reshape(2, 6).astype(np.float64))
    compare_onnx_transform(obj, np.arange(4).reshape(4, 1).astype(np.float64))


def test_multi_label():
    obj = lb.MultiLabelObjective()
    g, h = obj.gradient(
        cn.array([[1, 0], [0, 1]]),
        cn.array([[[0.5, 0.5], [0.5, 0.5]], [[0.5, 0.5], [0.5, 0.5]]]),
    )
    assert cn.allclose(g, cn.array([[-0.5, 0.5], [0.5, -0.5]]))
    assert cn.allclose(h, cn.array([[0.25, 0.25], [0.25, 0.25]]))

    with pytest.raises(ValueError, match=r"Expected labels to be in \[0, 1\]"):
        obj.initialise_prediction(cn.array([[1], [2]]), cn.array([[1.0], [1.0]]), False)

    compare_onnx_transform(obj, np.arange(12).reshape(2, 6).astype(np.float64))
