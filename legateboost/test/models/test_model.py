import numpy as np
import pytest

import cupynumeric as cn
import legateboost as lb


@pytest.mark.parametrize("Model", [M for M in lb.models.BaseModel.__subclasses__()])
def test_batch_inference(Model):
    rs = np.random.RandomState(0)
    X = rs.random((1000, 10))
    g = rs.normal(size=(X.shape[0], 5))
    h = rs.random(g.shape) + 0.1
    X, g, h = cn.array(X), cn.array(g), cn.array(h)
    model_a = Model().set_random_state(np.random.RandomState(2)).fit(X, g, h)
    g_b = cn.array(rs.normal(size=(X.shape[0], 5)))
    model_b = Model().set_random_state(np.random.RandomState(2)).fit(X, g_b, h)

    normal_pred = model_a.predict(X) + model_b.predict(X)
    batch_pred = Model.batch_predict([model_a, model_b], X)
    assert np.allclose(normal_pred, batch_pred)
