from copy import deepcopy

import numpy as np

import cunumeric as cn


def check_determinism(model):
    rs = cn.random.RandomState(79)
    X = cn.array(rs.random((10000, 10)))
    g = cn.array(rs.normal(size=(X.shape[0], 5)))
    h = cn.array(rs.random(g.shape) + 0.1)
    preds = []
    models = []
    for _ in range(0, 5):
        models.append(deepcopy(model).set_random_state(np.random.RandomState(0)))
        models[-1].fit(X, g, h)
        # for some reason this needs to be converted to np
        # otherwise we end up with changing values in the arrays
        preds.append(np.array(models[-1].predict(X)))

    assert all((np.array(p) == preds[0]).all() for p in preds)
    assert all(m == models[0] for m in models)
