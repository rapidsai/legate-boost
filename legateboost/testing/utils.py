import numpy as np

import cunumeric as cn
import legateboost as lb

__all__: list[str] = []


def non_increasing(x, tol=1e-3):
    return all(x - y > -tol for x, y in zip(x, x[1:]))


def non_decreasing(x):
    return all(x <= y for x, y in zip(x, x[1:]))


def sanity_check_models(model):
    trees = [m for m in model.models_ if isinstance(m, lb.models.Tree)]
    linear_models = [m for m in model.models_ if isinstance(m, lb.models.Linear)]
    krr_models = [m for m in model.models_ if isinstance(m, lb.models.KRR)]
    nn_models = [m for m in model.models_ if isinstance(m, lb.models.NN)]

    for m in trees:
        # Check that we have no 0 hessian splits
        split_nodes = m.feature != -1
        assert cn.all(m.hessian[split_nodes] > 0.0)

        # Check gain is positive
        assert cn.all(m.gain[split_nodes] > 0.0)

        # Check that hessians of leaves add up to root.
        leaves = (m.feature == -1) & (m.hessian[:, 0] > 0.0)
        leaf_sum = m.hessian[leaves].sum(axis=0)
        assert np.isclose(leaf_sum, m.hessian[0]).all()

    for m in linear_models:
        assert cn.isfinite(m.betas_).all()

    for m in nn_models:
        for c, b in zip(m.coefficients_, m.biases_):
            assert cn.isfinite(c).all()
            assert cn.isfinite(b).all()

    for m in krr_models:
        assert cn.isfinite(m.betas_).all()
