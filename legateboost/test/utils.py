import numpy as np

import cunumeric as cn
import legateboost as lb


def non_increasing(x, tol=1e-3):
    return all(x - y > -tol for x, y in zip(x, x[1:]))


def non_decreasing(x):
    return all(x <= y for x, y in zip(x, x[1:]))


def sanity_check_tree_stats(model):
    trees = [m for m in model.models_ if isinstance(m, lb.models.Tree)]
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
