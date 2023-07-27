import numpy as np

import cunumeric as cn


def non_increasing(x, tol=1e-3):
    return all(x - y > -tol for x, y in zip(x, x[1:]))


def non_decreasing(x):
    return all(x <= y for x, y in zip(x, x[1:]))


def sanity_check_tree_stats(trees):
    for tree in trees:
        # Check that we have no 0 hessian splits
        split_nodes = tree.feature != -1
        assert cn.all(tree.hessian[split_nodes] > 0.0)

        # Check gain is positive
        assert cn.all(tree.gain[split_nodes] > 0.0)

        # Check that hessians of leaves add up to root.
        leaves = (tree.feature == -1) & (tree.hessian[:, 0] > 0.0)
        leaf_sum = tree.hessian[leaves].sum(axis=0)
        assert np.isclose(leaf_sum, tree.hessian[0]).all()
