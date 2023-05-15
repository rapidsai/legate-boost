from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Union

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_random_state

import cunumeric as cn
from legate.core import Store, types

from .library import user_context, user_lib
from .objectives import objectives


class LegateBoostOpCode(IntEnum):
    BUILD_TREE = user_lib.cffi.BUILD_TREE


@dataclass
class TreeSplit:
    """A proposal to add a new split to the tree.

    Contains split information, gradient statistics the left and right
    branches and the indices of rows following the left and right
    branches.
    """

    feature: int = -1
    gain: float = -np.inf
    split_value: float = 0.0
    G_l: float = 0.0
    H_l: float = 0.0
    G_r: float = 0.0
    H_r: float = 0.0
    row_set_left: cn.ndarray = None
    row_set_right: cn.ndarray = None


class TreeStructure:
    """A structure of arrays representing a decision tree.

    A leaf node has value -1 at left_child[node_idx]
    """

    left_child: cn.ndarray
    right_child: cn.ndarray
    leaf_value: cn.ndarray
    feature: cn.ndarray
    split_value: cn.ndarray

    def expand(self, n: int) -> None:
        """Add n extra storage."""
        self.left_child = cn.concatenate(
            (self.left_child, cn.full(n, -1, dtype=cn.int32))
        )
        self.right_child = cn.concatenate(
            (self.right_child, cn.full(n, -1, dtype=cn.int32))
        )
        self.leaf_value = cn.concatenate((self.leaf_value, cn.full(n, 0.0)))
        self.feature = cn.concatenate((self.feature, cn.full(n, -1, dtype=cn.int32)))
        self.split_value = cn.concatenate((self.split_value, cn.full(n, 0.0)))

    def __init__(self, base_score: float, reserve: int) -> None:  # noqa: no-redef
        """Tree is initialised with a single leaf set to base_score no splits.

        Reserve is the amount of storage first initialised. Set to a
        larger number to prevent resizing repeatedly during expansion.
        """
        assert reserve > 0
        self.left_child = cn.full(reserve, -1, dtype=cn.int32)
        self.right_child = cn.full(reserve, -1, dtype=cn.int32)
        self.leaf_value = cn.full(reserve, 0.0)
        self.feature = cn.full(reserve, -1, dtype=cn.int32)
        self.split_value = cn.full(reserve, 0.0)
        self.leaf_value[0] = base_score

    @classmethod
    def from_arrays(
        clf,
        left_child: cn.ndarray,
        right_child: cn.ndarray,
        leaf_value: cn.ndarray,
        feature: cn.ndarray,
        split_value: cn.ndarray,
    ) -> TreeStructure:
        """Initialise from existing storage."""
        tree = clf(0.0, 1)
        tree.left_child = left_child
        tree.right_child = right_child
        tree.leaf_value = leaf_value
        tree.feature = feature
        tree.split_value = split_value
        return tree

    def add_split(self, id: int, split: TreeSplit, learning_rate: float) -> None:
        """Expand node with two new children.

        Storage is expanded if necessary.
        """
        assert split.H_l > 0.0 and split.H_r > 0.0
        if self.left_child.size <= id * 2 + 2:
            self.expand(max(self.left_child.size * 4, id * 4))
        self.left_child[id] = id * 2 + 1
        self.right_child[id] = id * 2 + 2
        self.leaf_value[id] = 0.0
        self.feature[id] = split.feature
        self.split_value[id] = split.split_value

        # set leaves
        self.leaf_value[id * 2 + 1] = -split.G_l / split.H_l * learning_rate
        self.leaf_value[id * 2 + 2] = -split.G_r / split.H_r * learning_rate

    def is_leaf(self, id: int) -> Any:
        return self.left_child[id] == -1

    def predict(self, X: cn.ndarray) -> cn.ndarray:
        """Vectorised decision tree prediction."""
        id = cn.zeros(X.shape[0], dtype=cn.int32)
        while True:
            at_leaf = self.is_leaf(id)
            if cn.all(at_leaf):
                break
            else:
                id_subset = id[~at_leaf]
                id[~at_leaf] = cn.where(
                    X[~at_leaf, self.feature[id_subset]] <= self.split_value[id_subset],
                    self.left_child[id_subset],
                    self.right_child[id_subset],
                )
        return self.leaf_value[id]

    def __str__(self) -> str:
        def recurse_print(id: int, depth: int) -> str:
            if self.is_leaf(id):
                text = "\t" * depth + "{}:leaf={}\n".format(id, self.leaf_value[id])
            else:
                text = "\t" * depth + "{}:[f{}<={}] yes={} no={}\n".format(
                    id,
                    self.feature[id],
                    self.split_value[id],
                    self.left_child[id],
                    self.right_child[id],
                )
                text += recurse_print(self.left_child[id], depth + 1)
                text += recurse_print(self.right_child[id], depth + 1)
            return text

        return recurse_print(0, 0)


def build_tree_python(
    X: cn.ndarray,
    g: cn.ndarray,
    h: cn.ndarray,
    learning_rate: float,
    max_depth: int,
    random_state: np.random.RandomState,
) -> TreeStructure:
    """Build a single decision tree in a GBDT ensemble.

    Accepts gradients and a dataset, trains a tree model.
    """

    assert g.size == h.size == X.shape[0]
    # store indices of training rows in each node
    row_sets = {0: cn.arange(g.shape[0])}
    # queue of nodes to be opened, including their sum gradient statistics
    candidates = [(0, g.sum(), h.sum())]
    # base_score is the default prediction if we dont expand the tree at all
    base_score = -candidates[0][1] / candidates[0][2] * learning_rate
    tree = TreeStructure(base_score, 256)
    while candidates:
        id, G, H = candidates.pop()

        # depth_check
        depth = cn.log2(id + 1)
        if depth >= max_depth:
            continue

        best_split = get_split(X, g, h, G, H, random_state, row_sets[id])
        if best_split:
            tree.add_split(id, best_split, learning_rate)
            row_sets[id * 2 + 1] = best_split.row_set_left
            row_sets[id * 2 + 2] = best_split.row_set_right
            candidates.append((id * 2 + 1, best_split.G_l, best_split.H_l))
            candidates.append((id * 2 + 2, best_split.G_r, best_split.H_r))
    return tree


def get_split(
    X: cn.ndarray,
    g: cn.ndarray,
    h: cn.ndarray,
    G: float,
    H: float,
    random_state: np.random.RandomState,
    row_set: cn.ndarray,
) -> Union[TreeSplit, None]:
    """Given a subset of rows, randomly choose an instance, then attempt to
    split on each of its features.

    Take whichever feature value improves the objective function the
    most as the split.
    """

    # take the subsets of gradient statistics for this row set
    g_set = g[row_set]
    h_set = h[row_set]

    # select a random row
    # each element of this row vector is a potential split
    # we will choose 1 by checking the change in objective function for each
    i = random_state.randint(0, row_set.shape[0])
    splits = X[row_set[i]]

    # test split condition for each training instance and each proposed split
    # result is a boolean matrix
    left_mask = X[row_set] <= splits

    # sum up the gradient statistics in the left partition for each proposed split
    # G_l/H_l is a vector
    G_l = cn.where(left_mask, g_set[:, None], 0.0).sum(axis=0)
    H_l = cn.where(left_mask, h_set[:, None], 0.0).sum(axis=0)

    # find the sum in the right partition by subtracting the from the parent sum
    G_r = G - G_l
    H_r = H - H_l

    # calculate improvement in objective function
    # see below for explanation of gain formula
    # https://xgboost.readthedocs.io/en/stable/tutorials/model.html

    gain = 1 / 2 * (G_l**2 / H_l + G_r**2 / H_r - G**2 / H)

    # it is possible to have a partition with no instances in it
    # guard against divide by 0
    gain[~cn.isfinite(gain)] = 0.0

    # select the best from the proposed split and return
    best_idx = cn.argmax(gain)
    left_set = row_set[left_mask[:, best_idx]]
    right_set = row_set[~left_mask[:, best_idx]]
    best = TreeSplit(
        best_idx,
        gain[best_idx],
        splits[best_idx],
        G_l[best_idx],
        H_l[best_idx],
        G_r[best_idx],
        H_r[best_idx],
        left_set,
        right_set,
    )
    if best.gain <= 0.0:
        return None
    else:
        # we should not have empty partitions at this point
        assert left_set.size > 0 and right_set.size > 0
        return best


def _get_legate_store(input: Any) -> Store:
    """Extracts a Legate store from any object implementing the legete data
    interface.

    Args:
        input (Any): The input object

    Returns:
        Store: The extracted Legate store
    """
    if isinstance(input, Store):
        return input
    data = input.__legate_data_interface__["data"]
    field = next(iter(data))
    array = data[field]
    _, store = array.stores()
    return store


def build_tree_native(
    X: cn.ndarray,
    g: cn.ndarray,
    h: cn.ndarray,
    learning_rate: float,
    max_depth: int,
    random_state: np.random.RandomState,
) -> TreeStructure:
    task = user_context.create_auto_task(
        LegateBoostOpCode.BUILD_TREE,
    )
    task.add_input(_get_legate_store(X))
    task.add_broadcast(_get_legate_store(X), axes=0)
    task.add_input(_get_legate_store(g))
    task.add_input(_get_legate_store(h))
    task.add_scalar_arg(learning_rate, types.float64)
    task.add_scalar_arg(max_depth, types.int32)
    task.add_scalar_arg(random_state.randint(0, 2**32), types.uint64)

    left_child = user_context.create_store(types.int32, ndim=1)
    right_child = user_context.create_store(types.int32, ndim=1)
    leaf_value = user_context.create_store(types.float64, ndim=1)
    feature = user_context.create_store(types.int32, ndim=1)
    split_value = user_context.create_store(types.float32, ndim=1)

    task.add_output(left_child)
    task.add_output(right_child)
    task.add_output(leaf_value)
    task.add_output(feature)
    task.add_output(split_value)
    task.execute()

    return TreeStructure.from_arrays(
        cn.array(left_child, copy=False),
        cn.array(right_child, copy=False),
        cn.array(leaf_value, copy=False),
        cn.array(feature, copy=False),
        cn.array(split_value, copy=False),
    )


def _check_sample_weight(sample_weight: Any, n: int) -> cn.ndarray:
    if sample_weight is None:
        sample_weight = cn.ones(n)
    elif cn.isscalar(sample_weight):
        sample_weight = cn.full(n, sample_weight)
    elif not isinstance(sample_weight, cn.ndarray):
        sample_weight = cn.array(sample_weight)
    if sample_weight.shape != (n,):
        raise ValueError(
            "Incorrect sample weight shape: "
            + str(sample_weight.shape)
            + ", expected: ("
            + str(n)
            + ",)"
        )
    return sample_weight


class LBBase(BaseEstimator):
    def __init__(
        self,
        n_estimators: int = 100,
        objective: str = "squared_error",
        learning_rate: float = 0.1,
        init: Union[str, None] = "average",
        verbose: int = 0,
        random_state: np.random.RandomState = None,
        max_depth: int = 3,
        version: str = "native",
    ) -> None:
        self.n_estimators = n_estimators
        self.objective = objective
        self.learning_rate = learning_rate
        self.init = init
        self.verbose = verbose
        self.random_state = random_state
        self.max_depth = max_depth
        self.version = version

    def _more_tags(self) -> Any:
        return {
            "_xfail_checks": {
                "check_sample_weights_invariance": (
                    "zero sample_weight is not equivalent to removing samples"
                ),
            }
        }

    def fit(
        self, X: cn.ndarray, y: cn.ndarray, sample_weight: cn.ndarray = None
    ) -> "LBRegressor":
        sample_weight = _check_sample_weight(sample_weight, X.shape[0])
        self.n_features_in_ = X.shape[1]
        self.models_ = []

        objective = objectives[self.objective]()
        if self.init is None:
            self.model_init_ = 0.0
        else:
            # initialise the model to some good average value
            # this is equivalent to a tree with a single leaf and learning rate 1.0
            g, h = objective.gradient(y, cn.zeros_like(y), sample_weight)
            H = h.sum()
            self.model_init_ = 0.0
            if H > 0.0:
                self.model_init_ = -g.sum() / H

        # current model prediction
        pred = cn.full(y.shape, self.model_init_)
        self._metric = objective.metric()
        self.train_score_ = []
        for i in range(self.n_estimators):
            # obtain gradients
            g, h = objective.gradient(y, objective.transform(pred), sample_weight)

            # build new tree
            if self.version == "native":
                tree = build_tree_native(
                    X,
                    g,
                    h,
                    self.learning_rate,
                    self.max_depth,
                    check_random_state(self.random_state),
                )
            else:
                tree = build_tree_python(
                    X,
                    g,
                    h,
                    self.learning_rate,
                    self.max_depth,
                    check_random_state(self.random_state),
                )
            self.models_.append(tree)

            # update current predictions
            pred += self.models_[-1].predict(X)

            # evaluate our progress
            self.train_score_.append(
                self._metric.metric(y, objective.transform(pred), sample_weight)
            )
            if self.verbose:
                print(
                    "i: {} {}: {}".format(i, self._metric.name(), self.train_score_[-1])
                )
        self.is_fitted_ = True
        return self

    def predict(self, X: cn.ndarray) -> cn.ndarray:
        X = check_array(X)
        check_is_fitted(self, "is_fitted_")
        pred = cn.full(X.shape[0], self.model_init_)
        for m in self.models_:
            pred += m.predict(X)
        return pred

    def dump_trees(self) -> str:
        check_is_fitted(self, "is_fitted_")
        text = "init={}\n".format(self.model_init_)
        for m in self.models_:
            text += str(m)
        return text


class LBRegressor(LBBase, RegressorMixin):
    def __init__(
        self,
        n_estimators: int = 100,
        objective: str = "squared_error",
        learning_rate: float = 0.1,
        init: Union[str, None] = "average",
        verbose: int = 0,
        random_state: np.random.RandomState = None,
        max_depth: int = 3,
        version: str = "native",
    ) -> None:
        super().__init__(
            n_estimators=n_estimators,
            objective=objective,
            learning_rate=learning_rate,
            init=init,
            verbose=verbose,
            random_state=random_state,
            max_depth=max_depth,
            version=version,
        )


class LBClassifier(LBBase, ClassifierMixin):
    def __init__(
        self,
        n_estimators: int = 100,
        objective: str = "log_loss",
        learning_rate: float = 0.1,
        init: Union[str, None] = "average",
        verbose: int = 0,
        random_state: np.random.RandomState = None,
        max_depth: int = 3,
        version: str = "native",
    ) -> None:
        super().__init__(
            n_estimators=n_estimators,
            objective=objective,
            learning_rate=learning_rate,
            init=init,
            verbose=verbose,
            random_state=random_state,
            max_depth=max_depth,
            version=version,
        )

    def predict_proba(self, X: cn.ndarray) -> cn.ndarray:
        objective = objectives[self.objective]()
        return objective.transform(self.super().predict(X))

    def predict(self, X: cn.ndarray) -> cn.ndarray:
        return self.predict_proba(X) >= 0.5
