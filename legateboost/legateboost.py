from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Union

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import (
    check_array,
    check_is_fitted,
    check_random_state,
    check_X_y,
)

import cunumeric as cn

from .library import user_lib


class LegateBoostOpCode(IntEnum):
    QUANTILE = user_lib.cffi.QUANTILE
    QUANTILE_REDUCE = user_lib.cffi.QUANTILE_REDUCE
    QUANTILE_OUTPUT = user_lib.cffi.QUANTILE_OUTPUT
    QUANTISE_DATA = user_lib.cffi.QUANTISE_DATA


class SquaredErrorObjective:
    def gradient(self, y: cn.ndarray, pred: cn.ndarray, w: cn.ndarray) -> cn.ndarray:
        return pred - y, w


class MSEMetric:
    def metric(self, y: cn.ndarray, pred: cn.ndarray, w: cn.ndarray) -> float:
        return float(((y - pred) ** 2 * w).sum() / w.sum())

    def name(self) -> str:
        return "MSE"


@dataclass
class TreeSplit:
    feature: int = -1
    gain: float = -np.inf
    split_value: float = 0.0
    left_leaf: float = -1
    right_leaf: float = -1
    row_set_left: cn.ndarray = None
    row_set_right: cn.ndarray = None


@dataclass
class TreeNode:
    left_child: int = -1
    right_child: int = -1
    leaf_value: float = 0.0
    feature: int = -1
    split_value: float = 0.0

    def is_leaf(self) -> bool:
        return self.left_child == -1


class Tree:
    def __init__(
        self,
        X: cn.ndarray,
        g: cn.ndarray,
        h: cn.ndarray,
        learning_rate: float,
        max_depth: int,
        random_state: np.random.RandomState,
    ) -> None:
        self.base = -g.sum() / h.sum() * learning_rate
        row_sets = {0: cn.arange(g.shape[0])}
        candidates = [0]
        self.tree = {0: TreeNode(-1, -1, self.base)}
        while candidates:
            id = candidates.pop()

            # depth_check
            depth = cn.log2(id + 1)
            if depth >= max_depth:
                continue

            best_split = self.get_split(X, g, h, random_state, row_sets[id])
            if best_split:
                self.tree[id] = TreeNode(
                    id * 2 + 1,
                    id * 2 + 2,
                    0.0,
                    best_split.feature,
                    best_split.split_value,
                )
                self.tree[id * 2 + 1] = TreeNode(
                    -1, -1, best_split.left_leaf * learning_rate
                )
                self.tree[id * 2 + 2] = TreeNode(
                    -1, -1, best_split.right_leaf * learning_rate
                )

                row_sets[id * 2 + 1] = best_split.row_set_left
                row_sets[id * 2 + 2] = best_split.row_set_right
                candidates.append(id * 2 + 1)
                candidates.append(id * 2 + 2)

    def get_split(
        self,
        X: cn.ndarray,
        g: cn.ndarray,
        h: cn.ndarray,
        random_state: np.random.RandomState,
        row_set: cn.ndarray,
    ) -> Union[TreeSplit, None]:
        best = TreeSplit()
        for j in range(X.shape[1]):
            i = random_state.randint(0, row_set.shape[0])
            split = X[row_set[i], j]
            left_mask = X[row_set, j] <= split
            left_indices = row_set[left_mask]
            right_indices = row_set[~left_mask]

            G_l = g[left_indices].sum()
            H_l = h[left_indices].sum()
            G_r = g[right_indices].sum()
            H_r = h[right_indices].sum()

            if H_l <= 0.0 or H_r <= 0.0:
                continue

            gain = 1 / 2 * (G_l**2 / H_l + G_r**2 / H_r)
            if gain > best.gain:
                best = TreeSplit(
                    j, gain, split, -G_l / H_l, -G_r / H_r, left_indices, right_indices
                )

        if best.gain <= 0.0:
            return None
        else:
            return best

    def predict(self, X: cn.ndarray) -> cn.ndarray:
        pred = cn.zeros(X.shape[0])
        for i in range(X.shape[0]):
            node = self.tree[0]
            while True:
                if node.is_leaf():
                    pred[i] = node.leaf_value
                    break
                else:
                    if X[i, node.feature] <= node.split_value:
                        node = self.tree[node.left_child]
                    else:
                        node = self.tree[node.right_child]

        return pred

    def __str__(self) -> str:
        def recurse_print(id: int, depth: int) -> str:
            node = self.tree[id]
            if node.is_leaf():
                text = "\t" * depth + "{}:leaf={}\n".format(id, node.leaf_value)
            else:
                text = "\t" * depth + "{}:[f{}<={}] yes={} no={}\n".format(
                    id,
                    node.feature,
                    node.split_value,
                    node.left_child,
                    node.right_child,
                )
                text += recurse_print(node.left_child, depth + 1)
                text += recurse_print(node.right_child, depth + 1)
            return text

        return recurse_print(0, 0)


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


class LBRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        n_estimators: int = 100,
        objective: str = "squared_error",
        learning_rate: float = 0.1,
        init: Union[str, None] = "average",
        verbose: int = 0,
        random_state: np.random.RandomState = None,
        max_depth: int = 3,
    ) -> None:
        self.n_estimators = n_estimators
        self.objective = objective
        self.learning_rate = learning_rate
        self.init = init
        self.verbose = verbose
        self.random_state = random_state
        self.max_depth = max_depth

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
        X, y = check_X_y(X, y, y_numeric=True)
        sample_weight = _check_sample_weight(sample_weight, X.shape[0])
        self.n_features_in_ = X.shape[1]
        self.models_ = []

        objective = SquaredErrorObjective()
        if self.init is None:
            self.model_init_ = 0.0
        else:
            g, h = objective.gradient(y, cn.zeros_like(y), sample_weight)
            H = h.sum()
            self.model_init_ = 0.0
            if H > 0.0:
                self.model_init_ = -g.sum() / H

        pred = cn.full(y.shape, self.model_init_)
        self._metric = MSEMetric()
        self.train_score_ = []
        for i in range(self.n_estimators):
            g, h = objective.gradient(y, pred, sample_weight)
            self.models_.append(
                Tree(
                    X,
                    g,
                    h,
                    self.learning_rate,
                    self.max_depth,
                    check_random_state(self.random_state),
                )
            )
            pred += self.models_[-1].predict(X)
            self.train_score_.append(self._metric.metric(y, pred, sample_weight))
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
