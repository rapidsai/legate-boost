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
    G_l: float = 0.0
    H_l: float = 0.0
    G_r: float = 0.0
    H_r: float = 0.0
    row_set_left: cn.ndarray = None
    row_set_right: cn.ndarray = None


class TreeStructure:
    left_child: cn.ndarray
    right_child: cn.ndarray
    leaf_value: cn.ndarray
    feature: cn.ndarray
    split_value: cn.ndarray

    # expand by n elements
    def expand(self, n: int) -> None:
        self.left_child = cn.concatenate(
            (self.left_child, cn.full(n, -1, dtype=cn.int32))
        )
        self.right_child = cn.concatenate(
            (self.right_child, cn.full(n, -1, dtype=cn.int32))
        )
        self.leaf_value = cn.concatenate((self.leaf_value, cn.full(n, 0.0)))
        self.feature = cn.concatenate((self.feature, cn.full(n, -1, dtype=cn.int32)))
        self.split_value = cn.concatenate((self.split_value, cn.full(n, 0.0)))

    def __init__(self, base_score: float, reserve: int) -> None:
        assert reserve > 0
        self.left_child = cn.full(reserve, -1, dtype=cn.int32)
        self.right_child = cn.full(reserve, -1, dtype=cn.int32)
        self.leaf_value = cn.full(reserve, 0.0)
        self.feature = cn.full(reserve, -1, dtype=cn.int32)
        self.split_value = cn.full(reserve, 0.0)
        self.leaf_value[0] = base_score

    def add_split(self, id: int, split: TreeSplit, learning_rate: float) -> None:
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
        assert g.size == h.size == X.shape[0]
        row_sets = {0: cn.arange(g.shape[0])}
        candidates = [(0, g.sum(), h.sum())]
        base_score = -candidates[0][1] / candidates[0][2] * learning_rate
        self.tree = TreeStructure(base_score, 256)
        while candidates:
            id, G, H = candidates.pop()

            # depth_check
            depth = cn.log2(id + 1)
            if depth >= max_depth:
                continue

            best_split = self.get_split(X, g, h, G, H, random_state, row_sets[id])
            if best_split:
                self.tree.add_split(id, best_split, learning_rate)
                row_sets[id * 2 + 1] = best_split.row_set_left
                row_sets[id * 2 + 2] = best_split.row_set_right
                candidates.append((id * 2 + 1, best_split.G_l, best_split.H_l))
                candidates.append((id * 2 + 2, best_split.G_r, best_split.H_r))

    def get_split(
        self,
        X: cn.ndarray,
        g: cn.ndarray,
        h: cn.ndarray,
        G: float,
        H: float,
        random_state: np.random.RandomState,
        row_set: cn.ndarray,
    ) -> Union[TreeSplit, None]:
        i = random_state.randint(0, row_set.shape[0])
        splits = X[row_set[i]]
        g_set = g[row_set]
        h_set = h[row_set]
        left_mask = X[row_set] <= splits
        G_l = cn.where(left_mask, g_set[:, None], 0.0).sum(axis=0)
        H_l = cn.where(left_mask, h_set[:, None], 0.0).sum(axis=0)
        G_r = G - G_l
        H_r = H - H_l
        gain = 1 / 2 * (G_l**2 / H_l + G_r**2 / H_r - G**2 / H)
        gain[~cn.isfinite(gain)] = 0.0
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
            assert left_set.size > 0 and right_set.size > 0
            return best

    def predict(self, X: cn.ndarray) -> cn.ndarray:
        id = cn.zeros(X.shape[0], dtype=cn.int32)
        while True:
            at_leaf = self.tree.is_leaf(id)
            if cn.all(at_leaf):
                break
            else:
                id_subset = id[~at_leaf]
                id[~at_leaf] = cn.where(
                    X[~at_leaf, self.tree.feature[id_subset]]
                    <= self.tree.split_value[id_subset],
                    self.tree.left_child[id_subset],
                    self.tree.right_child[id_subset],
                )
        return self.tree.leaf_value[id]

    def __str__(self) -> str:
        def recurse_print(id: int, depth: int) -> str:
            if self.tree.is_leaf(id):
                text = "\t" * depth + "{}:leaf={}\n".format(
                    id, self.tree.leaf_value[id]
                )
            else:
                text = "\t" * depth + "{}:[f{}<={}] yes={} no={}\n".format(
                    id,
                    self.tree.feature[id],
                    self.tree.split_value[id],
                    self.tree.left_child[id],
                    self.tree.right_child[id],
                )
                text += recurse_print(self.tree.left_child[id], depth + 1)
                text += recurse_print(self.tree.right_child[id], depth + 1)
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
