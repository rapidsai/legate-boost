from enum import IntEnum
from typing import Union

import numpy as np

import cunumeric as cn

from .library import user_lib


class LegateBoostOpCode(IntEnum):
    QUANTILE = user_lib.cffi.QUANTILE
    QUANTILE_REDUCE = user_lib.cffi.QUANTILE_REDUCE
    QUANTILE_OUTPUT = user_lib.cffi.QUANTILE_OUTPUT
    QUANTISE_DATA = user_lib.cffi.QUANTISE_DATA


class SquaredErrorObjective:
    def gradient(self, y: cn.array, pred: cn.array, w: cn.array) -> cn.array:
        return pred - y, w


class MSEMetric:
    def metric(self, y: cn.array, pred: cn.array, w: cn.array) -> float:
        return float(((y - pred) ** 2 * w).sum() / w.sum())

    def name(self) -> str:
        return "MSE"


class Tree:
    def __init__(
        self,
        X: cn.array,
        g: cn.array,
        h: cn.array,
        learning_rate: float,
        max_depth: int,
        random_state: np.random.RandomState,
    ) -> None:
        self.base = -g.sum() / h.sum() * learning_rate
        # choose random split
        self.split_feature = 0
        best_gain = -np.inf
        self.best_split = 0.0
        self.left_leaf = 0.0
        self.right_leaf = 0.0
        for j in range(X.shape[1]):
            i = random_state.randint(0, X.shape[0])
            split = X[i, j]
            left_mask = X[:, j] <= split

            right_mask = ~left_mask
            G_l = g[left_mask].sum()
            H_l = h[left_mask].sum()
            G_r = g[right_mask].sum()
            H_r = h[right_mask].sum()
            # Not enough data
            if H_l == 0.0 or H_r == 0.0:
                continue

            gain = 1 / 2 * (G_l**2 / H_l + G_r**2 / H_r)
            if gain > best_gain:
                self.split_feature = j
                best_gain = gain
                self.best_split = split
                self.left_leaf = -G_l / H_l
                self.right_leaf = -G_r / H_r

    def predict(self, X: cn.array) -> cn.array:
        pred = cn.zeros(X.shape[0])
        left = X[:, self.split_feature] <= self.best_split
        pred[left] = self.left_leaf
        pred[~left] = self.right_leaf
        return pred


class LBBase:
    pass


class LBRegressor:
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
        if random_state is not None:
            self.random_state = random_state
        else:
            self.random_state = np.random.RandomState()
        self.max_depth = max_depth

    def fit(
        self, X: cn.array, y: cn.array, w: Union[float, cn.array] = 1.0
    ) -> "LBRegressor":
        self.model = []
        if self.init is None:
            self.init_model = 0.0
        else:
            self.init_model = cn.mean(y)

        if cn.isscalar(w):
            w = cn.full(X.shape[0], w)
        pred = cn.full(y.shape, self.init_model)
        objective = SquaredErrorObjective()
        metric = MSEMetric()
        for i in range(self.n_estimators):
            g, h = objective.gradient(y, pred, w)
            self.model.append(
                Tree(X, g, h, self.learning_rate, self.max_depth, self.random_state)
            )
            pred += self.model[-1].predict(X)
            loss = metric.metric(y, pred, w)
            if self.verbose:
                print("i: {} {}: {}".format(i, metric.name(), loss))
        return self

    def predict(self, X: cn.array) -> cn.array:
        pred = cn.full(X.shape[0], self.init_model)
        for m in self.model:
            pred += m.predict(X)
        return pred
