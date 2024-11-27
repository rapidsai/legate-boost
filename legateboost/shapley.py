from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np
from sklearn.base import is_regressor
from sklearn.utils.validation import check_random_state

import cupynumeric as cn

from .metrics import BaseMetric

# provide definitions for mypy without circular import at runtime
if TYPE_CHECKING:
    from .legateboost import LBBase

__all__: List[str] = []


def global_shapley_attributions(
    model: "LBBase",
    X: cn.array,
    y: cn.array,
    metric_in: Optional[BaseMetric] = None,
    random_state: Optional[np.random.RandomState] = None,
    n_samples: int = 5,
    check_efficiency: bool = False,
) -> Tuple[cn.array, cn.array]:
    def predict_fn(X: cn.array) -> cn.array:
        fn = model.predict if is_regressor(model) else model.predict_proba
        return fn(X)

    metric = metric_in if metric_in is not None else model._metrics[0]

    random_state_ = check_random_state(random_state)
    w = cn.ones(y.shape[0])
    gen = cn.random.default_rng(seed=random_state_.randint(2**32))

    # antithetic sampling
    v_a = cn.empty((X.shape[1] + 1, n_samples))
    v_b = cn.empty((X.shape[1] + 1, n_samples))

    def eval_sample(p: cn.array, v: cn.array) -> None:
        # cupynumeric has no shuffle as of writing
        # with replacement should be fine
        X_temp = X[gen.integers(0, X.shape[0], X.shape[0])]
        null_loss = metric.metric(y, predict_fn(X_temp), w)
        v[-1] = null_loss
        previous_loss = null_loss
        for feature in p:
            X_temp[:, feature] = X[:, feature]
            loss = metric.metric(y, predict_fn(X_temp), w)
            v[feature] = loss - previous_loss
            previous_loss = loss

    for i in range(n_samples):
        p_a = random_state_.permutation(X.shape[1])
        p_b = cn.flip(p_a)
        eval_sample(p_a, v_a[:, i])
        eval_sample(p_b, v_b[:, i])

    v = (v_a + v_b) / 2
    shapley_values = cn.mean(v, axis=1)
    se = cn.std(v, axis=1, ddof=1) / cn.sqrt(n_samples)

    if check_efficiency:
        full_coalition_loss = metric.metric(y, predict_fn(X), cn.ones(y.shape[0]))
        if not cn.isclose(cn.sum(shapley_values), full_coalition_loss):
            raise ValueError("Shapley values do not sum up to the full coalition loss")
    return shapley_values, se


def local_shapley_attributions(
    model: "LBBase",
    X: cn.array,
    X_background: cn.array,
    random_state: Optional[np.random.RandomState] = None,
    n_samples: int = 5,
    check_efficiency: bool = False,
) -> Tuple[cn.array, cn.array]:
    def predict_fn(X: cn.array) -> cn.array:
        fn = model.predict if is_regressor(model) else model.predict_proba
        p = fn(X)
        if p.ndim == 1:
            return p.reshape(-1, 1)
        return p

    random_state_ = check_random_state(random_state)
    n_background_samples = 5
    gen = cn.random.default_rng(seed=random_state_.randint(2**32))

    n_outputs = predict_fn(X[:1, ...]).shape[1]

    # antithetic sampling
    # perhaps we can do a running mean/se to avoid the last dimension
    v_a = cn.zeros((X.shape[0], X.shape[1] + 1, n_outputs, n_samples))
    v_b = cn.zeros((X.shape[0], X.shape[1] + 1, n_outputs, n_samples))

    def eval_sample(p: cn.array, v: cn.array) -> None:
        X_temp = X_background[
            gen.integers(
                0, X_background.shape[0], size=(X.shape[0], n_background_samples)
            )
        ]
        null_pred = predict_fn(X_temp.reshape(X.shape[0] * n_background_samples, -1))
        v[:, -1, :, i] = null_pred.reshape(
            X.shape[0], n_background_samples, n_outputs
        ).mean(axis=1)
        previous_pred = null_pred
        for feature in p:
            X_temp[:, :, feature] = X[:, cn.newaxis, feature]
            pred = predict_fn(X_temp.reshape(X.shape[0] * n_background_samples, -1))
            v[:, feature, :, i] = (
                (pred - previous_pred)
                .reshape(X.shape[0], n_background_samples, n_outputs)
                .mean(axis=1)
            )
            previous_pred = pred

    for i in range(n_samples):
        p_a = random_state_.permutation(X.shape[1])
        p_b = cn.flip(p_a)
        eval_sample(p_a, v_a)
        eval_sample(p_b, v_b)

    v = (v_a + v_b) / 2
    shapley_values = cn.mean(v, axis=-1)
    se = cn.std(v, axis=-1, ddof=1) / cn.sqrt(n_samples)

    if check_efficiency:
        pred = predict_fn(X)
        if not cn.allclose(cn.sum(shapley_values, axis=1), pred):
            raise ValueError("Shapley values do not sum up to the predictions")

    if n_outputs == 1:
        shapley_values = shapley_values[:, :, 0]
        se = se[:, :, 0]
    return shapley_values, se
