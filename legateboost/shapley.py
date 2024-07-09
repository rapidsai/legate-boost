from typing import Optional

from sklearn.base import is_regressor
from sklearn.utils.validation import check_random_state

import cunumeric as cn

from .legateboost import BaseModel
from .metrics import BaseMetric


def global_shapley_attributions(
    model: BaseModel,
    X: cn.array,
    y: cn.array,
    metric: Optional[BaseMetric] = None,
    n_background_samples=100,
    random_state=None,
    n_samples: int = 5,
    assert_efficiency: bool = False,
):
    def predict_fn(X):
        fn = model.predict if is_regressor(model) else model.predict_proba
        return fn(X)

    if metric is None:
        metric = model._metrics[0]
    random_state = check_random_state(random_state)
    background_samples = random_state.choice(
        X.shape[0], n_background_samples, replace=True
    )
    X_background = X[background_samples]
    y_temp = cn.repeat(y, X_background.shape[0], axis=0)
    w = cn.ones(y_temp.shape[0])
    X_temp = cn.zeros((X.shape[0], X_background.shape[0], X.shape[1]))
    X_temp[:, :, :] = X_background[cn.newaxis, :, :]
    # null coalition
    pred = predict_fn(X_temp.reshape(X.shape[0] * X_background.shape[0], -1))
    null_loss = metric.metric(y_temp, pred, w)

    # antithetic sampling
    v_a = cn.zeros((X.shape[1], n_samples))
    v_b = cn.zeros((X.shape[1], n_samples))

    def eval_sample(p, v):
        X_temp[:, :, :] = X_background[cn.newaxis, :, :]
        previous_loss = null_loss
        for feature in p:
            X_temp[:, :, feature] = X[:, cn.newaxis, feature]
            pred = predict_fn(X_temp.reshape(X.shape[0] * X_background.shape[0], -1))
            loss = metric.metric(y_temp, pred, w)
            v[feature, i] = loss - previous_loss
            previous_loss = loss

    for i in range(n_samples):
        p_a = random_state.permutation(X.shape[1])
        p_b = cn.flip(p_a)
        eval_sample(p_a, v_a)
        eval_sample(p_b, v_b)

    v = (v_a + v_b) / 2
    shapley_values = cn.mean(v, axis=1)
    std = cn.std(v, axis=1, ddof=1) / cn.sqrt(n_samples)

    if assert_efficiency:
        full_coalition_loss = metric.metric(y, predict_fn(X), cn.ones(y.shape[0]))
        assert cn.isclose(null_loss + cn.sum(shapley_values), full_coalition_loss)
    return null_loss, shapley_values, std


def local_shapley_attributions(
    model: BaseModel,
    X: cn.array,
    X_background: cn.array,
    random_state=None,
    n_samples: int = 5,
    assert_efficiency: bool = False,
):
    def predict_fn(X):
        fn = model.predict if is_regressor(model) else model.predict_proba
        p = fn(X)
        if p.ndim == 1:
            return p.reshape(-1, 1)
        return p

    random_state = check_random_state(random_state)
    X_temp = cn.zeros((X.shape[0], X_background.shape[0], X.shape[1]))
    X_temp[:, :, :] = X_background[cn.newaxis, :, :]
    # null coalition
    null_pred = predict_fn(X_temp.reshape(X.shape[0] * X_background.shape[0], -1))
    n_outputs = null_pred.shape[1]

    # antithetic sampling
    v_a = cn.zeros((X.shape[0], X.shape[1], n_outputs, n_samples))
    v_b = cn.zeros((X.shape[0], X.shape[1], n_outputs, n_samples))

    def eval_sample(p, v):
        X_temp[:, :, :] = X_background[cn.newaxis, :, :]
        previous_pred = null_pred
        for feature in p:
            X_temp[:, :, feature] = X[:, cn.newaxis, feature]
            pred = predict_fn(X_temp.reshape(X.shape[0] * X_background.shape[0], -1))
            v[:, feature, :, i] = (
                (pred - previous_pred)
                .reshape(X.shape[0], X_background.shape[0], n_outputs)
                .mean(axis=1)
            )
            previous_pred = pred

    for i in range(n_samples):
        p_a = random_state.permutation(X.shape[1])
        p_b = cn.flip(p_a)
        eval_sample(p_a, v_a)
        eval_sample(p_b, v_b)

    v = (v_a + v_b) / 2
    shapley_values = cn.mean(v, axis=-1)
    print(shapley_values.shape)
    std = cn.std(v, axis=-1, ddof=1) / cn.sqrt(n_samples)

    null_shap = null_pred.reshape(X.shape[0], X_background.shape[0], n_outputs).mean(
        axis=1
    )
    if assert_efficiency:
        pred = predict_fn(X)
        assert cn.allclose(null_shap + cn.sum(shapley_values, axis=1), pred)

    if n_outputs == 1:
        shapley_values = shapley_values[:, :, 0]
        std = std[:, :, 0]
    return null_shap, shapley_values, std
