from abc import ABC, abstractmethod
from typing import Tuple

import cunumeric as cn

from .metrics import (
    BaseMetric,
    ExponentialMetric,
    LogLossMetric,
    MSEMetric,
    NormalLLMetric,
)


class BaseObjective(ABC):
    """The base class for objective functions.

    Implement this class to create custom objectives.
    """

    @abstractmethod
    def gradient(
        self, y: cn.ndarray, pred: cn.ndarray
    ) -> Tuple[cn.ndarray, cn.ndarray]:
        """Computes the functional gradient and hessian of the squared error
        objective function.

        Args:
            y : The true labels.
            pred : The predicted labels.

        Returns:
            The functional gradient and hessian of the squared error
            objective function.
        """  # noqa: E501
        pass

    def transform(self, pred: cn.ndarray) -> cn.ndarray:
        """Transforms the predicted labels. E.g. sigmoid for log loss.

        Args:
            pred : The predicted labels.

        Returns:
            The transformed labels.
        """
        return pred

    @abstractmethod
    def metric(self) -> BaseMetric:
        """Returns the default error metric for the objective function.

        Returns:
            The default error metric for the objective function.
        """
        pass

    def check_labels(self, y: cn.ndarray) -> int:
        """Checks the validity of the labels for this objective function.
        Return the number of outputs for raw prediction.

        Args:
            y : The labels.

        Returns:
            The number of outputs for raw prediction.
        """
        pass


class SquaredErrorObjective(BaseObjective):
    """The Squared Error objective function for regression problems.

    This objective function computes the mean squared error between the
    predicted and true labels.

    :math:`L(y_i, p_i) = \\frac{1}{2} (y_i - p_i)^2`

    See also:
        :class:`legateboost.metrics.MSEMetric`
    """

    def gradient(
        self, y: cn.ndarray, pred: cn.ndarray
    ) -> Tuple[cn.ndarray, cn.ndarray]:
        return pred - y, cn.ones(pred.shape)

    def transform(self, pred: cn.ndarray) -> cn.ndarray:
        return pred

    def metric(self) -> MSEMetric:
        return MSEMetric()

    def check_labels(self, y: cn.ndarray) -> int:
        return y.shape[1]


class NormalObjective(BaseObjective):
    """The normal distribution objective function for regression problems.

    This objective fits both mean and standard deviation parameters, where :class:`SquaredErrorObjective` only fits the mean.

    The objective minimised is the negative log likelihood of the normal distribution.

    :math:`L(y_i, p_i) = -log(\\frac{1}{\\sqrt{2\\pi p_{i, 1}^2}} exp(-\\frac{(y_i - p_{i, 0})^2}{2 p_{i, 1}^2}))`

    Where :math:`p_{i, 0}` is the mean and :math:`p_{i, 1}` is the standard deviation.

    See also:
        :class:`legateboost.metrics.NormalLLMetric`
    """  # noqa: E501

    def check_labels(self, y: cn.ndarray) -> int:
        return y.shape[1] * 2

    def gradient(self, y: cn.ndarray, pred: cn.ndarray) -> cn.ndarray:
        grad = cn.zeros((y.shape[0], y.shape[1], 2))
        hess = cn.ones((y.shape[0], y.shape[1], 2))
        mean = pred[:, :, 0]
        sd = pred[:, :, 1]
        var = sd * sd

        diff = mean - y
        grad[:, :, 0] = diff / var
        hess[:, :, 0] = 1 / var  # fisher information

        # do not update the variance on first iteration when predictions are 0
        if not cn.all(mean == 0.0):
            grad[:, :, 1] = 1 / sd - (diff * diff) / (var * sd)
            hess[:, :, 1] = 2 / var  # fisher information
        return grad.reshape(grad.shape[0], -1), hess.reshape(hess.shape[0], -1)

    def metric(self) -> NormalLLMetric:
        return NormalLLMetric()

    def transform(self, pred) -> cn.ndarray:
        # normally predictions start at 0.0
        # this is not allowed for variance
        # renormalize to start at 1.0
        copy = pred.copy().reshape(pred.shape[0], pred.shape[1] // 2, 2)
        copy[:, :, 1] += 1.0
        return copy


class LogLossObjective(BaseObjective):
    """The Log Loss objective function for binary and multi-class
    classification problems.

    This objective function computes the log loss between the predicted and true labels.

    :math:`L(y_i, p_i) = -y_i log(p_i) - (1 - y_i) log(1 - p_i)`

    See also:
        :class:`legateboost.metrics.LogLossMetric`
    """

    def gradient(
        self, y: cn.ndarray, pred: cn.ndarray
    ) -> Tuple[cn.ndarray, cn.ndarray]:
        assert pred.ndim == 2
        eps = 1e-15
        # binary case
        if pred.shape[1] == 1:
            return pred - y, cn.maximum(pred * (1.0 - pred), eps)

        # multi-class case
        label = y.astype(cn.int32).squeeze()
        h = pred * (1.0 - pred)
        g = pred.copy()
        g[cn.arange(y.size), label] -= 1.0
        return g, cn.maximum(h, eps)

    def transform(self, pred: cn.ndarray) -> cn.ndarray:
        assert len(pred.shape) == 2
        if pred.shape[1] == 1:
            return 1.0 / (1.0 + cn.exp(-pred))
        # softmax function
        s = cn.max(pred, axis=1)
        e_x = cn.exp(pred - s[:, cn.newaxis])
        div = cn.sum(e_x, axis=1)
        return e_x / div[:, cn.newaxis]

    def metric(self) -> LogLossMetric:
        return LogLossMetric()

    def check_labels(self, y: cn.ndarray) -> int:
        if not cn.all((y == cn.floor(y)) & (y >= 0)):
            raise ValueError("Expected labels to be non-zero whole numbers")
        return cn.max(y) + 1


class ExponentialObjective(BaseObjective):
    """Exponential loss objective function for binary classification.
    Equivalent to the AdaBoost multiclass exponential loss in [1].

    Defined as:

    :math:`L(y_i, p_i) = exp(-\\frac{1}{K} y_i^T p_i)`

    where :math:`K` is the number of classes, and
    :math:`y_{i,k} = 1` if :math:`k` is the label and :math:`y_{i,k} = -1/(K-1)` otherwise.

    See also:
        :class:`legateboost.metrics.ExponentialMetric`

    References
    ----------
    [1] Hastie, Trevor, et al. "Multi-class adaboost." Statistics and its Interface 2.3 (2009): 349-360.
    """  # noqa: E501

    def gradient(self, y: cn.ndarray, pred: cn.ndarray) -> cn.ndarray:
        assert pred.ndim == 2

        # binary case
        if pred.shape[1] == 1:
            adjusted_y = 2 * y - 1.0
            f = 0.5 * cn.log(pred / (1 - pred))  # undo sigmoid
            exp = cn.exp(-f * adjusted_y)
            return -adjusted_y * exp, exp

        # multi-class case
        K = pred.shape[1]  # number of classes
        f = cn.log(pred) * (K - 1)  # undo softmax
        y_k = cn.full((y.size, K), -1.0 / (K - 1.0))
        labels = y.astype(cn.int32).squeeze()
        y_k[cn.arange(y.size), labels] = 1.0
        exp = cn.exp(-1 / K * cn.sum(y_k * f, axis=1))

        return (
            -1 / K * y_k * exp[:, cn.newaxis],
            (1 / K**2) * y_k * y_k * exp[:, cn.newaxis],
        )

    def transform(self, pred: cn.ndarray) -> cn.ndarray:
        logloss = LogLossObjective()
        if pred.shape[1] == 1:
            return logloss.transform(2 * pred)
        K = pred.shape[1]  # number of classes
        return logloss.transform((1 / (K - 1)) * pred)

    def metric(self) -> ExponentialMetric:
        return ExponentialMetric()

    def check_labels(self, y: cn.ndarray) -> int:
        if not cn.all((y == cn.floor(y)) & (y >= 0)):
            raise ValueError("Expected labels to be non-zero whole numbers")
        return cn.max(y) + 1


objectives = {
    "squared_error": SquaredErrorObjective,
    "normal": NormalObjective,
    "log_loss": LogLossObjective,
    "exp": ExponentialObjective,
}
