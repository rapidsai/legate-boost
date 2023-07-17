from abc import ABC, abstractmethod

import cunumeric as cn


class BaseMetric(ABC):
    """The base class for metrics.

    Implement this class to create custom metrics.
    """

    @abstractmethod
    def metric(self, y: cn.ndarray, pred: cn.ndarray, w: cn.ndarray) -> float:
        """Computes the metric between the true labels `y` and predicted labels
        `pred`, weighted by `w`.

        Args:
            y (cn.ndarray): True labels.
            pred (cn.ndarray): Predicted labels.
            w (cn.ndarray): Weights for each sample.

        Returns:
            float: The metric between the true labels `y` and predicted labels
            `pred`, weighted by `w`.
        """
        pass

    def requires_probability(self) -> bool:
        """Returns whether or not the metric requires predicted probabilities.

        Returns:
            bool: True if the metric requires predicted probabilities, False otherwise.
        """
        return False

    @abstractmethod
    def name(self) -> str:
        """Returns the name of the metric as a string.

        Returns:
            str: The name of the metric.
        """
        pass


class MSEMetric(BaseMetric):
    """Class for computing the mean squared error (MSE) metric between the true
    labels and predicted labels.

    :math:`MSE(y, p) = \\frac{1}{n} \\sum_{i=1}^{n} (y_i - p_i)^2`

    See also:
        :class:`legateboost.objectives.SquaredErrorObjective`
    """

    def metric(self, y: cn.ndarray, pred: cn.ndarray, w: cn.ndarray) -> float:
        assert w.ndim == 1
        y = y.reshape(pred.shape)
        w_sum = w.sum()
        if w_sum == 0:
            return 0

        if y.ndim == 2:
            w = w[:, cn.newaxis]
        numerator = ((y - pred) ** 2 * w).sum(axis=0)

        numerator = numerator / w_sum
        return float(numerator.mean())

    def name(self) -> str:
        return "mse"


class LogLossMetric(BaseMetric):
    """Class for computing the logarithmic loss (logloss) metric between the
    true labels and predicted labels.

    For binary classification:

    :math:`logloss(y, p) = -\\frac{1}{n} \\sum_{i=1}^{n} [y_i \\log(p_i) + (1 - y_i) \\log(1 - p_i)]`

    For multi-class classification:

    :math:`logloss(y, p) = -\\frac{1}{n} \\sum_{i=1}^{n} \\sum_{j=1}^{k} y_{ij} \\log(p_{ij})`

    where `n` is the number of samples, `k` is the number of classes, `y` is the
    true labels, and `p` is the predicted probabilities.

    See also:
        :class:`legateboost.objectives.LogLossObjective`
    """  # noqa: E501

    def metric(self, y: cn.ndarray, pred: cn.ndarray, w: cn.ndarray) -> float:
        y = y.squeeze()
        eps = cn.finfo(pred.dtype).eps
        cn.clip(pred, eps, 1 - eps, out=pred)

        # binary case
        if pred.ndim == 1 or pred.shape[1] == 1:
            pred = pred.squeeze()
            logloss = -(y * cn.log(pred) + (1 - y) * cn.log(1 - pred))
            return float((logloss * w).sum() / w.sum())

        # multi-class case
        assert pred.ndim == 2
        label = y.astype(cn.int32)
        logloss = -cn.log(pred[cn.arange(label.size), label])
        return float((logloss * w).sum() / w.sum())

    def requires_probability(self) -> bool:
        return True

    def name(self) -> str:
        return "log_loss"


class ExponentialMetric(BaseMetric):
    """Class for computing the exponential loss metric.

    :math:`exp(y, p) = \\sum_{i=1}^{n} \\exp(-\\frac{1}{K}  y_i^T p_i)`

    where :math:`K` is the number of classes, and
    :math:`y_{i,k} = 1` if :math:`k` is the label and :math:`y_{i,k} = -1/(K-1)` otherwise.
    :math:`p_{i,k}` is not a probability, but the raw model output.

    See also:
        :class:`legateboost.objectives.ExponentialObjective`
    """  # noqa: E501

    def metric(self, y: cn.ndarray, pred: cn.ndarray, w: cn.ndarray) -> float:
        y = y.squeeze()
        # binary case
        if pred.ndim == 1 or pred.shape[1] == 1:
            pred = pred.squeeze()
            adjusted_y = 2 * y - 1.0
            exp = cn.exp(-pred * adjusted_y)
            return float((exp * w).sum() / w.sum())

        # multi-class case
        K = pred.shape[1]  # number of classes
        y_k = cn.full((y.size, K), -1.0 / (K - 1.0))
        y_k[cn.arange(y.size), y.astype(cn.int32)] = 1.0

        exp = cn.exp(-1 / K * cn.sum(y_k * pred, axis=1))
        return float((exp * w).sum() / w.sum())

    def requires_probability(self) -> bool:
        return False

    def name(self) -> str:
        return "exp"


metrics = {"log_loss": LogLossMetric, "mse": MSEMetric, "exp": ExponentialMetric}
