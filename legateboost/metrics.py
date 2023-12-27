from abc import ABC, abstractmethod
from typing import Tuple

from typing_extensions import Self

import cunumeric as cn

from .utils import pick_col_by_idx, sample_average, set_col_by_idx


class BaseMetric(ABC):
    """The base class for metrics.

    Implement this class to create custom metrics.
    """

    @abstractmethod
    def metric(self, y: cn.ndarray, pred: cn.ndarray, w: cn.ndarray) -> float:
        """Computes the metric between the true labels `y` and predicted labels
        `pred`, weighted by `w`.

        Args:
            y : True labels.
            pred : Predicted labels.
            w : Weights for each sample.

        Returns:
            The metric between the true labels `y` and predicted labels
            `pred`, weighted by `w`.
        """
        pass

    @abstractmethod
    def name(self) -> str:
        """Returns the name of the metric as a string.

        Returns:
            The name of the metric.
        """
        pass

    @classmethod
    def create(cls) -> Self:
        return cls()


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

        # per output
        mse = numerator / w_sum
        # average over outputs
        return float(mse.mean())

    def name(self) -> str:
        return "mse"


def check_normal(y: cn.ndarray, pred: cn.ndarray) -> Tuple[cn.ndarray, cn.ndarray]:
    """Checks for normal distribution inputs."""
    if y.size * 2 != pred.size:
        raise ValueError("Expected pred to contain mean and sd for each y_i")
    if y.ndim == 1:
        y = y.reshape((y.size, 1))
    pred = pred.reshape((y.shape[0], y.shape[1], 2))
    return y, pred


class NormalLLMetric(BaseMetric):
    """The mean negative log likelihood of the labels, given mean and variance
    parameters.

    :math:`L(y_i, p_i) = -log(\\frac{1}{\\sqrt{2\\pi exp(p_{i, 1})}} exp(-\\frac{(y_i - p_{i, 0})^2}{2 exp(2 p_{i, 1})}))`

    Where :math:`p_{i, 0}` is the mean and :math:`p_{i, 1}` is the log standard deviation.

    See also:
        :class:`legateboost.objectives.NormalObjective`
    """  # noqa: E501

    def metric(self, y: cn.ndarray, pred: cn.ndarray, w: cn.ndarray) -> float:
        y, pred = check_normal(y, pred)
        w_sum = w.sum()
        if w_sum == 0:
            return 0
        if y.ndim == 2:
            w = w[:, cn.newaxis]
        mean = pred[:, :, 0]
        diff = mean - y
        log_sigma = pred[:, :, 1]
        neg_ll = 0.5 * (
            cn.exp(-2 * log_sigma) * diff * diff + 2 * log_sigma + cn.log(2 * cn.pi)
        )
        # ll = -0.5 * cn.log(2 * cn.pi) + 2 * log_sigma - 0.5 * (diff * diff) / var
        neg_ll = (neg_ll * w).sum(axis=0) / w_sum
        # average over output
        return float(neg_ll.mean())

    def name(self) -> str:
        return "normal_neg_ll"


def erf(x: cn.ndarray) -> cn.ndarray:
    """Element-wise error function.

    Parameters
    ----------
    x :
        Input array.

    Returns :
        The error function applied element-wise to the input array.
    """
    # Code from https://www.johndcook.com/blog/python_erf/
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911

    # Save the sign of x
    sign = cn.ones(shape=x.shape, dtype=cn.int8)
    sign[x < 0.0] = -1
    x = cn.abs(x)

    # A&S formula 7.1.26
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * cn.exp(-x * x)

    return sign * y


def norm_cdf(x: cn.ndarray) -> cn.ndarray:
    """CDF function for standard normal distribution."""
    return 0.5 * (1.0 + erf(x / cn.sqrt(2.0)))


def norm_pdf(x: cn.ndarray) -> cn.ndarray:
    """PDF function for standard normal distribution."""
    return cn.exp(-0.5 * (x) ** 2) / (cn.sqrt(2.0 * cn.pi))


class NormalCRPSMetric(BaseMetric):
    """Continuous Ranked Probability Score for normal distribution. Can be used
    with :py:class:`~legateboost.objectives.NormalObjective`.

    References
    ----------
    [1] Tilmann Gneiting, Adrian E. Raftery (2007)
        `Strictly Proper Scoring Rules, Prediction, and Estimation`
    """

    def metric(self, y: cn.ndarray, pred: cn.ndarray, w: cn.ndarray) -> float:
        y, pred = check_normal(y, pred)
        loc = pred[:, :, 0]
        # `NormalObjective` outputs variance instead of scale.
        scale = cn.sqrt(pred[:, :, 1])
        z = (y - loc) / scale
        # This is negating the definition in [1] to make it a loss.
        v = scale * (z * (2 * norm_cdf(z) - 1) + 2 * norm_pdf(z) - 1 / cn.sqrt(cn.pi))
        return sample_average(v, w)

    def name(self) -> str:
        return "normal_crps"


class GammaDevianceMetric(BaseMetric):
    """The mean gamma deviance.

    :math:`E = 2[(\\ln{\\frac{p_i}{y_i}} + \\frac{y_i}{p_i} - 1)w_i]`
    """

    def metric(self, y: cn.ndarray, pred: cn.ndarray, w: cn.ndarray) -> float:
        eps = 1e-15
        y = y + eps
        pred = pred + eps
        d = 2.0 * (cn.log(pred / y) + y / pred - 1.0)
        result = sample_average(d, w)
        if result.size > 1:
            result = cn.average(result)
        return float(result)

    def name(self) -> str:
        return "deviance_gamma"


class QuantileMetric(BaseMetric):
    """The quantile loss, otherwise known as check loss or pinball loss.

    :math:`L(y, p) = \\frac{1}{n}\\sum_{i=1}^{n} \\frac{1}{k}\\sum_{j=1}^{k} (q_j - \\mathbb{1})(y_i - p_{i, j})`

    where

    :math:`\\mathbb{1} = 1` if :math:`y_i - p_{i, j} \\leq 0` and :math:`\\mathbb{1} = 0` otherwise.

    See also:
        :class:`legateboost.objectives.QuantileObjective`
    """  # noqa

    def __init__(self, quantiles: cn.ndarray = cn.array([0.25, 0.5, 0.75])) -> None:
        super().__init__()
        assert cn.all(0.0 <= quantiles) and cn.all(quantiles <= 1.0)
        self.quantiles = quantiles

    def metric(self, y: cn.ndarray, pred: cn.ndarray, w: cn.ndarray) -> float:
        assert w.ndim == 1
        assert y.shape[1] == 1
        assert pred.shape[1] == self.quantiles.size
        diff = y - pred
        indicator = diff <= 0
        loss = (self.quantiles[cn.newaxis, :] - indicator) * diff
        return ((loss * w[:, cn.newaxis]).sum() / self.quantiles.size) / w.sum()

    def name(self) -> str:
        return "quantile"


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

        w_sum = w.sum()
        if w_sum == 0:
            return 0.0

        # binary case
        if pred.ndim == 1 or pred.shape[1] == 1:
            pred = pred.squeeze()
            logloss = -(y * cn.log(pred) + (1 - y) * cn.log(1 - pred))
            return float((logloss * w).sum() / w_sum)

        # multi-class case
        assert pred.ndim == 2
        label = y.astype(cn.int32)

        logloss = -cn.log(pick_col_by_idx(pred, label))
        # logloss = -cn.log(pred[cn.arange(label.size), label])

        return float((logloss * w).sum() / w_sum)

    def name(self) -> str:
        return "log_loss"


class ExponentialMetric(BaseMetric):
    """Class for computing the exponential loss metric.

    :math:`exp(y, p) = \\sum_{i=1}^{n} \\exp(-\\frac{1}{K}  y_i^T f_i)`
    where :math:`K` is the number of classes, and
    :math:`y_{i,k} = 1` if :math:`k` is the label and :math:`y_{i,k} = -1/(K-1)` otherwise.
    :math:`f_{i,k} = ln(p_{i, k}) * (K - 1)` with :math:`p_{i, k}` a probability.

    See also:
        :class:`legateboost.objectives.ExponentialObjective`
    """  # noqa: E501

    def metric(self, y: cn.ndarray, pred: cn.ndarray, w: cn.ndarray) -> float:
        y = y.squeeze()
        # binary case
        if pred.ndim == 1 or pred.shape[1] == 1:
            pred = pred.squeeze()
            exp = cn.power(pred / (1 - pred), 0.5 - y)
            return float((exp * w).sum() / w.sum())

        # multi-class case
        # note that exp loss is invariant to adding a constant to prediction
        K = pred.shape[1]  # number of classes
        f = cn.log(pred) * (K - 1)  # undo softmax
        y_k = cn.full((y.size, K), -1.0 / (K - 1.0))

        set_col_by_idx(y_k, y.astype(cn.int32), 1.0)
        # y_k[cn.arange(y.size), y.astype(cn.int32)] = 1.0

        exp = cn.exp(-1 / K * cn.sum(y_k * f, axis=1))
        return float((exp * w).sum() / w.sum())

    def name(self) -> str:
        return "exp"


metrics = {
    "log_loss": LogLossMetric,
    "mse": MSEMetric,
    "exp": ExponentialMetric,
    "normal_neg_ll": NormalLLMetric,
    "normal_crps": NormalCRPSMetric,
    "deviance_gamma": GammaDevianceMetric,
}
