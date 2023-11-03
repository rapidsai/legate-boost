from abc import ABC, abstractmethod
from typing import Tuple

from scipy.stats import norm

import cunumeric as cn

from .metrics import (
    BaseMetric,
    ExponentialMetric,
    GammaDevianceMetric,
    LogLossMetric,
    MSEMetric,
    NormalLLMetric,
    QuantileMetric,
)
from .utils import mod_col_by_idx, preround, set_col_by_idx


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

    @abstractmethod
    def initialise_prediction(
        self, y: cn.ndarray, w: cn.ndarray, boost_from_average: bool
    ) -> cn.ndarray:
        """Initializes the base score of the model. May also validate labels.

        Args:
            y : The target values.
            w : The sample weights.
            boost_from_average (bool): Whether to initialize the predictions
              from the average of the target values.

        Returns:
            The initial predictions for a single example.
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

    def initialise_prediction(
        self, y: cn.ndarray, w: cn.ndarray, boost_from_average: bool
    ) -> cn.ndarray:
        assert y.ndim == 2
        if boost_from_average:
            y = preround(y)
            w = preround(w)
            return cn.sum(y * w[:, None], axis=0) / cn.sum(w)
        else:
            return cn.zeros(y.shape[1])


class NormalObjective(BaseObjective):
    """The normal distribution objective function for regression problems.

    This objective fits both mean and variance parameters, where :class:`SquaredErrorObjective` only fits the mean.

    The objective minimised is the negative log likelihood of the normal distribution.

    :math:`L(y_i, p_i) = -log(\\frac{1}{\\sqrt{2\\pi exp(p_{i, 1})}} exp(-\\frac{(y_i - p_{i, 0})^2}{2 exp(p_{i, 1})}))`

    Where :math:`p_{i, 0}` is the mean and :math:`p_{i, 1}` is the log standard deviation.

    The variance is clipped to a minimum of 1e-5 to avoid numerical instability.

    See also:
        :class:`legateboost.metrics.NormalLLMetric`
    """  # noqa: E501

    def gradient(self, y: cn.ndarray, pred: cn.ndarray) -> cn.ndarray:
        grad = cn.zeros((y.shape[0], y.shape[1], 2))
        hess = cn.ones((y.shape[0], y.shape[1], 2))
        mean = pred[:, :, 0]
        log_sigma = pred[:, :, 1]
        inv_var = cn.exp(-2 * log_sigma)
        assert log_sigma.ndim == 2
        diff = mean - y
        grad[:, :, 0] = diff * inv_var
        hess[:, :, 0] = inv_var  # fisher information

        grad[:, :, 1] = 1 - inv_var * diff * diff
        hess[:, :, 1] = 2  # fisher information
        return grad.reshape(grad.shape[0], -1), hess.reshape(hess.shape[0], -1)

    def metric(self) -> NormalLLMetric:
        return NormalLLMetric()

    def initialise_prediction(
        self, y: cn.ndarray, w: cn.ndarray, boost_from_average: bool
    ) -> cn.ndarray:
        assert y.ndim == 2
        pred = cn.zeros((y.shape[1], 2))
        if boost_from_average:
            y = preround(y)
            w = preround(w)
            mean = cn.sum(y * w[:, None], axis=0) / cn.sum(w)
            var = (y - mean) * (y - mean) * w[:, None]
            var = cn.sum(preround(var), axis=0) / cn.sum(w)
            pred[:, 0] = mean
            pred[:, 1] = cn.log(var) / 2
        return pred.reshape(-1)

    def transform(self, pred: cn.ndarray) -> cn.ndarray:
        # internally there is no third dimension
        # reshape this nicely for the user so mean and variance have their own dimension
        pred = pred.reshape((pred.shape[0], pred.shape[1] // 2, 2))
        # don't let the variance go to zero
        pred[:, :, 1] = cn.clip(pred[:, :, 1], -5, 5)
        return pred


class FitInterceptRegMixIn(BaseObjective):
    def one_step_newton(
        self, y: cn.ndarray, w: cn.ndarray, boost_from_average: bool, n_targets: int
    ) -> cn.ndarray:
        if boost_from_average:
            # take 1 newton step (we could iterate here to get a better estimate)
            g, h = self.gradient(
                y,
                self.transform(cn.zeros((y.shape[0], n_targets))),
            )
            g = g * w[:, None]
            h = h * w[:, None]
            return -preround(g).sum(axis=0) / preround(h).sum(axis=0)
        return cn.zeros(n_targets)


class GammaDevianceObjective(FitInterceptRegMixIn):
    """Gamma regression with the log link function. For the expression of the
    deviance, see :py:class:`legateboost.metrics.GammaDevianceMetric`.

    The response :math:`y` variable should be positive values.
    """

    def gradient(
        self, y: cn.ndarray, pred: cn.ndarray
    ) -> Tuple[cn.ndarray, cn.ndarray]:
        # p = exp(u)
        #
        # g = dL/du   = 1 - y / exp(u)
        # h = d^2L/du = y / exp(u)
        h = y / pred
        g = 1.0 - h
        return g, h

    def metric(self) -> GammaDevianceMetric:
        return GammaDevianceMetric()

    def transform(self, pred: cn.ndarray) -> cn.ndarray:
        """Inverse log link."""
        return cn.exp(pred)

    def initialise_prediction(
        self, y: cn.ndarray, w: cn.ndarray, boost_from_average: bool
    ) -> cn.ndarray:
        if not (y > 0.0).all():
            raise ValueError("y is expected to be positive.")
        if y.ndim == 1 or y.shape[1] <= 1:
            n_targets = 1
        else:
            n_targets = y.shape[1]
        return self.one_step_newton(y, w, boost_from_average, n_targets)


class QuantileObjective(BaseObjective):
    """Minimises the quantile loss, otherwise known as check loss or pinball
    loss.

    :math:`L(y_i, p_i) = \\frac{1}{k}\\sum_{j=1}^{k} (q_j - \\mathbb{1})(y_i - p_{i, j})`

    where

    :math:`\\mathbb{1} = 1` if :math:`y_i - p_{i, j} \\leq 0` and :math:`\\mathbb{1} = 0` otherwise.

    This objective function is non-smooth and therefore can converge significantly slower than other objectives.

    See also:
        :class:`legateboost.metrics.QuantileMetric`
    """  # noqa

    def __init__(self, quantiles: cn.ndarray = cn.array([0.25, 0.5, 0.75])) -> None:
        super().__init__()
        assert cn.all(0.0 < quantiles) and cn.all(quantiles < 1.0)
        self.quantiles = quantiles

    def gradient(self, y: cn.ndarray, pred: cn.ndarray) -> cn.ndarray:
        diff = y - pred
        indicator = diff <= 0
        # Apply the polyak step size rule for subgradient descent.
        # Notice that this scales the gradient magnitude relative to the loss
        # function. If we don't do this, the gradient sizes are constant with
        # respect to the size of the input labels. E.g. if the labels are very
        # large and we take 0.5 size steps, convergence takes forever.
        polyak_step_size = (
            ((self.quantiles[cn.newaxis, :] - indicator) * diff).sum() * 2 / pred.size
        )
        return (indicator - self.quantiles[cn.newaxis, :]) * polyak_step_size, cn.ones(
            pred.shape
        )

    def metric(self) -> BaseMetric:
        return QuantileMetric(self.quantiles)

    def initialise_prediction(
        self, y: cn.ndarray, w: cn.ndarray, boost_from_average: bool
    ) -> cn.ndarray:
        assert y.shape[1] == 1, "Quantile loss does not support multi-output"
        # We don't have a way to calculate weighted quantiles easily in cunumeric.
        # In any case, it would require slow global sort.
        # Instead fit a normal distribution to the data and use that
        # to estimate quantiles.
        if boost_from_average:
            y = preround(y)
            w = preround(w)
            mean = cn.sum(y * w[:, None], axis=0) / cn.sum(w)
            var = cn.sum((y - mean) * (y - mean) * w[:, None], axis=0) / cn.sum(w)
            init = cn.array(
                norm.ppf(self.quantiles, loc=mean[0], scale=cn.sqrt(var[0]))
            )
            return init
        return cn.zeros_like(self.quantiles)


class LogLossObjective(FitInterceptRegMixIn):
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
        # binary case
        if pred.shape[1] == 1:
            return pred - y, pred * (1.0 - pred)

        # multi-class case
        label = y.astype(cn.int32).squeeze()
        h = pred * (1.0 - pred)
        g = pred.copy()
        mod_col_by_idx(g, label, -1.0)
        # g[cn.arange(y.size), label] -= 1.0
        return g, h

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

    def initialise_prediction(
        self, y: cn.ndarray, w: cn.ndarray, boost_from_average: bool
    ) -> cn.ndarray:
        if not cn.all((y == cn.floor(y)) & (y >= 0)):
            raise ValueError("Expected labels to be non-zero whole numbers")
        num_class = int(cn.max(y) + 1)
        n_targets = num_class if num_class > 2 else 1
        return self.one_step_newton(y, w, boost_from_average, n_targets)


class ExponentialObjective(FitInterceptRegMixIn):
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
        set_col_by_idx(y_k, labels, 1.0)
        # y_k[cn.arange(y.size), labels] = 1.0
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

    def initialise_prediction(
        self, y: cn.ndarray, w: cn.ndarray, boost_from_average: bool
    ) -> cn.ndarray:
        if not cn.all((y == cn.floor(y)) & (y >= 0)):
            raise ValueError("Expected labels to be non-zero whole numbers")
        num_class = int(cn.max(y) + 1)
        n_targets = num_class if num_class > 2 else 1
        return self.one_step_newton(y, w, boost_from_average, n_targets)


objectives = {
    "squared_error": SquaredErrorObjective,
    "normal": NormalObjective,
    "log_loss": LogLossObjective,
    "exp": ExponentialObjective,
    "quantile": QuantileObjective,
    "gamma_deviance": GammaDevianceObjective,
}
