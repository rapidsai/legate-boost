from typing import Tuple

import cunumeric as cn

from .metrics import LogLossMetric, MSEMetric


class SquaredErrorObjective:
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
        """Computes the functional gradient and hessian of the squared error
        objective function.

        Args:
            y (cn.ndarray): The true labels.
            pred (cn.ndarray): The predicted labels.

        Returns:
            cn.ndarray: The functional gradient and hessian of the squared error
            objective function.
        """
        return pred - y, cn.ones(pred.shape)

    def transform(self, pred: cn.ndarray) -> cn.ndarray:
        """Transforms the predicted labels. Identity function for MSE.

        Args:
            pred (cn.ndarray): The predicted labels.

        Returns:
            cn.ndarray: The transformed predicted labels.
        """
        return pred

    def metric(self) -> MSEMetric:
        """Returns default error metric.

        Returns:
            MSEMetric: The mean squared error metric.
        """
        return MSEMetric()


class LogLossObjective:
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
        """Computes the functional gradient and hessian of the log loss
        objective function.

        Args:
            y (cn.ndarray): The true labels.
            pred (cn.ndarray): The predicted labels.

        Returns:
            Tuple[cn.ndarray, cn.ndarray]: The functional gradient and hessian
            of the log loss objective function.
        """
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
        """Transforms the predicted labels using the sigmoid function for
        binary classification and the softmax function for multi-class
        classification.

        Args:
            pred (cn.ndarray): The predicted labels.

        Returns:
            cn.ndarray: The transformed predicted labels.
        """
        assert len(pred.shape) == 2
        if pred.shape[1] == 1:
            return 1.0 / (1.0 + cn.exp(-pred))
        # softmax function
        s = cn.max(pred, axis=1)
        e_x = cn.exp(pred - s[:, cn.newaxis])
        div = cn.sum(e_x, axis=1)
        return e_x / div[:, cn.newaxis]

    def metric(self) -> LogLossMetric:
        """Returns the metric object for the Log Loss objective function.

        Returns:
            LogLossMetric: The metric object for the Log Loss objective function.
        """
        return LogLossMetric()


objectives = {"squared_error": SquaredErrorObjective, "log_loss": LogLossObjective}
