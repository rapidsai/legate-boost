import cunumeric as cn


class MSEMetric:
    """Class for computing the mean squared error (MSE) metric between the true
    labels and predicted labels.

    :math:`MSE(y, p) = \\frac{1}{n} \\sum_{i=1}^{n} (y_i - p_i)^2`

    Methods:
        metric(y: cn.ndarray, pred: cn.ndarray, w: cn.ndarray) -> float:
            Computes the MSE metric between the true labels `y` and predicted
            labels `pred`, weighted by `w`.

        name() -> str:
            Returns the name of the metric as a string.
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
        """Returns the name of the metric as a string.

        Returns:
            str: The name of the metric.
        """
        return "MSE"

    def requires_probability(self) -> bool:
        return False


class LogLossMetric:
    """Class for computing the logarithmic loss (logloss) metric between the
    true labels and predicted labels.

    For binary classification:

    :math:`logloss(y, p) = -\\frac{1}{n} \\sum_{i=1}^{n} [y_i \\log(p_i) + (1 - y_i) \\log(1 - p_i)]` # noqa: E501

    For multi-class classification:

    :math:`logloss(y, p) = -\\frac{1}{n} \\sum_{i=1}^{n} \\sum_{j=1}^{k} y_{ij} \\log(p_{ij})` # noqa: E501

    where `n` is the number of samples, `k` is the number of classes, `y` is the
    true labels, and `p` is the predicted probabilities.
    """

    def metric(self, y: cn.ndarray, pred: cn.ndarray, w: cn.ndarray) -> float:
        """Computes the logloss metric between the true labels `y` and
        predicted probabilities `pred`, weighted by `w`.

        Args:
            y (cn.ndarray): True labels.
            pred (cn.ndarray): Predicted probabilities.
            w (cn.ndarray): Weights for each sample.

        Returns:
            float: The logloss metric between the true labels `y` and predicted
            probabilities `pred`, weighted by `w`.
        """
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
        """Returns the name of the metric as a string.

        Returns:
            str: The name of the metric.
        """
        return "logloss"


class ExponentialMetric:
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


metrics = {"logloss": LogLossMetric, "mse": MSEMetric, "exp": ExponentialMetric}
