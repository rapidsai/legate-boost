import cunumeric as cn

from .metrics import ExponentialMetric, LogLossMetric, MSEMetric


class SquaredErrorObjective:
    def gradient(self, y: cn.ndarray, pred: cn.ndarray) -> cn.ndarray:
        return pred - y, cn.ones(pred.shape)

    def transform(self, pred: cn.ndarray) -> cn.ndarray:
        return pred

    def metric(self) -> MSEMetric:
        return MSEMetric()

    def check_labels(self, y) -> None:
        return


class LogLossObjective:
    def gradient(self, y: cn.ndarray, pred: cn.ndarray) -> cn.ndarray:
        assert pred.ndim == 2
        pred = self.transform(pred)
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

    def check_labels(self, y) -> None:
        if not cn.all((y == cn.floor(y)) & (y >= 0)):
            raise ValueError("Log loss expected labels to be non-zero whole numbers")


class ExponentialObjective:
    """Exponential loss objective function for binary classification.
    Equivalent to the AdaBoost exponential loss.

    In AdaBoost, the exponential loss is defined as exp(-y * pred),
    where y is in {-1, 1}

    In our case y is in {0, 1} so we adjust the loss to be exp(-(2 * y - 1) * pred).
    """

    def gradient(self, y: cn.ndarray, pred: cn.ndarray) -> cn.ndarray:
        assert pred.ndim == 2
        if pred.shape[1] == 1:
            pred = pred.squeeze()
            adjusted_y = 2 * y - 1.0
            exp = cn.exp(-pred * adjusted_y)
            return -adjusted_y * exp, exp

    def transform(self, pred: cn.ndarray) -> cn.ndarray:
        return 1 / (1 + cn.exp(-2 * pred))

    def metric(self) -> ExponentialMetric:
        return ExponentialMetric()

    def check_labels(self, y) -> None:
        if not cn.all((y == cn.floor(y)) & (y >= 0)):
            raise ValueError("Log loss expected labels to be non-zero whole numbers")


objectives = {
    "squared_error": SquaredErrorObjective,
    "log_loss": LogLossObjective,
    "exp": ExponentialObjective,
}
