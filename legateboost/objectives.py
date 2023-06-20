import cunumeric as cn

from .metrics import LogLossMetric, MSEMetric


class SquaredErrorObjective:
    def gradient(self, y: cn.ndarray, pred: cn.ndarray) -> cn.ndarray:
        return pred - y, cn.ones(pred.shape)

    def transform(self, pred: cn.ndarray) -> cn.ndarray:
        return pred

    def metric(self) -> MSEMetric:
        return MSEMetric()


class LogLossObjective:
    def gradient(self, y: cn.ndarray, pred: cn.ndarray) -> cn.ndarray:
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


objectives = {"squared_error": SquaredErrorObjective, "log_loss": LogLossObjective}
