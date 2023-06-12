import cunumeric as cn

from .metrics import LogLossMetric, MSEMetric


class SquaredErrorObjective:
    def gradient(self, y: cn.ndarray, pred: cn.ndarray, w: cn.ndarray) -> cn.ndarray:
        if y.ndim == 2:
            w = cn.repeat(w[:, cn.newaxis], y.shape[1], axis=1)
        return pred - y, w

    def transform(self, pred: cn.ndarray) -> cn.ndarray:
        return pred

    def metric(self) -> MSEMetric:
        return MSEMetric()


class LogLossObjective:
    def gradient(self, y: cn.ndarray, pred: cn.ndarray, w: cn.ndarray) -> cn.ndarray:
        return pred - y, cn.maximum(pred * (1.0 - pred) * w, 1e-5)

    def transform(self, pred: cn.ndarray) -> cn.ndarray:
        return 1.0 / (1.0 + cn.exp(-pred))

    def metric(self) -> LogLossMetric:
        return LogLossMetric()


objectives = {"squared_error": SquaredErrorObjective, "log_loss": LogLossObjective}
