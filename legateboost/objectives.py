import cunumeric as cn

from .metrics import LogLossMetric, MSEMetric


class SquaredErrorObjective:
    def gradient(self, y: cn.ndarray, pred: cn.ndarray, w: cn.ndarray) -> cn.ndarray:
        return pred - y, w

    def transform(self, pred: cn.ndarray) -> cn.ndarray:
        return pred

    def metric(self) -> MSEMetric:
        return MSEMetric()


class LogLossObjective:
    def gradient(self, y: cn.ndarray, pred: cn.ndarray, w: cn.ndarray) -> cn.ndarray:
        return pred - y, pred * (1.0 - pred) * w

    def transform(self, pred: cn.ndarray) -> cn.ndarray:
        return 1 / (1 + cn.exp(-pred))

    def metric(self) -> LogLossMetric:
        return LogLossMetric()


objectives = {"squared_error": SquaredErrorObjective, "log_loss": LogLossObjective}
