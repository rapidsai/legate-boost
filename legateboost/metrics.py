import cunumeric as cn


class MSEMetric:
    def metric(self, y: cn.ndarray, pred: cn.ndarray, w: cn.ndarray) -> float:
        w_sum = w.sum()
        if w_sum == 0:
            return 0
        numerator = ((y - pred) ** 2 * w).sum()
        return float(numerator / w_sum)

    def name(self) -> str:
        return "MSE"


class LogLossMetric:
    def metric(self, y: cn.ndarray, pred: cn.ndarray, w: cn.ndarray) -> float:
        logloss = -(y * cn.log(pred) + (1 - y) * cn.log(1 - pred))
        return float((logloss * w).sum() / w.sum())

    def name(self) -> str:
        return "logloss"
