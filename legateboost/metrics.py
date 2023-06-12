import cunumeric as cn


class MSEMetric:
    def metric(self, y: cn.ndarray, pred: cn.ndarray, w: cn.ndarray) -> float:
        """In multi-output case, return the average of the MSE of each
        output."""
        assert w.ndim == 1
        y = y.reshape(pred.shape)
        w_sum = w.sum()
        if w_sum == 0:
            return 0

        if y.ndim == 2:
            w = w[:, cn.newaxis]
        numerator = ((y - pred) ** 2 * w).sum(axis=0)

        numerator = numerator / w_sum
        return numerator.mean()

    def name(self) -> str:
        return "MSE"


class LogLossMetric:
    def metric(self, y: cn.ndarray, pred: cn.ndarray, w: cn.ndarray) -> float:
        logloss = -(y * cn.log(pred) + (1 - y) * cn.log(1 - pred))
        return float((logloss * w).sum() / w.sum())

    def name(self) -> str:
        return "logloss"
