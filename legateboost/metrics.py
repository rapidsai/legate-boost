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
        return float(numerator.mean())

    def name(self) -> str:
        return "MSE"

    def requires_probability(self) -> bool:
        return False


class LogLossMetric:
    def metric(self, y: cn.ndarray, pred: cn.ndarray, w: cn.ndarray) -> float:
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
        return "logloss"


class ExponentialMetric:
    def metric(self, y: cn.ndarray, pred: cn.ndarray, w: cn.ndarray) -> float:
        y = y.squeeze()
        pred = pred.squeeze()
        adjusted_y = 2 * y - 1.0
        exp = cn.exp(-pred * adjusted_y)
        return float((exp * w).sum() / w.sum())

    def requires_probability(self) -> bool:
        return False

    def name(self) -> str:
        return "exp"


metrics = {"logloss": LogLossMetric, "mse": MSEMetric, "exp": ExponentialMetric}
