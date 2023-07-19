import cunumeric as cn
import legateboost as lb


class MyMetric(lb.BaseMetric):
    def metric(self, y: cn.ndarray, pred: cn.ndarray, w: cn.ndarray) -> float:
        return cn.sqrt(((y - pred) ** 2 * w).sum() / w.sum())

    def name(self):
        return "rmse"


class MyObjective(lb.BaseObjective):
    def gradient(self, y, pred):
        return 0.5 * pred - y, 0.5 * cn.ones(pred.shape)

    def metric(self):
        return MyMetric()


X = cn.random.random((100, 10))
y = cn.random.random(X.shape[0])
model = lb.LBRegressor(verbose=1, objective=MyObjective(), n_estimators=10).fit(X, y)

model = lb.LBRegressor(
    verbose=1, objective="squared_error", metric=MyMetric(), n_estimators=10
).fit(X, y)
