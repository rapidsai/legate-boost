import cunumeric as cn

from .base_model import BaseModel


class Linear(BaseModel):
    def __init__(self, alpha: float = 0.0) -> None:
        self.alpha = alpha

    def fit(
        self,
        X: cn.ndarray,
        g: cn.ndarray,
        h: cn.ndarray,
    ) -> "Linear":

        num_outputs = g.shape[1]
        self.bias_ = cn.zeros(num_outputs)
        self.betas_ = cn.zeros((X.shape[1], num_outputs))
        for k in range(num_outputs):
            W = cn.sqrt(h[:, k])
            Xw = cn.ones((X.shape[0], X.shape[1] + 1))
            Xw[:, 1:] = X
            Xw = Xw * W[:, cn.newaxis]
            diag = cn.eye(Xw.shape[1]) * self.alpha
            diag[0, 0] = 0
            XtX = cn.dot(Xw.T, Xw) + diag
            yw = W * (-g[:, k] / h[:, k])
            result = cn.linalg.solve(XtX, cn.dot(Xw.T, yw))
            self.bias_[k] = result[0]
            self.betas_[:, k] = result[1:]
        return self

    def clear(self) -> None:
        self.bias_.fill(0)
        self.betas_.fill(0)

    def update(
        self,
        X: cn.ndarray,
        g: cn.ndarray,
        h: cn.ndarray,
    ) -> "Linear":
        return self.fit(X, g, h)

    def predict(self, X: cn.ndarray) -> cn.ndarray:
        return self.bias_ + X.dot(self.betas_)

    def __str__(self) -> str:
        return "Bias: " + str(self.bias_) + "\nCoefficients: " + str(self.betas_) + "\n"

    def __eq__(self, other: object) -> bool:
        return (other.betas_ == self.betas_).all()
