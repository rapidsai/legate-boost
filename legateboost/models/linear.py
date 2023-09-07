import cunumeric as cn

from .base_model import BaseModel


class Linear(BaseModel):
    def fit(
        self,
        X: cn.ndarray,
        g: cn.ndarray,
        h: cn.ndarray,
    ) -> "Linear":

        num_outputs = g.shape[1]
        self.bias = -g.sum(axis=0) / h.sum(axis=0)
        g = g + self.bias[cn.newaxis, :] * h
        self.betas = cn.zeros((X.shape[1], num_outputs))
        for k in range(num_outputs):
            W = cn.sqrt(h[:, k])
            Xw = X * W[:, cn.newaxis]
            yw = W * (-g[:, k] / h[:, k])
            self.betas[:, k] = cn.linalg.lstsq(Xw, yw)[0]
        return self

    def clear(self) -> None:
        self.bias.fill(0)
        self.betas.fill(0)

    def update(
        self,
        X: cn.ndarray,
        g: cn.ndarray,
        h: cn.ndarray,
    ) -> "Linear":
        return self.fit(X, g, h)

    def predict(self, X: cn.ndarray) -> cn.ndarray:
        return self.bias + X.dot(self.betas)

    def __str__(self) -> str:
        return "Bias: " + str(self.bias) + "\nCoefficients: " + str(self.betas) + "\n"

    def __eq__(self, other: object) -> bool:
        return (other.betas == self.betas).all()
