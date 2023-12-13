import cunumeric as cn

from ..utils import solve_singular
from .base_model import BaseModel


class Linear(BaseModel):
    """Generalised linear model. Boosting linear models is equivalent to
    fitting a single linear model where each boosting iteration is a newton
    step. Note that the l2 penalty is applied to the weights of each model, as
    opposed to the sum of all models. This can lead to different results when
    compared to fitting a linear model with sklearn.

    It is recommended to normalize the data before fitting. This ensures
    regularisation is evening applied to all features and prevents numerical issues.

    Parameters
    ----------
    alpha : L2 regularization parameter.

    Attributes
    ----------
    bias_ : ndarray of shape (n_outputs,)
        Intercept term.
    betas_ : ndarray of shape (n_features, n_outputs)
        Coefficients of the linear model.
    """

    def __init__(self, alpha: float = 1e-5) -> None:
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
            result = solve_singular(XtX, cn.dot(Xw.T, yw))
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
        if not isinstance(other, Linear):
            raise NotImplementedError()
        return (other.betas_ == self.betas_).all()
