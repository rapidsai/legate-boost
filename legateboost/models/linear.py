import copy
from typing import Any, List, Sequence, Tuple, cast

import cupynumeric as cn

from ..utils import lbfgs, solve_singular
from .base_model import BaseModel


class Linear(BaseModel):
    """Generalised linear model. Boosting linear models is equivalent to
    fitting a single linear model where each boosting iteration is a newton
    step. Note that the l2 penalty is applied to the weights of each model, as
    opposed to the sum of all models. This can lead to different results when
    compared to fitting a linear model with sklearn.

    It is recommended to normalize the data before fitting. This ensures
    regularisation is evenly applied to all features and prevents numerical issues.

    Two solvers are available. A direct numerical solver that can be faster, but
    uses more memory, and an iterative L-BFGS solver that uses less memory
    but can be slower.

    Parameters
    ----------
    alpha : L2 regularization parameter.
    solver : "direct" or "lbfgs"
        If "direct", use a direct solver. If "lbfgs", use the lbfgs solver.

    Attributes
    ----------
    bias_ : ndarray of shape (n_outputs,)
        Intercept term.
    betas_ : ndarray of shape (n_features, n_outputs)
        Coefficients of the linear model.
    """

    def __init__(self, alpha: float = 1e-5, solver: str = "direct") -> None:
        self.alpha = alpha
        self.solver = solver

    def _fit_solve(self, X: cn.ndarray, g: cn.ndarray, h: cn.ndarray) -> None:
        self.betas_ = cn.zeros((X.shape[1] + 1, g.shape[1]))
        num_outputs = g.shape[1]
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
            self.betas_[:, k] = result

    def _loss_grad(
        self, betas: cn.ndarray, X: cn.ndarray, g: cn.ndarray, h: cn.ndarray
    ) -> Tuple[float, cn.ndarray]:
        self.betas_ = betas.reshape(self.betas_.shape)
        pred = self.predict(X)
        loss = (pred * (g + 0.5 * h * pred)).sum(axis=0).mean()
        # make sure same type as X, else a copy is made
        delta = (g + h * pred).astype(X.dtype)
        grads = cn.empty(self.betas_.shape, dtype=X.dtype)
        grads[0] = delta.sum(axis=0)
        grads[1:] = cn.dot(X.T, delta) + self.alpha * self.betas_[1:]
        grads /= X.shape[0]
        assert grads.shape == self.betas_.shape
        return loss, grads.ravel()

    def _fit_lbfgs(self, X: cn.ndarray, g: cn.ndarray, h: cn.ndarray) -> None:
        self.betas_ = cn.zeros((X.shape[1] + 1, g.shape[1]))
        result = lbfgs(
            self.betas_.ravel(),
            self._loss_grad,
            args=(X, g, h),
            verbose=0,
            gtol=1e-5,
            max_iter=100,
        )
        self.betas_ = result.x.reshape(self.betas_.shape)

    def fit(
        self,
        X: cn.ndarray,
        g: cn.ndarray,
        h: cn.ndarray,
    ) -> "Linear":
        if self.solver == "lbfgs":
            self._fit_lbfgs(X, g, h)
        elif self.solver == "direct":
            self._fit_solve(X, g, h)
        else:
            raise ValueError(f"Unknown solver {self.solver}")

        return self

    def clear(self) -> None:
        self.betas_.fill(0)

    def update(
        self,
        X: cn.ndarray,
        g: cn.ndarray,
        h: cn.ndarray,
    ) -> "Linear":
        return self.fit(X, g, h)

    def predict(self, X: cn.ndarray) -> cn.ndarray:
        return self.betas_[0] + X.dot(self.betas_[1:].astype(X.dtype))

    @staticmethod
    def batch_predict(models: Sequence[BaseModel], X: cn.ndarray) -> cn.ndarray:
        assert all(isinstance(m, Linear) for m in models)
        models = cast(List[Linear], models)
        # summing together the coeffiecients of each model then predicting
        # saves a lot of work
        betas = cn.sum([model.betas_ for model in models], axis=0)
        return betas[0] + X.dot(betas[1:].astype(X.dtype))

    def __str__(self) -> str:
        return (
            "Bias: "
            + str(self.betas_[1])
            + "\nCoefficients: "
            + str(self.betas_[1:])
            + "\n"
        )

    def __mul__(self, scalar: Any) -> "Linear":
        new = copy.deepcopy(self)
        new.betas_ *= scalar
        return new
