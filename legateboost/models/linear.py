import copy
import warnings
from typing import Any, List, Sequence, Tuple, cast

import cupynumeric as cn

from ..utils import lbfgs, solve_singular
from .base_model import BaseModel


class Linear(BaseModel):
    """Generalised linear model. Boosting linear models is equivalent to fitting a
    single linear model where each boosting iteration is a newton step. Note that
    the l2 penalty is applied to the weights of each model, as opposed to the sum
    of all models. This can lead to different results when compared to fitting a
    linear model with sklearn.

    It is recommended to normalize the data before fitting. This ensures
    regularisation is evenly applied to all features and prevents numerical issues.

    Two solvers are available. A direct numerical solver that can be faster, but
    uses more memory, and an iterative L-BFGS solver that uses less memory
    but can be slower.

    Parameters
    ----------
    l2_regularization :
        An L2 penalty applied to the coefficients.
    alpha : deprecated
        Deprecated, use `l2_regularization` instead.
    solver : "direct" or "lbfgs"
        If "direct", use a direct solver. If "lbfgs", use the lbfgs solver.

    Attributes
    ----------
    bias_ : ndarray of shape (n_outputs,)
        Intercept term.
    betas_ : ndarray of shape (n_features, n_outputs)
        Coefficients of the linear model.
    """

    def __init__(
        self,
        *,
        l2_regularization: float = 1e-5,
        alpha: Any = "deprecated",
        solver: str = "direct",
    ) -> None:
        self.alpha = alpha
        self.l2_regularization = l2_regularization
        self.solver = solver
        if alpha != "deprecated":
            warnings.warn(
                "`alpha` was renamed to `l2_regularization` in 23.03"
                " and will be removed in 23.05",
                FutureWarning,
            )
            self.l2_regularization = alpha

    def _fit_solve(self, X: cn.ndarray, g: cn.ndarray, h: cn.ndarray) -> None:
        self.betas_ = cn.zeros((X.shape[1] + 1, g.shape[1]), dtype=X.dtype)
        num_outputs = g.shape[1]
        for k in range(num_outputs):
            W = cn.sqrt(h[:, k])
            Xw = cn.ones((X.shape[0], X.shape[1] + 1))
            Xw[:, 1:] = X
            Xw = Xw * W[:, cn.newaxis]
            diag = cn.eye(Xw.shape[1]) * self.l2_regularization
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
        grads[1:] = cn.dot(X.T, delta) + self.l2_regularization * self.betas_[1:]
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
        betas = betas.astype(X.dtype)
        return betas[0] + X.dot(betas[1:])

    def __str__(self) -> str:
        return (
            "Bias: "
            + str(self.betas_[0])
            + "\nCoefficients: "
            + str(self.betas_[1:])
            + "\n"
        )

    def __mul__(self, scalar: Any) -> "Linear":
        new = copy.deepcopy(self)
        new.betas_ *= scalar
        return new

    def to_onnx(self, X: cn.array) -> Any:
        import onnx

        X_type_text = "double" if X.dtype == cn.float64 else "float"
        onnx_text = f"""
        LinearModel ({X_type_text}[N, M] X_in, double[N, K] predictions_in) => ({X_type_text}[N, M] X_out, double[N, K] predictions_out)
        {{
            X_out = Identity(X_in)
            mult = MatMul(X_in, betas)
            result = Add(mult, intercept)
            result_double = Cast<to=11>(result)
            predictions_out = Add(result_double, predictions_in)
        }}
        """  # noqa: E501
        graph = onnx.parser.parse_graph(onnx_text)
        graph.initializer.extend(
            [
                onnx.numpy_helper.from_array(self.betas_[1:].__array__(), name="betas"),
                onnx.numpy_helper.from_array(
                    self.betas_[0].__array__(), name="intercept"
                ),
            ]
        )
        return graph
