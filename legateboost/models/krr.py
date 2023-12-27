from __future__ import annotations

from scipy.special import lambertw

import cunumeric as cn

from ..utils import lbfgs
from .base_model import BaseModel


def l2(X: cn.ndarray, Y: cn.ndarray) -> cn.ndarray:
    XX = cn.einsum("ij,ij->i", X, X)[:, cn.newaxis]
    YY = cn.einsum("ij,ij->i", Y, Y)
    XY = 2 * cn.dot(X, Y.T)
    return cn.maximum(XX + YY - XY, 0.0)


class KRR(BaseModel):
    """Kernel Ridge Regression model using the Nyström approximation. The
    accuracy of the approximation is governed by the parameter `n_components`
    <= `n`. Effectively, `n_components` rows will be randomly sampled (without
    replacement) from X in each boosting iteration.

    The kernel is fixed to be the RBF kernel:

    :math:`k(x_i, x_j) = \\exp(-\\frac{||x_i - x_j||^2}{2\\sigma^2})`

    Standardising data is recommended.

    The sigma parameter, if not given, is estimated using the method described in:
    Allerbo, Oskar, and Rebecka Jörnsten. "Bandwidth Selection for Gaussian Kernel
    Ridge Regression via Jacobian Control." arXiv preprint arXiv:2205.11956 (2022).


    See the following reference for more details on gradient boosting with
    kernel ridge regression:
    Sigrist, Fabio. "KTBoost: Combined kernel and tree boosting."
    Neural Processing Letters 53.2 (2021): 1147-1160.


    Parameters
    ----------
    n_components :
        Number of components to use in the model.
    alpha :
        Regularization parameter.
    sigma :
        Kernel bandwidth parameter. If None, use the mean squared distance.
    solver :
        Solver to use for solving the linear system.
        Options are , 'lbfgs', and 'direct'.

    Attributes
    ----------
    betas_ : ndarray of shape (n_train_samples, n_outputs)
        Coefficients of the regression model.
    X_train : ndarray of shape (n_components, n_features)
        Training data used to fit the model.
    indices : ndarray of shape (n_components,)
        Indices of the training data used to fit the model.
    """

    def __init__(
        self,
        n_components: int = 100,
        alpha: float = 1e-5,
        sigma: float | None = None,
        solver: str = "direct",
    ):
        self.num_components = n_components
        self.alpha = alpha
        self.sigma = sigma
        self.solver = solver
        self.num_components = n_components
        self.alpha = alpha
        self.sigma = sigma

    def _apply_kernel(self, X: cn.ndarray) -> cn.ndarray:
        return self.rbf_kernel(X, self.X_train)

    def _direct_solve(self, X: cn.ndarray, g: cn.ndarray, h: cn.ndarray) -> "KRR":
        # fit with fixed set of components
        K_nm = self._apply_kernel(X)
        K_mm = self._apply_kernel(self.X_train)
        num_outputs = g.shape[1]
        self.betas_ = cn.zeros((self.X_train.shape[0], num_outputs))

        for k in range(num_outputs):
            W = cn.sqrt(h[:, k])
            # Make sure we are working in 64 bit for numerical stability
            Kw = K_nm.astype(cn.float64) * W[:, cn.newaxis]
            yw = W * (-g[:, k] / h[:, k])
            self.betas_[:, k] = cn.linalg.lstsq(
                Kw.T.dot(Kw) + self.alpha * K_mm, cn.dot(Kw.T, yw), rcond=None
            )[0]
        return self

    def _loss_grad(self, betas, K_nm, K_mm, g, h):
        self.betas_ = betas.reshape(self.betas_.shape)
        pred = K_nm.dot(self.betas_.astype(K_nm.dtype))
        loss = (pred * (g + 0.5 * h * pred)).sum(axis=0).mean()
        delta = g + h * pred
        grads = cn.dot(K_nm.T, delta) + self.alpha * K_mm.dot(self.betas_)
        grads /= K_nm.shape[0]
        assert grads.shape == self.betas_.shape
        return loss, grads.ravel()

    def _lbfgs_solve(self, X, g, h) -> "KRR":
        self.betas_ = cn.zeros((self.X_train.shape[0], g.shape[1]))
        K_nm = self._apply_kernel(X)
        K_mm = self._apply_kernel(self.X_train)
        result = lbfgs(
            self.betas_.ravel(),
            self._loss_grad,
            args=(K_nm, K_mm, g, h),
            verbose=0,
        )
        self.betas_ = result.x.reshape(self.betas_.shape)
        return self

    def opt_sigma(self, D_2: cn.ndarray) -> cn.ndarray:
        n = D_2.shape[1]
        assert self.X_train.shape[0] > 1, "Need at least 2 components to estimate sigma"
        mins = self.X_train.min(axis=0)
        maxs = self.X_train.max(axis=0)
        lmax = cn.mean(maxs - mins)
        p = self.X_train.shape[1]
        d = 2 * lmax / (((n - 1) ** (1 / p) - 1) * cn.pi)

        w_arg = -self.alpha * cn.exp(0.5) / (2 * n)
        if w_arg < -cn.exp(-1):
            return d * cn.sqrt(3 / 2)
        w_0 = cn.real(lambertw(w_arg, k=0))
        sigma = d / cn.sqrt(2) * cn.sqrt(1 - 2 * w_0)
        return sigma

    def rbf_kernel(self, X: cn.ndarray, Y: cn.ndarray) -> cn.ndarray:
        D_2 = l2(X, Y)

        if self.sigma is None:
            self.sigma = self.opt_sigma(D_2)
        return cn.exp(-D_2 / (2 * self.sigma * self.sigma))

    def _fit_components(self, X, g, h) -> "KRR":
        if self.solver == "direct":
            return self._direct_solve(X, g, h)
        elif self.solver == "lbfgs":
            return self._lbfgs_solve(X, g, h)
        else:
            raise ValueError(f"Unknown solver {self.solver}")

    def fit(
        self,
        X: cn.ndarray,
        g: cn.ndarray,
        h: cn.ndarray,
    ) -> "KRR":
        usable_num_components = min(X.shape[0], self.num_components)
        self.indices = self.random_state.permutation(X.shape[0])[:usable_num_components]
        self.X_train = X[self.indices]
        return self._fit_components(X, g, h)

    def predict(self, X: cn.ndarray) -> cn.ndarray:
        K = self._apply_kernel(X)
        return K.dot(self.betas_.astype(K.dtype))

    def clear(self) -> None:
        self.betas_.fill(0)

    def update(
        self,
        X: cn.ndarray,
        g: cn.ndarray,
        h: cn.ndarray,
    ) -> "KRR":
        return self._fit_components(X, g, h)

    def __str__(self) -> str:
        return (
            "Sigma:"
            + str(self.sigma)
            + "\n"
            + "Components: "
            + str(self.X_train)
            + "\nCoefficients: "
            + str(self.betas_)
            + "\n"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, KRR):
            raise NotImplementedError()
        return (other.betas_ == self.betas_).all() and (
            other.X_train == self.X_train
        ).all()
