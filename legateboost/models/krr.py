import cunumeric as cn

from ..utils import solve_singular
from .base_model import BaseModel


def l2(X, Y):
    XX = cn.einsum("ij,ij->i", X, X)[:, cn.newaxis]
    YY = cn.einsum("ij,ij->i", Y, Y)
    XY = 2 * cn.dot(X, Y.T)
    return XX + YY - XY


def rbf_kernel(X, Y, sigma=None):
    K = l2(X, Y)
    if sigma is None:
        sigma_sq = K.mean()
    else:
        sigma_sq = sigma**2
    return cn.exp(-K / (2 * sigma_sq))


class KRR(BaseModel):
    def __init__(self, n_components=10, alpha=1e-5, sigma=None):
        self.num_components = n_components
        self.alpha = alpha
        self.sigma = sigma

    def _apply_kernel(self, X):
        return rbf_kernel(X, self.X_train, sigma=self.sigma)

    def _fit_components(self, X, g, h) -> "KRR":
        # fit with fixed set of components
        K_mm = self._apply_kernel(self.X_train)
        K_nm = self._apply_kernel(X)
        num_outputs = g.shape[1]
        self.betas_ = cn.zeros((self.X_train.shape[0], num_outputs))

        for k in range(num_outputs):
            W = cn.sqrt(h[:, k])
            Kw = K_nm * W[:, cn.newaxis]
            yw = W * (-g[:, k] / h[:, k])
            self.betas_[:, k] = solve_singular(
                Kw.T.dot(Kw) + self.alpha * K_mm, cn.dot(Kw.T, yw)
            )
        return self

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

    def predict(self, X):
        K = self._apply_kernel(X)
        return K.dot(self.betas_)

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
            "Components: "
            + str(self.X_train)
            + "\nCoefficients: "
            + str(self.betas_)
            + "\n"
        )

    def __eq__(self, other: object) -> bool:
        return (other.betas_ == self.betas_).all() and (
            other.X_train == self.X_train
        ).all()
