from __future__ import annotations

import copy
import warnings
from typing import Any, List, Sequence, Set, Tuple, cast

import numpy as np
from scipy.special import lambertw

import cupynumeric as cn
from legate.core import get_legate_runtime, types

from ..library import user_context, user_lib
from ..utils import gather, get_store, lbfgs
from .base_model import BaseModel


def l2(X: cn.ndarray, Y: cn.ndarray) -> cn.ndarray:
    XX = cn.einsum("ij,ij->i", X, X)[:, cn.newaxis]
    YY = cn.einsum("ij,ij->i", Y, Y)
    XY = cn.dot(X, Y.T)
    XY *= -2.0
    XY += XX
    XY += YY
    return cn.maximum(XY, 0.0, out=XY)


def rbf(x: cn.ndarray, sigma: float) -> cn.ndarray:
    task = get_legate_runtime().create_auto_task(user_context, user_lib.cffi.RBF)
    task.add_input(get_store(x))
    task.add_scalar_arg(sigma, types.float64)
    task.add_output(get_store(x))
    task.execute()
    return x


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
    l2_regularization :
        l2 regularization parameter on the weights.
    alpha :
        Deprecated. Use `l2_regularization` instead.
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
        *,
        n_components: int = 100,
        alpha: Any = "deprecated",
        l2_regularization: float = 1e-5,
        sigma: float | None = None,
        solver: str = "direct",
    ):
        self.num_components = n_components
        self.alpha = alpha
        self.l2_regularization = l2_regularization
        self.sigma = sigma
        self.solver = solver
        if alpha != "deprecated":
            warnings.warn(
                "`alpha` was renamed to `l2_regularization` in 23.03"
                " and will be removed in 23.05",
                FutureWarning,
            )
            self.l2_regularization = alpha

    def _apply_kernel(self, X: cn.ndarray) -> cn.ndarray:
        return self.rbf_kernel(X, self.X_train)

    def _direct_solve(self, X: cn.ndarray, g: cn.ndarray, h: cn.ndarray) -> "KRR":
        # fit with fixed set of components
        K_nm = self._apply_kernel(X)
        K_mm = self._apply_kernel(self.X_train)
        num_outputs = g.shape[1]
        self.betas_ = cn.zeros((self.X_train.shape[0], num_outputs), dtype=X.dtype)

        for k in range(num_outputs):
            W = cn.sqrt(h[:, k]).astype(X.dtype)
            Kw = K_nm * W[:, cn.newaxis]
            yw = W * (-g[:, k] / h[:, k]).astype(X.dtype)
            self.betas_[:, k] = cn.linalg.lstsq(
                Kw.T.dot(Kw) + self.l2_regularization * K_mm,
                cn.dot(Kw.T, yw),
                rcond=None,
            )[0]
        return self

    def _loss_grad(
        self,
        betas: cn.ndarray,
        K_nm: cn.ndarray,
        K_mm: cn.ndarray,
        g: cn.ndarray,
        h: cn.ndarray,
    ) -> Tuple[float, cn.ndarray]:
        self.betas_ = betas.reshape(self.betas_.shape)
        pred = K_nm.dot(self.betas_.astype(K_nm.dtype))
        loss = (pred * (g + 0.5 * h * pred)).sum(axis=0).mean()
        delta = g + h * pred
        grads = cn.dot(K_nm.T, delta) + self.l2_regularization * K_mm.dot(self.betas_)
        grads /= K_nm.shape[0]
        assert grads.shape == self.betas_.shape
        return loss, grads.ravel()

    def _lbfgs_solve(self, X: cn.ndarray, g: cn.ndarray, h: cn.ndarray) -> "KRR":
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

        w_arg = -self.l2_regularization * cn.exp(0.5) / (2 * n)
        if w_arg < -cn.exp(-1):
            return d * cn.sqrt(3 / 2)
        w_0 = cn.real(lambertw(w_arg, k=0))
        sigma = d / cn.sqrt(2) * cn.sqrt(1 - 2 * w_0)
        return sigma

    def rbf_kernel(self, X: cn.ndarray, Y: cn.ndarray) -> cn.ndarray:
        D_2 = l2(X, Y)

        if self.sigma is None:
            self.sigma = self.opt_sigma(D_2)
        return rbf(D_2, self.sigma)

    def _sample_components(self, X: cn.ndarray) -> cn.ndarray:
        usable_num_components = min(X.shape[0], self.num_components)
        if usable_num_components == X.shape[0]:
            return X
        selected: Set[int] = set()
        # numpy.choice is not efficient for small number of
        # samples from large population
        while len(selected) < usable_num_components:
            selected.add(self.random_state.randint(0, X.shape[0]))
        return gather(X, tuple(np.fromiter(selected, int, len(selected))))

    def _fit_components(self, X: cn.ndarray, g: cn.ndarray, h: cn.ndarray) -> "KRR":
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
        self.X_train = self._sample_components(X)
        return self._fit_components(X, g, h)

    def predict(self, X: cn.ndarray) -> cn.ndarray:
        K = self._apply_kernel(X)
        return K.dot(self.betas_.astype(K.dtype))

    @staticmethod
    def batch_predict(models: Sequence[BaseModel], X: cn.ndarray) -> cn.ndarray:
        assert all(isinstance(m, KRR) for m in models)
        models = cast(List[KRR], models)
        return sum(model.predict(X) for model in models)

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

    def __mul__(self, scalar: Any) -> "KRR":
        new = copy.deepcopy(self)
        self.betas_ *= scalar
        return new

    def to_onnx(self) -> Any:
        from onnx import numpy_helper
        from onnx.checker import check_model
        from onnx.helper import (
            make_graph,
            make_model,
            make_node,
            make_tensor_value_info,
            np_dtype_to_tensor_dtype,
        )

        assert self.X_train.dtype == self.betas_.dtype

        def make_constant_node(value: cn.array, name: str) -> Any:
            return make_node(
                "Constant",
                inputs=[],
                value=numpy_helper.from_array(value, name=name),
                outputs=[name],
            )

        nodes = []

        # model constants
        betas = numpy_helper.from_array(self.betas_.__array__(), name="betas")
        X_train = numpy_helper.from_array(self.X_train.__array__(), name="X_train")

        # pred inputs
        X = make_tensor_value_info(
            "X",
            np_dtype_to_tensor_dtype(self.betas_.dtype),
            [None, self.X_train.shape[1]],
        )
        pred = make_tensor_value_info(
            "pred",
            np_dtype_to_tensor_dtype(self.betas_.dtype),
            [None, self.betas_.shape[1]],
        )

        # exanded l2 distance
        # distance = np.sum(X**2, axis=1)[:, np.newaxis] - 2 * np.dot(X, self.X_train.T)
        # + np.sum(self.X_train**2, axis=1)
        make_tensor_value_info(
            "XX", np_dtype_to_tensor_dtype(self.betas_.dtype), [None]
        )
        make_tensor_value_info(
            "YY",
            np_dtype_to_tensor_dtype(self.betas_.dtype),
            [self.X_train.shape[0], 1],
        )
        make_tensor_value_info(
            "XY_reshaped",
            np_dtype_to_tensor_dtype(self.betas_.dtype),
            [1, self.X_train.shape[0]],
        )
        make_tensor_value_info(
            "XY",
            np_dtype_to_tensor_dtype(self.betas_.dtype),
            [None, self.X_train.shape[0]],
        )
        nodes.append(make_constant_node(np.array([1]), "axis1"))
        nodes.append(make_node("ReduceSumSquare", ["X", "axis1"], ["XX"]))
        nodes.append(make_node("Gemm", ["X", "X_train"], ["XY"], alpha=-2.0, transB=1))
        nodes.append(make_node("ReduceSumSquare", ["X_train", "axis1"], ["YY"]))
        nodes.append(make_constant_node(np.array([1, -1]), "reshape"))
        nodes.append(make_node("Reshape", ["YY", "reshape"], ["YY_reshaped"]))
        nodes.append(make_node("Add", ["XX", "XY"], ["add0"]))
        make_tensor_value_info(
            "l2",
            np_dtype_to_tensor_dtype(self.betas_.dtype),
            [None, self.X_train.shape[0]],
        )
        nodes.append(make_node("Add", ["YY_reshaped", "add0"], ["l2"]))
        nodes.append(make_constant_node(np.array([0.0], self.betas_.dtype), "zero"))
        make_tensor_value_info(
            "l2_clipped",
            np_dtype_to_tensor_dtype(self.betas_.dtype),
            [None, self.X_train.shape[0]],
        )
        nodes.append(make_node("Max", ["l2", "zero"], ["l2_clipped"]))

        # RBF kernel
        # K = np.exp(-distance / (2 * self.sigma**2))
        make_tensor_value_info(
            "rbf0",
            np_dtype_to_tensor_dtype(self.betas_.dtype),
            [None, self.X_train.shape[0]],
        )
        if self.sigma is None:
            raise ValueError("sigma is None. Has fit been called?")
        nodes.append(
            make_constant_node(
                np.array([-2.0 * self.sigma**2], self.betas_.dtype), "denominator"
            )
        )
        nodes.append(make_node("Div", ["l2_clipped", "denominator"], ["rbf0"]))
        make_tensor_value_info(
            "K",
            np_dtype_to_tensor_dtype(self.betas_.dtype),
            [None, self.X_train.shape[0]],
        )
        nodes.append(make_node("Exp", ["rbf0"], ["K"]))

        # prediction
        # pred = np.dot(K, self.betas_)
        nodes.append(make_node("MatMul", ["K", "betas"], ["pred"]))
        graph = make_graph(
            nodes, "legateboost.model.KRR", [X], [pred], [betas, X_train]
        )
        onnx_model = make_model(graph)
        check_model(onnx_model)
        return onnx_model
