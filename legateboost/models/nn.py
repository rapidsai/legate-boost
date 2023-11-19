# import cunumeric as cn
import numpy as cn

# from ..utils import lbfgs
from .base_model import BaseModel


class NN(BaseModel):
    def __init__(self, max_iter=100, hidden_layer_sizes=(100,), verbose=False):
        self.max_iter = max_iter
        self.hidden_layer_sizes = hidden_layer_sizes
        self.verbose = verbose

    def tanh(self, x):
        return cn.tanh(x)

    def tanh_prime(self, x):
        return 1 - cn.tanh(x) ** 2

    def forward(self, X):
        activations = []
        Z = []
        Z.append(X.dot(self.coefficients_[0]))
        for i in range(1, len(self.hidden_layer_sizes) + 1):
            activations.append(self.tanh(Z[-1]))
            Z.append(activations[-1].dot(self.coefficients_[i]))
        return activations, Z

    def cost(self, pred, y, sample_weight):
        result = (
            0.5
            * cn.square(pred - y.reshape(pred.shape) * sample_weight).sum(axis=0)
            / sample_weight.sum(axis=0)
        )
        return float(result.mean())

    def cost_prime(self, pred, y, sample_weight):
        return (pred - y.reshape(pred.shape)) * sample_weight

    def backward(self, X, y, sample_weight):
        activations, Z = self.forward(X)
        grads = [None] * (len(self.hidden_layer_sizes) + 1)
        E = self.cost_prime(Z[-1], y, sample_weight=sample_weight)
        for i in range(len(self.hidden_layer_sizes), 0, -1):
            grads[i] = activations[i - 1].T.dot(E)
            E = E.dot(self.coefficients_[i].T) * self.tanh_prime(Z[i - 1])
        grads[0] = X.T.dot(E)

        for g, c in zip(grads, self.coefficients_):
            assert g.shape == c.shape, (g.shape, c.shape)
        return self.cost(Z[-1].squeeze(), y, sample_weight), grads

    def _loss_grad_lbfgs(self, packed, X, y, sample_weight=None):
        self._unpack(packed)
        loss, grads = self.backward(X, y, sample_weight)
        packed_grad = self._pack(grads)
        assert packed_grad.shape == packed.shape
        return loss, packed_grad

    def _pack(self, xs):
        return cn.concatenate([x.ravel() for x in xs])

    def _unpack(self, packed_coef):
        offset = 0
        for i in range(len(self.hidden_layer_sizes) + 1):
            self.coefficients_[i] = packed_coef[
                offset : offset + self.coefficients_[i].size
            ].reshape(self.coefficients_[i].shape)
            offset += self.coefficients_[i].size

    def _fitlbfgs(self, X, y, sample_weight=None):
        assert y.ndim == 2
        assert sample_weight.ndim == 2
        packed = self._pack(self.coefficients_)
        # result = lbfgs(packed, self._loss_grad_lbfgs, args=(X, y, sample_weight,),
        #    verbose=self.verbose, max_iter=self.max_iter)
        from scipy.optimize import minimize

        result = minimize(
            self._loss_grad_lbfgs,
            packed,
            args=(
                cn.array(X),
                cn.array(y),
                cn.array(sample_weight),
            ),
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": self.max_iter, "disp": self.verbose},
        )
        self._unpack(result.x)
        if self.verbose:
            print(result)

    def fit(self, X, g, h):
        X = cn.concatenate([cn.ones((X.shape[0], 1)), X], axis=1)
        # init layers with glorot initialization
        self.coefficients_ = []
        for i in range(0, len(self.hidden_layer_sizes) + 1):
            n = self.hidden_layer_sizes[i - 1] if i > 0 else X.shape[1]
            m = self.hidden_layer_sizes[i] if i < len(self.hidden_layer_sizes) else 1
            factor = 2.0
            init_bound = cn.sqrt(factor / (n + m))
            self.coefficients_.append(
                self.random_state.uniform(-init_bound, init_bound, size=(n, m))
            )

        y = -g / h
        self._fitlbfgs(X, y, sample_weight=h)
        return self

    def predict(self, X):
        X = cn.concatenate([cn.ones((X.shape[0], 1)), X], axis=1)
        return self.forward(X)[1][-1]

    def clear(self) -> None:
        for c in self.coefficients_:
            c.fill(0)

    def update(
        self,
        X: cn.ndarray,
        g: cn.ndarray,
        h: cn.ndarray,
    ) -> "NN":
        return self.fit(X, g, h)

    def __str__(self) -> str:
        result = "Coefficients:\n"
        for c in self.coefficients_:
            result += str(c) + "\n"
        return result

    def __eq__(self, other: object) -> bool:
        if len(other.coefficients_) != len(self.coefficients_):
            return False
        return all([x == y for x, y in zip(self.coefficients_, other.coefficients_)])
