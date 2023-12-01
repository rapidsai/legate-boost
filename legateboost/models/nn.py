import cunumeric as cn

from ..utils import lbfgs
from .base_model import BaseModel


class NN(BaseModel):
    def __init__(self, max_iter=100, hidden_layer_sizes=(100,), verbose=False, m=10):
        self.max_iter = max_iter
        self.hidden_layer_sizes = hidden_layer_sizes
        self.verbose = verbose
        self.m = m

    def tanh(self, x):
        return cn.tanh(x, out=x)

    def tanh_prime(self, H, delta):
        delta *= 1 - H**2

    def forward(self, X, activations):
        for i in range(len(self.hidden_layer_sizes) + 1):
            activations[i + 1] = activations[i].dot(self.coefficients_[i])
            activations[i + 1] += self.biases_[i]
            if i + 1 < len(self.hidden_layer_sizes) + 1:
                activations[i + 1] = self.tanh(activations[i + 1])
        return activations

    def cost(self, pred, y, sample_weight):
        result = (
            0.5
            * cn.square((pred - y.reshape(pred.shape)) * sample_weight).sum(axis=0)
            / sample_weight.sum(axis=0)
        )
        if pred.shape[1] > 1:
            return result.mean()
        return result

    def cost_prime(self, pred, y, sample_weight):
        return (pred - y.reshape(pred.shape)) * sample_weight

    def backward(
        self, X, y, sample_weight, coeff_grads, bias_grads, deltas, activations
    ):
        activations = self.forward(X, activations)
        cost = self.cost(activations[-1], y, sample_weight)
        deltas[-1] = self.cost_prime(activations[-1], y, sample_weight)
        # todo: scale by weight?
        coeff_grads[-1] = activations[-2].T.dot(deltas[-1]) / X.shape[0]
        bias_grads[-1] = deltas[-1].mean(axis=0)
        for i in range(len(self.hidden_layer_sizes), 0, -1):
            deltas[i - 1] = deltas[i].dot(self.coefficients_[i].T)
            self.tanh_prime(activations[i], deltas[i - 1])
            coeff_grads[i - 1] = activations[i - 1].T.dot(deltas[i - 1]) / X.shape[0]
            bias_grads[i - 1] = deltas[i - 1].mean(axis=0)
        for g, c in zip(coeff_grads, self.coefficients_):
            assert g.shape == c.shape, (g.shape, c.shape)
        return cost, bias_grads, coeff_grads

    def _loss_grad_lbfgs(
        self, packed, X, y, sample_weight, coeff_grads, bias_grads, deltas, activations
    ):
        self._unpack(packed)
        loss, bias_grads, coeff_grads = self.backward(
            X, y, sample_weight, coeff_grads, bias_grads, deltas, activations
        )
        packed_grad = self._pack(bias_grads + coeff_grads)
        assert packed_grad.shape == packed.shape
        return loss, packed_grad

    def _pack(self, xs):
        return cn.concatenate([x.ravel() for x in xs])

    def _unpack(self, packed_coef):
        offset = 0
        for i in range(len(self.hidden_layer_sizes) + 1):
            self.biases_[i] = packed_coef[
                offset : offset + self.biases_[i].size
            ].reshape(self.biases_[i].shape)
            offset += self.biases_[i].size
        for i in range(len(self.hidden_layer_sizes) + 1):
            self.coefficients_[i] = packed_coef[
                offset : offset + self.coefficients_[i].size
            ].reshape(self.coefficients_[i].shape)
            offset += self.coefficients_[i].size

    def _fitlbfgs(self, X, y, sample_weight=None):
        assert y.ndim == 2
        assert sample_weight.ndim == 2
        packed = self._pack(self.biases_ + self.coefficients_)
        coeff_grads = [None] * (len(self.hidden_layer_sizes) + 1)
        bias_grads = [None] * (len(self.hidden_layer_sizes) + 1)
        deltas = [None] * (len(self.hidden_layer_sizes) + 1)
        activations = [X] + [None] * len(self.hidden_layer_sizes) + [None]
        result = lbfgs(
            packed,
            self._loss_grad_lbfgs,
            args=(
                X,
                y,
                sample_weight,
                coeff_grads,
                bias_grads,
                deltas,
                activations,
            ),
            verbose=self.verbose,
            max_iter=self.max_iter,
            m=self.m,
        )
        """From scipy.optimize import minimize.

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
        """
        self._unpack(result.x)
        if self.verbose:
            print(result)

    def fit(self, X, g, h):
        # init layers with glorot initialization
        self.coefficients_ = []
        self.biases_ = []
        for i in range(0, len(self.hidden_layer_sizes) + 1):
            n = self.hidden_layer_sizes[i - 1] if i > 0 else X.shape[1]
            m = self.hidden_layer_sizes[i] if i < len(self.hidden_layer_sizes) else 1
            factor = 6.0
            init_bound = cn.sqrt(factor / (n + m))
            self.coefficients_.append(
                self.random_state.uniform(-init_bound, init_bound, size=(n, m))
            )
            self.biases_.append(
                self.random_state.uniform(-init_bound, init_bound, size=(m,))
            )

        y = -g / h
        self._fitlbfgs(X, y, sample_weight=h)
        return self

    def predict(self, X):
        activations = [X] + [None] * len(self.hidden_layer_sizes) + [None]
        return self.forward(X, activations)[-1]

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
