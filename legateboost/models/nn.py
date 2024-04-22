import cunumeric as cn
from legate.core import TaskTarget, get_legate_runtime, types

from ..library import user_context, user_lib
from ..utils import get_store
from .base_model import BaseModel


class NN(BaseModel):
    def __init__(
        self,
        max_iter=100,
        hidden_layer_sizes=(100,),
        alpha=0.0001,
        verbose=False,
        m=10,
        gtol=1e-5,
    ):
        self.max_iter = max_iter
        self.hidden_layer_sizes = hidden_layer_sizes
        self.verbose = verbose
        self.m = m
        self.alpha = alpha
        self.gtol = gtol

    def tanh(self, x):
        return cn.tanh(x, out=x)

    def tanh_prime(self, H, delta):
        delta *= 1 - H**2

    def forward(self, X, activations):
        for i in range(len(self.hidden_layer_sizes) + 1):
            activations[i + 1] = activations[i].dot(self.coefficients_[i])
            activations[i + 1] += self.biases_[i][0]
            if i + 1 < len(self.hidden_layer_sizes) + 1:
                activations[i + 1] = self.tanh(activations[i + 1])
        return activations

    def _fit_lbfgs(self, X, g, h):
        task = get_legate_runtime().create_auto_task(
            user_context, user_lib.cffi.BUILD_NN
        )

        X_ = get_store(X).promote(2, g.shape[1])

        g_ = get_store(g.astype(cn.float64)).promote(1, X.shape[1])
        h_ = get_store(h.astype(cn.float64)).promote(1, X.shape[1])

        task.add_scalar_arg(X.shape[0], types.int64)
        task.add_scalar_arg(self.gtol, types.float64)
        task.add_scalar_arg(self.verbose, types.int32)
        task.add_scalar_arg(self.m, types.int32)
        task.add_scalar_arg(self.max_iter, types.int32)
        task.add_scalar_arg(self.alpha, types.float64)
        task.add_input(X_)
        task.add_input(g_)
        task.add_input(h_)
        task.add_alignment(g_, h_)
        task.add_alignment(g_, X_)
        for c, b in zip(self.coefficients_, self.biases_):
            b_ = get_store(b).project(0, 0)
            task.add_input(get_store(c))
            task.add_input(b_)
            task.add_output(get_store(c))
            task.add_output(b_)
            task.add_broadcast(get_store(c))
            task.add_broadcast(b_)
        if get_legate_runtime().machine.count(TaskTarget.GPU) > 1:
            task.add_nccl_communicator()
        if get_legate_runtime().machine.count() > 1:
            task.add_cpu_communicator()
        task.execute()

    def fit(self, X, g, h):
        # init layers with glorot initialization
        self.coefficients_ = []
        self.biases_ = []
        for i in range(0, len(self.hidden_layer_sizes) + 1):
            n = self.hidden_layer_sizes[i - 1] if i > 0 else X.shape[1]
            m = (
                self.hidden_layer_sizes[i]
                if i < len(self.hidden_layer_sizes)
                else g.shape[1]
            )
            factor = 6.0
            init_bound = cn.sqrt(factor / (n + m))
            self.coefficients_.append(
                cn.array(
                    self.random_state.uniform(-init_bound, init_bound, size=(n, m)),
                    dtype=X.dtype,
                )
            )
            # https://github.com/nv-legate/legate.core.internal/issues/584
            # bias cannot be size 1 - give it an extra useless dimension
            self.biases_.append(
                cn.array(
                    self.random_state.uniform(
                        -init_bound,
                        init_bound,
                        size=(
                            2,
                            m,
                        ),
                    ),
                    dtype=X.dtype,
                )
            )

        self._fit_lbfgs(X, g, h)
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
        return self._fitlbfgs(X, g, h)

    def __str__(self) -> str:
        result = "Coefficients:\n"
        for c in self.coefficients_:
            result += str(c) + "\n"
        return result

    def __eq__(self, other: object) -> bool:
        if len(other.coefficients_) != len(self.coefficients_):
            return False
        return all([x == y for x, y in zip(self.coefficients_, other.coefficients_)])
