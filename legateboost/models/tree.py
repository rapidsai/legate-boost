import math
from enum import IntEnum
from typing import Any

import cunumeric as cn
from legate.core import TaskTarget, get_legate_runtime, types

from ..library import user_context, user_lib
from ..utils import gather, get_store
from .base_model import BaseModel


class LegateBoostOpCode(IntEnum):
    BUILD_TREE = user_lib.cffi.BUILD_TREE
    PREDICT = user_lib.cffi.PREDICT
    UPDATE_TREE = user_lib.cffi.UPDATE_TREE


class Tree(BaseModel):
    """A structure of arrays representing a decision tree.

    A leaf node has value -1 at feature[node_idx]
    """

    leaf_value: cn.ndarray
    feature: cn.ndarray
    split_value: cn.ndarray
    gain: cn.ndarray
    hessian: cn.ndarray

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Tree):
            return NotImplemented
        eq = [cn.all(self.leaf_value == other.leaf_value)]
        eq.append(cn.all(self.feature == other.feature))
        eq.append(cn.all(self.split_value == other.split_value))
        eq.append(cn.all(self.gain == other.gain))
        eq.append(cn.all(self.hessian == other.hessian))
        return all(eq)

    def num_procs_to_use(self, num_rows: int) -> int:
        min_rows_per_worker = 10
        available_procs = len(get_legate_runtime().machine)
        return min(available_procs, int(math.ceil(num_rows / min_rows_per_worker)))

    def __init__(
        self,
        max_depth: int,
    ) -> None:
        self.max_depth = max_depth

    def fit(
        self,
        X: cn.ndarray,
        g: cn.ndarray,
        h: cn.ndarray,
    ) -> "Tree":
        # dont let legate create a future - make sure at least 2 sample rows
        sample_rows = cn.array(
            self.random_state.randint(0, X.shape[0], max(2, self.max_depth))
        )
        split_proposals = gather(X, sample_rows)

        num_outputs = g.shape[1]

        task = get_legate_runtime().create_auto_task(
            user_context, LegateBoostOpCode.BUILD_TREE
        )

        # inputs
        X_ = get_store(X).promote(2, g.shape[1])
        g_ = get_store(g).promote(1, X.shape[1])
        h_ = get_store(h).promote(1, X.shape[1])

        task.add_scalar_arg(self.max_depth, types.int32)
        task.add_input(X_)
        task.add_broadcast(X_, 1)
        task.add_input(g_)
        task.add_input(h_)
        task.add_alignment(g_, h_)
        task.add_alignment(g_, X_)
        task.add_input(get_store(split_proposals))
        task.add_broadcast(get_store(split_proposals))

        # outputs
        max_nodes = 2 ** (self.max_depth + 1)
        task.add_scalar_arg(max_nodes, types.int32)
        leaf_value = get_legate_runtime().create_store(
            types.float64, (max_nodes, num_outputs)
        )
        feature = get_legate_runtime().create_store(types.int32, (max_nodes,))
        split_value = get_legate_runtime().create_store(types.float64, (max_nodes,))
        gain = get_legate_runtime().create_store(types.float64, (max_nodes,))
        hessian = get_legate_runtime().create_store(
            types.float64, (max_nodes, num_outputs)
        )
        task.add_output(leaf_value)
        task.add_output(feature)
        task.add_output(split_value)
        task.add_output(gain)
        task.add_output(hessian)
        if get_legate_runtime().machine.count(TaskTarget.GPU) > 1:
            task.add_nccl_communicator()
        elif get_legate_runtime().machine.count() > 1:
            task.add_cpu_communicator()
        task.execute()

        self.leaf_value = cn.array(leaf_value, copy=False)
        self.feature = cn.array(feature, copy=False)
        self.split_value = cn.array(split_value, copy=False)
        self.gain = cn.array(gain, copy=False)
        self.hessian = cn.array(hessian, copy=False)
        return self

    def clear(self) -> None:
        self.leaf_value.fill(0)
        self.hessian.fill(0)

    def update(
        self,
        X: cn.ndarray,
        g: cn.ndarray,
        h: cn.ndarray,
    ) -> "Tree":
        task = get_legate_runtime().create_auto_task(
            user_context, LegateBoostOpCode.UPDATE_TREE
        )

        # inputs
        X_ = get_store(X).promote(2, g.shape[1])
        g_ = get_store(g).promote(1, X.shape[1])
        h_ = get_store(h).promote(1, X.shape[1])
        task.add_scalar_arg(self.max_depth, types.int32)
        task.add_input(X_)
        task.add_broadcast(X_, 1)
        task.add_input(g_)
        task.add_input(h_)
        task.add_alignment(g_, h_)
        task.add_alignment(g_, X_)

        # broadcast the tree structure
        task.add_input(get_store(self.feature))
        task.add_broadcast(get_store(self.feature))
        task.add_input(get_store(self.split_value))
        task.add_broadcast(get_store(self.split_value))

        leaf_value = get_legate_runtime().create_store(
            types.float64, self.leaf_value.shape
        )
        hessian = get_legate_runtime().create_store(types.float64, self.hessian.shape)

        task.add_output(leaf_value)
        task.add_output(hessian)

        # Update task has only a CPU implementation
        task.add_cpu_communicator()

        task.execute()
        self.leaf_value = cn.array(leaf_value, copy=False)
        self.hessian = cn.array(hessian, copy=False)
        return self

    def predict(self, X: cn.ndarray) -> cn.ndarray:
        n_rows = X.shape[0]
        n_features = X.shape[1]
        n_outputs = self.leaf_value.shape[1]
        task = get_legate_runtime().create_auto_task(
            user_context, LegateBoostOpCode.PREDICT
        )

        pred = get_legate_runtime().create_store(types.float64, (n_rows, n_outputs))
        X_ = get_store(X).promote(2, n_outputs)
        pred_ = get_store(pred).promote(1, n_features)
        task.add_input(X_)
        task.add_broadcast(X_, 1)

        # broadcast the tree structure
        leaf_value_ = get_store(self.leaf_value)
        feature_ = get_store(self.feature)
        split_value_ = get_store(self.split_value)
        task.add_input(leaf_value_)
        task.add_input(feature_)
        task.add_input(split_value_)
        task.add_broadcast(leaf_value_)
        task.add_broadcast(feature_)
        task.add_broadcast(split_value_)

        task.add_output(pred_)

        task.add_alignment(X_, pred_)
        task.execute()

        return cn.array(pred, copy=False)

    def is_leaf(self, id: int) -> Any:
        return self.feature[id] == -1

    def left_child(self, id: int) -> int:
        return id * 2 + 1

    def right_child(self, id: int) -> int:
        return id * 2 + 2

    def __str__(self) -> str:
        def format_vector(v: cn.ndarray) -> str:
            if cn.isscalar(v):
                return "{:0.4f}".format(v)
            return "[" + ",".join(["{:0.4f}".format(x) for x in v]) + "]"

        def recurse_print(id: int, depth: int) -> str:
            if self.is_leaf(id):
                text = "\t" * depth + "{}:leaf={},hess={}\n".format(
                    id,
                    format_vector(self.leaf_value[id]),
                    format_vector(self.hessian[id]),
                )
            else:
                text = (
                    "\t" * depth
                    + "{}:[f{}<={:0.4f}] yes={},no={},gain={:0.4f},hess={}\n".format(
                        id,
                        self.feature[id],
                        self.split_value[id],
                        self.left_child(id),
                        self.right_child(id),
                        self.gain[id],
                        self.hessian[id],
                    )
                )
                text += recurse_print(self.left_child(id), depth + 1)
                text += recurse_print(self.right_child(id), depth + 1)
            return text

        return recurse_print(0, 0)
