import math
from enum import IntEnum
from typing import Any, Tuple

import cunumeric as cn
from legate.core import Future, Rect, get_legate_runtime, types

from ..library import user_context, user_lib
from ..utils import get_store
from .base_model import BaseModel


class LegateBoostOpCode(IntEnum):
    BUILD_TREE = user_lib.cffi.BUILD_TREE
    PREDICT = user_lib.cffi.PREDICT
    UPDATE_TREE = user_lib.cffi.UPDATE_TREE


# handle the case of 1 input row, where the store can be a future
# calls to partition_by_tiling will fail
def partition_if_not_future(array: cn.ndarray, shape: Tuple[int, int]) -> Any:
    store = get_store(array)
    if store.kind == Future:
        return store
    return store.partition_by_tiling(shape)


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
        # choose possible splits
        sample_rows = self.random_state.randint(0, X.shape[0], self.max_depth)
        split_proposals = X[sample_rows]  # may not be efficient, maybe write new task
        num_features = X.shape[1]
        num_outputs = g.shape[1]
        n_rows = X.shape[0]
        num_procs = self.num_procs_to_use(n_rows)
        use_gpu = get_legate_runtime().machine.preferred_kind == 1
        rows_per_tile = int(cn.ceil(n_rows / num_procs))

        task = user_context.create_manual_task(
            LegateBoostOpCode.BUILD_TREE, launch_domain=Rect((num_procs, 1))
        )

        # Defining a projection function (even the identity) prevents legate
        # from trying to assign empty tiles to workers
        # in the case where the number of tiles is less than the launch grid
        def proj(x: Tuple[int, int]) -> Tuple[int, int]:
            return (x[0], 0)  # everything crashes if this is lambda x: x ????

        # inputs
        task.add_scalar_arg(self.max_depth, types.int32)

        task.add_input(
            partition_if_not_future(X, (rows_per_tile, num_features)), proj=proj
        )
        task.add_input(
            partition_if_not_future(g, (rows_per_tile, num_outputs)), proj=proj
        )
        task.add_input(
            partition_if_not_future(h, (rows_per_tile, num_outputs)), proj=proj
        )
        task.add_input(get_store(split_proposals))

        # outputs
        # force 1d arrays to be 2d otherwise we get the dreaded assert proj_id == 0
        max_nodes = 2 ** (self.max_depth + 1)
        leaf_value = user_context.create_store(types.float64, (max_nodes, num_outputs))
        feature = user_context.create_store(types.int32, (max_nodes, 1))
        split_value = user_context.create_store(types.float64, (max_nodes, 1))
        gain = user_context.create_store(types.float64, (max_nodes, 1))
        hessian = user_context.create_store(types.float64, (max_nodes, num_outputs))

        # All outputs belong to a single tile on worker 0
        task.add_output(
            leaf_value.partition_by_tiling((max_nodes, num_outputs)), proj=proj
        )
        task.add_output(feature.partition_by_tiling((max_nodes, 1)), proj=proj)
        task.add_output(split_value.partition_by_tiling((max_nodes, 1)), proj=proj)
        task.add_output(gain.partition_by_tiling((max_nodes, 1)), proj=proj)
        task.add_output(
            hessian.partition_by_tiling((max_nodes, num_outputs)), proj=proj
        )

        if num_procs > 1:
            if use_gpu:
                task.add_nccl_communicator()
            else:
                task.add_cpu_communicator()

        task.execute()

        self.leaf_value = cn.array(leaf_value, copy=False)
        self.feature = cn.array(feature, copy=False).squeeze()
        self.split_value = cn.array(split_value, copy=False).squeeze()
        self.gain = cn.array(gain, copy=False).squeeze()
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
        num_features = X.shape[1]
        num_outputs = g.shape[1]
        n_rows = X.shape[0]
        num_procs = self.num_procs_to_use(n_rows)
        use_gpu = get_legate_runtime().machine.preferred_kind == 1
        rows_per_tile = int(cn.ceil(n_rows / num_procs))

        task = user_context.create_manual_task(
            LegateBoostOpCode.UPDATE_TREE, launch_domain=Rect((num_procs, 1))
        )

        def proj(x: Tuple[int, int]) -> Tuple[int, int]:
            return (x[0], 0)

        task.add_input(
            partition_if_not_future(X, (rows_per_tile, num_features)), proj=proj
        )
        task.add_input(
            partition_if_not_future(g, (rows_per_tile, num_outputs)), proj=proj
        )
        task.add_input(
            partition_if_not_future(h, (rows_per_tile, num_outputs)), proj=proj
        )

        # broadcast the tree structure
        task.add_input(get_store(self.feature))
        task.add_input(get_store(self.split_value))

        leaf_value = user_context.create_store(types.float64, self.leaf_value.shape)
        hessian = user_context.create_store(types.float64, self.hessian.shape)

        # All tree outputs belong to a single tile on worker 0
        task.add_output(
            leaf_value.partition_by_tiling(self.leaf_value.shape), proj=proj
        )
        task.add_output(hessian.partition_by_tiling(self.hessian.shape), proj=proj)

        if num_procs > 1:
            if use_gpu:
                task.add_nccl_communicator()
            else:
                task.add_cpu_communicator()

        task.execute()
        self.leaf_value = cn.array(leaf_value, copy=False)
        self.hessian = cn.array(hessian, copy=False)
        return self

    def predict(self, X: cn.ndarray) -> cn.ndarray:
        n_rows = X.shape[0]
        n_features = X.shape[1]
        n_outputs = self.leaf_value.shape[1]
        num_procs = self.num_procs_to_use(n_rows)
        rows_per_tile = int(cn.ceil(n_rows / num_procs))
        task = user_context.create_manual_task(
            LegateBoostOpCode.PREDICT, Rect((num_procs, 1))
        )

        def proj(x: Tuple[int, int]) -> Tuple[int, int]:
            return (x[0], 0)

        task.add_input(
            partition_if_not_future(X, (rows_per_tile, n_features)), proj=proj
        )

        # broadcast the tree structure
        task.add_input(get_store(self.leaf_value))
        task.add_input(get_store(self.feature))
        task.add_input(get_store(self.split_value))

        pred = user_context.create_store(types.float64, (n_rows, n_outputs))
        task.add_output(
            partition_if_not_future(pred, (rows_per_tile, n_outputs)), proj=proj
        )

        task.execute()
        return cn.array(pred)

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
