import copy
import warnings
from enum import IntEnum
from typing import Any, Callable, List, Sequence, Union, cast

import numpy as np

import cupynumeric as cn
from legate.core import TaskTarget, get_legate_runtime, types

from ..library import user_context, user_lib
from ..utils import get_store
from .base_model import BaseModel


class LegateBoostOpCode(IntEnum):
    BUILD_TREE = user_lib.cffi.BUILD_TREE
    PREDICT = user_lib.cffi.PREDICT
    UPDATE_TREE = user_lib.cffi.UPDATE_TREE


class Tree(BaseModel):
    """Decision tree model for gradient boosting.

    Instead of exhaustive search over all possible split values, a random sample
    of size `split_samples` is taken from the dataset and used as split candidates.
    The tree learner matches very closely a histogram type algorithm from
    XGBoost/LightGBM, where the `split_samples` parameter can be tuned like
    the number of bins.

    Parameters
    ----------
    max_depth : int
        The maximum depth of the tree.
    split_samples : int
        The number of data points to sample for each split decision.
        Max value is 2048 due to constraints on shared memory in GPU kernels.
    feature_fraction :
        If float, the subsampled fraction of features considered in building this model.
        Features are sampled without replacement, the number of features is
        rounded up and at least 1.
        Users may implement an arbitrary function returning a cupynumeric array of
        booleans of shape ``(n_features,)`` to specify the feature subset.
    l1_regularization : float
        The L1 regularization parameter applied to leaf weights.
    l2_regularization : float
        The L2 regularization parameter applied to leaf weights.
    alpha : deprecated
        Deprecated, use `l2_regularization` instead.
    min_split_gain : float
        The minimum improvement in the loss function required to make a split.
        Increasing this value generates smaller trees. Equivalent to the `gamma`
        parameter from XGBoost. Is applied on a per output basis e.g. if there
        are 3 output classes then the gain must be greater than 3 * min_split_gain.
    """

    leaf_value: cn.ndarray
    feature: cn.ndarray
    split_value: cn.ndarray
    gain: cn.ndarray
    hessian: cn.ndarray

    def __init__(
        self,
        *,
        max_depth: int = 8,
        split_samples: int = 256,
        feature_fraction: Union[float, Callable[[], cn.array]] = 1.0,
        l1_regularization: float = 0.0,
        l2_regularization: float = 1.0,
        min_split_gain: float = 0.0,
        alpha: Any = "deprecated",
    ) -> None:
        self.max_depth = max_depth
        if split_samples > 2048:
            raise ValueError("split_samples must be <= 2048")
        self.split_samples = split_samples
        self.alpha = alpha
        self.feature_fraction = feature_fraction
        self.l1_regularization = l1_regularization
        self.l2_regularization = l2_regularization
        self.min_split_gain = min_split_gain

        if alpha != "deprecated":
            warnings.warn(
                "`alpha` was renamed to `l2_regularization` in 23.03"
                " and will be removed in 23.05",
                FutureWarning,
            )
            self.l2_regularization = alpha

    def num_nodes(self) -> int:
        return int(cn.sum(self.hessian > 0.0))

    def is_leaf(self, id: cn.array) -> cn.array:
        return self.feature[id] == -1

    def left_child(self, id: cn.array) -> cn.array:
        return id * 2 + 1

    def right_child(self, id: cn.array) -> cn.array:
        return id * 2 + 2

    def fit(
        self,
        X: cn.ndarray,
        g: cn.ndarray,
        h: cn.ndarray,
    ) -> "Tree":
        num_outputs = g.shape[1]

        task = get_legate_runtime().create_auto_task(
            user_context, LegateBoostOpCode.BUILD_TREE
        )

        # inputs
        X_ = get_store(X).promote(2, g.shape[1])
        g_ = get_store(g).promote(1, X.shape[1])
        h_ = get_store(h).promote(1, X.shape[1])

        task.add_scalar_arg(self.max_depth, types.int32)
        max_nodes = 2 ** (self.max_depth + 1) - 1
        task.add_scalar_arg(max_nodes, types.int32)
        task.add_scalar_arg(self.split_samples, types.int32)
        task.add_scalar_arg(self.random_state.randint(0, 2**31), types.int32)
        task.add_scalar_arg(X.shape[0], types.int64)
        task.add_scalar_arg(self.l1_regularization, types.float64)
        task.add_scalar_arg(self.l2_regularization, types.float64)
        task.add_scalar_arg(self.min_split_gain, types.float64)

        task.add_input(X_)
        task.add_broadcast(X_, 1)
        task.add_broadcast(X_, 2)
        task.add_input(g_)
        task.add_input(h_)
        task.add_alignment(g_, h_)
        task.add_alignment(g_, X_)

        # sample features
        if callable(self.feature_fraction):
            feature_set = self.feature_fraction()
            if feature_set.shape != (X.shape[1],) or feature_set.dtype != bool:
                raise ValueError(
                    "feature_fraction must return a boolean array of"
                    " shape (n_features,)"
                )
            task.add_input(get_store(feature_set))
            task.add_broadcast(get_store(feature_set))
        elif self.feature_fraction < 1.0:
            feature_set = cn.zeros(X.shape[1], dtype=bool)
            n_sampled_features = max(1, int(X.shape[1] * self.feature_fraction))
            # use numpy here for sampling as cupynumeric seed is currently unreliable
            # https://github.com/nv-legate/cupynumeric/issues/964
            selection = self.random_state.choice(X.shape[1], n_sampled_features, False)
            feature_set[selection] = True
            task.add_input(get_store(feature_set))
            task.add_broadcast(get_store(feature_set))

        # outputs
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
        task.add_broadcast(leaf_value)
        task.add_broadcast(feature)
        task.add_broadcast(split_value)
        task.add_broadcast(gain)
        task.add_broadcast(hessian)

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
        task.add_scalar_arg(self.l1_regularization, types.float64)
        task.add_scalar_arg(self.l2_regularization, types.float64)
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
        return Tree.batch_predict([self], X)

    @staticmethod
    def batch_predict(models: Sequence[BaseModel], X: cn.ndarray) -> cn.ndarray:
        assert all(isinstance(m, Tree) for m in models)
        models = cast(List[Tree], models)
        n_rows = X.shape[0]
        n_features = X.shape[1]
        n_outputs = models[0].leaf_value.shape[1]
        task = get_legate_runtime().create_auto_task(
            user_context, LegateBoostOpCode.PREDICT
        )

        pred = get_legate_runtime().create_store(types.float64, (n_rows, n_outputs))

        X_ = get_store(X).promote(2, n_outputs)
        pred_ = get_store(pred).promote(1, n_features)
        task.add_input(X_)
        task.add_broadcast(X_, 1)

        # add and broadcast the tree structures
        for m in models:
            leaf_value_ = get_store(m.leaf_value)
            feature_ = get_store(m.feature)
            split_value_ = get_store(m.split_value)
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

    def __mul__(self, scalar: Any) -> "Tree":
        new = copy.deepcopy(self)
        new.leaf_value *= scalar
        return new

    def to_onnx(self) -> Any:
        import onnx
        from onnx import numpy_helper
        from onnx.checker import check_model
        from onnx.helper import (
            make_graph,
            make_model,
            make_tensor,
            make_tensor_value_info,
        )

        onnx_nodes = []

        # We map the legate-boost tree representation to the TreeEnsemble ONNX operator
        # the features array, splits array, and leaf weights can be passed unchanged
        # ONNX then requires some extra arrays to represent the tree structure
        # - nodes_truenodeidx is the index of the left child for a given node
        # - nodes_falsenodeidx is the index of the right child for a given node
        # - nodes_modes indicates that nodes use a <= comparison operator
        # - nodes_trueleafs indicates that the left child is a leaf node
        # - nodes_falseleafs indicates that the right child is a leaf node
        # - leaf_targetids indicates which output the leaf node corresponds to
        # ONNX does not support vector leaf so we will repeat the tree n_outputs
        # times, each time with a different constant for leaf_targetids
        # This is not ideal but I don't see a better way

        tree_max_nodes = self.feature.size
        all_nodes_idx = np.arange(tree_max_nodes)
        nodes_featureids = self.feature.__array__()
        nodes_splits = numpy_helper.from_array(self.split_value.__array__())
        nodes_truenodeids = self.left_child(all_nodes_idx)
        # get the left child of each node and check if it is a leaf
        # if the node is already leaf then its child can go off the end of the array
        # use np.minimum to avoid this
        nodes_trueleafs = self.is_leaf(
            np.minimum(tree_max_nodes - 1, self.left_child(all_nodes_idx))
        ).astype(int)
        nodes_falsenodeids = self.right_child(all_nodes_idx)
        nodes_falseleafs = self.is_leaf(
            np.minimum(tree_max_nodes - 1, self.right_child(all_nodes_idx))
        ).astype(int)

        for output_idx in range(0, self.leaf_value.shape[1]):
            leaf_targetids = np.full(self.feature.size, output_idx, dtype=np.int64)
            leaf_weights = numpy_helper.from_array(
                self.leaf_value[:, output_idx].__array__()
            )

            onnx_nodes.append(
                onnx.helper.make_node(
                    "TreeEnsemble",
                    ["X"],
                    ["pred" + str(output_idx)],
                    domain="ai.onnx.ml",
                    n_targets=self.leaf_value.shape[1],
                    membership_values=None,
                    nodes_missing_value_tracks_true=None,
                    nodes_hitrates=None,
                    aggregate_function=1,
                    post_transform=0,
                    tree_roots=[0],
                    nodes_modes=make_tensor(
                        "nodes_modes",
                        onnx.TensorProto.UINT8,
                        self.feature.shape,
                        np.zeros_like(self.feature, dtype=np.uint8),
                    ),
                    nodes_featureids=nodes_featureids,
                    nodes_splits=nodes_splits,
                    nodes_truenodeids=nodes_truenodeids,
                    nodes_trueleafs=nodes_trueleafs,
                    nodes_falsenodeids=nodes_falsenodeids,
                    nodes_falseleafs=nodes_falseleafs,
                    leaf_targetids=leaf_targetids,
                    leaf_weights=leaf_weights,
                )
            )

            if output_idx == 0:
                accumulated_pred = make_tensor_value_info(
                    "accumulated_pred0", onnx.TensorProto.DOUBLE, [None, None]
                )
                onnx_nodes.append(
                    onnx.helper.make_node(
                        "Identity",
                        ["pred" + str(output_idx)],
                        ["accumulated_pred0"],
                    )
                )
            else:
                accumulated_pred = make_tensor_value_info(
                    "accumulated_pred" + str(output_idx),
                    onnx.TensorProto.DOUBLE,
                    [None, None],
                )
                onnx_nodes.append(
                    onnx.helper.make_node(
                        "Add",
                        [
                            "accumulated_pred" + str(output_idx - 1),
                            "pred" + str(output_idx),
                        ],
                        ["accumulated_pred" + str(output_idx)],
                    )
                )

        # pred inputs
        X = make_tensor_value_info("X", onnx.TensorProto.DOUBLE, [None, None])
        graph = make_graph(
            onnx_nodes, "legateboost.models.Tree", [X], [accumulated_pred]
        )
        model = make_model(
            graph,
            opset_imports=[
                onnx.helper.make_opsetid("ai.onnx.ml", 5),
                onnx.helper.make_opsetid("", 14),
            ],
        )
        check_model(model)
        return model
