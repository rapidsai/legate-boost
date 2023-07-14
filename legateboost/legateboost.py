from __future__ import annotations

import warnings
from enum import IntEnum
from typing import Any, Tuple, Union

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.exceptions import DataConversionWarning
from sklearn.utils.validation import check_is_fitted, check_random_state

import cunumeric as cn
from legate.core import Future, Rect, Store, get_legate_runtime, types

from .input_validation import check_sample_weight, check_X_y
from .library import user_context, user_lib
from .metrics import metrics
from .objectives import objectives


class LegateBoostOpCode(IntEnum):
    BUILD_TREE = user_lib.cffi.BUILD_TREE
    PREDICT = user_lib.cffi.PREDICT


class _PickleCunumericMixin:
    """When reading back from pickle, convert numpy arrays to cunumeric
    arrays."""

    def __getstate__(self) -> dict[str, Any]:
        return self.__dict__.copy()

    def __setstate__(self, state: dict[str, Any]) -> None:
        def replace(data: Any) -> None:
            if isinstance(data, (dict, list)):
                for k, v in data.items() if isinstance(data, dict) else enumerate(data):
                    if isinstance(v, np.ndarray):
                        data[k] = cn.asarray(v)
                    replace(v)

        replace(state)
        self.__dict__.update(state)


# handle the case of 1 input row, where the store can be a future
# calls to partition_by_tiling will fail
def partition_if_not_future(array: cn.ndarray, shape: Tuple[int, int]) -> Any:
    store = _get_store(array)
    if store.kind == Future:
        return store
    return store.partition_by_tiling(shape)


class TreeStructure(_PickleCunumericMixin):
    """A structure of arrays representing a decision tree.

    A leaf node has value -1 at feature[node_idx]
    """

    leaf_value: cn.ndarray
    feature: cn.ndarray
    split_value: cn.ndarray
    gain: cn.ndarray
    hessian: cn.ndarray

    def is_leaf(self, id: int) -> Any:
        return self.feature[id] == -1

    def left_child(self, id: int) -> int:
        return id * 2 + 1

    def right_child(self, id: int) -> int:
        return id * 2 + 2

    def __init__(
        self,
        leaf_value: cn.ndarray,
        feature: cn.ndarray,
        split_value: cn.ndarray,
        gain: cn.ndarray,
        hessian: cn.ndarray,
    ) -> None:
        """Initialise from existing storage."""
        self.leaf_value = leaf_value
        assert leaf_value.dtype == cn.float64
        self.feature = feature
        assert feature.dtype == cn.int32
        self.split_value = split_value
        assert split_value.dtype == cn.float64
        self.hessian = hessian
        assert hessian.dtype == cn.float64
        self.gain = gain
        assert gain.dtype == cn.float64

    def predict(self, X: cn.ndarray) -> cn.ndarray:
        n_rows = X.shape[0]
        n_features = X.shape[1]
        n_outputs = self.leaf_value.shape[1]
        num_procs = len(get_legate_runtime().machine)
        # dont launch more tasks than rows
        num_procs = min(num_procs, n_rows)
        rows_per_tile = int(cn.ceil(n_rows / num_procs))
        task = user_context.create_manual_task(
            LegateBoostOpCode.PREDICT, Rect((num_procs, 1))
        )
        task.add_input(partition_if_not_future(X, (rows_per_tile, n_features)))

        # broadcast the tree structure
        task.add_input(_get_store(self.leaf_value))
        task.add_input(_get_store(self.feature))
        task.add_input(_get_store(self.split_value))

        pred = user_context.create_store(types.float64, (n_rows, n_outputs))
        task.add_output(partition_if_not_future(pred, (rows_per_tile, n_outputs)))
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


def _get_store(input: Any) -> Store:
    """Extracts a Legate store from any object implementing the legete data
    interface.

    Args:
        input (Any): The input object

    Returns:
        Store: The extracted Legate store
    """
    if isinstance(input, Store):
        return input
    data = input.__legate_data_interface__["data"]
    field = next(iter(data))
    array = data[field]
    _, store = array.stores()
    return store


def build_tree_native(
    X: cn.ndarray,
    g: cn.ndarray,
    h: cn.ndarray,
    learning_rate: float,
    max_depth: int,
    random_state: np.random.RandomState,
) -> TreeStructure:

    # choose possible splits
    split_proposals = X[
        random_state.randint(0, X.shape[0], max_depth)
    ]  # may not be efficient, maybe write new task
    num_features = X.shape[1]
    num_outputs = g.shape[1]
    n_rows = X.shape[0]
    num_procs = len(get_legate_runtime().machine)
    use_gpu = get_legate_runtime().machine.preferred_kind == 1
    # dont launch more tasks than rows
    num_procs = min(num_procs, n_rows)
    rows_per_tile = int(cn.ceil(n_rows / num_procs))

    task = user_context.create_manual_task(
        LegateBoostOpCode.BUILD_TREE, launch_domain=Rect((num_procs, 1))
    )

    # inputs
    task.add_scalar_arg(learning_rate, types.float64)
    task.add_scalar_arg(max_depth, types.int32)
    task.add_scalar_arg(random_state.randint(0, 2**32), types.uint64)

    task.add_input(partition_if_not_future(X, (rows_per_tile, num_features)))
    task.add_input(partition_if_not_future(g, (rows_per_tile, num_outputs)))
    task.add_input(partition_if_not_future(h, (rows_per_tile, num_outputs)))
    task.add_input(_get_store(split_proposals))

    # outputs
    # force 1d arrays to be 2d otherwise we get the dreaded assert proj_id == 0
    max_nodes = 2 ** (max_depth + 1)
    leaf_value = user_context.create_store(types.float64, (max_nodes, num_outputs))
    feature = user_context.create_store(types.int32, (max_nodes, 1))
    split_value = user_context.create_store(types.float64, (max_nodes, 1))
    gain = user_context.create_store(types.float64, (max_nodes, 1))
    hessian = user_context.create_store(types.float64, (max_nodes, num_outputs))

    # All outputs belong to a single tile on worker 0
    # Defining a projection function (even the identity) prevents legate
    # from trying to assign empty tiles to workers
    # in the case where the number of tiles is less than the launch grid
    def proj(x: Tuple[int, int]) -> Tuple[int, int]:
        return (x[0], 0)  # everything crashes if this is lambda x: x ????

    task.add_output(leaf_value.partition_by_tiling((max_nodes, num_outputs)), proj=proj)
    task.add_output(feature.partition_by_tiling((max_nodes, 1)), proj=proj)
    task.add_output(split_value.partition_by_tiling((max_nodes, 1)), proj=proj)
    task.add_output(gain.partition_by_tiling((max_nodes, 1)), proj=proj)
    task.add_output(hessian.partition_by_tiling((max_nodes, num_outputs)), proj=proj)

    if num_procs > 1:
        if use_gpu:
            task.add_nccl_communicator()
        else:
            task.add_cpu_communicator()

    task.execute()

    return TreeStructure(
        cn.array(leaf_value, copy=False),
        cn.array(feature, copy=False).squeeze(),
        cn.array(split_value, copy=False).squeeze(),
        cn.array(gain, copy=False).squeeze(),
        cn.array(hessian, copy=False),
    )


class LBBase(BaseEstimator, _PickleCunumericMixin):
    def __init__(
        self,
        n_estimators: int = 100,
        objective: str = "squared_error",
        metric: str = "default",
        learning_rate: float = 0.1,
        init: Union[str, None] = "average",
        verbose: int = 0,
        random_state: np.random.RandomState = None,
        max_depth: int = 3,
        version: str = "native",
    ) -> None:
        self.n_estimators = n_estimators
        self.objective = objective
        self.metric = metric
        self.learning_rate = learning_rate
        self.init = init
        self.verbose = verbose
        self.random_state = random_state
        self.max_depth = max_depth
        self.version = version

    def _more_tags(self) -> Any:
        return {
            "_xfail_checks": {
                "check_sample_weights_invariance": (
                    "zero sample_weight is not equivalent to removing samples"
                ),
                "check_complex_data": (
                    "LegateBoost does not currently support complex data."
                ),
                "check_dtype_object": ("object type data not supported."),
            },
        }

    def fit(
        self, X: cn.ndarray, y: cn.ndarray, sample_weight: cn.ndarray = None
    ) -> "LBBase":
        """Build a gradient boosting model from the training set (X, y).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values (class labels) as integers or as floating point numbers.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted.

        Returns
        -------
        self : object
            Returns self.
        """
        sample_weight = check_sample_weight(sample_weight, len(y))
        self.n_features_in_ = X.shape[1]
        self.models_ = []

        objective = objectives[self.objective]()
        objective.check_labels(y)
        self.model_init_ = cn.zeros(self.n_margin_outputs_, dtype=cn.float64)
        if self.init == "average":
            # initialise the model to some good average value
            # this is equivalent to a tree with a single leaf and learning rate 1.0
            pred = cn.tile(self.model_init_, (y.shape[0], 1))
            g, h = objective.gradient(y, pred)
            H = h.sum()
            if H > 0.0:
                self.model_init_ = -g.sum(axis=0) / H

        # current model prediction
        pred = cn.tile(self.model_init_, (y.shape[0], 1))
        self._metric = (
            objective.metric() if self.metric == "default" else metrics[self.metric]()
        )
        self.train_metric_ = []
        for i in range(self.n_estimators):
            # obtain gradients
            g, h = objective.gradient(y, pred)
            assert g.dtype == h.dtype == cn.float64, "g.dtype={}, h.dtype={}".format(
                g.dtype, h.dtype
            )
            assert g.shape == h.shape
            g = g * sample_weight[:, None]
            h = h * sample_weight[:, None]

            # build new tree
            tree = build_tree_native(
                X,
                g,
                h,
                self.learning_rate,
                self.max_depth,
                check_random_state(self.random_state),
            )
            self.models_.append(tree)

            # update current predictions
            pred += self.models_[-1].predict(X)

            # evaluate our progress
            metric_pred = (
                objective.transform(pred)
                if self._metric.requires_probability()
                else pred
            )
            self.train_metric_.append(
                self._metric.metric(y, metric_pred, sample_weight)
            )
            if self.verbose:
                print(
                    "i: {} {}: {}".format(
                        i, self._metric.name(), self.train_metric_[-1]
                    )
                )
        self.is_fitted_ = True
        return self

    def predict(self, X: cn.ndarray) -> cn.ndarray:
        X = check_X_y(X)
        check_is_fitted(self, "is_fitted_")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                "X.shape[1] = {} should be equal to {}".format(
                    X.shape[1], self.n_features_in_
                )
            )
        pred = cn.tile(self.model_init_, (X.shape[0], 1))
        for m in self.models_:
            pred += m.predict(X)
        return pred

    def dump_trees(self) -> str:
        check_is_fitted(self, "is_fitted_")
        text = "init={}\n".format(self.model_init_)
        for m in self.models_:
            text += str(m)
        return text


class LBRegressor(LBBase, RegressorMixin):
    """Implementation of a gradient boosting algorithm for regression problems.
    Uses decision trees as weak learners and iteratively improves the model by
    minimizing a loss function.

    Parameters
    ----------
    n_estimators : int, default=100
        The number of boosting stages to perform.
    objective : str, default='squared_error'
        The loss function to optimize. Possible values are ['squared_error'].
    learning_rate : float, default=0.1
        The learning rate shrinks the contribution of each tree.
    init : str or None, default='average'
        The initial prediction of the model. If `None`, the initial prediction
        is zero. If 'average', the initial prediction minimises a second order
        approximation of the loss-function (simply the mean label in the case of
        regression).
    verbose : int, default=0
        Controls the verbosity when fitting and predicting.
    random_state : np.random.RandomState or None, default=None
        Controls the randomness of the estimator. Pass an int for reproducible
        results across multiple function calls.
    max_depth : int, default=3
        The maximum depth of the decision trees.

    Attributes
    ----------
    n_features_in_ : int
        The number of features when `fit` is performed.
    is_fitted_ : bool
        Whether the estimator has been fitted.
    models_ :
        list of models from each iteration.
    train_metric_ :
        evaluated training metrics from each iteration.


    See Also
    --------
    LBClassifier

    Examples
    --------
    >>> import cunumeric as cn
    >>> import legateboost as lbst
    >>> X = cn.random.random((1000, 10))
    >>> y = cn.random.random(X.shape[0])
    >>> model = lbst.LBRegressor(verbose=1,
    ... n_estimators=100, random_state=0, max_depth=2).fit(X, y)
    >>> model.predict(X)
    """

    def __init__(
        self,
        n_estimators: int = 100,
        objective: str = "squared_error",
        metric: str = "default",
        learning_rate: float = 0.1,
        init: Union[str, None] = "average",
        verbose: int = 0,
        random_state: np.random.RandomState = None,
        max_depth: int = 3,
    ) -> None:
        super().__init__(
            n_estimators=n_estimators,
            objective=objective,
            metric=metric,
            learning_rate=learning_rate,
            init=init,
            verbose=verbose,
            random_state=random_state,
            max_depth=max_depth,
        )

    def _more_tags(self) -> Any:
        return {
            "multioutput": True,
        }

    def fit(
        self, X: cn.ndarray, y: cn.ndarray, sample_weight: cn.ndarray = None
    ) -> "LBRegressor":
        self.n_margin_outputs_ = 1
        X, y = check_X_y(X, y)
        if y.ndim > 1:
            self.n_margin_outputs_ = y.shape[1]
        return super().fit(X, y, sample_weight)

    def predict(self, X: cn.ndarray) -> cn.ndarray:
        """Predict labels for samples in X.

        Parameters
        ----------
        X : cn.ndarray
            Input data.

        Returns
        -------
        cn.ndarray
            Predicted labels for X.
        """
        pred = super().predict(X)
        if pred.shape[1] == 1:
            pred = pred.squeeze(axis=1)
        return pred


class LBClassifier(LBBase, ClassifierMixin):
    """Implements a gradient boosting algorithm for classification problems.

    Parameters
    ----------
    n_estimators : int, default=100
        The number of boosting stages to perform.
    objective : str, default='log_loss'
        The loss function to be optimized. Possible values: ['log_loss'].
    learning_rate : float, default=0.1
        The learning rate shrinks the contribution of each tree by `learning_rate`.
    init : str or None, default='average'
        The initial prediction of the model. If `None`, the initial prediction
        is zero. If 'average', the initial prediction minimises a second order
        approximation of the loss-function.
    verbose : int, default=0
        Controls the verbosity of the boosting process.
    random_state : np.random.RandomState or None, default=None
        Controls the randomness of the estimator. Pass an int for reproducible output
        across multiple function calls.
    max_depth : int, default=3
        The maximum depth of the individual trees.

    Attributes
    ----------
    classes_ : array of shape (n_classes,)
        The classes labels.
    n_features_ : int
        The number of features.
    n_classes_ : int
        The number of classes.
    models_ : list of models from each iteration.
    train_metric_ : evaluated training metrics from each iteration.

    See Also
    --------
    LBRegressor

    Examples
    --------
    >>> import cunumeric as cn
    >>> import legateboost as lbst
    >>> X = cn.random.random((1000, 10))
    >>> y = cn.random.randint(0, 2, X.shape[0])
    >>> model = lbst.LBClassifier(verbose=1, n_estimators=100,
    ...     random_state=0, max_depth=2).fit(X, y)
    >>> model.predict(X)
    """

    def __init__(
        self,
        n_estimators: int = 100,
        objective: str = "log_loss",
        metric: str = "default",
        learning_rate: float = 0.1,
        init: Union[str, None] = "average",
        verbose: int = 0,
        random_state: np.random.RandomState = None,
        max_depth: int = 3,
    ) -> None:
        super().__init__(
            n_estimators=n_estimators,
            objective=objective,
            metric=metric,
            learning_rate=learning_rate,
            init=init,
            verbose=verbose,
            random_state=random_state,
            max_depth=max_depth,
        )

    def fit(
        self, X: cn.ndarray, y: cn.ndarray, sample_weight: cn.ndarray = None
    ) -> "LBClassifier":
        if hasattr(y, "ndim") and y.ndim > 1:
            warnings.warn(
                "A column-vector y was passed when a 1d array was expected.",
                DataConversionWarning,
            )
        X, y = check_X_y(X, y)

        # Validate classifier inputs
        if y.size <= 1:
            raise ValueError("y has only 1 sample in classifer training.")

        self.classes_ = cn.unique(y.squeeze())
        num_classes = int(self.classes_.max() + 1)
        assert np.issubdtype(self.classes_.dtype, np.integer) or np.issubdtype(
            self.classes_.dtype, np.floating
        ), "y must be integer or floating type"

        if np.issubdtype(self.classes_.dtype, np.floating):
            whole_numbers = cn.all(self.classes_ == cn.floor(self.classes_))
            if not whole_numbers:
                raise ValueError("Unknown label type: ", self.classes_)

        self.n_margin_outputs_ = num_classes if num_classes > 2 else 1
        super().fit(X, y, sample_weight=sample_weight)
        return self

    def predict_raw(self, X: cn.ndarray) -> cn.ndarray:
        """Predict pre-transformed values for samples in X. E.g. before
        applying a sigmoid function.

        Parameters
        ----------

        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------

        y : ndarray of shape (n_samples,)
            The predicted raw values for each sample in X.
        """
        return super().predict(X)

    def predict_proba(self, X: cn.ndarray) -> cn.ndarray:
        """Predict class probabilities for samples in X.

        Parameters
        ----------

        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------

        y : ndarray of shape (n_samples, n_classes)
            The predicted class probabilities for each sample in X.
        """
        objective = objectives[self.objective]()
        pred = objective.transform(super().predict(X))
        if self.n_margin_outputs_ == 1:
            pred = pred.squeeze()
            pred = cn.stack([1.0 - pred, pred], axis=1)
        return pred

    def predict(self, X: cn.ndarray) -> cn.ndarray:
        """Predict class labels for samples in X.

        Parameters
        ----------

        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------

        y : ndarray of shape (n_samples,)
            The predicted class labels for each sample in X.
        """
        return cn.argmax(self.predict_proba(X), axis=1)
