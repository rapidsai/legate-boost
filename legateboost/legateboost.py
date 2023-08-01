from __future__ import annotations

import math
import warnings
from enum import IntEnum
from typing import Any, List, Optional, Tuple, Union

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.exceptions import DataConversionWarning
from sklearn.utils.validation import check_is_fitted, check_random_state

import cunumeric as cn
from legate.core import Future, Rect, Store, get_legate_runtime, types

from .input_validation import check_sample_weight, check_X_y
from .library import user_context, user_lib
from .metrics import BaseMetric, metrics
from .objectives import BaseObjective, objectives


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

    def num_procs_to_use(self, num_rows: int) -> int:
        min_rows_per_worker = 10
        available_procs = len(get_legate_runtime().machine)
        return min(available_procs, int(math.ceil(num_rows / min_rows_per_worker)))

    def __init__(
        self,
        X: cn.ndarray,
        g: cn.ndarray,
        h: cn.ndarray,
        learning_rate: float,
        max_depth: int,
        random_state: np.random.RandomState,
    ) -> None:
        # choose possible splits
        sample_rows = random_state.randint(0, X.shape[0], max_depth)
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
        task.add_scalar_arg(learning_rate, types.float64)
        task.add_scalar_arg(max_depth, types.int32)
        task.add_scalar_arg(random_state.randint(0, 2**32), types.uint64)

        task.add_input(
            partition_if_not_future(X, (rows_per_tile, num_features)), proj=proj
        )
        task.add_input(
            partition_if_not_future(g, (rows_per_tile, num_outputs)), proj=proj
        )
        task.add_input(
            partition_if_not_future(h, (rows_per_tile, num_outputs)), proj=proj
        )
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
        task.add_input(_get_store(self.leaf_value))
        task.add_input(_get_store(self.feature))
        task.add_input(_get_store(self.split_value))

        pred = user_context.create_store(types.float64, (n_rows, n_outputs))
        task.add_output(
            partition_if_not_future(pred, (rows_per_tile, n_outputs)), proj=proj
        )
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


class LBBase(BaseEstimator, _PickleCunumericMixin):
    def __init__(
        self,
        n_estimators: int = 100,
        objective: Union[str, BaseObjective] = "squared_error",
        metric: Union[str, BaseMetric, list[Union[str, BaseMetric]]] = "default",
        learning_rate: float = 0.1,
        init: Union[str, None] = "average",
        verbose: int = 0,
        random_state: Optional[np.random.RandomState] = None,
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

    def _setup_metrics(self) -> list[BaseMetric]:
        iterable = (self.metric,) if not isinstance(self.metric, list) else self.metric
        metric_instances = []
        for metric in iterable:
            if isinstance(metric, str):
                if metric == "default":
                    metric_instances.append(self._objective_instance.metric())
                else:
                    metric_instances.append(metrics[metric]())
            elif isinstance(metric, BaseMetric):
                metric_instances.append(metric)
            else:
                raise ValueError(
                    "Expected metric to be a string or instance of BaseMetric"
                )
        return metric_instances

    def _compute_metrics(
        self,
        iteration: int,
        pred: cn.ndarray,
        y: cn.ndarray,
        sample_weight: cn.ndarray,
        metrics: list[BaseMetric],
        verbose: int,
        eval_set: List[Tuple[cn.ndarray, cn.ndarray, cn.ndarray]],
        eval_result: dict,
    ) -> None:
        # make sure dict is initialised
        if not eval_result:
            eval_result.update({"train": {metric.name(): [] for metric in metrics}})
            for i, _ in enumerate(eval_set):
                eval_result[f"eval-{i}"] = {metric.name(): [] for metric in metrics}

        def add_metric(
            metric_pred: cn.ndarray,
            y: cn.ndarray,
            sample_weight: cn.ndarray,
            metric: BaseMetric,
            name: str,
        ) -> None:
            eval_result[name][metric.name()].append(
                metric.metric(y, metric_pred, sample_weight)
            )
            if verbose:
                print(
                    "i: {} {} {}: {}".format(
                        iteration,
                        name,
                        metric.name(),
                        eval_result[name][metric.name()][-1],
                    )
                )

        # add the training metrics
        for metric in metrics:
            add_metric(pred, y, sample_weight, metric, "train")

        # add any eval metrics, if they exist
        for i, (X_eval, y_eval, sample_weight_eval) in enumerate(eval_set):
            for metric in metrics:
                eval_pred = self._objective_instance.transform(self._predict(X_eval))
                add_metric(
                    eval_pred, y_eval, sample_weight_eval, metric, "eval-{}".format(i)
                )

    # check the types of the eval set and add sample weight if none
    def _process_eval_set(
        self, eval_set: List[Tuple[cn.ndarray, ...]]
    ) -> List[Tuple[cn.ndarray, cn.ndarray, cn.ndarray]]:
        new_eval_set: List[Tuple[cn.ndarray, cn.ndarray, cn.ndarray]] = []
        for i, tuple in enumerate(eval_set):
            assert len(tuple) in [2, 3]
            if len(tuple) == 2:
                new_eval_set.append(
                    check_X_y(tuple[0], tuple[1]) + (cn.ones(tuple[1].shape[0]),)
                )
            else:
                new_eval_set.append(
                    check_X_y(tuple[0], tuple[1])
                    + (check_sample_weight(tuple[2], tuple[1].shape[0]),)
                )

        return new_eval_set

    def _partial_fit(
        self,
        X: cn.ndarray,
        y: cn.ndarray,
        sample_weight: Optional[cn.ndarray] = None,
        eval_set: List[Tuple[cn.ndarray, ...]] = [],
        eval_result: dict = {},
    ) -> "LBBase":

        # check inputs
        X, y = check_X_y(X, y)
        _eval_set = self._process_eval_set(eval_set)

        sample_weight = check_sample_weight(sample_weight, y.shape[0])

        if not hasattr(self, "is_fitted_"):
            return self.fit(X, y, sample_weight)

        if self.n_features_in_ != X.shape[1]:
            raise ValueError(
                "X.shape[1] = {} should be equal to {}".format(
                    X.shape[1], self.n_features_in_
                )
            )

        # avoid appending to an existing eval result
        eval_result.clear()

        # current model prediction
        pred = self._predict(X)
        for _ in range(self.n_estimators):
            # obtain gradients
            # check input dimensions are consistent
            assert y.ndim == pred.ndim == 2
            g, h = self._objective_instance.gradient(
                y, self._objective_instance.transform(pred)
            )
            assert g.ndim == h.ndim == 2
            assert g.dtype == h.dtype == cn.float64, "g.dtype={}, h.dtype={}".format(
                g.dtype, h.dtype
            )
            assert g.shape == h.shape

            # apply weights
            g = g * sample_weight[:, None]
            h = h * sample_weight[:, None]

            # build new tree
            self.models_.append(
                TreeStructure(
                    X,
                    g,
                    h,
                    self.learning_rate,
                    self.max_depth,
                    self.random_state_,
                )
            )

            # update current predictions
            pred += self.models_[-1].predict(X)

            # evaluate our progress
            model_idx = len(self.models_) - 1
            self._compute_metrics(
                model_idx,
                self._objective_instance.transform(pred),
                y,
                sample_weight,
                self._metrics,
                self.verbose,
                _eval_set,
                eval_result,
            )
        return self

    def fit(
        self,
        X: cn.ndarray,
        y: cn.ndarray,
        sample_weight: cn.ndarray,
        eval_set: List[Tuple[cn.ndarray, ...]] = [],
        eval_result: dict = {},
    ) -> "LBBase":
        """Build a gradient boosting model from the training set (X, y).

        Parameters
        ----------
        X :
            The training input samples.
        y :
            The target values (class labels) as integers or as floating point numbers.
        sample_weight :
            Sample weights. If None, then samples are equally weighted.
        eval_set :
            A list of (X, y) or (X, y, w) tuples.
            The metric will be evaluated on each tuple.
        eval_result :
            Returns evaluation result dictionary on training completion.
        Returns
        -------
        self :
            Returns self.
        """
        sample_weight = check_sample_weight(sample_weight, len(y))
        self.n_features_in_ = X.shape[1]
        self.models_: List[TreeStructure] = []
        # initialise random state if an integer was passed
        self.random_state_ = check_random_state(self.random_state)

        # setup objective
        if isinstance(self.objective, str):
            self._objective_instance = objectives[self.objective]()
        elif isinstance(self.objective, BaseObjective):
            self._objective_instance = self.objective
        else:
            raise ValueError(
                "Expected objective to be a string or instance of BaseObjective"
            )

        self._metrics = self._setup_metrics()

        n_outputs = int(self._objective_instance.check_labels(y))
        self.model_init_ = cn.zeros(n_outputs, dtype=cn.float64)
        if self.init == "average":
            # initialise the model to some good average value
            # this is equivalent to a tree with a single leaf and learning rate 1.0
            pred = cn.tile(self.model_init_, (y.shape[0], 1))
            g, h = self._objective_instance.gradient(
                y, self._objective_instance.transform(pred)
            )
            # apply weights
            g = g * sample_weight[:, None]
            h = h * sample_weight[:, None]
            H = h.sum(axis=0)
            if cn.all(H > 0.0):
                self.model_init_ = -g.sum(axis=0) / H

        self.is_fitted_ = True

        return self._partial_fit(X, y, sample_weight, eval_set, eval_result)

    def _predict(self, X: cn.ndarray) -> cn.ndarray:
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
    n_estimators :
        The number of boosting stages to perform.
    objective :
        The loss function to optimize. Possible values are ['squared_error'].
    metric :
        Metric for evaluation. 'default' indicates for the objective function to choose
        the accompanying metric. Possible values: ['mse'] or instance of BaseMetric. Can
        be a list multiple metrics.
    learning_rate :
        The learning rate shrinks the contribution of each tree.
    init :
        The initial prediction of the model. If `None`, the initial prediction
        is zero. If 'average', the initial prediction minimises a second order
        approximation of the loss-function (simply the mean label in the case of
        regression).
    verbose :
        Controls the verbosity when fitting and predicting.
    random_state :
        Controls the randomness of the estimator. Pass an int for reproducible
        results across multiple function calls.
    max_depth :
        The maximum depth of the decision trees.

    Attributes
    ----------
    n_features_in_ :
        The number of features when `fit` is performed.
    is_fitted_ :
        Whether the estimator has been fitted.
    models_ :
        list of models from each iteration.


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
        objective: Union[str, BaseObjective] = "squared_error",
        metric: Union[str, BaseMetric, list[Union[str, BaseMetric]]] = "default",
        learning_rate: float = 0.1,
        init: Union[str, None] = "average",
        verbose: int = 0,
        random_state: Optional[np.random.RandomState] = None,
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

    def partial_fit(
        self,
        X: cn.ndarray,
        y: cn.ndarray,
        sample_weight: cn.ndarray = None,
        eval_set: List[Tuple[cn.ndarray, ...]] = [],
        eval_result: dict = {},
    ) -> LBBase:
        """This method is used for incremental (online) training of the model.
        An additional `n_estimators` models will be added to the ensemble.

        Parameters
        ----------
        X :
            The input samples.
        y : cn.ndarray
            The target values.
        sample_weight :
            Individual weights for each sample. If None, then samples
            are equally weighted.
        eval_set :
            A list of (X, y) or (X, y, w) tuples.
            The metric will be evaluated on each tuple.
        eval_result :
            Returns evaluation result dictionary on training completion.
        Returns
        -------
        self :
            Returns self.
        """
        return super()._partial_fit(X, y, sample_weight, eval_set, eval_result)

    def fit(
        self,
        X: cn.ndarray,
        y: cn.ndarray,
        sample_weight: cn.ndarray = None,
        eval_set: List[Tuple[cn.ndarray, ...]] = [],
        eval_result: dict = {},
    ) -> "LBRegressor":
        X, y = check_X_y(X, y)
        return super().fit(X, y, sample_weight, eval_set, eval_result)

    def predict(self, X: cn.ndarray) -> cn.ndarray:
        """Predict labels for samples in X.

        Parameters
        ----------
        X :
            Input data.

        Returns
        -------
        cn.ndarray
            Predicted labels for X.
        """
        pred = self._objective_instance.transform(super()._predict(X))
        if pred.shape[1] == 1:
            pred = pred.squeeze(axis=1)
        return pred


class LBClassifier(LBBase, ClassifierMixin):
    """Implements a gradient boosting algorithm for classification problems.

    Parameters
    ----------
    n_estimators :
        The number of boosting stages to perform.
    objective :
        The loss function to be optimized. Possible values: ['log_loss', 'exp']
        or instance of BaseObjective.
    metric :
        Metric for evaluation. 'default' indicates for the objective function to
        choose the accompanying metric. Possible values: ['log_loss', 'exp'] or
        instance of BaseMetric. Can be a list multiple metrics.
    learning_rate :
        The learning rate shrinks the contribution of each tree by `learning_rate`.
    init :
        The initial prediction of the model. If `None`, the initial prediction
        is zero. If 'average', the initial prediction minimises a second order
        approximation of the loss-function.
    verbose :
        Controls the verbosity of the boosting process.
    random_state :
        Controls the randomness of the estimator. Pass an int for reproducible output
        across multiple function calls.
    max_depth :
        The maximum depth of the individual trees.

    Attributes
    ----------
    classes_ :
        The class labels.
    n_features_ :
        The number of features.
    n_classes_ :
        The number of classes.
    models_ :
        list of models from each iteration.

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
        objective: Union[str, BaseObjective] = "log_loss",
        metric: Union[str, BaseMetric, list[Union[str, BaseMetric]]] = "default",
        learning_rate: float = 0.1,
        init: Union[str, None] = "average",
        verbose: int = 0,
        random_state: Optional[np.random.RandomState] = None,
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

    def partial_fit(
        self,
        X: cn.ndarray,
        y: cn.ndarray,
        classes: Optional[cn.ndarray] = None,
        sample_weight: cn.ndarray = None,
        eval_set: List[Tuple[cn.ndarray, ...]] = [],
        eval_result: dict = {},
    ) -> LBBase:
        """This method is used for incremental fitting on a batch of samples.
        Requires the classes to be provided up front, as they may not be
        inferred from the first batch.

        Parameters
        ----------
        X :
            The training input samples.
        y :
            The target values
        classes :
            The unique labels of the target. Must be provided at the first call.
        sample_weight :
            Weights applied to individual samples (1D array). If None, then
            samples are equally weighted.
        eval_set :
            A list of (X, y) or (X, y, w) tuples.
            The metric will be evaluated on each tuple.
        eval_result :
            Returns evaluation result dictionary on training completion.

        Returns
        -------
        self :
            Returns self.

        Raises
        ------
        ValueError
            If the classes provided are not whole numbers, or
            if provided classes do not match previous fit.
        """
        if classes is not None and not hasattr(self, "classes_"):
            self.classes_ = classes
            assert np.issubdtype(self.classes_.dtype, np.integer) or np.issubdtype(
                self.classes_.dtype, np.floating
            ), "y must be integer or floating type"

            if np.issubdtype(self.classes_.dtype, np.floating):
                whole_numbers = cn.all(self.classes_ == cn.floor(self.classes_))
                if not whole_numbers:
                    raise ValueError("Unknown label type: ", self.classes_)

        else:
            assert self.is_fitted_
            if classes is not None and cn.any(self.classes_ != classes):
                raise ValueError("classes must match previous fit")

        return super()._partial_fit(X, y, sample_weight, eval_set, eval_result)

    def fit(
        self,
        X: cn.ndarray,
        y: cn.ndarray,
        sample_weight: cn.ndarray = None,
        eval_set: List[Tuple[cn.ndarray, ...]] = [],
        eval_result: dict = {},
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
        assert np.issubdtype(self.classes_.dtype, np.integer) or np.issubdtype(
            self.classes_.dtype, np.floating
        ), "y must be integer or floating type"

        if np.issubdtype(self.classes_.dtype, np.floating):
            whole_numbers = cn.all(self.classes_ == cn.floor(self.classes_))
            if not whole_numbers:
                raise ValueError("Unknown label type: ", self.classes_)

        super().fit(
            X,
            y,
            sample_weight=sample_weight,
            eval_set=eval_set,
            eval_result=eval_result,
        )
        return self

    def predict_raw(self, X: cn.ndarray) -> cn.ndarray:
        """Predict pre-transformed values for samples in X. E.g. before
        applying a sigmoid function.

        Parameters
        ----------

        X :
            The input samples.

        Returns
        -------

        y :
            The predicted raw values for each sample in X.
        """
        return super()._predict(X)

    def predict_proba(self, X: cn.ndarray) -> cn.ndarray:
        """Predict class probabilities for samples in X.

        Parameters
        ----------

        X :
            The input samples.

        Returns
        -------

        y :
            The predicted class probabilities for each sample in X.
        """
        check_is_fitted(self, "is_fitted_")
        pred = self._objective_instance.transform(super()._predict(X))
        if pred.shape[1] == 1:
            pred = pred.squeeze()
            pred = cn.stack([1.0 - pred, pred], axis=1)
        return pred

    def predict(self, X: cn.ndarray) -> cn.ndarray:
        """Predict class labels for samples in X.

        Parameters
        ----------

        X :
            The input samples.

        Returns
        -------

        y :
            The predicted class labels for each sample in X.
        """
        return cn.argmax(self.predict_proba(X), axis=1)
