from __future__ import annotations

import warnings
from copy import deepcopy
from typing import Any, List, Optional, Tuple, Union

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.exceptions import DataConversionWarning
from sklearn.utils.validation import check_is_fitted, check_random_state
from typing_extensions import Self, TypeAlias

import cunumeric as cn

from .input_validation import check_sample_weight, check_X_y
from .metrics import BaseMetric, metrics
from .models import BaseModel, Tree
from .objectives import BaseObjective, objectives
from .utils import PickleCunumericMixin, preround

EvalResult: TypeAlias = dict[str, dict[str, list[float]]]


class LBBase(BaseEstimator, PickleCunumericMixin):
    def __init__(
        self,
        n_estimators: int = 100,
        objective: Union[str, BaseObjective] = "squared_error",
        metric: Union[str, BaseMetric, list[Union[str, BaseMetric]]] = "default",
        learning_rate: float = 0.1,
        init: Union[str, None] = "average",
        base_models: Tuple[BaseModel, ...] = (Tree(max_depth=3),),
        verbose: int = 0,
        random_state: Optional[np.random.RandomState] = None,
        version: str = "native",
    ) -> None:
        self.n_estimators = n_estimators
        self.objective = objective
        self.metric = metric
        self.learning_rate = learning_rate
        self.init = init
        self.verbose = verbose
        self.random_state = random_state
        self.version = version
        self.model_init_: cn.ndarray
        self.base_models = base_models

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
                    metric_instances.append(metrics[metric].create())
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
        eval_preds: List[cn.ndarray],
        y: cn.ndarray,
        sample_weight: cn.ndarray,
        metrics: list[BaseMetric],
        verbose: int,
        eval_set: List[Tuple[cn.ndarray, cn.ndarray, cn.ndarray]],
        eval_result: EvalResult,
    ) -> None:
        # make sure dict is initialised
        if not eval_result:
            eval_result.update({"train": {metric.name(): [] for metric in metrics}})
            for i, _ in enumerate(eval_preds):
                eval_result[f"eval-{i}"] = {metric.name(): [] for metric in metrics}

        def add_metric(
            metric_pred: cn.ndarray,
            y: cn.ndarray,
            sample_weight: cn.ndarray,
            metric: BaseMetric,
            name: str,
        ) -> None:
            eval_result[name][metric.name()].append(
                metric.metric(
                    y, self._objective_instance.transform(metric_pred), sample_weight
                )
            )

        # add the training metrics
        for metric in metrics:
            add_metric(pred, y, sample_weight, metric, "train")

        # add any eval metrics, if they exist
        for i, (X_eval, y_eval, sample_weight_eval) in enumerate(eval_set):
            for metric in metrics:
                add_metric(
                    eval_preds[i],
                    y_eval,
                    sample_weight_eval,
                    metric,
                    "eval-{}".format(i),
                )

        # print the metrics
        if verbose:

            def format(set_name: str, metric_name: str, value: float) -> str:
                return "\t{}-{}:".format(set_name, metric_name) + f"{value: 8.4f}"

            msg = "[{}]".format(iteration)
            for k, v in eval_result.items():
                for m, values in v.items():
                    msg += format(k, str(m), values[-1])
            print(msg)

    # check the types of the eval set and add sample weight if none
    def _process_eval_set(
        self, eval_set: List[Tuple[cn.ndarray, ...]]
    ) -> List[Tuple[cn.ndarray, cn.ndarray, cn.ndarray]]:
        new_eval_set: List[Tuple[cn.ndarray, cn.ndarray, cn.ndarray]] = []
        for tuple in eval_set:
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

    def _get_weighted_gradient(
        self,
        y: cn.ndarray,
        pred: cn.ndarray,
        sample_weight: cn.ndarray,
        learning_rate: float,
    ) -> Tuple[cn.ndarray, cn.ndarray]:
        """Computes the weighted gradient and Hessian for the given predictions
        and labels.

        Also applies a pre-rounding step to ensure reproducible floating
        point summation.
        """
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

        # apply weights and learning rate
        g = g * sample_weight[:, None] * learning_rate
        # ensure hessians are not too small for numerical stability
        h = cn.maximum(h * sample_weight[:, None], 1e-8)
        return preround(g), preround(h)

    def _partial_fit(
        self,
        X: cn.ndarray,
        y: cn.ndarray,
        sample_weight: Optional[cn.ndarray] = None,
        eval_set: List[Tuple[cn.ndarray, ...]] = [],
        eval_result: EvalResult = {},
    ) -> Self:
        # check inputs
        X, y = check_X_y(X, y)
        _eval_set = self._process_eval_set(eval_set)

        sample_weight = check_sample_weight(sample_weight, y.shape[0])

        if not hasattr(self, "is_fitted_"):
            return self.fit(
                X,
                y,
                sample_weight=sample_weight,
                eval_set=eval_set,
                eval_result=eval_result,
            )

        if self.n_features_in_ != X.shape[1]:
            raise ValueError(
                "X.shape[1] = {} should be equal to {}".format(
                    X.shape[1], self.n_features_in_
                )
            )

        # avoid appending to an existing eval result
        eval_result.clear()

        # current model prediction
        train_pred = self._predict(X)
        eval_preds = [self._predict(X_eval) for X_eval, _, _ in _eval_set]
        for i in range(self.n_estimators):
            # obtain gradients
            g, h = self._get_weighted_gradient(
                y, train_pred, sample_weight, self.learning_rate
            )

            # build new model
            self.models_.append(
                deepcopy(self.base_models[i % len(self.base_models)]).set_random_state(
                    self.random_state_
                )
            )
            self.models_[-1].fit(X, g, h)

            # update current predictions
            train_pred += self.models_[-1].predict(X)
            for i, (X_eval, _, _) in enumerate(_eval_set):
                eval_preds[i] += self.models_[-1].predict(X_eval)

            # evaluate our progress
            model_idx = len(self.models_) - 1
            self._compute_metrics(
                model_idx,
                train_pred,
                eval_preds,
                y,
                sample_weight,
                self._metrics,
                self.verbose,
                _eval_set,
                eval_result,
            )
        return self

    def update(
        self,
        X: cn.ndarray,
        y: cn.ndarray,
        sample_weight: Optional[cn.ndarray] = None,
        eval_set: List[Tuple[cn.ndarray, ...]] = [],
        eval_result: EvalResult = {},
    ) -> Self:
        """Update a gradient boosting model from the training set (X, y). This
        method does not add any new models to the ensemble, only updates
        existing models to fit the new data.

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

        # check inputs
        X, y = check_X_y(X, y)
        _eval_set = self._process_eval_set(eval_set)

        sample_weight = check_sample_weight(sample_weight, y.shape[0])

        assert hasattr(self, "is_fitted_") and self.is_fitted_

        if self.n_features_in_ != X.shape[1]:
            raise ValueError(
                "X.shape[1] = {} should be equal to {}".format(
                    X.shape[1], self.n_features_in_
                )
            )

        # avoid appending to an existing eval result
        eval_result.clear()

        # update the model initialisation
        self.model_init_ = self._objective_instance.initialise_prediction(
            y, sample_weight, self.init == "average"
        )

        for m in self.models_:
            m.clear()

        # current model prediction
        train_pred = self._predict(X)
        eval_preds = [self._predict(X_eval) for X_eval, _, _ in _eval_set]

        for i, m in enumerate(self.models_):
            # obtain gradients
            g, h = self._get_weighted_gradient(
                y, train_pred, sample_weight, self.learning_rate
            )

            m.update(X, g, h)

            train_pred += m.predict(X)
            for i, (X_eval, _, _) in enumerate(_eval_set):
                eval_preds[i] += self.models_[-1].predict(X_eval)

            # evaluate our progress
            self._compute_metrics(
                i,
                train_pred,
                eval_preds,
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
        eval_result: EvalResult = {},
    ) -> Self:
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
        self.models_: List[BaseModel] = []
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

        self.model_init_ = self._objective_instance.initialise_prediction(
            y, sample_weight, self.init == "average"
        )
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
        pred = cn.repeat(self.model_init_[cn.newaxis, :], X.shape[0], axis=0)
        for m in self.models_:
            pred += m.predict(X)
        return pred

    def dump_models(self) -> str:
        check_is_fitted(self, "is_fitted_")
        text = "init={}\n".format(self.model_init_)
        for m in self.models_:
            text += str(m)
        return text


class LBRegressor(LBBase, RegressorMixin):
    """Implementation of a gradient boosting algorithm for regression problems.
    Learns component models to iteratively improve a loss function.

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
        The learning rate shrinks the contribution of each model.
    init :
        The initial prediction of the model. If `None`, the initial prediction
        is zero. If 'average', the initial prediction minimises a second order
        approximation of the loss-function (simply the mean label in the case of
        regression).
    base_models :
        The base models to use for each iteration. The model used in each iteration
        i is base_models[i % len(base_models)].
    verbose :
        Controls the verbosity when fitting and predicting.
    random_state :
        Controls the randomness of the estimator. Pass an int for reproducible
        results across multiple function calls.

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
        base_models: Tuple[BaseModel, ...] = (Tree(max_depth=3),),
        verbose: int = 0,
        random_state: Optional[np.random.RandomState] = None,
    ) -> None:
        super().__init__(
            n_estimators=n_estimators,
            objective=objective,
            metric=metric,
            learning_rate=learning_rate,
            init=init,
            base_models=base_models,
            verbose=verbose,
            random_state=random_state,
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
        eval_result: EvalResult = {},
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
        return super()._partial_fit(
            X,
            y,
            sample_weight=sample_weight,
            eval_set=eval_set,
            eval_result=eval_result,
        )

    def fit(
        self,
        X: cn.ndarray,
        y: cn.ndarray,
        sample_weight: cn.ndarray = None,
        eval_set: List[Tuple[cn.ndarray, ...]] = [],
        eval_result: EvalResult = {},
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
        check_is_fitted(self, "is_fitted_")
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
        The learning rate shrinks the contribution of each model.
    init :
        The initial prediction of the model. If `None`, the initial prediction
        is zero. If 'average', the initial prediction minimises a second order
        approximation of the loss-function.
    base_models:
        The base models to use for each iteration. The model used in each iteration
        i is base_models[i % len(base_models)].
    verbose :
        Controls the verbosity of the boosting process.
    random_state :
        Controls the randomness of the estimator. Pass an int for reproducible output
        across multiple function calls.

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
        base_models: Tuple[BaseModel, ...] = (Tree(max_depth=3),),
        verbose: int = 0,
        random_state: Optional[np.random.RandomState] = None,
    ) -> None:
        super().__init__(
            n_estimators=n_estimators,
            objective=objective,
            metric=metric,
            learning_rate=learning_rate,
            init=init,
            base_models=base_models,
            verbose=verbose,
            random_state=random_state,
        )

    def partial_fit(
        self,
        X: cn.ndarray,
        y: cn.ndarray,
        classes: Optional[cn.ndarray] = None,
        sample_weight: cn.ndarray = None,
        eval_set: List[Tuple[cn.ndarray, ...]] = [],
        eval_result: EvalResult = {},
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

        return super()._partial_fit(
            X,
            y,
            sample_weight=sample_weight,
            eval_set=eval_set,
            eval_result=eval_result,
        )

    def fit(
        self,
        X: cn.ndarray,
        y: cn.ndarray,
        sample_weight: cn.ndarray = None,
        eval_set: List[Tuple[cn.ndarray, ...]] = [],
        eval_result: EvalResult = {},
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
