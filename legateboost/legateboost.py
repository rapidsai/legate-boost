from __future__ import annotations

import warnings
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.exceptions import DataConversionWarning
from sklearn.utils import Tags
from sklearn.utils.validation import check_is_fitted, check_random_state, validate_data
from typing_extensions import Self, TypeAlias

import cupynumeric as cn

from .input_validation import _lb_check_X, _lb_check_X_y, check_sample_weight
from .metrics import BaseMetric, metrics
from .models import BaseModel, Tree
from .objectives import OBJECTIVES_MAP, BaseObjective
from .shapley import global_shapley_attributions, local_shapley_attributions
from .utils import AddableMixin, AddMember, PickleCupynumericMixin

if TYPE_CHECKING:
    from .callbacks import TrainingCallback

EvalResult: TypeAlias = dict[str, dict[str, list[float]]]

__all__ = ["LBBase", "LBClassifier", "LBRegressor"]


class LBBase(BaseEstimator, PickleCupynumericMixin, AddableMixin):
    def __init__(
        self,
        *,
        n_estimators: int = 100,
        objective: Union[str, BaseObjective] = "squared_error",
        metric: Union[str, BaseMetric, list[Union[str, BaseMetric]]] = "default",
        learning_rate: float = 0.1,
        subsample: float = 1.0,
        init: Union[str, None] = "average",
        base_models: Tuple[BaseModel, ...] = (Tree(max_depth=3),),
        callbacks: Sequence[TrainingCallback] = (),
        verbose: int = 0,
        random_state: Optional[np.random.RandomState] = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.objective = objective
        self.metric = metric
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.init = init
        self.verbose = verbose
        self.random_state = random_state
        self.model_init_: cn.ndarray
        self.callbacks = callbacks
        self.metrics_: list[BaseMetric]
        if not isinstance(base_models, tuple):
            warnings.warn("base_models should be a tuple")
        self.base_models = base_models

        # define what happens to the attributes when two models are added
        self._add_behaviour.update(
            {
                "models_": AddMember.ADD,
                "model_init_": AddMember.ADD,
                "n_features_in_": AddMember.ASSERT_SAME,
                "is_fitted_": AddMember.ASSERT_SAME,
                "n_estimators": AddMember.ADD,
                "objective": AddMember.ASSERT_SAME,
                "metric": AddMember.PREFER_A,
                "learning_rate": AddMember.PREFER_A,
                "subsample": AddMember.PREFER_A,
                "init": AddMember.PREFER_A,
                "base_models": AddMember.PREFER_A,
                "callbacks": AddMember.PREFER_A,
                "verbose": AddMember.PREFER_A,
                "random_state_": AddMember.PREFER_A,
                "random_state": AddMember.PREFER_A,
                "_objective_instance": AddMember.PREFER_A,
                "_metrics": AddMember.PREFER_A,
            }
        )

    def __sklearn_tags__(self) -> Tags:
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = False
        tags.input_tags.categorical = False
        tags.input_tags.string = False
        return tags

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
                float(
                    metric.metric(
                        y,
                        self._objective_instance.transform(metric_pred),
                        sample_weight,
                    )
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
                    _lb_check_X_y(tuple[0], tuple[1]) + (cn.ones(tuple[1].shape[0]),)
                )
            else:
                new_eval_set.append(
                    _lb_check_X_y(tuple[0], tuple[1])
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
        and labels."""
        # check input dimensions are consistent
        assert y.ndim == pred.ndim == 2, (y.shape, pred.shape)
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

        # apply subsample
        if self.subsample < 1.0:
            generator = cn.random.Generator(
                cn.random.XORWOW(seed=self.random_state_.randint(0, 2**32))
            )
            mask = generator.binomial(1, self.subsample, size=y.shape[0])
            g *= mask[:, None]
            h *= mask[:, None]

        return g, h

    def _partial_fit(
        self,
        X: cn.ndarray,
        y: cn.ndarray,
        *,
        sample_weight: Optional[cn.ndarray] = None,
        eval_set: List[Tuple[cn.ndarray, ...]] = [],
        eval_result: EvalResult = {},
    ) -> Self:
        # check inputs
        X, y = _lb_check_X_y(X, y)
        validate_data(self, X, reset=False, skip_check_array=True)
        _eval_set = self._process_eval_set(eval_set)
        sample_weight = check_sample_weight(sample_weight, y.shape[0])
        if not isinstance(self.base_models, tuple) or len(self.base_models) == 0:
            raise ValueError("base_models should be a tuple")

        # avoid appending to an existing eval result
        eval_result.clear()

        # current model prediction
        train_pred = self._predict(X)
        eval_preds = [self._predict(X_eval) for X_eval, _, _ in _eval_set]

        # callbacks before training
        for c in self.callbacks:
            c.before_training(self)

        for i in range(self.n_estimators):

            # callbacks before iteration
            if any((c.before_iteration(self, i, eval_result) for c in self.callbacks)):
                break

            # obtain gradients
            g, h = self._get_weighted_gradient(
                y, train_pred, sample_weight, self.learning_rate
            )

            # build new model
            self.models_.append(
                deepcopy(self.base_models[i % len(self.base_models)])
                .set_random_state(self.random_state_)
                .fit(X, g, h)
            )

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

            # callbacks after iteration
            if any(
                [
                    c.after_iteration(self, model_idx, eval_result)
                    for c in self.callbacks
                ]
            ):
                break

        # callbacks after training
        for c in self.callbacks:
            c.after_training(self)

        return self

    def update(
        self,
        X: cn.ndarray,
        y: cn.ndarray,
        *,
        sample_weight: Optional[cn.ndarray] = None,
        eval_set: List[Tuple[cn.ndarray, ...]] = [],
        eval_result: EvalResult = {},
    ) -> Self:
        """Update a gradient boosting model from the training set (X, y). This
        method does not add any new models to the ensemble, only updates existing
        models to fit the new data.

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
        X, y = _lb_check_X_y(X, y)
        validate_data(self, X, y, reset=False, skip_check_array=True)
        _eval_set = self._process_eval_set(eval_set)

        sample_weight = check_sample_weight(sample_weight, y.shape[0])

        assert hasattr(self, "is_fitted_") and self.is_fitted_

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
        *,
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
            self._objective_instance = OBJECTIVES_MAP[self.objective]()
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

        return self._partial_fit(
            X,
            y,
            sample_weight=sample_weight,
            eval_set=eval_set,
            eval_result=eval_result,
        )

    def __len__(self) -> int:
        """Returns the number of models in the ensemble.

        Returns:
            int: The number of models in the `models_` attribute.
        """
        check_is_fitted(self, "is_fitted_")
        return len(self.models_)

    def __getitem__(self, i: int) -> BaseModel:
        """Retrieve the model at the specified index.

        Args:
            i (int): The index of the model to retrieve.

        Returns:
            BaseModel: The model at the specified index.
        """
        check_is_fitted(self, "is_fitted_")
        return self.models_[i]

    def __iter__(self) -> Any:
        """Returns an iterator over the models in the estimator.

        Yields:
            Any: An iterator over the models in the `models_` attribute.
        """
        check_is_fitted(self, "is_fitted_")
        return iter(self.models_)

    def __mul__(self, scalar: Any) -> Self:
        """Gradient boosted models are linear in the predictions before the non-
        linear link function is applied. This means that the model can be
        multiplied by a scalar, which subsequently scales all raw output
        predictions. This is useful for ensembling models.

        Parameters
        ----------
        scalar : numeric
            The scalar value to multiply with the model.
        Returns
        -------
        new : object
            A new instance of the model with all internal models and initial model
            multiplied by the given scalar.
        Raises
        ------
        ValueError
            If the provided scalar is not a numeric value.
        """
        check_is_fitted(self, "is_fitted_")

        if not np.isscalar(scalar):
            raise ValueError("Can only multiply by scalar")
        new = deepcopy(self)
        new.models_ = [m * scalar for m in self.models_]
        new.model_init_ = self.model_init_ * scalar
        return new

    def _predict(self, X: cn.ndarray) -> cn.ndarray:
        check_is_fitted(self, "is_fitted_")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                "X.shape[1] = {} should be equal to {}".format(
                    X.shape[1], self.n_features_in_
                )
            )
        pred = cn.empty((X.shape[0],) + self.model_init_.shape, dtype=cn.float64)
        pred[:] = self.model_init_

        # get a list of models for each type
        model_types: Dict[type[BaseModel], List[BaseModel]] = {}
        for m in self.models_:
            if type(m) not in model_types:
                model_types[type(m)] = []
            model_types[type(m)].append(m)
        # call batch prediction for each model type
        # this may be faster than adding predictions one by one
        # e.g. linear models can be added together first then predict
        for Type, models in model_types.items():
            pred += Type.batch_predict(models, X)
        return pred

    def predict_raw(self, X: cn.ndarray) -> cn.ndarray:
        """Predict pre-transformed values for samples in X. E.g. before applying a
        sigmoid function.

        Parameters
        ----------

        X :
            The input samples.

        Returns
        -------

        y :
            The predicted raw values for each sample in X.
        """
        X = _lb_check_X(X)
        validate_data(self, X, reset=False, skip_check_array=True)
        return self._predict(X)

    def dump_models(self) -> str:
        """Dumps the models in the current instance to a string.

        Returns:
            str: A string representation of the models.
        """
        check_is_fitted(self, "is_fitted_")
        text = "init={}\n".format(self.model_init_)
        for m in self.models_:
            text += str(m)
        return text

    def _make_onnx_init(self, X_dtype):
        # turn self.model_init_ into an ONNX model
        from onnx import TensorProto, numpy_helper
        from onnx.checker import check_model
        from onnx.helper import (
            make_graph,
            make_model,
            make_node,
            make_opsetid,
            make_tensor_value_info,
            np_dtype_to_tensor_dtype,
        )

        # model constants
        X_in = make_tensor_value_info(
            "X_in", np_dtype_to_tensor_dtype(X_dtype), [None, self.n_features_in_]
        )
        nodes = []
        nodes.append(make_node("Shape", ["X_in"], ["n_rows"], end=1))
        one = numpy_helper.from_array(np.array([1], dtype=np.int64), name="one")
        nodes.append(make_node("Concat", ["n_rows", "one"], ["tile_repeat"], axis=0))
        init = numpy_helper.from_array(
            np.atleast_2d(self.model_init_.__array__()), name="init"
        )
        prediction_out = make_tensor_value_info(
            "predictions_out",
            TensorProto.DOUBLE,
            [None, self.model_init_.shape[0]],
        )
        nodes.append(make_node("Tile", ["init", "tile_repeat"], ["predictions_out"]))
        X_out = make_tensor_value_info(
            "X_out",
            np_dtype_to_tensor_dtype(X_dtype),
            [None, None],
        )
        nodes.append(make_node("Identity", ["X_in"], ["X_out"]))
        graph = make_graph(
            nodes,
            "legateboost estimator init",
            [X_in],
            [X_out, prediction_out],
            [init, one],
        )
        onnx_model = make_model(
            graph,
            opset_imports=[
                make_opsetid("", 21),
            ],
        )
        check_model(onnx_model)

        return onnx_model

    def to_onnx(self, X: cn.ndarray, predict_function="predict"):
        """Converts the estimator to an ONNX model which is expected to produce
        equivalent predictions to `predict_function` up to reasonable floating
        point tolerance. The ONNX model is hard coded to the X input data type,
        separate models should be generated for float and double. The ONNX model
        takes "X_in" as input and produces "predictions_out" as output.

        Parameters
        ----------
        X:
            Example input data. Use to infer input data characteristics.
            A model produced for float32 will not accept float64 input and vice versa.
        predict_function : str
            The serialised ONNX model can produce output equivalent to 'predict',
            'predict_proba', or 'predict_raw'.
            The default is "predict".
        Returns
        -------
        Any
            The ONNX model.

        Examples
        --------
        >>> import numpy as np
        >>> import legateboost as lb
        >>> X = np.random.random((1000, 10))
        >>> y = np.random.random(X.shape[0])
        >>> model = lb.LBRegressor(n_estimators=5).fit(X, y)
        >>> import onnxruntime as ort
        >>> sess = ort.InferenceSession(model.to_onnx(X.dtype).SerializeToString())
        >>> onnx_pred = sess.run(None, {"X_in": X})[0]
        >>> assert np.allclose(model.predict(X), onnx_pred, atol=1e-6)
        >>>
        """
        if predict_function not in ["predict", "predict_proba", "predict_raw"]:
            raise ValueError(
                "predict_function should be one of "
                "['predict', 'predict_proba', 'predict_raw']"
            )

        from onnx import TensorProto, numpy_helper
        from onnx.checker import check_model
        from onnx.compose import merge_models
        from onnx.helper import (
            make_graph,
            make_model,
            make_node,
            make_opsetid,
            make_tensor_value_info,
        )

        model = self._make_onnx_init(X.dtype)
        if self.models_ is not None and len(self.models_) > 0:
            model = merge_models(
                model,
                self.models_[0].to_onnx(X),
                io_map=[("X_out", "X_in"), ("predictions_out", "predictions_in")],
                prefix2="model_0_",
            )

        for i in range(1, len(self.models_)):
            model = merge_models(
                model,
                self.models_[i].to_onnx(X),
                io_map=[
                    ("model_{}_X_out".format(i - 1), "X_in"),
                    ("model_{}_predictions_out".format(i - 1), "predictions_in"),
                ],
                prefix2="model_{}_".format(i),
            )

        # remove the X_out output, we only need the predictions
        # add a transform operator
        model.graph.output.remove(model.graph.output[0])

        # add any transform from the objective
        if predict_function == "predict":
            model = merge_models(
                model,
                self._objective_instance.onnx_transform(),
                io_map=[
                    (
                        "model_{}_predictions_out".format(len(self.models_) - 1),
                        "predictions_in",
                    )
                ],
                prefix2="transform_",
            )
            # coerce the output shape to be the same as the equivalent predict function
            test_pred = getattr(self, predict_function)(X[0:1])

            extra_out_shape = [] if test_pred.ndim == 1 else list(test_pred.shape[1:])
            shape = numpy_helper.from_array(
                np.array([-1] + extra_out_shape), name="shape"
            )

            reshape_predictions_in = make_tensor_value_info(
                "reshape_predictions_in",
                TensorProto.DOUBLE,
                [None, None],
            )
            reshaped_predictions = make_tensor_value_info(
                "reshaped_predictions",
                TensorProto.DOUBLE,
                shape=[None] + list(extra_out_shape),
            )
            nodes = [
                make_node(
                    "Reshape",
                    ["reshape_predictions_in", "shape"],
                    ["reshaped_predictions"],
                )
            ]
            graph = make_graph(
                nodes,
                "legateboost estimator transform",
                [reshape_predictions_in],
                [reshaped_predictions],
                [shape],
            )
            model = merge_models(
                model,
                make_model(graph, opset_imports=[make_opsetid("", 21)]),
                io_map=[("transform_predictions_out", "reshape_predictions_in")],
            )

        check_model(model)
        return model

    def global_attributions(
        self,
        X: cn.array,
        y: cn.array,
        *,
        metric: Optional[BaseMetric] = None,
        random_state: Optional[np.random.RandomState] = None,
        n_samples: int = 5,
        check_efficiency: bool = False,
    ) -> Tuple[cn.array, cn.array]:
        r"""Compute global feature attributions for the model. Global attributions
        show the effect of a feature on a model's loss function.

        We use a Shapley value approach to compute the attributions:
        :math:`Sh_i(v)=\frac{1}{|N|!} \sum_{\sigma \in \mathfrak{S}_d} \big[ v([\sigma]_{i-1} \cup\{i\}) - v([\sigma]_{i-1}) \big],`
        where :math:`v` is the model's loss function, :math:`N` is the set of features, and :math:`\mathfrak{S}_d` is the set of all permutations of the features.
        :math:`[\sigma]_{i-1}` represents the set of players ranked lower than :math:`i` in the ordering :math:`\sigma`.

        In effect the shapley value shows the effect of adding a feature to the model, averaged over all possible orderings of the features. In our case the above function is approximated using an antithetic-sampling method [#]_, where `n_samples` corresponds to pairs of permutation samples. This method also returns the standard error, which decreases according to :math:`1/\sqrt{n\_samples}`.

        This definition of attributions requires removing a feature from the active set. We use a random sample of values from X to fill in the missing feature values. This choice of background distribution corresponds to an 'interventional' Shapley value approach discussed in [#]_.


        .. [#] Mitchell, Rory, et al. "Sampling permutations for shapley value estimation." Journal of Machine Learning Research 23.43 (2022): 1-46.
        .. [#] Covert, Ian, Scott M. Lundberg, and Su-In Lee. "Understanding global feature contributions with additive importance measures." Advances in Neural Information Processing Systems 33 (2020): 17212-17223.

        The method uses memory (and time) proportional to :math:`n\_samples \times n\_features \times n\_background\_samples`. Reduce the number of background samples or the size of X to speed up computation and reduce memory usage. X does not need to be the entire training set to get useful estimates.

        See the method :func:`~legateboost.BaseModel.local_attributions` for the effect of features on individual prediction outputs.

        Parameters
        ----------
        X : cn.array
            The input data.
        y : cn.array
            The target values.
        metric : BaseMetric, optional
            The metric to evaluate the model. If None, the model default metric is used.
        random_state : int, optional
            The random state for reproducibility.
        n_samples : int, optional
            The number of sample pairs to use in the antithetic sampling method.
        check_efficiency : bool, optional
            If True, check that shapley values + null coalition add up to the final loss for X, y (the so called efficiency property of Shapley values)'.

        Returns
        -------
        cn.array
            The Shapley value estimates for each feature. The last value is the null coalition loss. The sum of this array results in the loss for X, y.
        cn.array
            The standard error of the Shapley value esimates, with respect to `n_samples`. The standard error decreases according to :math:`1/\sqrt{n\_samples}`.
        """  # noqa: E501
        check_is_fitted(self, "is_fitted_")
        return global_shapley_attributions(
            self,
            X,
            y,
            metric,
            random_state,
            n_samples,
            check_efficiency,
        )

    def local_attributions(
        self,
        X: cn.array,
        X_background: cn.array,
        *,
        random_state: Optional[np.random.RandomState] = None,
        n_samples: int = 5,
        check_efficiency: bool = False,
    ) -> Tuple[cn.array, cn.array]:
        r"""Local feature attributions for model predictions. Shows the effect of a
        feature on each output prediction. See the definition of Shapley values in
        :func:`~legateboost.BaseModel.global_attributions`, where the :math:`v`
        function is here the model prediction instead of the loss function.

        Parameters
        ----------
        X : cn.array
            The input data.
        X_background : cn.array
            The background data to use for missing feature values. This could be a random sample of training data (e.g. between 10-100 instances).
        random_state : int, optional
            The random state for reproducibility.
        n_samples : int
            The number of sample pairs to use in the antithetic sampling method.
        check_efficiency : bool
            If True, check that shapley values + null prediction add up to the final predictions for X (the so called efficiency property of Shapley values).


        Returns
        -------
        cn.array
            The Shapley value estimates for each feature. The final value is the 'null prediction', where all features are turned off. The sum of this array results in the model prediction.
        cn.array
            The standard error of the Shapley value esimates, with respect to `n_samples`. The standard error decreases according to :math:`1/\sqrt{n\_samples}`.
        """  # noqa: E501
        check_is_fitted(self, "is_fitted_")
        return local_shapley_attributions(
            self,
            X,
            X_background,
            random_state,
            n_samples,
            check_efficiency,
        )


class LBRegressor(RegressorMixin, LBBase):
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
    subsample :
        The fraction of samples to be used for fitting the individual base models.
    init :
        The initial prediction of the model. If `None`, the initial prediction
        is zero. If 'average', the initial prediction minimises a second order
        approximation of the loss-function (simply the mean label in the case of
        regression).
    base_models :
        The base models to use for each iteration. The model used in each iteration
        i is base_models[i % len(base_models)].
    callbacks :
        List of callbacks to apply during training e.g. early stopping.
        See `callbacks` module for more information.
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
    >>> import cupynumeric as cn
    >>> import legateboost as lb
    >>> X = cn.random.random((1000, 10))
    >>> y = cn.random.random(X.shape[0])
    >>> model = lb.LBRegressor(n_estimators=5).fit(X, y)
    >>>
    """

    def __init__(
        self,
        *,
        n_estimators: int = 100,
        objective: Union[str, BaseObjective] = "squared_error",
        metric: Union[str, BaseMetric, list[Union[str, BaseMetric]]] = "default",
        learning_rate: float = 0.1,
        subsample: float = 1.0,
        init: Union[str, None] = "average",
        base_models: Tuple[BaseModel, ...] = (Tree(max_depth=3),),
        callbacks: Sequence[TrainingCallback] = (),
        verbose: int = 0,
        random_state: Optional[np.random.RandomState] = None,
    ) -> None:
        super().__init__(
            n_estimators=n_estimators,
            objective=objective,
            metric=metric,
            learning_rate=learning_rate,
            subsample=subsample,
            init=init,
            base_models=base_models,
            callbacks=callbacks,
            verbose=verbose,
            random_state=random_state,
        )

    def __sklearn_tags__(self) -> Tags:
        tags = super().__sklearn_tags__()
        tags.target_tags.multi_output = True
        return tags

    def partial_fit(
        self,
        X: cn.ndarray,
        y: cn.ndarray,
        *,
        sample_weight: cn.ndarray = None,
        eval_set: List[Tuple[cn.ndarray, ...]] = [],
        eval_result: EvalResult = {},
    ) -> LBBase:
        """This method is used for incremental (online) training of the model. An
        additional `n_estimators` models will be added to the ensemble.

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
        if not hasattr(self, "is_fitted_"):
            return self.fit(
                X,
                y,
                sample_weight=sample_weight,
                eval_set=eval_set,
                eval_result=eval_result,
            )
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
        *,
        sample_weight: cn.ndarray = None,
        eval_set: List[Tuple[cn.ndarray, ...]] = [],
        eval_result: EvalResult = {},
    ) -> "LBRegressor":
        X, y = _lb_check_X_y(X, y)
        validate_data(self, X, y, skip_check_array=True)
        return super().fit(
            X,
            y,
            sample_weight=sample_weight,
            eval_set=eval_set,
            eval_result=eval_result,
        )

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
        X = _lb_check_X(X)
        validate_data(self, X, reset=False, skip_check_array=True)
        check_is_fitted(self, "is_fitted_")
        pred = self._objective_instance.transform(super()._predict(X))
        if pred.shape[1] == 1:
            pred = pred.squeeze(axis=1)
        return pred


class LBClassifier(ClassifierMixin, LBBase):
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
    subsample :
        The fraction of samples to be used for fitting the individual base models.
    init :
        The initial prediction of the model. If `None`, the initial prediction
        is zero. If 'average', the initial prediction minimises a second order
        approximation of the loss-function.
    base_models:
        The base models to use for each iteration. The model used in each iteration
        i is base_models[i % len(base_models)].
    callbacks :
        List of callbacks to apply during training e.g. early stopping.
        See `callbacks` module for more information.
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
    >>> import cupynumeric as cn
    >>> import legateboost as lb
    >>> X = cn.random.random((1000, 10))
    >>> y = cn.random.randint(0, 2, X.shape[0])
    >>> model = lb.LBClassifier(n_estimators=5).fit(X, y)
    >>>
    """

    def __init__(
        self,
        *,
        n_estimators: int = 100,
        objective: Union[str, BaseObjective] = "log_loss",
        metric: Union[str, BaseMetric, list[Union[str, BaseMetric]]] = "default",
        learning_rate: float = 0.1,
        subsample: float = 1.0,
        init: Union[str, None] = "average",
        base_models: Tuple[BaseModel, ...] = (Tree(max_depth=3),),
        callbacks: Sequence[TrainingCallback] = (),
        verbose: int = 0,
        random_state: Optional[np.random.RandomState] = None,
    ) -> None:
        super().__init__(
            n_estimators=n_estimators,
            objective=objective,
            metric=metric,
            learning_rate=learning_rate,
            subsample=subsample,
            init=init,
            base_models=base_models,
            callbacks=callbacks,
            verbose=verbose,
            random_state=random_state,
        )
        # two models cannot be added if they have different classes
        self._add_behaviour.update({"classes_": AddMember.ASSERT_SAME})

    def partial_fit(
        self,
        X: cn.ndarray,
        y: cn.ndarray,
        *,
        classes: Optional[cn.ndarray] = None,
        sample_weight: cn.ndarray = None,
        eval_set: List[Tuple[cn.ndarray, ...]] = [],
        eval_result: EvalResult = {},
    ) -> LBBase:
        """This method is used for incremental fitting on a batch of samples.
        Requires the classes to be provided up front, as they may not be inferred
        from the first batch.

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

        if not hasattr(self, "is_fitted_"):
            return self.fit(
                X,
                y,
                sample_weight=sample_weight,
                eval_set=eval_set,
                eval_result=eval_result,
            )

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
        *,
        sample_weight: cn.ndarray = None,
        eval_set: List[Tuple[cn.ndarray, ...]] = [],
        eval_result: EvalResult = {},
    ) -> "LBClassifier":
        if hasattr(y, "ndim") and y.ndim > 1:
            warnings.warn(
                "A column-vector y was passed when a 1d array was expected.",
                DataConversionWarning,
            )
        X, y = _lb_check_X_y(X, y)
        validate_data(self, X, y, skip_check_array=True)

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

    def predict_proba(self, X: cn.ndarray) -> cn.ndarray:
        """Predict class probabilities for samples in X.

        Parameters
        ----------

        X :
            The input samples.

        Returns
        -------

        probabilities:
            The predicted class probabilities for each sample in X.
        """
        X = _lb_check_X(X)
        validate_data(self, X, reset=False, skip_check_array=True)
        check_is_fitted(self, "is_fitted_")
        pred = self._objective_instance.transform(super()._predict(X))
        if pred.shape[1] == 1:
            pred = pred.reshape(-1)
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
        check_is_fitted(self)
        return self._objective_instance.output_class(self.predict_proba(X))
