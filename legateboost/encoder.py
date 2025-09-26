from typing import Optional, Tuple, Union

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import Tags, check_random_state
from sklearn.utils.validation import check_is_fitted

import cupynumeric as cn
from legate.core import get_legate_runtime, types

from .input_validation import _lb_check_X, check_array
from .library import user_context, user_lib
from .utils import PickleCupynumericMixin, get_store

__all__ = ["TargetEncoder"]


class TargetEncoder(TransformerMixin, BaseEstimator, PickleCupynumericMixin):
    """TargetEncoder is a transformer that encodes categorical features using
    the mean of the target variable. When `fit_transform` is called, a cross-
    validation procedure is used to generate encodings for each training fold,
    which are then applied to the test fold. `fit().transform()` differs from
    `fit_transform()` in that the former fits the encoder on all the data and
    generates encodings for each feature. This encoder is modelled on the
    sklearn TargetEncoder with only minor differences in how the CV folds are
    generated. As it is difficult to rearrange and gather data from each fold
    in distributed environment, training rows are kept in place and then
    assigned a cv fold by generating a random integer in the range [0,
    n_folds). As per sklearn, when smooth="auto", an empirical Bayes estimate
    per [#]_ is used to avoid overfitting.

    .. [#] Micci-Barreca, Daniele. "A preprocessing scheme for high-cardinality categorical attributes in classification and prediction problems." ACM SIGKDD explorations newsletter 3.1 (2001): 27-32.

    Parameters
    ----------
    target_type : str
        The type of target variable. Must be one of {"continuous", "binary", "multiclass"}.
    smooth : float, default=1.0
        Smoothing parameter to avoid overfitting. If "auto", the smoothing parameter is determined automatically.
    cv : int, default=5
        Number of cross-validation folds. If 0, no cross-validation is performed.
    shuffle : bool, default=True
        Whether to shuffle the data before splitting into folds.
    random_state : int or None, default=None
        Seed for the random number generator.

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during fit.
    categories_ : list of arrays
        List of unique categories for each feature.
    categories_sparse_matrix_ : array
        Concatenated array of unique categories for all features.
    categories_row_pointers_ : array
        Array of row pointers for the concatenated categories array.
    encodings_ : list of arrays
        List of encoding arrays for each feature.
    target_mean_ : array
        Mean of the target variable.
    """  # noqa: E501

    def __init__(
        self,
        target_type: str,
        smooth: Union[str, float] = "auto",
        cv: int = 5,
        shuffle: bool = True,
        random_state: Optional[np.random.RandomState] = None,
    ):
        self.smooth = smooth
        self.target_type = target_type
        self.cv = cv
        self.shuffle = shuffle
        self.random_state = random_state

    def __sklearn_tags__(self) -> Tags:
        tags = super().__sklearn_tags__()
        tags.input_tags.categorical = True
        tags.input_tags.allow_nan = False
        tags.target_tags.required = True
        tags.input_tags.string = False
        return tags

    # integer multiclass labels must be one-hot encoded
    # force 2-d y
    def _maybe_expand_labels(self, y: cn.ndarray) -> cn.ndarray:
        if self.target_type == "multiclass":
            # expand y to one-hot encoding
            n_outputs = y.max() + 1
            range = cn.arange(n_outputs)
            # one hot encode labels
            return y[:, cn.newaxis] == range[cn.newaxis, :].astype(y.dtype)
        if len(y.shape) == 1:
            return y.reshape(-1, 1)
        return y

    def fit(self, X: cn.array, y: cn.array) -> "TargetEncoder":
        """Fit the encoder to the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        y : array-like of shape (n_samples,)
            The target values. Cannot be None.

        Returns
        -------
        self : object
            Fitted encoder.

        Raises
        ------
        ValueError
            If the target `y` is None.
        """
        if y is None:
            raise ValueError("requires y to be passed, but the target y is None")
        self.random_state_ = check_random_state(self.random_state)
        X = _lb_check_X(X)
        y = check_array(y)
        self.n_features_in_ = X.shape[1]
        self.categories_ = []
        for column in X.T:
            self.categories_.append(cn.unique(column))
        self.categories_sparse_matrix_ = cn.concatenate(self.categories_)
        self.categories_row_pointers_ = cn.cumsum(
            cn.array([0] + [len(c) for c in self.categories_]), dtype=cn.int64
        )

        y_ = self._maybe_expand_labels(y)
        # no cross validation
        self.encodings_, self.target_mean_ = self._get_encoding(X, y_, None, 0)
        return self

    def fit_transform(self, X: cn.array, y: cn.array) -> cn.array:
        X = _lb_check_X(X)
        y = check_array(y)
        # fit on all of the data first
        # to get the target mean, categories
        # and an encoding over the entire dataset
        # this all-data encoding gets used for transform
        self.fit(X, y)

        # for the output of this function however we generate new encodings
        # for each fold to reduce overfitting
        y_ = self._maybe_expand_labels(y)
        X_out = cn.empty(
            (X.shape[0], X.shape[1], y_.shape[1]),
            dtype=X.dtype,
        )

        if self.cv == 0:
            return self.transform(X)

        if self.shuffle:
            # assign each instance to a fold
            seed = self.random_state_.randint(0, 2**32)
            # need to seed both due to
            # https://github.com/nv-legate/cupynumeric/issues/964
            np.random.seed(seed)
            cn.random.seed(seed)
            cv_indices = cn.random.randint(0, self.cv, size=X.shape[0], dtype=cn.int64)
        else:
            fold_size = int(np.ceil(X.shape[0] / self.cv))
            cv_indices = cn.arange(X.shape[0]) // fold_size
        for fold_idx in range(self.cv):
            encoding, y_mean = self._get_encoding(X, y_, cv_indices, fold_idx)
            self._transform_X(X, X_out, encoding, cv_indices, fold_idx, y_mean)

        return X_out.reshape(X.shape[:-1] + (-1,))

    def transform(self, X: cn.array) -> cn.array:
        """Transforms the input data X using the target encoding.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data to transform.

        Returns
        -------
        X_out : ndarray of shape (n_samples, n_features * encoding_dim)
            The transformed data with target encoding applied.

        Raises
        ------
        ValueError
            If the number of features in X does not match the number of features
            the encoder was fitted with.
        """
        X = _lb_check_X(X)
        check_is_fitted(self)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                "X has {} features, but TargetEncoder"
                " is expecting {} features as input".format(
                    X.shape[1], self.n_features_in_
                )
            )
        X_out = cn.empty(
            (X.shape[0], X.shape[1], self.encodings_[0].shape[0]),
            dtype=X.dtype,
        )
        self._transform_X(
            X,
            X_out,
            self.encodings_,
            None,
            0,
            self.target_mean_,
        )
        return X_out.reshape(X.shape[:-1] + (-1,))

    # if cv_indices is None, use all data
    def _transform_X(
        self,
        X_in: cn.array,
        X_out: cn.array,
        encodings: cn.array,
        cv_indices: Optional[cn.array],
        cv_fold_idx: int,
        y_mean: cn.array,
    ) -> cn.array:
        task = get_legate_runtime().create_auto_task(
            user_context, user_lib.cffi.TARGET_ENCODER_ENCODE
        )
        # inputs
        task.add_scalar_arg(cv_fold_idx, types.int64)
        do_cv = cv_indices is not None
        task.add_scalar_arg(do_cv, types.bool_)

        X_in_ = get_store(X_in).promote(2, X_out.shape[2])
        task.add_input(X_in_)
        task.add_input(get_store(self.categories_sparse_matrix_))
        task.add_input(get_store(self.categories_row_pointers_))
        task.add_broadcast(get_store(self.categories_sparse_matrix_))
        task.add_broadcast(get_store(self.categories_row_pointers_))
        task.add_input(get_store(encodings))
        task.add_broadcast(get_store(encodings))
        task.add_input(get_store(y_mean))
        task.add_broadcast(get_store(y_mean))

        # output
        task.add_output(get_store(X_out))
        task.add_alignment(X_in_, get_store(X_out))

        if do_cv:
            cv_indices_ = (
                get_store(cv_indices)
                .promote(1, X_in.shape[1])
                .promote(2, X_out.shape[2])
            )
            task.add_input(cv_indices_)
            task.add_alignment(cv_indices_, get_store(X_out))

        task.execute()
        return X_out

    def _get_category_means(
        self, X: cn.array, y: cn.array, cv_indices: Optional[cn.array], cv_fold_idx: int
    ) -> cn.array:
        """Compute some label summary statistics for each category in the input
        data.

        Returns a 3D array of shape (n_categories, n_outputs, 2)
        containing the sum, count of the labels for each category.
        """
        task = get_legate_runtime().create_auto_task(
            user_context, user_lib.cffi.TARGET_ENCODER_MEAN
        )
        # inputs
        task.add_scalar_arg(cv_fold_idx, types.int64)
        do_cv = cv_indices is not None
        task.add_scalar_arg(do_cv, types.bool_)
        X_ = get_store(X).promote(2, y.shape[1])
        y_ = get_store(y.astype(X.dtype, copy=False)).promote(1, X.shape[1])
        task.add_input(X_)
        task.add_input(y_)

        task.add_input(get_store(self.categories_sparse_matrix_))
        task.add_input(get_store(self.categories_row_pointers_))
        task.add_broadcast(get_store(self.categories_sparse_matrix_))
        task.add_broadcast(get_store(self.categories_row_pointers_))

        # output array contains label sums and counts for each category
        means = cn.zeros(
            (len(self.categories_sparse_matrix_), y.shape[1], 2), dtype=cn.float64
        )
        task.add_reduction(get_store(means), types.ReductionOpKind.ADD)

        task.add_alignment(X_, y_)
        if do_cv:
            cv_indices_ = (
                get_store(cv_indices).promote(1, X.shape[1]).promote(2, y.shape[1])
            )
            task.add_input(cv_indices_)
            task.add_alignment(cv_indices_, X_)
        task.execute()
        return means

    def _get_category_variances(
        self,
        X: cn.array,
        y: cn.array,
        means: cn.array,
        y_mean: cn.array,
        cv_indices: Optional[cn.array],
        cv_fold_idx: int,
    ) -> Tuple[cn.array, cn.array]:
        # means is the sum, count of the labels for each category and output
        # y_mean is the mean of the labels for this train fold
        task = get_legate_runtime().create_auto_task(
            user_context, user_lib.cffi.TARGET_ENCODER_VARIANCE
        )
        # inputs
        # inputs
        task.add_scalar_arg(cv_fold_idx, types.int64)
        do_cv = cv_indices is not None
        task.add_scalar_arg(do_cv, types.bool_)
        task.add_scalar_arg(X.shape[0], types.int64)
        X_ = get_store(X).promote(2, y.shape[1])
        y_ = get_store(y.astype(X.dtype, copy=False)).promote(1, X.shape[1])
        task.add_input(X_)
        task.add_input(y_)

        task.add_input(get_store(self.categories_sparse_matrix_))
        task.add_input(get_store(self.categories_row_pointers_))
        task.add_broadcast(get_store(self.categories_sparse_matrix_))
        task.add_broadcast(get_store(self.categories_row_pointers_))
        task.add_input(get_store(means))
        task.add_broadcast(get_store(means))
        task.add_input(get_store(y_mean))
        task.add_broadcast(get_store(y_mean))

        # output array contains label sums and counts for each category
        variances_sum = cn.zeros(
            (len(self.categories_sparse_matrix_), y.shape[1]), dtype=cn.float64
        )
        y_variances_sum = cn.zeros(y.shape[1], dtype=cn.float64)
        task.add_reduction(get_store(variances_sum), types.ReductionOpKind.ADD)
        task.add_reduction(get_store(y_variances_sum), types.ReductionOpKind.ADD)
        task.add_alignment(X_, y_)
        if do_cv:
            cv_indices_ = (
                get_store(cv_indices).promote(1, X.shape[1]).promote(2, y.shape[1])
            )
            task.add_input(cv_indices_)
            task.add_alignment(cv_indices_, X_)
        task.execute()
        return variances_sum / means[:, :, 1], y_variances_sum / means[:, :, 1].sum()

    def _get_encoding(
        self, X: cn.array, y: cn.array, cv_indices: Optional[cn.array], cv_fold_idx: int
    ) -> Tuple[cn.array, cn.array]:
        means = self._get_category_means(X, y, cv_indices, cv_fold_idx)
        y_mean = means[:, :, 0].sum(axis=0) / means[:, :, 1].sum(axis=0)
        if self.smooth != "auto":
            sums = means[:, :, 0]
            counts = means[:, :, 1]
            encoding = (sums + self.smooth * y_mean) / (counts + self.smooth)
            zero_count = counts[:, 0] == 0
            encoding[zero_count] = y_mean
            return encoding, y_mean
        else:
            variances, y_variance = self._get_category_variances(
                X, y, means, y_mean, cv_indices, cv_fold_idx
            )
            sums = means[:, :, 0]
            counts = means[:, :, 1]
            lambda_ = (y_variance * counts) / (y_variance * counts + variances)
            means = sums / counts
            encoding = lambda_ * means + (1 - lambda_) * y_mean
            nans = cn.isnan(encoding)
            encoding[nans] = y_mean
            return encoding, y_mean
