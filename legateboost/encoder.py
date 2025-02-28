from dataclasses import dataclass

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import Tags, check_random_state
from sklearn.utils.validation import check_is_fitted

import cupynumeric as cn
from legate.core import get_legate_runtime, types

from .input_validation import _lb_check_X, check_array
from .library import user_context, user_lib
from .utils import PickleCupynumericMixin, get_store

__all__ = ["TargetEncoder", "KFold"]


# This uniquely defines a cross-validation split
# It is passed to the legate tasks which call something like
# fold = rng(seed).advance(row_idx).rantint(0, n_folds)
# thus each instance is assigned to a fold
# It differs from the sklearn KFold, which shuffles the data
# The size of each of our folds can vary slightly, but still
# each instance has equal chance of being assigned to each fold
# If n_folds is set to 0, the test and train sets both contain
# all instances
@dataclass
class KFold:
    seed: np.int64
    n_folds: int


class TargetEncoder(TransformerMixin, BaseEstimator, PickleCupynumericMixin):
    def __init__(
        self,
        target_type: str,
        smooth: float = 1.0,
        cv=5,
        shuffle=True,
        random_state=None,
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
    def _maybe_expand_labels(self, y):
        if self.target_type == "multiclass":
            # expand y to one-hot encoding
            n_outputs = y.max() + 1
            range = cn.arange(n_outputs)
            # one hot encode labels
            return y[:, cn.newaxis] == range[cn.newaxis, :].astype(y.dtype)
        if len(y.shape) == 1:
            return y.reshape(-1, 1)
        return y

    # fit on all data
    def fit(self, X, y):
        if y is None:
            raise ValueError("requires y to be passed, but the target y is None")
        self.random_state_ = check_random_state(self.random_state)
        X = _lb_check_X(X)
        y = check_array(y)
        self.n_features_in_ = X.shape[1]
        self.categories_ = []
        self.encodings_ = []
        for column in X.T:
            self.categories_.append(cn.unique(column))
        self.categories_sparse_matrix_ = cn.concatenate(self.categories_)
        self.categories_row_pointers_ = cn.cumsum(
            cn.array([0] + [len(c) for c in self.categories_]), dtype=cn.int64
        )

        y_ = self._maybe_expand_labels(y)
        # no cross validation
        self.encodings_, self.target_mean_ = self._get_encoding(
            X, y_, KFold(seed=0, n_folds=0), 0
        )
        return self

    # fit on cv splits
    def fit_transform(self, X, y):
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

        cv = KFold(seed=self.random_state_.randint(0, 2**32), n_folds=self.cv)
        for fold_idx in range(self.cv):
            encoding, y_mean = self._get_encoding(X, y_, cv, fold_idx)
            self._transform_X(X, X_out, encoding, cv, fold_idx, y_mean)

        return X_out.reshape(X.shape[:-1] + (-1,))

    def transform(self, X):
        X = _lb_check_X(X)
        check_is_fitted(self)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                "X has {} features, but TargetEncoder "
                " is expecting {} features as input".format(
                    X.shape[1], self.n_features_in_
                )
            )
        X_out = cn.empty(
            (X.shape[0], X.shape[1], self.encodings_[0].shape[0]),
            dtype=X.dtype,
        )
        self._transform_X(
            X, X_out, self.encodings_, KFold(seed=0, n_folds=0), 0, self.target_mean_
        )
        return X_out.reshape(X.shape[:-1] + (-1,))

    def _transform_X(self, X_in, X_out, encodings, cv: KFold, fold: int, y_mean):
        task = get_legate_runtime().create_auto_task(
            user_context, user_lib.cffi.TARGET_ENCODER_ENCODE
        )
        # inputs
        task.add_scalar_arg(cv.seed, types.int64)
        task.add_scalar_arg(cv.n_folds, types.int64)
        task.add_scalar_arg(fold, types.int64)

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

        task.execute()
        return X_out

    def _get_category_means(self, X, y, cv: KFold, fold: int):
        """Compute some label summary statistics for each category in the input
        data.

        Returns a 3D array of shape (n_categories, n_outputs, 2)
        containing the sum, count of the labels for each category.
        """
        task = get_legate_runtime().create_auto_task(
            user_context, user_lib.cffi.TARGET_ENCODER_MEAN
        )
        # inputs
        task.add_scalar_arg(cv.seed, types.int64)
        task.add_scalar_arg(cv.n_folds, types.int64)
        task.add_scalar_arg(fold, types.int64)
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
        task.execute()
        return means

    def _get_category_variances(self, X, y, means, cv: KFold, fold: int):
        return None, None

    def _get_encoding(self, X, y, cv: KFold, fold: int):
        means = self._get_category_means(X, y, cv, fold)
        y_mean = means[:, :, 0].sum(axis=0) / means[:, :, 1].sum(axis=0)
        if self.smooth != "auto":
            sums = means[:, :, 0]
            counts = means[:, :, 1]
            encoding = (sums + self.smooth * y_mean) / (counts + self.smooth)
            zero_count = counts[:, 0] == 0
            encoding[zero_count] = y_mean
            return encoding, y_mean
        else:
            variances, y_variance = self._get_category_variances(X, y, means)
            sums = means[:, :, 0]
            counts = means[:, :, 1]
            lambda_ = (y_variance * counts) / (y_variance * counts + variances)
            means = sums / counts
            encoding = lambda_ * means + (1 - lambda_) * y_mean
            nans = cn.isnan(lambda_)
            encoding[nans] = y_mean
            return encoding, y_mean
