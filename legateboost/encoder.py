import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import Tags, check_random_state
from sklearn.utils.validation import check_is_fitted

import cupynumeric as cn

from .input_validation import _lb_check_X, _lb_check_X_y

__all__ = ["TargetEncoder", "KFold"]


# simple cupynumeric implementation of sklearn kfold
class KFold:
    def __init__(self, n_splits=5, shuffle=True, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X):
        n_samples = X.shape[0]
        random_state = check_random_state(self.random_state)
        if self.shuffle:
            if self.random_state is not None:
                # annoyingly we have to also seed numpy
                # https://github.com/nv-legate/cupynumeric/issues/964
                seed = random_state.randint(0, 2**32 - 1)
                np.random.seed(seed)
                cn.random.seed(seed)
            # no permutation at the time of writing in cupynumeric
            indices = cn.random.randint(0, 2**63, size=n_samples).argsort()
        else:
            indices = cn.arange(n_samples)
        for i in range(self.n_splits):
            start = i * n_samples // self.n_splits
            end = (i + 1) * n_samples // self.n_splits
            yield cn.concatenate([indices[:start], indices[end:]]), indices[start:end]


class TargetEncoder(TransformerMixin, BaseEstimator):
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
        return tags

    # integer multiclass labels must be one-hot encoded
    # force 2-d y
    def _maybe_expand_labels(self, y):
        if self.target_type == "multiclass":
            # expand y to one-hot encoding
            n_outputs = y.max() + 1
            range = cn.arange(n_outputs)
            # one hot encode labels
            y_ = y[:, cn.newaxis] == range[cn.newaxis, :].astype(y.dtype)
        else:
            if len(y.shape) == 1:
                y_ = y.reshape(-1, 1)
        return y_

    # fit on all data
    def fit(self, X, y):
        _lb_check_X_y(X, y)
        self.n_features_in_ = X.shape[1]
        self.categories_ = []
        self.encodings_ = []
        for column in X.T:
            self.categories_.append(cn.unique(column))

        y_ = self._maybe_expand_labels(y)
        self.target_mean_ = y_.mean(axis=0)
        self.encodings_ = self._get_encoding(X, y_, self.target_mean_)
        return self

    # fit on cv splits
    def fit_transform(self, X, y):
        _lb_check_X_y(X, y)
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
        # unknown classes are encoded as the target mean
        X_out[:] = self.target_mean_

        cv = KFold(
            n_splits=self.cv, shuffle=self.shuffle, random_state=self.random_state
        )
        for train, test in cv.split(X):
            X_train, y_train = X[train], y_[train]
            encoding = self._get_encoding(X_train, y_train, self.target_mean_)
            self._transform_X(X, X_out, test, encoding)

        return X_out.reshape(X.shape[:-1] + (-1,))

    def transform(self, X):
        _lb_check_X(X)
        check_is_fitted(self)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but TargetEncoder"
                " is expecting {self.n_features_in_} features as input"
            )
        X_out = cn.empty(
            (X.shape[0], X.shape[1], self.encodings_[0][0].shape[0]),
            dtype=X.dtype,
        )
        # unknown classes are encoded as the target mean
        X_out[:] = self.target_mean_
        self._transform_X(X, X_out, cn.arange(X.shape[0]), self.encodings_)
        return X_out.reshape(X.shape[:-1] + (-1,))

    def _transform_X(self, X_in, X_out, row_indices, encodings):
        for (
            feature_idx,
            levels,
        ) in enumerate(self.categories_):
            for ordinal, level in enumerate(levels):
                mask = X_in[:, feature_idx] == level
                row_mask = cn.zeros(X_in.shape[0], dtype=bool)
                row_mask[row_indices] = True
                mask &= row_mask
                X_out[mask, feature_idx] = encodings[feature_idx][ordinal]
        return X_out

    def _get_encoding(self, X, y, target_mean):
        encodings = []
        for (
            feature_idx,
            levels,
        ) in enumerate(self.categories_):
            category_encodings = []
            for level in levels:
                mask = X[:, feature_idx] == level
                level_count = mask.sum()
                sum = y[mask, :].sum(axis=0)
                encoding = (sum + self.smooth * target_mean) / (
                    level_count + self.smooth
                )
                category_encodings.append(encoding)
            encodings.append(category_encodings)
        return encodings
