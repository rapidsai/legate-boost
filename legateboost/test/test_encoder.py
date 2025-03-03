import numpy as np
import pytest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import TargetEncoder
from sklearn.utils.estimator_checks import check_estimator

import cupynumeric as cn
import legateboost as lb


class TestTargetEncoder:
    def test_without_cv(self):
        encoder = lb.encoder.TargetEncoder(target_type="binary", smooth=0.0)
        sklearn_encoder = TargetEncoder(smooth=0.0)
        X = cn.array([[0.0], [0.0], [1.0]])
        y = cn.array([0, 1, 0])
        X_encoded_sklearn = sklearn_encoder.fit(X, y).transform(X)
        X_encoded = encoder.fit(X, y).transform(X)
        assert X_encoded.shape == X.shape
        assert cn.all(X_encoded == X_encoded_sklearn)
        assert cn.all(X_encoded == cn.array([[0.5], [0.5], [0.0]]))

        encoder = lb.encoder.TargetEncoder(target_type="binary", smooth=1.0)
        sklearn_encoder = TargetEncoder(smooth=1.0)
        X_encoded_sklearn = sklearn_encoder.fit(X, y).transform(X)
        X_encoded = encoder.fit(X, y).transform(X)
        assert cn.all(
            X_encoded
            == cn.array(
                [[(1 + 1.0 / 3) / 3], [(1 + 1.0 / 3) / 3], [(0.0 + 1.0 / 3) / 2]]
            )
        )
        assert cn.all(X_encoded == X_encoded_sklearn)

    def test_multiclass(self):
        X = cn.array([[0.0], [0.0], [1.0]])
        encoder = lb.encoder.TargetEncoder(target_type="multiclass", smooth=0.0)
        y = cn.array([2, 1, 2])
        X_encoded = encoder.fit(X, y).transform(X)
        assert X_encoded.shape == (3, 3)
        assert cn.all(
            X_encoded == cn.array([[0.0, 0.5, 0.5], [0.0, 0.5, 0.5], [0.0, 0.0, 1.0]])
        )

    def test_unseen_category(self):
        X = cn.array([[0.0], [0.0], [1.0]])
        encoder = lb.encoder.TargetEncoder(target_type="binary", smooth=0.0)
        y = cn.array([0, 1, 0])
        X_encoded = encoder.fit(X, y).transform(cn.array([[2.0]]))
        assert cn.all(X_encoded == y.mean())

    def test_cv(self):
        encoder = lb.encoder.TargetEncoder(target_type="binary", smooth=0.0, cv=0)
        X = cn.array([[0.0], [0.0], [1.0]])
        y = cn.array([0, 1, 0])
        # with 0 cv folds should be the same as without cv
        X_encoded = encoder.fit(X, y).transform(X)
        X_encoded_fit_transform = encoder.fit_transform(X, y)
        assert cn.all(X_encoded == X_encoded_fit_transform)

    def test_sklearn_api(self):
        expected_failed_checks = {
            "check_dtype_object": "legate-boost does not " "support non-numeric inputs",
            "check_transformer_data_not_an_array": "legate-boost"
            " does not support non-numeric inputs",
            "check_positive_only_tag_during_fit": "This test " "fails on a numpy error",
        }
        check_estimator(
            lb.encoder.TargetEncoder(target_type="binary", cv=0),
            expected_failed_checks=expected_failed_checks,
        )

    @pytest.mark.parametrize(
        "y, target_type",
        [(cn.array([3.4] * 20), "continuous"), (cn.array([0] * 20), "binary")],
        ids=["continuous", "binary"],
    )
    @pytest.mark.parametrize("smooth", ["auto", 4.0, 0.0])
    def test_constant_target_and_feature(self, y, target_type, smooth):
        X = cn.full((20, 1), 1)
        y_mean = y.mean()
        enc = lb.encoder.TargetEncoder(
            cv=2, smooth=smooth, random_state=0, target_type=target_type
        )
        X_trans = enc.fit_transform(X, y)
        assert cn.allclose(X_trans, cn.full(X.shape, y_mean))
        assert cn.allclose(enc.encodings_[0][0], y_mean)
        assert cn.allclose(enc.target_mean_[0], y_mean)

        X_test = cn.array([[1], [0]])
        X_test_trans = enc.transform(X_test)
        assert cn.allclose(X_test_trans, cn.full(X_test.shape, y_mean))

    def test_shuffle_false(self):
        # cv=2 creates splits of length 6 and 5
        X = cn.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]).reshape(-1, 1)
        y = cn.array([2.1, 4.3, 1.2, 3.1, 1.0, 9.0, 10.3, 14.2, 13.3, 15.0, 12.0])
        enc = lb.encoder.TargetEncoder(
            smooth=0.0, shuffle=False, cv=2, target_type="continuous"
        )
        X_trans = enc.fit_transform(X, y)
        # cv without shuffle means the first half should have the mean
        # of the second half and vice versa
        assert cn.isclose(X_trans[0, 0], cn.mean(y[6:]))
        assert cn.isclose(X_trans[-1, 0], cn.mean(y[:6]))

    def test_mean_variance_tasks(self):
        X = cn.array([0, 0, 1, 1, 1, 0, 0, 1, 1, 1], dtype=cn.float64).reshape(-1, 1)
        y = cn.array([0, 1, 0, 0, 1, 0, 0, 1, 1, 1], dtype=cn.float64).reshape(-1, 1)
        enc = lb.encoder.TargetEncoder(
            target_type="binary", cv=2, random_state=0, shuffle=False
        )
        enc.fit(X, y)
        from legateboost.encoder import KFold

        # fold 0: train instances 5, 6, 7, 8, 9
        sums_counts = enc._get_category_means(X, y, KFold(0, 2, shuffle=False), 0)
        sums = sums_counts[:, 0, 0]
        counts = sums_counts[:, 0, 1]
        assert cn.allclose(sums, cn.array([0, 3]))
        assert cn.allclose(counts, cn.array([2, 3]))

        y_mean = sums_counts[:, :, 0].sum(axis=0) / sums_counts[:, :, 1].sum(axis=0)
        variances, y_variance = enc._get_category_variances(
            X, y, sums_counts, y_mean, KFold(0, 2, shuffle=False), 0
        )
        assert cn.isclose(y_variance[0], y[5:].var())
        assert cn.allclose(variances[:, 0], cn.array([0.0, 0.0]))

        # fold 1: train instances 0, 1, 2, 3, 4
        sums_counts = enc._get_category_means(X, y, KFold(0, 2, shuffle=False), 1)
        sums = sums_counts[:, 0, 0]
        counts = sums_counts[:, 0, 1]
        assert cn.allclose(sums, cn.array([1, 1]))
        assert cn.allclose(counts, cn.array([2, 3]))

        y_mean = sums_counts[:, :, 0].sum(axis=0) / sums_counts[:, :, 1].sum(axis=0)
        variances, y_variance = enc._get_category_variances(
            X, y, sums_counts, y_mean, KFold(0, 2, shuffle=False), 1
        )
        assert cn.isclose(y_variance[0], y[:5].var())
        assert cn.allclose(variances[:, 0], cn.array([y[0:2].var(), y[2:5].var()]))

    @pytest.mark.parametrize("smooth", [0.0, 1e3, "auto"])
    def test_invariance_of_encoding_under_label_permutation(self, smooth):
        # this test adapted directly from sklearn TargetEncoder's tests
        # randomly reassign the X categories and check the encoding is invariant
        rng = np.random.RandomState(0)
        y = rng.normal(size=1000)
        n_categories = 10
        X = rng.randint(0, n_categories, size=y.shape[0]).reshape(-1, 1)

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=3)

        permutated_labels = rng.permutation(n_categories)
        X_train_permuted = permutated_labels[X_train]
        X_test_permuted = permutated_labels[X_test]

        target_encoder = lb.encoder.TargetEncoder(
            smooth=smooth, random_state=0, target_type="continuous"
        )
        X_train_encoded = target_encoder.fit_transform(X_train, y_train)
        X_test_encoded = target_encoder.transform(X_test)

        X_train_permuted_encoded = target_encoder.fit_transform(
            X_train_permuted, y_train
        )
        X_test_permuted_encoded = target_encoder.transform(X_test_permuted)

        assert cn.allclose(X_train_encoded, X_train_permuted_encoded)
        assert cn.allclose(X_test_encoded, X_test_permuted_encoded)
