from sklearn.preprocessing import TargetEncoder
from sklearn.utils.estimator_checks import check_estimator

import cupynumeric as cn
import legateboost as lb


def test_kfold():

    def all_data_used(train, test, X):
        used = set(train) ^ set(test)
        for i in X:
            if i not in used:
                return False
        return True

    cv = lb.encoder.KFold(n_splits=5, shuffle=True, random_state=0)
    X = cn.arange(5)
    for train, test in cv.split(X):
        assert len(train) == 4
        assert len(test) == 1
        assert not cn.any(cn.in1d(X[train], X[test]))
        assert not cn.any(cn.in1d(X[test], X[train]))
        assert all_data_used(train, test, X)

    X = cn.arange(2)
    for train, test in cv.split(X):
        assert all_data_used(train, test, X)

    X = cn.arange(7)
    for train, test in cv.split(X):
        assert all_data_used(train, test, X)


def test_kfold_seed():
    cv0 = lb.encoder.KFold(n_splits=5, shuffle=True, random_state=0)
    cv1 = lb.encoder.KFold(n_splits=5, shuffle=True, random_state=0)
    for (train0, test0), (train1, test1) in zip(
        cv0.split(cn.arange(10)), cv1.split(cn.arange(10))
    ):
        assert cn.all(train0 == train1)
        assert cn.all(test0 == test1)


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

    def test_sklearn_api(self):
        expected_failed_checks = {
            "check_dtype_object": "legate-boost does not " "support non-numeric inputs",
            "check_transformer_data_not_an_array": "legate-boost"
            " does not support non-numeric inputs",
            "check_positive_only_tag_during_fit": "This test " "fails on a numpy error",
        }
        check_estimator(
            lb.encoder.TargetEncoder(target_type="binary", cv=3),
            expected_failed_checks=expected_failed_checks,
        )
