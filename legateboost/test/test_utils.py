import numpy as np
import pytest
from scipy import optimize

import cunumeric as cn
from legateboost.utils import gather, lbfgs, preround, sample_average


def test_sample_average() -> None:
    x = cn.array([1, 2, 3])
    w = cn.array([1, 1, 1])
    mean = sample_average(x, w)
    assert cn.isclose(mean, cn.array([2.0]))

    mean = sample_average(x)
    assert cn.isclose(mean, cn.array([2.0]))

    x = cn.array([[1, 2, 3], [1, 2, 3]]).T
    mean = sample_average(x)
    assert cn.isclose(mean, cn.array([2.0, 2.0])).all()

    mean = sample_average(x, w)
    assert cn.isclose(mean, cn.array([2.0, 2.0])).all()


def test_lbfgs():
    def f(x):
        return optimize.rosen(x), optimize.rosen_der(x)

    rs = np.random.RandomState(1)
    test_points = [rs.normal(size=3) for _ in range(10)]
    for x in test_points:
        result = lbfgs(x, f, max_iter=100)
        assert result.eval < 1e-5
        assert result.norm < 1e-5
        assert result.num_iter < 100
        # The number of function evaluations should be not too
        # much larger than the number of iterations.
        # If not, the line search is inefficient
        assert result.feval < 200
        assert cn.allclose(result.x, cn.array([1.0, 1.0, 1.0]))


@pytest.mark.parametrize("dtype", [cn.float32, cn.float64])
def test_gather(dtype):
    X = cn.array([[1, 2, 3], [4, 5, 6]], dtype=dtype)

    def check_gather(X, rows):
        a = gather(X, rows)
        b = X[rows]
        assert a.dtype == b.dtype
        assert a.shape == b.shape
        assert (a == b).all()

    check_gather(X, cn.array([0, 1]))
    check_gather(X, cn.array([1, 0]))

    X = cn.array([[1]])
    check_gather(X, cn.array([0]))

    rs = np.random.RandomState(1)
    X = cn.array(rs.randn(1000, 100).astype(dtype))
    rows = cn.array(rs.randint(0, 1000, size=100))
    check_gather(X, rows)


@pytest.mark.parametrize("dtype", [cn.float32, cn.float64])
def test_preround(dtype):
    rng = np.random.RandomState(1)
    xs = [cn.array(rng.randn(10000, 10).astype(dtype)) for _ in range(3)]
    convential_sums = [x.sum() for x in xs]
    sums = []
    for _ in range(10):
        sums.append([x.sum() for x in preround(xs)])
    assert np.all(s == sums[0] for s in sums)
    assert np.isclose(convential_sums, sums[0]).all()
