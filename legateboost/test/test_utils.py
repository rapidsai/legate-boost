import numpy as np
from scipy import optimize

import cunumeric as cn
from legateboost.utils import lbfgs, sample_average


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

    rs = np.random.RandomState(0)
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
