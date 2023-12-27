import cunumeric as cn
from legateboost.utils import sample_average


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
