import pytest

import cunumeric as cn
import legateboost as lb


def test_normal():
    obj = lb.NormalObjective()
    y = cn.array([[1.0], [2.0], [3.0]])
    init = obj.initialise_prediction(y, cn.array([1.0, 1.0, 1.0]), True)
    assert cn.allclose(init, cn.array([y.mean(), y.var()]))


def test_log_loss():
    obj = lb.LogLossObjective()
    # binary
    g, h = obj.gradient(cn.array([[1], [0]]), cn.array([[0.5], [0.5]]))
    assert cn.allclose(g, cn.array([[-0.5], [0.5]]))
    assert cn.allclose(h, cn.array([[0.25], [0.25]]))

    # multi
    g, h = obj.gradient(
        cn.array([[2], [1], [0]]),
        cn.array([[0.3, 0.3, 0.4], [0.3, 0.4, 0.3], [0.4, 0.3, 0.3]]),
    )
    assert cn.allclose(
        g, cn.array([[0.3, 0.3, -0.6], [0.3, -0.6, 0.3], [-0.6, 0.3, 0.3]])
    )
    assert cn.allclose(
        h, cn.array([[0.21, 0.21, 0.24], [0.21, 0.24, 0.21], [0.24, 0.21, 0.21]])
    )

    with pytest.raises(
        ValueError, match="Expected labels to be non-zero whole numbers"
    ):
        obj.initialise_prediction(
            cn.array([[1], [2.5]]), cn.array([[1.0], [1.0]]), False
        )

    with pytest.raises(
        ValueError, match="Expected labels to be non-zero whole numbers"
    ):
        obj.initialise_prediction(
            cn.array(
                [
                    [-1],
                ]
            ),
            cn.array(
                [
                    [1.0],
                ]
            ),
            False,
        )


def test_exp():
    obj = lb.ExponentialObjective()
    # binary
    g, h = obj.gradient(cn.array([[1], [0]]), cn.array([[0.5], [0.5]]))
    assert cn.allclose(g, cn.array([[-1.0], [1.0]]))
    assert cn.allclose(h, cn.array([[1.0], [1.0]]))

    with pytest.raises(
        ValueError, match="Expected labels to be non-zero whole numbers"
    ):
        obj.initialise_prediction(
            cn.array([[1], [2.5]]), cn.array([[1.0], [1.0]]), False
        )

    with pytest.raises(
        ValueError, match="Expected labels to be non-zero whole numbers"
    ):
        obj.initialise_prediction(
            cn.array(
                [
                    [-1],
                ]
            ),
            cn.array(
                [
                    [1.0],
                ]
            ),
            False,
        )
