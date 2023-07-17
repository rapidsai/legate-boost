import pytest

import cunumeric as cn
import legateboost as lb


def test_log_loss():
    obj = lb.LogLossObjective()
    # binary
    g, h = obj.gradient(cn.array([[1], [0]]), cn.array([[0.0], [0.0]]))
    assert cn.allclose(g, cn.array([[-0.5], [0.5]]))
    assert cn.allclose(h, cn.array([[0.25], [0.25]]))

    with pytest.raises(
        ValueError, match="Expected labels to be non-zero whole numbers"
    ):
        obj.check_labels(cn.array([[0.2], [1.0]]))

    with pytest.raises(
        ValueError, match="Expected labels to be non-zero whole numbers"
    ):
        obj.check_labels(cn.array([[-1]]))


def test_exp():
    obj = lb.ExponentialObjective()
    # binary
    g, h = obj.gradient(cn.array([[1], [0]]), cn.array([[0.0], [0.0]]))
    assert cn.allclose(g, cn.array([[-1.0], [1.0]]))
    assert cn.allclose(h, cn.array([[1.0], [1.0]]))

    with pytest.raises(
        ValueError, match="Expected labels to be non-zero whole numbers"
    ):
        obj.check_labels(cn.array([[1], [2.5]]))
