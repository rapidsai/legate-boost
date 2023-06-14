import cunumeric as cn
import legateboost as lb


def test_log_loss():
    obj = lb.LogLossObjective()
    # binary
    g, h = obj.gradient(
        cn.array([[1], [0]]), cn.array([[0.5], [0.5]]), cn.array([1, 1])
    )
    assert cn.allclose(g, cn.array([[-0.5], [0.5]]))
    assert cn.allclose(h, cn.array([[0.25], [0.25]]))

    # multi
    g, h = obj.gradient(
        cn.array([[2], [1], [0]]),
        cn.array([[0.3, 0.3, 0.4], [0.3, 0.4, 0.3], [0.4, 0.3, 0.3]]),
        cn.array([1, 1, 1]),
    )
    assert cn.allclose(
        g, cn.array([[0.3, 0.3, -0.6], [0.3, -0.6, 0.3], [-0.6, 0.3, 0.3]])
    )
    assert cn.allclose(
        h, cn.array([[0.21, 0.21, 0.24], [0.21, 0.24, 0.21], [0.24, 0.21, 0.21]])
    )
