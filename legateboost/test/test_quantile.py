import numpy as np
import pytest

import cunumeric as cun
import legateboost as lbst


def test_quantile_basic():
    x = cun.array([[1.0], [2.0], [3.0]], dtype=cun.float32)
    quantiles, ptr, quantised = lbst.quantise(x, 4)
    assert quantised.dtype == cun.uint16
    assert cun.array_equal(ptr, cun.array([0, 4])), ptr
    assert cun.array_equal(quantised, cun.array([[0], [1], [2]])), quantiles
    assert quantiles.size == 4

    quantiles, ptr, quantised = lbst.quantise(x, 10)
    assert cun.array_equal(quantised, cun.array([[0], [1], [2]]))
    assert quantiles.size == 4


@pytest.mark.parametrize("n_bins", [32, 128, 256])
@pytest.mark.parametrize("n", [10000])
def test_quantile(n_bins, n):
    dtype = cun.float32
    cun.random.seed(10)
    X = cun.zeros((n, 2), dtype=dtype)
    X[:, 0] = cun.random.normal(size=n, scale=100)
    X[:, 1] = cun.random.randint(0, int(cun.ceil(n_bins / 2)), size=n)

    quantiles, ptr, quantised = lbst.quantise(X, n_bins)
    for col in range(X.shape[1]):
        unique, counts = np.unique(quantised[:, col], return_counts=True)
        acceptable_bin_error = 0.3
        expected_bins = ptr[col + 1] - ptr[col] - 1

        def is_sorted(a):
            return np.all(a[:-1] <= a[1:])

        assert is_sorted(quantiles[ptr[col] : ptr[col + 1]])
        assert cun.array_equal(unique, cun.arange(expected_bins))
        assert unique.size <= n_bins
        expected_bin_count = n // expected_bins
        assert cun.all(
            cun.abs(counts - expected_bin_count)
            < expected_bin_count * acceptable_bin_error
        )
