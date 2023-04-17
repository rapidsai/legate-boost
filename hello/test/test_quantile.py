import cunumeric as np
import numpy as old_np
import hello


def test_quantile():
    x = np.array([1.0,2.0,3.0],dtype=np.float32)
    quantiles, quantised = hello.quantise(x, 4)
    assert quantised.dtype == np.uint16
    assert np.array_equal(quantised,np.array([0,1,2])), quantiles
    assert quantiles.size == 5

    quantiles, quantised = hello.quantise(x, 10)
    assert np.array_equal(quantised,np.array([0,1,2]))
    assert quantiles.size == 5

    np.random.seed(10)
    n_bins = 10
    x = np.random.normal(size=10000).astype(np.float32)
    quantiles, quantised = hello.quantise(x, n_bins)
    unique, counts = old_np.unique(quantised,return_counts=True)

    acceptable_bin_error = 0.05
    assert np.array_equal(unique, np.arange(10))
    expected_bin_size = x.size//n_bins
    assert np.all(np.abs(counts - expected_bin_size) < expected_bin_size * acceptable_bin_error)
    print(quantiles)
    print(unique)
    print(counts)




