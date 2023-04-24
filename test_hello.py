import legateboost 
import cunumeric as np
import numpy as old_np



n_bins = 10
np.random.seed(1)
x = np.random.normal(size=(1000,2)).astype(np.float32)
quantiles, ptr, quantised = legateboost.quantise(x, n_bins)


unique, counts = old_np.unique(quantised,return_counts=True)

acceptable_bin_error = 0.05
assert np.array_equal(unique, np.arange(10))
expected_bin_size = x.size//n_bins
assert np.all(np.abs(counts - expected_bin_size) < expected_bin_size * acceptable_bin_error)
print(quantiles)
print(counts)