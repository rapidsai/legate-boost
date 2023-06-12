# Development principles

The following general principles should be followed when developing legateboost.

- Strive for simple and clear design, appropriate for a reference implementation.
- Algorithm accuracy and reliability is more important than speed.
    - e.g. do not replace double precision floats with single precision in order to achieve small constant factor implementation speedups.
    - Do not be afraid to use 64 bit integers for indexing if it means avoiding any possible overflow issues.
- Avoid optimisation where possible in favour of clear implementation
- Favour cunumeric implementations where appropriate. e.g. elementwise or matrix operations
- Legateboost will strive for acceptable to good performance on single machine and state-of-the-art performance in a distributed setting.
- Accepting performance improvements will depend on how maintainable the changes are versus the improvement for a single machine and distributed setting, with a heavier weighting towards the distributed setting.
- In deciding what level of performance optimisation is appropriate, see the below performance guidelines
    - Legateboost should be expected to run faster than equivalent python based implementations on a single machine e.g. Sklearn.
    - Legateboost <em>should not</em> be expected to run faster than highly optimised native implementations on a single machine. e.g. LightGBM/XGBoost.
    - Legateboost <em>should</em> compete with the above implementions in a distributed setting.
