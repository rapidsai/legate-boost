# Contributing to legateboost

For editable installation
```
pip install -e .
```
To include test dependencies
```
pip install -e .[test]
```

## Running tests
```
legate --module pytest legateboost/test
```

## Change default CUDA architectures

By default, builds here default to `CMAKE_CUDA_ARCHITECTURES=native` (whatever GPU exists on the system where the build is running).

If installing with `pip`, set the `CUDAARCHS` environment variable, as described in the CMake docs ([link](https://cmake.org/cmake/help/latest/variable/CMAKE_CUDA_ARCHITECTURES.html)).

```shell
CUDAARCHS="70;80" \
    pip install .
```

For CMake-based builds, pass `CMAKE_CUDA_ARCHITECTURES`.

```shell
cmake -B build -S . -DCMAKE_CUDA_ARCHITECTURES="70;80"
cmake --build build -j
```

## Pre-commit hooks

The pre-commit package is used for linting, formatting and type checks. This project uses strict mypy type checking.

Install pre-commit.
```
pip install pre-commit
```
Run all checks manually.
```
pre-commit run --all-files
```
## Development principles

The following general principles should be followed when developing legateboost.

### Coding style

- Strive for simple and clear design, appropriate for a reference implementation.
- Algorithm accuracy and reliability is more important than speed.
    - e.g. do not replace double precision floats with single precision in order to achieve small constant factor implementation speedups.
    - Do not be afraid to use 64 bit integers for indexing if it means avoiding any possible overflow issues.
- Avoid optimisation where possible in favour of clear implementation
- Favour cunumeric implementations where appropriate. e.g. elementwise or matrix operations
- Use mypy type annotations if at all possible. The typing can be checked by running the following command under the project root:
```
mypy ./legateboost --config-file ./setup.cfg --exclude=legateboost/test --exclude=install_info
```

### Performance

- Memory usage is more often a limiting factor than computation time in large distributed training runs. E.g. A proposal that improves runtime by 2x but increases memory usage by 1.5x is likely to be rejected.
- Legateboost should support CPUs and GPUs as first class citizens.
- Legateboost will strive for acceptable to good performance on single machine and state-of-the-art performance in a distributed setting.
- Accepting performance improvements will depend on how maintainable the changes are versus the improvement for a single machine and distributed setting, with a heavier weighting towards the distributed setting.
- In deciding what level of performance optimisation is appropriate, see the below performance guidelines
    - Legateboost should be expected to run faster than equivalent python based implementations on a single machine e.g. Sklearn.
    - Legateboost <em>should not</em> be expected to run faster than highly optimised native implementations on a single machine. e.g. LightGBM/XGBoost.
    - Legateboost <em>should</em> compete with the above implementions in a distributed setting.

### Testing

- High level interfaces (e.g. estimators) should be tested using property based testing (e.g. the hypothesis library in python). These tests will automatically test a wide range of inputs.
- Test run times should be optimised. Minimise the number of boosting rounds or dataset size required to achieve a test result. Cache datasets, preprocessing or other commonly used functionality.
- Enable santisers in CI to check for various C++ errors.

### Supported platforms

- Platform support: legateboost will support the same platforms as the legate ecosystem. Legateboost will also support conda or pip following the legate ecosystem.
- Installation should be as simple as possible. e.g. `pip install legateboost` or `conda install legateboost`.
- Dependency minimisation will facilitate the above.

### Data science considerations
TODO: review by experts
- Minimise the number of hyperparameters. e.g. XGBoost supports 3 different types of column sampling, where columns are sampled for each tree, each layer or each node. Are the differences between these methods statistically significant? Do we only need one?
- Some functionality (e.g. preprocessing data) can be deffered to other libraries, although direct implementation can sometimes significantly improve usability or performance (e.g. cross validation).
- Categorical support is important.
- Support for sparse data is important. Applications such as NLP involve very sparse data.

### Non-goals
- Federated learning or privacy preserving machine learning. The literature is not advanced enough to indicate what the best approach is here.
- External memory. Legateboost will defer data management to legate/legion. Legateboost will not implement its own external memory algorithms, unless the functionality is already implemented in legate.
