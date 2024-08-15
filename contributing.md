# Contributing to legate-boost

`legate-boost` depends on some libraries that are not easily installable with `pip`.

Use `conda` to create a development environment that includes them.

```shell
# CUDA 12.2
conda env create \
    --name legate-boost-dev \
    -f ./conda/environments/all_cuda-122.yaml

source activate legate-boost-dev
```

The easiest way to develop is to compile the shared library separately, then build
and install an editable wheel that uses it.

```shell
./build.sh legate-boost --editable
```

## Running tests

CPU:

```shell
ci/run_pytests_cpu.sh
```

GPU:

```shell
ci/run_pytests_gpu.sh
```

## Add new tests

Test cases should go in `legateboost/test`.

Utility code re-used by multiple tests should be added in `legateboost/testing`.

## Change default CUDA architectures

By default, builds here default to `CMAKE_CUDA_ARCHITECTURES=native` (whatever GPU exists on the system where the build is running).

If installing with `pip`, set the `CUDAARCHS` environment variable, as described in the CMake docs ([link](https://cmake.org/cmake/help/latest/variable/CMAKE_CUDA_ARCHITECTURES.html)).

```shell
CUDAARCHS="70;80" \
    pip install --no-build-isolation --no-deps .
```

For CMake-based builds, pass `CMAKE_CUDA_ARCHITECTURES`.

```shell
cmake -B build -S . -DCMAKE_CUDA_ARCHITECTURES="70;80"
cmake --build build -j
```

## Build conda packages locally

Before doing this, be sure to remove any other left-over build artifacts.

```shell
git clean -d -f -X
```

Build inside a container using one of the RAPIDS CI images.

```shell
docker run \
  --rm \
  -v $(pwd):/opt/legate-boost:ro \
  -w /opt/legate-boost \
  -it rapidsai/ci-conda:cuda12.5.1-ubuntu22.04-py3.11 \
  bash

CMAKE_GENERATOR=Ninja \
CONDA_OVERRIDE_CUDA="${RAPIDS_CUDA_VERSION}" \
rapids-conda-retry mambabuild \
    --channel legate \
    --channel conda-forge \
    --channel nvidia \
    --no-force-upload \
    conda/recipes/legate-boost
```

Once that completes, you can work with the packages.
Environment variable `RAPIDS_CONDA_BLD_OUTPUT_DIR` points to a location with the packages and
all the necessary data to be used as a full conda channel.

For example:

```shell
# list the package contents
cph list \
    "$(echo ${RAPIDS_CONDA_BLD_OUTPUT_DIR}/linux-64/legate-boost-*_gpu.tar.bz2)"

# check that the dependency metadata is correct
conda search \
    --override-channels \
    --channel ${RAPIDS_CONDA_BLD_OUTPUT_DIR} \
    --info \
        legate-boost

# create an environment with the package installed
conda create \
    --name legate-boost-test \
    -c legate \
    -c conda-forge \
    -c "${RAPIDS_CONDA_BLD_OUTPUT_DIR}" \
        legate-boost
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

## Change the project version

The `VERSION` file at the root of the repo is the single source for `legate-boost`'s version.
Modify that file to change the version for wheels, conda packages, the CMake project, etc.

## Development principles

The following general principles should be followed when developing `legate-boost`.

### Coding style

- Strive for simple and clear design, appropriate for a reference implementation.
- Algorithm accuracy and reliability is more important than speed.
    - e.g. do not replace double precision floats with single precision in order to achieve small constant factor implementation speedups.
    - Do not be afraid to use 64 bit integers for indexing if it means avoiding any possible overflow issues.
- Avoid optimisation where possible in favour of clear implementation
- Favour cunumeric implementations where appropriate. e.g. elementwise or matrix operations
- Use mypy type annotations if at all possible. The typing can be checked by running the following command under the project root:

```shell
ci/run_mypy.sh
```

### Performance

- Memory usage is more often a limiting factor than computation time in large distributed training runs. E.g. A proposal that improves runtime by 2x but increases memory usage by 1.5x is likely to be rejected.
- `legate-boost` should support CPUs and GPUs as first class citizens.
- `legate-boost` will strive for acceptable to good performance on single machine and state-of-the-art performance in a distributed setting.
- Accepting performance improvements will depend on how maintainable the changes are versus the improvement for a single machine and distributed setting, with a heavier weighting towards the distributed setting.
- In deciding what level of performance optimisation is appropriate, see the below performance guidelines
    - `legate-boost` should be expected to run faster than equivalent python based implementations on a single machine e.g. Sklearn.
    - `legate-boost` <em>should not</em> be expected to run faster than highly optimised native implementations on a single machine. e.g. LightGBM/XGBoost.
    - `legate-boost` <em>should</em> compete with the above implementions in a distributed setting.

### Testing

- High level interfaces (e.g. estimators) should be tested using property based testing (e.g. the hypothesis library in python). These tests will automatically test a wide range of inputs.
- Test run times should be optimised. Minimise the number of boosting rounds or dataset size required to achieve a test result. Cache datasets, preprocessing or other commonly used functionality.
- Enable santisers in CI to check for various C++ errors.

### Supported platforms

- Platform support:
    - `legate-boost` will support the same platforms as the legate ecosystem.
    - `legate-boost` will also support conda or pip following the legate ecosystem.
- Installation should be as simple as possible. e.g. `pip install legate-boost` or `conda install legate-boost`.
- Dependency minimisation will facilitate the above.

### Data science considerations
TODO: review by experts
- Minimise the number of hyperparameters. e.g. XGBoost supports 3 different types of column sampling, where columns are sampled for each tree, each layer or each node. Are the differences between these methods statistically significant? Do we only need one?
- Some functionality (e.g. preprocessing data) can be deffered to other libraries, although direct implementation can sometimes significantly improve usability or performance (e.g. cross validation).
- Categorical support is important.
- Support for sparse data is important. Applications such as NLP involve very sparse data.

### Non-goals
- Federated learning or privacy preserving machine learning. The literature is not advanced enough to indicate what the best approach is here.
- External memory.
    - `legate-boost` will defer data management to legate/legion.
    - `legate-boost` will not implement its own external memory algorithms, unless the functionality is already implemented in legate.
