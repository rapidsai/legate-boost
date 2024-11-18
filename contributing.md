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

The project's version is determined by git tags.
To see how to change the version, read "Releasing" below.

The `VERSION` file checked into source control is intended for use by local builds during development, and
so should be kept up to date with those git tags.

## Work with the conda packages

Run the commands in this section in a container using the same base image as CI.

```shell
# NOTE: remove '--gpus' to test the CPU-only version
docker run \
  --rm \
  --gpus 1 \
  -v $(pwd):/opt/legate-boost \
  -w /opt/legate-boost \
  -it rapidsai/ci-conda:cuda12.5.1-ubuntu22.04-py3.11 \
  bash
```

### Build conda packages locally

Before doing this, be sure to remove any other left-over build artifacts.

```shell
git clean -d -f -X
```

Build the packages.

```shell
CMAKE_GENERATOR=Ninja \
CONDA_OVERRIDE_CUDA="${RAPIDS_CUDA_VERSION}" \
LEGATEBOOST_PACKAGE_VERSION=$(head -1 ./VERSION) \
rapids-conda-retry mambabuild \
    --channel legate \
    --channel conda-forge \
    --channel nvidia \
    --no-force-upload \
    conda/recipes/legate-boost
```

### Download conda package created in CI

Packages built in CI are hosted on the GitHub Artifact Store.

To start, authenticate with the GitHub CLI.
By default, this will require interactively entering a code in a browser window.
That can be avoided by setting environment variable `GH_TOKEN`, as described in the
GitHub docs ([link](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/about-authentication-to-github)).

```shell
# authenticate with the GitHub CLI
# (can skip this by providing GH_TOKEN environment variable)
gh auth login
```

Next, select a CI run whose artifacts you want to test.
The run IDs can be found in the URLs at https://github.com/rapidsai/legate-boost/actions/workflows/github-actions.yml.
For example, given a URL like

```text
https://github.com/rapidsai/legate-boost/actions/runs/10566116913
```

The run ID is `10566116913`.

```shell
# choose a specific CI run ID
RUN_ID=10566116913
```

It's possible to omit the run ID and just have these commands download whatever
the latest artifact produced was.
For details on that, see the GitHub docs ([link](https://cli.github.com/manual/gh_run_download)).

Download the packages.
This will download and unpack a single artifact which contains all of the conda packages
built for a particular combination of CUDA version, CPU architecture, and Python version.

```shell
gh run download \
    --dir "${RAPIDS_CONDA_BLD_OUTPUT_DIR}" \
    --repo rapidsai/legate-boost \
    --name "legate-boost-conda-cuda${RAPIDS_CUDA_VERSION}-amd64-py${PYTHON_VERSION}" \
    "${RUN_ID}"
```

### Work with conda packages locally

After using either of the above approaches, use the tips in this section to work
with those local conda packages.

Environment variable `RAPIDS_CONDA_BLD_OUTPUT_DIR` points to a location with the packages and
all the necessary data to be used as a full conda channel.

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

## Releasing

NOTE: some steps in this section require direct write access to the repo (including its `main` branch).

### Create a stable release

NOTE: this assumes that the `VERSION` file on the `main` branch already holds the 3-part version number like `24.09.00`.

1. Push a git tag like `v24.09.00` ... that tag push will trigger a new release

```shell
git checkout main
git pull upstream main
git tag -a v24.09.00 -m 'v24.09.00'
git push upstream 'v24.09.00'
```

2. Update the `VERSION` file again, to the base version for the anticipated next release (without a leading `v`). Open a pull request with that change. Ensure the pull request title ends with `[skip ci]`.

```shell
git checkout -b update-version
echo "24.12.00" > ./VERSION
git commit -m "start v24.12 development"
git push origin update-version
```

3. Merge the pull request.
4. Tag that commit with a dev version

```shell
git checkout main
git pull upstream main
git tag -a v24.12.00.dev -m "v24.12.00.dev"
git push upstream v24.12.00.dev
```

From that point forward, all packages produced by CI from the `main` branch will have versions like `v24.12.00.dev{n}`,
where `{n}` is "number of new commits since the one tagged `v24.12.00.dev`".

### Hotfixes

Imagine that `v24.09.00` has been published, and at some later point a critical bug is found, which you want to package and release as `v24.09.01`.

Do the following.

1. Create a release branch, cut from the tag corresponding to the release you want to fix.

```shell
# get all the tags locally
git checkout main
git pull upstream main --tags

# create the new branch
git checkout v24.09.00
git checkout -b release/24.09
echo 'v24.09.01' > ./VERSION
git commit -m 'start v24.09.01 [skip ci]'
git push upstream release/24.09

# tag the first commit on the new branch as the beginning of the 24.09.01 series
git tag -a v24.09.01.dev -m 'v24.09.01.dev'
git push upstream v24.09.01.dev
```

2. Open pull requests targeting that branch and merge them into that branch.
3. When you feel the branch is ready to release, push a new tag.

```shell
git checkout release/v24.09
git pull upstream release/v24.09 --tags
git tag -a v24.09.01 -m 'v24.09.01'
git push upstream v24.09.01
```

With that hotfix release complete, merge the fixes into `main`.

1. create a new branch, cut from `main`

```shell
git checkout main
git pull upstream main
git checkout -b forward-merge-24.09-hotfixes
```

2. On that branch, use `git cherry-pick` to bring over the hotfix changes.

```shell
git cherry-pick release/v24.09
```

NOTE: The use of `cherry-pick` here is important because it re-writes the commit IDs. That avoids the situation where e.g. the
`v24.09.01` hotfix tag points to commits on the `main` branch during `v24.12` development (which could lead to those packages
incorrectly getting `v24.09.01.dev{n}` versions).

3. Open a pull request to merge that branch into `main`.
4. Perform a non-squash merge of that pull request.
5. Add a branch protection to prevent deletion of the `release/v24.09` branch, so you can return to it in the future if another hotfix is required.

## Development principles

The following general principles should be followed when developing `legate-boost`.

### Coding style

- Strive for simple and clear design, appropriate for a reference implementation.
- Algorithm accuracy and reliability is more important than speed.
    - e.g. do not replace double precision floats with single precision in order to achieve small constant factor implementation speedups.
    - Do not be afraid to use 64 bit integers for indexing if it means avoiding any possible overflow issues.
- Avoid optimisation where possible in favour of clear implementation
- Favour cupynumeric implementations where appropriate. e.g. elementwise or matrix operations
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
