#!/bin/bash

set -e -E -u -o pipefail

# shellcheck disable=SC1091
source rapids-configure-sccache

rapids-print-env

rapids-generate-version > ./VERSION

sccache --zero-stats

CMAKE_GENERATOR=Ninja \
CONDA_OVERRIDE_CUDA="${RAPIDS_CUDA_VERSION}" \
LEGATEBOOST_PACKAGE_VERSION=$(head -1 ./VERSION) \
rapids-conda-retry build \
    --channel legate \
    --channel legate/label/branch-25.01 \
    --channel legate/label/experimental \
    --channel conda-forge \
    --channel nvidia \
    --no-force-upload \
    conda/recipes/legate-boost

sccache --show-adv-stats

# echo package details to logs, to help with debugging
conda search \
    --override-channels \
    --channel "${RAPIDS_CONDA_BLD_OUTPUT_DIR}" \
    --info \
        legate-boost
