#!/bin/bash

set -e -E -u -o pipefail

rapids-print-env

rapids-generate-version > ./VERSION

CMAKE_GENERATOR=Ninja \
CONDA_OVERRIDE_CUDA="${RAPIDS_CUDA_VERSION}" \
LEGATEBOOST_PACKAGE_VERSION=$(head -1 ./VERSION) \
rapids-conda-retry mambabuild \
    --channel legate \
    --channel conda-forge \
    --channel nvidia \
    --no-force-upload \
    conda/recipes/legate-boost

# echo package details to logs, to help with debugging
conda search \
    --override-channels \
    --channel "${RAPIDS_CONDA_BLD_OUTPUT_DIR}" \
    --info \
        legate-boost
