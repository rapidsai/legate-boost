#!/bin/bash

set -e -E -u -o pipefail

rapids-print-env

rm -rf ./build ./dist ./_skbuild

CMAKE_GENERATOR=Ninja \
CONDA_OVERRIDE_CUDA="${RAPIDS_CUDA_VERSION}" \
rapids-conda-retry mambabuild \
    --channel legate \
    --channel conda-forge \
    --channel nvidia \
    --no-force-upload \
    conda/recipes/legate-boost

rapids-upload-conda-to-s3 python
