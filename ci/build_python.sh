#!/bin/bash

set -e -u -o pipefail

export CMAKE_GENERATOR=Ninja

# TODO: remove this, use RAPIDS images
export RAPIDS_CUDA_VERSION="12.2"

rm -rf ./build ./dist ./_skbuild

conda mambabuild \
    --channel conda-forge \
    --channel legate \
    --no-force-upload \
    conda/recipes/legateboost
