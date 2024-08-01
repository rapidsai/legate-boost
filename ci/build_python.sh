#!/bin/bash

set -e -u -o pipefail

export CMAKE_GENERATOR=Ninja

rapids-print-env

rm -rf ./build ./dist ./_skbuild

conda mambabuild \
    --channel conda-forge \
    --channel legate \
    --no-force-upload \
    conda/recipes/legateboost

rapids-upload-conda-to-s3 python
