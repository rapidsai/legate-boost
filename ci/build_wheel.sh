#!/bin/bash

set -e -u -o pipefail

./build.sh
python -m build \
    -no-build-isolation \
    --wheel \
    --outdir dist
