#!/bin/bash

set -e -u -o pipefail

./build.sh
python -m build \
    --no-isolation \
    --wheel \
    --outdir dist
