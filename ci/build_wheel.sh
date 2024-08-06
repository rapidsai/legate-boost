#!/bin/bash

set -e -u -o pipefail

python -m build \
    --no-isolation \
    --skip-dependency-check \
    --wheel \
    --outdir dist
