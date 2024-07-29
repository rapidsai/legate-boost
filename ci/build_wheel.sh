#!/bin/bash

set -e -u -o pipefail

${PYTHON} -m build \
    --no-isolation \
    --skip-dependency-check \
    --wheel \
    --outdir dist
