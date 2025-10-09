#!/bin/bash

# [description]
#
#   Run GPU tests.
#
#   This is intended for use by both CI and local development,
#   so shouldn't rely on any CI-specific details.
#
#   Put CI-specific details in 'test_python_gpu.sh'.
#
#   Additional arguments passed to this script are passed through to 'pytest'.
#

set -e -E -u -o pipefail

nvidia-smi

# Go into package folder to not import source package
cd legateboost/test

LEGATE_TEST=1 legate \
    --omps 0 \
    --gpus 1 \
    --fbmem 28000 \
    --sysmem 28000 \
    --module pytest \
    -sv \
    -k 'not sklearn' \
    --durations=0 \
    "${@}"
