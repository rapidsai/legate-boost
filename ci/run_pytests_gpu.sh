#!/bin/bash

# [description]
#
#   Run GPU tests. This is intended for use by both CI and local development,
#   so shouldn't rely on any CI-specific details.
#
#   Put CI-specific details in 'run_pytests_gpu.sh'.
#
#   Additional arguments passed to this script are passed through to 'pytest'.
#

set -e -E -u -o pipefail

legate \
    --gpus 1 \
    --fbmem 28000 \
    --sysmem 28000 \
    --module pytest legateboost/test/[!_]**.py \
    -sv \
    --durations=0 \
    -k 'not sklearn' \
    "${@}"
