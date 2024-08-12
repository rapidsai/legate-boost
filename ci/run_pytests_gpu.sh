#!/bin/bash

# [description]
#
#   Run GPU tests.
#
#   This is intended for use by both CI and local development,
#   so shouldn't rely on any CI-specific details.
#
#   Additional arguments passed to this script are passed through to 'pytest'.
#

set -e -E -u -o pipefail

nvidia-smi

legate \
    --gpus 1 \
    --fbmem 28000 \
    --sysmem 28000 \
    --module pytest \
    legateboost/test/[!_]**.py \
    -sv \
    --durations=0 \
    -k 'not sklearn' \
    "${@}"
