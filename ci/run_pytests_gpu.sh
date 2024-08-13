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

# run from somewhere other that the repo root, to ensure that
# "import legateboost" matches the installed package, not the local source diretory
cd ./legateboost

legate \
    --gpus 1 \
    --fbmem 28000 \
    --sysmem 28000 \
    --module pytest \
    test/[!_]**.py \
    -sv \
    --durations=0 \
    -k 'not sklearn' \
    "${@}"
