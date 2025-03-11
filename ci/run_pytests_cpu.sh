#!/bin/bash

# [description]
#
#   Run CPU tests.
#
#   This is intended for use by both CI and local development,
#   so shouldn't rely on any CI-specific details.
#
#   Put CI-specific details in 'test_python_cpu.sh'.
#
#   Additional arguments passed to this script are passed through to 'pytest'.
#

set -e -E -u -o pipefail

# Go into test folder to not not import source package
cd legateboost/test


legate \
    --sysmem 28000 \
    --gpus 0 \
    --cpus 8 \
    --module pytest \
    . \
    -sv \
    --durations=0 \
    "${@}"

# Run a subset of tests forcing legate to run in deferred mode.
# When run normally legate may not partition smaller tasks, such
# that no distributed code gets tested.
LEGATE_TEST=1 legate \
    --sysmem 28000 \
    --gpus 0 \
    --cpus 8 \
    --module pytest \
    . \
    -sv \
    --durations=0 \
    -k 'not sklearn' \
    "${@}"
