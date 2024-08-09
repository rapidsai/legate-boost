#!/bin/bash

# [description]
#
# Common environment-setup stuff used by different CI jobs that
# test conda packages.
#
# This is intended to be source'd by other test scripts.

set -e -E -u -o pipefail

# source conda settings so 'conda activate' will work
# shellcheck disable=SC1091
. /opt/conda/etc/profile.d/conda.sh

rapids-print-env

rapids-dependency-file-generator \
  --output conda \
  --file-key py_test \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}" \
| tee /tmp/env.yaml

rapids-mamba-retry env create \
    --yes \
    --file /tmp/env.yaml \
    --name test

# Temporarily allow unbound variables for conda activation.
set +u
conda activate test
set -u

rapids-logger "Downloading artifacts from build jobs"
PYTHON_CHANNEL=$(rapids-download-conda-from-s3 python)

rapids-print-env

# Install legate-boost conda package built in the previous CI job
rapids-mamba-retry install \
  --name test \
  --channel legate \
  --channel "${PYTHON_CHANNEL}" \
  legate-boost
