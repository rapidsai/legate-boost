#!/bin/bash

set -e -E -u -o pipefail

rapids-print-env

rapids-dependency-file-generator \
  --output conda \
  --file-key test \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}" \
| tee /tmp/env.yaml

mamba env create \
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

mamba install \
  --name test \
  --channel "${PYTHON_CHANNEL}" \
  legate-boost

rapids-logger "Running tests"

legate \
    --gpus 1 \
    --fbmem 28000 \
    --sysmem 28000 \
    --module pytest legateboost/test/[!_]**.py \
    -sv \
    --durations=0 \
    -k 'not sklearn'
