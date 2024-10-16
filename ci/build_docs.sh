#!/bin/bash

set -e -E -u -o pipefail

# source conda settings so 'conda activate' will work
# shellcheck disable=SC1091
. /opt/conda/etc/profile.d/conda.sh

LEGATEBOOST_VERSION=$(rapids-version)

rapids-print-env

rapids-dependency-file-generator \
  --output conda \
  --file-key py_docs \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}" \
| tee /tmp/env.yaml

rapids-mamba-retry env create \
    --yes \
    --file /tmp/env.yaml \
    --name docs-env

# Temporarily allow unbound variables for conda activation.
set +u
conda activate docs-env
set -u

rapids-print-env

# Install legate-boost conda package built in the previous CI job
rapids-mamba-retry install \
  --name docs-env \
  --channel "${RAPIDS_LOCAL_CONDA_CHANNEL}" \
  --channel legate \
  --channel conda-forge \
  "legate-boost=${LEGATEBOOST_VERSION}"

rapids-print-env

make -C docs html
