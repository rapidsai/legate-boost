#!/bin/bash

set -e -E -u -o pipefail

# source conda settings so 'conda activate' will work
# shellcheck disable=SC1091
. /opt/conda/etc/profile.d/conda.sh

rapids-generate-version > ./VERSION
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

# Evaluate conda channels in order, and prefer the first one found per-package.
# (this ensures that the 'legate-boost' built in CI is used, instead of one hosted on the 'legate' channel)
conda config --set channel_priority strict

# Install legate-boost conda package built in the previous CI job
rapids-mamba-retry install \
  --name docs-env \
  --override-channels \
  --channel "${RAPIDS_LOCAL_CONDA_CHANNEL}" \
  --channel legate \
  --channel legate/label/branch-25.01 \
  --channel legate/label/experimental \
  --channel conda-forge \
  --channel nvidia \
  "legate-boost<25"

rapids-print-env

# Force minimal resource use as auto-configure may fail with few cores.
LEGATE_CONFIG="--cpus=1 --gpus=0 --omps=0 --utility=1" \
make -C docs html
