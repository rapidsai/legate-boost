#!/bin/bash

# [description]
#
#   Run 'clang-tidy' static analysis.

set -e -E -u -o pipefail

# Ensure this is running from the root of the repo
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../;

# set up conda environment

# source conda settings so 'conda activate' will work
# shellcheck disable=SC1091
. /opt/conda/etc/profile.d/conda.sh

rapids-generate-version > ./VERSION

rapids-print-env

rapids-dependency-file-generator \
  --output conda \
  --file-key run_clang_tidy \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}" \
| tee /tmp/env.yaml

rapids-mamba-retry env create \
    --yes \
    --file /tmp/env.yaml \
    --name test-env

# Temporarily allow unbound variables for conda activation.
set +u
conda activate test-env
set -u

rapids-print-env

CMAKE_GENERATOR=Ninja \
./build.sh clang-tidy
