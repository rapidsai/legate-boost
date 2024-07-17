#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

set -euo pipefail

# install dependencies, so tools like 'mypy' can look deeper into the code
conda install \
    --yes \
    -c conda-forge \
    -c legate \
        cunumeric=24.06 \
        legate-core=24.06 \
        numpy \
        pre-commit \
        scikit-learn \
        scipy

git config --global --add safe.directory /opt/legate-boost

# Run pre-commit checks
pre-commit run --all-files
