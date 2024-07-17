#!/bin/bash

python -m build \
    --no-isolation \
    --wheel

#  \
#     --config-setting=cmake.define.BLAS_LIBRARIES="$CONDA_PREFIX/lib/libopenblas.so"
