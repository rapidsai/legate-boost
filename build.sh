#!/usr/bin/env bash

set -e -u -o pipefail

# ensure 'native' is used if CUDAARCHS isn't set
# (instead of the CMake default which is a specific architecture)
# ref: https://cmake.org/cmake/help/latest/variable/CMAKE_CUDA_ARCHITECTURES.html
declare -r CMAKE_CUDA_ARCHITECTURES="${CUDAARCHS:-native}"

legate_root=`python -c 'import legate.install_info as i; from pathlib import Path; print(Path(i.libpath).parent.resolve())'`
echo "Using Legate at $legate_root"
cmake -S . -B build -D legate_core_ROOT=$legate_root -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}
cmake --build build -j

python -m pip install --no-build-isolation --no-deps -e .
