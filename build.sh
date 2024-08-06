#!/usr/bin/env bash

set -e -u -o pipefail

NUMARGS=$#
ARGS=$*

HELP="build liblegateboost.so and a 'legate-boost' Python wheel, and install that wheel

  $0 [<flag> ...]

 where <flag> is any of:

   --editable  install Python wheel in editable mode
"

function hasArg {
    (( NUMARGS != 0 )) && (echo " ${ARGS} " | grep -q " $1 ")
}

if hasArg -h || hasArg --help; then
    echo "${HELP}"
    exit 0
fi

# Set defaults for vars modified by flags to this script
PIP_INSTALL_ARGS=(
    --no-build-isolation
    --no-deps
)

# ensure 'native' is used if CUDAARCHS isn't set
# (instead of the CMake default which is a specific architecture)
# ref: https://cmake.org/cmake/help/latest/variable/CMAKE_CUDA_ARCHITECTURES.html
declare -r CMAKE_CUDA_ARCHITECTURES="${CUDAARCHS:-native}"

legate_root=$(
    python -c 'import legate.install_info as i; from pathlib import Path; print(Path(i.libpath).parent.resolve())'
)
echo "Using Legate at '${legate_root}'"

cmake -S . -B build -Dlegate_core_ROOT="${legate_root}" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES}"
cmake --build build -j

if hasArg --editable; then
    PIP_INSTALL_ARGS+=("--editable")
fi

echo "building legate-boost Python package..."
python -m pip install "${PIP_INSTALL_ARGS[@]}" .
