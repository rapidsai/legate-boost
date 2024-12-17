#!/usr/bin/env bash

set -e -E -u -o pipefail

NUMARGS=$#
ARGS=$*

HELP="$0 [<target> ...] [<flag> ...]

  Build legateboost components.

 where <target> is any of:

    liblegateboost     - build the liblegateboost.so shared library
    legate-boost       - build and 'pip install' the legate-boost Python package
    clang-tidy         - run clang-tidy on the codebase

 where <flag> is any of:

   --editable        - install Python wheel in editable mode
   --fix             - clang-tidy will attempt to fix issues.
   -v                - Pass on verbosity to cmake and pip
   -h | --help       - print the help text
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
    --config-settings rapidsai.disable-cuda=true
)

# ensure 'native' is used if CUDAARCHS isn't set
# (instead of the CMake default which is a specific architecture)
# ref: https://cmake.org/cmake/help/latest/variable/CMAKE_CUDA_ARCHITECTURES.html
declare -r CMAKE_CUDA_ARCHITECTURES="${CUDAARCHS:-native}"

if hasArg --editable; then
    PIP_INSTALL_ARGS+=("--editable")
fi

if hasArg -v; then
    CMAKE_VERBOSE_FLAG="-v"
    PIP_INSTALL_ARGS+=(
         "-v"
         "--config-settings=build.verbose=true"
         "--config-settings=logging.level=INFO"
    )
    set -x
else
    CMAKE_VERBOSE_FLAG=""
fi

legate_root=$(
    python -c 'import legate.install_info as i; from pathlib import Path; print(Path(i.libpath).parent.resolve())'
)
if hasArg liblegateboost || hasArg --editable; then
    echo "building liblegateboost..."
    echo "Using Legate at '${legate_root}'"

    cmake -S . -B build -Dlegate_ROOT="${legate_root}" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES}"
    cmake --build build -j ${CMAKE_VERBOSE_FLAG}
    echo "done building liblegateboost"
fi

if hasArg clang-tidy; then
    echo "running clang-tidy..."
    # Build the project with clang
    CUDA_ROOT="$(dirname "$(dirname "$(which cuda-gdb)")")"
    echo "Using CUDA at '${CUDA_ROOT}'"
    cmake . -B build_clang_tidy -Dlegate_ROOT="${legate_root}" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES}" -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_CUDA_HOST_COMPILER=clang++ -DCMAKE_CUDA_COMPILER=clang++ -DCUDAToolkit_ROOT="${CUDA_ROOT}" -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
    FIX_ARG=""
    if hasArg --fix; then
        FIX_ARG="-fix"
    fi
    run-clang-tidy -p build_clang_tidy ${FIX_ARG} -exclude-header-filter='.*\/legate\/.*|.*\/libcudacxx\/.*' -header-filter='.*'
    echo "done running clang-tidy"
fi

if hasArg legate-boost; then
    echo "building legate-boost Python package..."
    CUDAARCHS="${CMAKE_CUDA_ARCHITECTURES}" \
        python -m pip install "${PIP_INSTALL_ARGS[@]}" .
    echo "done building legate-boost Python package"
fi
