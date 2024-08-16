#!/bin/bash

exit 0

# set -e -E -u -o pipefail

# # Ensure this is running from the root of the repo
# cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../;

# # Common setup steps shared by Python test jobs
# source ./ci/test_python_common.sh

# # run mypy
# rapids-logger "Running mypy"
# ./ci/run_mypy.sh

# # run the GPU tests
# rapids-logger "Running tests"
# ./ci/run_pytests_cpu.sh
