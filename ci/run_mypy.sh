#!/bin/bash

# [description]
#
#   Run 'mypy' type-checker.
#
#   This is intended for use by both CI and local development,
#   so shouldn't rely on any CI-specific details.
#
#   This is done in a separate script instead of via pre-commit because
#   running it in a non-isolated environment where all of the project's dependencies
#   are installed allows for more thorough type-checking.
#

set -e -E -u -o pipefail

mypy --version
mypy \
    --config-file ./pyproject.toml \
    --exclude=legateboost/test \
    --exclude=install_info \
    ./legateboost
