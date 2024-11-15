#!/bin/bash

# [description]
#
#   Run 'clang-tidy' static analysis.

set -e -E -u -o pipefail

./build.sh clang-tidy
