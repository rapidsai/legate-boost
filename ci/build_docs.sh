#!/bin/bash

set -e -E -u -o pipefail

rapids-print-env

make -C docs html
