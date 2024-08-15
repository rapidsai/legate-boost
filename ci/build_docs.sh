#!/bin/bash

set -e -E -u -o pipefail

apt-get update
apt-get install -y --no-install-recommends \
    make

make -C docs html
