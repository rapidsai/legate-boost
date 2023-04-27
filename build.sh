#!/usr/bin/env bash

cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
python -m pip install -e .
