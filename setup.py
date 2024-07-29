#!/usr/bin/env python3

# Copyright 2021-2022 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import os

from setuptools import find_packages
from skbuild import setup

# TODO: figure out a better way to handle this
#
#       importing legate in setup.py fails like this when building in an environment
#       that does not have GPUs:
#
#         > "libcuda.so.1: cannot open shared object file: No such file or directory"
#
# import legate.install_info as lg_install_info

# legate_dir = Path(lg_install_info.libpath).parent.as_posix()

# ensure 'native' is used if CUDAARCHS isn't set
# (instead of the CMake default which is a specific architecture)
# ref: https://cmake.org/cmake/help/latest/variable/CMAKE_CUDA_ARCHITECTURES.html
cuda_arch = os.getenv("CUDAARCHS", "native")

cmake_flags = [
    # f"-Dlegate_core_ROOT:STRING={legate_dir}",
    f"-DCMAKE_CUDA_ARCHITECTURES={cuda_arch}",
]

env_cmake_args = os.environ.get("CMAKE_ARGS")
if env_cmake_args is not None:
    cmake_flags.append(env_cmake_args)
os.environ["CMAKE_ARGS"] = " ".join(cmake_flags)


setup(
    packages=find_packages(
        where=".",
        include=["legateboost", "legateboost.*"],
    ),
    include_package_data=True,
    zip_safe=False,
)
