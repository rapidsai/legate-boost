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
from pathlib import Path

from setuptools import find_packages
from skbuild import setup

import legate.install_info as lg_install_info

legate_dir = Path(lg_install_info.libpath).parent.as_posix()

cmake_flags = [
    f"-Dlegate_core_ROOT:STRING={legate_dir}",
    "-DCMAKE_CUDA_ARCHITECTURES=native",
]

env_cmake_args = os.environ.get("CMAKE_ARGS")
if env_cmake_args is not None:
    cmake_flags.append(env_cmake_args)
os.environ["CMAKE_ARGS"] = " ".join(cmake_flags)

requires = [
    "cunumeric",
    "legate-core",
    "scikit-learn",
    "numpy",
    "typing_extensions",  # Required by legate.core as well.
]

extras_require = {
    "test": [
        "hypothesis",
        "pytest<8",
        "xgboost",
        "notebook",
        "nbconvert",
        "seaborn",
        "matplotlib",
        "mypy",
    ]
}

setup(
    name="legateboost",
    version="0.1",
    description="GBM libary on Legate",
    author="NVIDIA Corporation",
    license="Apache 2.0",
    classifiers=[
        "Intended Audience :: Developers",
        "Topic :: Database",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    install_requires=requires,
    extras_require=extras_require,
    packages=find_packages(
        where=".",
        include=["legateboost", "legateboost.*"],
    ),
    include_package_data=True,
    zip_safe=False,
)
