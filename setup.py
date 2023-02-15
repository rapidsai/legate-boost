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
import legate.install_info as lg_install_info
from pathlib import Path
import os
legate_dir = Path(lg_install_info.libpath).parent.as_posix()
os.environ["SKBUILD_CONFIGURE_OPTIONS"] = f"-Dlegate_core_ROOT:STRING={legate_dir}"

from setuptools import find_packages
from skbuild import setup

setup(
    name="legate_hello_world",
    version="0.1",
    description="A Hello World for Legate",
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
    packages=find_packages(
        where=".",
        include=["hello", "hello.*"],
    ),
    include_package_data=True,
    zip_safe=False,
)
