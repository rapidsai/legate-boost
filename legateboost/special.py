# Copyright 2024 NVIDIA Corporation
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
from __future__ import annotations
from enum import IntEnum

import cunumeric as cn
from legate.core import get_legate_runtime, types as ty

from .library import user_context, user_lib
from .utils import get_store


class _SpecialOpCode(IntEnum):
    ERF = user_lib.cffi.ERF
    LGAMMA = user_lib.cffi.LGAMMA
    TGAMMA = user_lib.cffi.TGAMMA
    DIGAMMA = user_lib.cffi.DIGAMMA


def _elementwise_fn(x: cn.ndarray, fn: _SpecialOpCode) -> cn.ndarray:
    xs = get_store(x)
    if xs.type not in (ty.float32, ty.float64):
        raise TypeError(f"{xs.type} is not supported.")

    task = get_legate_runtime().create_auto_task(
        user_context,
        fn,
    )
    task.add_input(xs)

    output = get_legate_runtime().create_store(dtype=xs.type, shape=xs.shape)
    task.add_output(output)
    task.add_alignment(xs, output)
    task.execute()

    return cn.array(output)


def erf(x: cn.ndarray) -> cn.ndarray:
    """Elementwise erf function."""
    return _elementwise_fn(x, _SpecialOpCode.ERF)


def loggamma(x: cn.ndarray) -> cn.ndarray:
    """Elementwise log-gamma function.

    :math:`x` should be greater than 0.
    """
    return _elementwise_fn(x, _SpecialOpCode.LGAMMA)


def gamma(x: cn.ndarray) -> cn.ndarray:
    """Elementwise gamma function."""
    return _elementwise_fn(x, _SpecialOpCode.TGAMMA)


def digamma(x: cn.ndarray) -> cn.ndarray:
    """Elementwise digamma function.

    Only real number is supported.
    """
    return _elementwise_fn(x, _SpecialOpCode.DIGAMMA)


def zeta(n: int | float, x: cn.ndarray) -> cn.ndarray:
    pass


def polygamma(n: int | float, x: cn.ndarray) -> cn.ndarray:
    """Polygamma functions."""

    fac2 = (-1.0) ** (n + 1) * gamma(n + 1.0) * zeta(n + 1, x)
    return cn.where(n == 0, digamma(x), fac2)
