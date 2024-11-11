# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import os
import platform
from ctypes import CDLL, RTLD_GLOBAL
from typing import Any

import cffi

from legate.core import get_legate_runtime


def dlopen_no_autoclose(ffi: Any, lib_path: str) -> Any:
    # Use an already-opened library handle, which cffi will convert to a
    # regular FFI object (using the definitions previously added using
    # ffi.cdef), but will not automatically dlclose() on collection.
    lib = CDLL(lib_path, mode=RTLD_GLOBAL)
    return ffi.dlopen(ffi.cast("void *", lib._handle))


class UserLibrary:
    def __init__(self, name: str) -> None:
        self.name = name
        self.shared_object: Any = None

        shared_lib_path = self.get_shared_library()
        if shared_lib_path is not None:
            ffi = cffi.FFI()
            header = self.get_c_header()
            if header is not None:
                ffi.cdef(header)
            # Don't use ffi.dlopen(), because that will call dlclose()
            # automatically when the object gets collected, thus removing
            # symbols that may be needed when destroying C++ objects later
            # (e.g. vtable entries, which will be queried for virtual
            # destructors), causing errors at shutdown.
            shared_lib = dlopen_no_autoclose(ffi, shared_lib_path)
            self.initialize(shared_lib)
            callback_name = self.get_registration_callback()
            callback = getattr(shared_lib, callback_name)
            callback()
        else:
            self.initialize(None)

    @property
    def cffi(self) -> Any:
        return self.shared_object

    def get_name(self) -> str:
        return self.name

    def get_shared_library(self) -> str:
        from .install_info import libpath

        return os.path.join(libpath, f"liblegateboost{self.get_library_extension()}")

    def get_c_header(self) -> str:
        from .install_info import header

        return str(header)

    def get_registration_callback(self) -> str:
        return "legateboost_perform_registration"

    def initialize(self, shared_object: Any) -> None:
        self.shared_object = shared_object

    def destroy(self) -> None:
        pass

    @staticmethod
    def get_library_extension() -> str:
        os_name = platform.system()
        if os_name == "Linux":
            return ".so"
        elif os_name == "Darwin":
            return ".dylib"
        raise RuntimeError(f"unknown platform {os_name!r}")


user_lib = UserLibrary("legateboost")
user_context = get_legate_runtime().find_library(user_lib.get_name())
