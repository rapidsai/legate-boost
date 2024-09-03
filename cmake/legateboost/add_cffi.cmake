#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#=============================================================================

include_guard(GLOBAL)

function(legateboost_add_cffi py_path)
  list(APPEND CMAKE_MESSAGE_CONTEXT "add_cffi")

  string(REPLACE "/" "_" target "${py_path}")
  string(REPLACE "/" "." py_import_path "${py_path}")

  set(fn_library "${CMAKE_CURRENT_SOURCE_DIR}/${py_path}/library.py")
  set(file_template
      [=[
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from legate.core import (
    get_legate_runtime,
)
import os
import platform
from typing import Any
from ctypes import CDLL, RTLD_GLOBAL

# TODO: Make sure we only have one ffi instance?
import cffi

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
        from @py_import_path@.install_info import libpath
        return os.path.join(libpath, f"lib@target@{self.get_library_extension()}")

    def get_c_header(self) -> str:
        from @py_import_path@.install_info import header

        return header

    def get_registration_callback(self) -> str:
        return "@target@_perform_registration"

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

user_lib = UserLibrary("@target@")
user_context = get_legate_runtime().find_library(user_lib.get_name())
]=])
  string(CONFIGURE "${file_template}" file_content @ONLY)
  file(WRITE "${fn_library}" "${file_content}")
endfunction()
