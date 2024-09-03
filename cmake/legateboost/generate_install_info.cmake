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

function(legateboost_generate_install_info header)
  list(APPEND CMAKE_MESSAGE_CONTEXT "generate_install_info")

  set(options)
  set(one_value_args TARGET PY_PATH)
  set(multi_value_args)
  cmake_parse_arguments(LEGATE_OPT "${options}" "${one_value_args}" "${multi_value_args}"
                        ${ARGN})

  # determine full Python path
  if(NOT DEFINED LEGATE_OPT_PY_PATH)
    set(py_path "${CMAKE_CURRENT_SOURCE_DIR}/${LEGATE_OPT_TARGET}")
  elseif(IS_ABSOLUTE LEGATE_OPT_PY_PATH)
    set(py_path "${LEGATE_OPT_PY_PATH}")
  else()
    set(py_path "${CMAKE_CURRENT_SOURCE_DIR}/${LEGATE_OPT_PY_PATH}")
  endif()

  # abbreviate for the function below
  set(target ${LEGATE_OPT_TARGET})
  set(install_info_in
      [=[
from pathlib import Path

def get_libpath():
    import os, sys, platform
    join = os.path.join
    exists = os.path.exists
    dirname = os.path.dirname
    cn_path = dirname(dirname(__file__))
    so_ext = {
        "": "",
        "Java": ".jar",
        "Linux": ".so",
        "Darwin": ".dylib",
        "Windows": ".dll"
    }[platform.system()]

    def find_lib(libdir):
        target = f"lib@target@{so_ext}*"
        search_path = Path(libdir)
        matches = [m for m in search_path.rglob(target)]
        if matches:
          return matches[0].parent
        return None

    return (
        find_lib("@libdir@") or
        find_lib(join(dirname(dirname(dirname(cn_path))), "lib")) or
        find_lib(join(dirname(dirname(sys.executable)), "lib")) or
        ""
    )

libpath: str = get_libpath()

header: str = """
  @header@
  void @target@_perform_registration();
"""
]=])
  set(install_info_py_in ${CMAKE_CURRENT_BINARY_DIR}/legate_${target}/install_info.py.in)
  set(install_info_py ${py_path}/install_info.py)
  file(WRITE ${install_info_py_in} "${install_info_in}")

  set(generate_script_content
      [=[
    execute_process(
      COMMAND ${CMAKE_C_COMPILER}
        -E
        -P @header@
      ECHO_ERROR_VARIABLE
      OUTPUT_VARIABLE header
      COMMAND_ERROR_IS_FATAL ANY
    )
    configure_file(
        @install_info_py_in@
        @install_info_py@
        @ONLY)
  ]=])

  set(generate_script ${CMAKE_CURRENT_BINARY_DIR}/gen_install_info.cmake)
  # I think this is a bug? It is complaining "Invalid form descriminator", which I believe
  # refers to the @ONLY. But that is valid for this function...
  #
  # cmake-lint: disable=E1126
  file(CONFIGURE OUTPUT ${generate_script} CONTENT "${generate_script_content}" @ONLY)

  if(DEFINED ${target}_BUILD_LIBDIR)
    # this must have been imported from an existing editable build
    set(libdir ${${target}_BUILD_LIBDIR})
  else()
    # libraries are built in a common spot
    set(libdir ${CMAKE_BINARY_DIR}/legate_${target})
  endif()
  add_custom_target("${target}_generate_install_info_py" ALL
                    COMMAND ${CMAKE_COMMAND} -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                            -Dtarget=${target} -Dlibdir=${libdir} -P ${generate_script}
                            OUTPUT ${install_info_py}
                    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                    COMMENT "Generating install_info.py"
                    DEPENDS ${header})
endfunction()
