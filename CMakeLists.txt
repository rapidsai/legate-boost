cmake_minimum_required(VERSION 3.24.0 FATAL_ERROR)

# read project version from VERSION file
file(READ "${CMAKE_CURRENT_LIST_DIR}/VERSION" _version_content)
if(_version_content MATCHES [[^([0-9]+)\.([0-9]+)\.([0-9]+)]])
  set(_legateboost_version "${CMAKE_MATCH_1}.${CMAKE_MATCH_2}.${CMAKE_MATCH_3}")
else()
  string(REPLACE "\n" "\n  " _legateboost_content_formatted "  ${_version_content}")
  message(
    FATAL_ERROR
      "Could not determine project version. Contents of VERSION file:\n${_legateboost_content_formatted}"
  )
endif()

project(legateboost VERSION "${_legateboost_version}" LANGUAGES C CXX CUDA)

option(SANITIZE "Build with address sanitizer" OFF)

# This is for convenience only when doing
# editable builds to avoid setting the flag
if (NOT legateboost_ROOT)
  set(legateboost_ROOT ${CMAKE_SOURCE_DIR}/build)
endif()

set(BUILD_SHARED_LIBS ON)

# Look for an existing C++ editable build
# Not required. We will build it if not found.
find_package(legateboost QUIET)
find_package(legate_core REQUIRED)
find_package(BLAS REQUIRED)

legate_add_cpp_subdirectory(src TARGET legateboost EXPORT legateboost-export)

legate_add_cffi(${CMAKE_SOURCE_DIR}/src/legateboost.h TARGET legateboost)
legate_python_library_template(legateboost)
legate_default_python_install(legateboost EXPORT legateboost-export)


if (SANITIZE)
  message(STATUS "Adding sanitizer flags")
  add_compile_options(-fsanitize=address)
  add_link_options(-fsanitize=address)
endif()
