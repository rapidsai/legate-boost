cmake_minimum_required(VERSION 3.24.0 FATAL_ERROR)

project(legategbm VERSION 1.0 LANGUAGES C CXX)

# This is for convenience only when doing
# editable builds to avoid setting the flag
if (NOT legategbm_ROOT)
  set(legategbm_ROOT ${CMAKE_SOURCE_DIR}/build)
endif()

if (NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

set(BUILD_SHARED_LIBS ON)

# Look for an existing C++ editable build
# Not required. We will build it if not found.
find_package(legategbm QUIET)
find_package(legate_core REQUIRED)

legate_add_cpp_subdirectory(src TARGET legategbm EXPORT legategbm-export)

legate_add_cffi(${CMAKE_SOURCE_DIR}/src/legategbm.h TARGET legategbm)
legate_python_library_template(legategbm)
legate_default_python_install(legategbm EXPORT legategbm-export)
