cmake_minimum_required(VERSION 3.24.0 FATAL_ERROR)

project(legateboost VERSION 1.0 LANGUAGES C CXX CUDA)

option(SANITIZE "Build with address sanitizer" OFF)

# This is for convenience only when doing
# editable builds to avoid setting the flag
if (NOT legateboost_ROOT)
  set(legateboost_ROOT ${CMAKE_SOURCE_DIR}/build)
endif()

if (NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

set(BUILD_SHARED_LIBS ON)

# Look for an existing C++ editable build
# Not required. We will build it if not found.
find_package(legateboost QUIET)
find_package(legate_core REQUIRED)

legate_add_cpp_subdirectory(src TARGET legateboost EXPORT legateboost-export)

legate_add_cffi(${CMAKE_SOURCE_DIR}/src/legateboost.h TARGET legateboost)
legate_python_library_template(legateboost)
legate_default_python_install(legateboost EXPORT legateboost-export)

if (SANITIZE)
  message(STATUS "Adding sanitizer flags")
  add_compile_options(-fsanitize=address)
  add_link_options(-fsanitize=address)
endif()
