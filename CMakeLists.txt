cmake_minimum_required(VERSION 3.24.0 FATAL_ERROR)

project(hello VERSION 1.0 LANGUAGES C CXX)

# This is for convenience only when doing 
# editable builds to avoid setting the flag
if (NOT hello_ROOT)
  set(hello_ROOT ${CMAKE_SOURCE_DIR}/build)
endif()

set(BUILD_SHARED_LIBS ON)

find_package(legate_core REQUIRED)
legate_add_cpp_subdirectory(src hello EXPORT hello-export)
legate_add_cffi(${CMAKE_SOURCE_DIR}/src/hello_world.h)
legate_default_python_install(hello EXPORT hello-export)

