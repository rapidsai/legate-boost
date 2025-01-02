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

include(cmake/legateboost/get_rapids_cmake.cmake)
legateboost_get_rapids_cmake()

rapids_cuda_init_architectures(legateboost)
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
find_package(legate REQUIRED)

if(Legion_USE_CUDA)
  enable_language(CUDA)
endif()

find_package(BLAS REQUIRED)

if (SANITIZE)
  message(STATUS "Adding sanitizer flags")
  add_compile_options(-fsanitize=address)
  add_link_options(-fsanitize=address)
endif()

add_subdirectory(src)

include(cmake/legateboost/generate_install_info.cmake)
legateboost_generate_install_info(${CMAKE_CURRENT_SOURCE_DIR}/src/legateboost.h TARGET legateboost)

include(GNUInstallDirs)

rapids_cmake_install_lib_dir(lib_dir)

install(TARGETS legateboost DESTINATION ${lib_dir} EXPORT legateboost-export)

install(DIRECTORY cmake/legateboost/
        DESTINATION "${lib_dir}/cmake/legateboost" FILES_MATCHING
        PATTERN "*.cmake")

rapids_export(INSTALL legateboost
  EXPORT_SET legateboost-export
  GLOBAL_TARGETS legateboost
  NAMESPACE legate::)

# build export targets
rapids_export(BUILD legateboost
  EXPORT_SET legateboost-export
  GLOBAL_TARGETS legateboote
  NAMESPACE legate::)

if(SKBUILD)
  add_library(legateboost_python INTERFACE)
  # NOTE(seberg): Had a local error with the below, but not sure if it is needed:
  # add_library(legate::legateboost_python ALIAS legateboost_python)
  target_link_libraries(legateboost_python INTERFACE legate::legate legate::legateboost)

  install(TARGETS legateboost_python DESTINATION ${lib_dir} EXPORT legateboost-export)

  rapids_export(INSTALL legateboost_python EXPORT_SET legateboost-export
    GLOBAL_TARGETS legateboost_python NAMESPACE legate::)
endif()
