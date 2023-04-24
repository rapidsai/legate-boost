
legate_cpp_library_template(legateboost TEMPLATE_SOURCES)

add_library(
  legateboost
  legateboost.h
  quantile.cc
  ${TEMPLATE_SOURCES}
)

set_property(TARGET PROPERTY legateboost COMPILE_WARNING_AS_ERROR ON)

target_include_directories(legateboost
  PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>
  INTERFACE
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(legateboost PRIVATE legate::core)

include(ExternalProject)
ExternalProject_Add(datasketches
  GIT_REPOSITORY https://github.com/apache/datasketches-cpp.git
  GIT_TAG 4.0.1
  GIT_SHALLOW true
  GIT_SUBMODULES ""
  INSTALL_DIR /tmp/datasketches-prefix
  CMAKE_ARGS -DBUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -DCMAKE_INSTALL_PREFIX=/tmp/datasketches-prefix

  # Override the install command to add DESTDIR
  # This is necessary to work around an oddity in the RPM (but not other) package
  # generation, as CMake otherwise picks up the Datasketch files when building
  # an RPM for a dependent package. (RPM scans the directory for files in addition to installing
  # those files referenced in an "install" rule in the cmake file)
  INSTALL_COMMAND env DESTDIR= ${CMAKE_COMMAND} --build . --target install
  UPDATE_DISCONNECTED True
)
ExternalProject_Get_property(datasketches INSTALL_DIR)
set(datasketches_INSTALL_DIR ${INSTALL_DIR})
target_include_directories(legateboost 
                          PRIVATE ${datasketches_INSTALL_DIR}/include/DataSketches)
add_dependencies(legateboost datasketches)