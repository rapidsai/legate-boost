
legate_cpp_library_template(legategbm TEMPLATE_SOURCES)

add_library(
  legategbm
  legategbm.h
  quantile.cc
  ${TEMPLATE_SOURCES}
)

target_include_directories(legategbm
  PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>
  INTERFACE
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(legategbm PRIVATE legate::core)

include(ExternalProject)
ExternalProject_Add(datasketches
  GIT_REPOSITORY https://github.com/apache/datasketches-cpp.git
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
)
ExternalProject_Get_property(datasketches INSTALL_DIR)
set(datasketches_INSTALL_DIR ${INSTALL_DIR})
message("Source dir of datasketches = ${datasketches_INSTALL_DIR}")
target_include_directories(legategbm 
                          PRIVATE ${datasketches_INSTALL_DIR}/include/DataSketches)
add_dependencies(legategbm datasketches)