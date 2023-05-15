
legate_cpp_library_template(legateboost TEMPLATE_SOURCES)

add_library(
  legateboost
  legateboost.h
  build_tree.cc
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
