
add_library(hello
  hello_world.cc
  hello_world.h)

target_include_directories(hello
  PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>
  INTERFACE
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(hello PRIVATE legate::core)

