
legate_cpp_library_template(legateboost TEMPLATE_SOURCES)

if(Legion_USE_CUDA)
  #find_package(NCCL REQUIRED)
  rapids_find_generate_module(NCCL
    HEADER_NAMES  nccl.h
    LIBRARY_NAMES nccl
  )

  # Currently NCCL has no CMake build-system so we require
  # it built and installed on the machine already
  rapids_find_package(NCCL REQUIRED)
endif()

set(legateboost_srcs
  legateboost
  legateboost.h
  build_tree.cc
  predict.cc
  utils.h
  utils.cc
)

if(Legion_USE_CUDA)
  list(APPEND legateboost_srcs
    predict.cu
  )
endif()

add_library(
  legateboost
  ${legateboost_srcs}
  ${TEMPLATE_SOURCES}
)

if (Legion_USE_CUDA)
  target_compile_definitions(legateboost PRIVATE LEGATEBOOST_USE_CUDA)
endif()

set_property(TARGET PROPERTY legateboost COMPILE_WARNING_AS_ERROR ON)

target_include_directories(legateboost
  PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>
  INTERFACE
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(legateboost PRIVATE legate::core $<TARGET_NAME_IF_EXISTS:NCCL::NCCL>)
