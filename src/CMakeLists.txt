
legate_cpp_library_template(legateboost TEMPLATE_SOURCES)
if(Legion_USE_CUDA)
  rapids_find_generate_module(NCCL
    HEADER_NAMES  nccl.h
    LIBRARY_NAMES nccl
  )

  # Currently NCCL has no CMake build-system so we require
  # it built and installed on the machine already
  rapids_find_package(NCCL REQUIRED)

  list(APPEND LB_CUDA_FLAGS --expt-extended-lambda )
  list(APPEND LB_CUDA_FLAGS --expt-relaxed-constexpr )

  if(CMAKE_BUILD_TYPE MATCHES Debug)
    list(APPEND LB_CUDA_FLAGS -G -Xcompiler=-rdynamic)
  else()
    list(APPEND LB_CUDA_FLAGS -lineinfo )
  endif()
endif()

set(legateboost_srcs
  legateboost
  legateboost.h

  models/tree/build_tree.cc
  models/tree/update_tree.cc
  models/tree/predict.cc
  cpp_utils/cpp_utils.h
  cpp_utils/cpp_utils.cc
  utils/gather.cc
)

if(Legion_USE_CUDA)
  list(APPEND legateboost_srcs
    models/tree/predict.cu
    models/tree/build_tree.cu
    special/special.cu
    utils/gather.cu
  )
endif()

add_library(
  legateboost
  ${legateboost_srcs}
  ${TEMPLATE_SOURCES}
)

set_target_properties(legateboost
  PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  POSITION_INDEPENDENT_CODE ON)

target_compile_options(legateboost
      PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${LB_CXX_FLAGS}>"
              "$<$<COMPILE_LANGUAGE:CUDA>:${LB_CUDA_FLAGS}>"
)



if (Legion_USE_CUDA)
  # CUDA_ARCH environment variable can override the CMAKE_CUDA_ARCHITECTURES
  if(DEFINED ENV{CUDA_ARCH})
    message(STATUS "legateboost: CUDA_ARCH=$ENV{CUDA_ARCH}")
    set_property(TARGET legateboost PROPERTY CUDA_ARCHITECTURES $ENV{CUDA_ARCH} )
  endif()
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
