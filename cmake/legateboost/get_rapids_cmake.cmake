#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#=============================================================================

include_guard(GLOBAL)

macro(legateboost_get_rapids_cmake)
  list(APPEND CMAKE_MESSAGE_CONTEXT "get_rapids_cmake")

  if(NOT rapids-cmake-version)
    # default
    set(rapids-cmake-version 24.08)
    set(rapids-cmake-sha "3cc764f287a6f3caeee5dd1c96c24b1710d4cdf1")
  endif()

  if(NOT EXISTS ${CMAKE_BINARY_DIR}/RAPIDS.cmake)
    file(DOWNLOAD
      https://raw.githubusercontent.com/rapidsai/rapids-cmake/branch-${rapids-cmake-version}/RAPIDS.cmake
      ${CMAKE_BINARY_DIR}/RAPIDS.cmake)
  endif()
  include(${CMAKE_BINARY_DIR}/RAPIDS.cmake)
  include(rapids-cmake)
  include(rapids-cpm)
  include(rapids-cuda)
  include(rapids-export)
  include(rapids-find)

  list(POP_BACK CMAKE_MESSAGE_CONTEXT)
endmacro()
