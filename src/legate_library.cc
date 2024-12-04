/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <legate.h>
#include <memory>
#include "legate_library.h"
#include "mapper.h"

namespace legateboost {

static const char* const library_name = "legateboost";

/*static*/ auto Registry::get_registrar() -> legate::TaskRegistrar&
{
  static legate::TaskRegistrar registrar;
  return registrar;
}

namespace {
void registration_callback()
{
  auto context = legate::Runtime::get_runtime()->create_library(
    library_name, legate::ResourceConfig{}, std::make_unique<LegateboostMapper>());

  Registry::get_registrar().register_all_tasks(context);
}
}  // namespace

}  // namespace legateboost

extern "C" {

void legateboost_perform_registration(void) { legateboost::registration_callback(); }
}
