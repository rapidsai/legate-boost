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

#pragma once

#include "legate.h"

namespace legateboost {

struct Registry {
  static auto get_registrar() -> legate::TaskRegistrar&;
};

template <typename T, int ID>
struct Task : public legate::LegateTask<T> {
 private:
  Task() = default;

 public:
  using Registrar = Registry;
  // NOLINTNEXTLINE(bugprone-dynamic-static-initializers)
  static inline const auto TASK_CONFIG = legate::TaskConfig{legate::LocalTaskID{ID}};
  friend T;
};

}  // namespace legateboost
