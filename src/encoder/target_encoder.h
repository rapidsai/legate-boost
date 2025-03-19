/* Copyright 2025 NVIDIA Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */
#pragma once
#include "legate_library.h"
#include "legateboost.h"

namespace legateboost {
class TargetEncoderMeanTask : public Task<TargetEncoderMeanTask, TARGET_ENCODER_MEAN> {
 public:
  static constexpr auto CPU_VARIANT_OPTIONS = legate::VariantOptions{}.with_has_allocations(false);
  static constexpr auto GPU_VARIANT_OPTIONS = legate::VariantOptions{}.with_has_allocations(false);
  static inline const auto TASK_CONFIG =
    legate::TaskConfig{legate::LocalTaskID{TARGET_ENCODER_MEAN}};

  static void cpu_variant(legate::TaskContext context);
#ifdef LEGATEBOOST_USE_CUDA
  static void gpu_variant(legate::TaskContext context);
#endif
};

class TargetEncoderVarianceTask : public Task<TargetEncoderVarianceTask, TARGET_ENCODER_VARIANCE> {
 public:
  static constexpr auto CPU_VARIANT_OPTIONS = legate::VariantOptions{}.with_has_allocations(false);
  static constexpr auto GPU_VARIANT_OPTIONS = legate::VariantOptions{}.with_has_allocations(false);
  static inline const auto TASK_CONFIG =
    legate::TaskConfig{legate::LocalTaskID{TARGET_ENCODER_VARIANCE}};

  static void cpu_variant(legate::TaskContext context);
#ifdef LEGATEBOOST_USE_CUDA
  static void gpu_variant(legate::TaskContext context);
#endif
};

class TargetEncoderEncodeTask : public Task<TargetEncoderEncodeTask, TARGET_ENCODER_ENCODE> {
 public:
  static constexpr auto CPU_VARIANT_OPTIONS = legate::VariantOptions{}.with_has_allocations(false);
  static constexpr auto GPU_VARIANT_OPTIONS = legate::VariantOptions{}.with_has_allocations(false);
  static inline const auto TASK_CONFIG =
    legate::TaskConfig{legate::LocalTaskID{TARGET_ENCODER_ENCODE}};

  static void cpu_variant(legate::TaskContext context);
#ifdef LEGATEBOOST_USE_CUDA
  static void gpu_variant(legate::TaskContext context);
#endif
};
}  // namespace legateboost
