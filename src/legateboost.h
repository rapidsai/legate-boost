/* Copyright 2024 NVIDIA Corporation
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

#ifndef SRC_LEGATEBOOST_H_
#define SRC_LEGATEBOOST_H_

enum LegateBoostOpCode {  // NOLINT(performance-enum-size)
  OP_CODE_BASE = 0,
  BUILD_TREE   = 1,
  PREDICT      = 2,
  UPDATE_TREE  = 3,
  /* special */
  ERF     = 4,
  LGAMMA  = 5,
  TGAMMA  = 6,
  DIGAMMA = 7,
  ZETA    = 8,
  /**/
  GATHER                  = 9,
  RBF                     = 10,
  BUILD_NN                = 11,
  TARGET_ENCODER_MEAN     = 12,
  TARGET_ENCODER_ENCODE   = 13,
  TARGET_ENCODER_VARIANCE = 14,
};

#endif  // SRC_LEGATEBOOST_H_
