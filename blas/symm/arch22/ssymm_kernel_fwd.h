/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <cstdint>
#include "cann_ops_blas_common.h"

#define GM_ADDR uint8_t*

struct SsymmRightCubeUnifiedConfig;
struct SsymmLeftCubeConfig;

void ssymm_kernel_do(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR tilingGm,
                     aclblasSideMode_t side, aclblasFillMode_t uplo,
                     uint32_t numBlocks, void *stream);

void ssymm_right_cube_unified_do(GM_ADDR a, GM_ADDR aDense, GM_ADDR b, GM_ADDR c, GM_ADDR scratch,
                                 GM_ADDR tilingGm, GM_ADDR configGm,
                                 const SsymmRightCubeUnifiedConfig &config, uint32_t numBlocks, void *stream);

void ssymm_right_cube_unified_postprocess_do(GM_ADDR scratch, GM_ADDR c, GM_ADDR configGm, void *stream);

void ssymm_left_cube_do(GM_ADDR aSym, GM_ADDR workspace, GM_ADDR b, GM_ADDR scratch,
                        GM_ADDR configGm, void *stream);

void ssymm_left_cube_postprocess_do(GM_ADDR scratch, GM_ADDR c, GM_ADDR configGm, void *stream);

