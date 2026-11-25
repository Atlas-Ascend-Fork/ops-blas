/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <cstdint>
#include "gemm_tiling_data.h"

#ifndef GM_ADDR
#define GM_ADDR __gm__ uint8_t*
#endif

// FP16 (half) GEMM kernel launchers. The cube kernel computes op(A)*op(B) with
// FLOAT32 accumulation in L0C; the epilogue kernels apply alpha/beta scaling
// in FLOAT32 precision and round the final result back to FLOAT16.
void hgemm_kernel_do(
    uint32_t numBlocks, void* stream, GM_ADDR a, GM_ADDR b, GM_ADDR c, int32_t outputFp32,
    const GemmTilingData& tilingData);

void hgemm_alpha_beta_do(
    uint32_t numBlocks, void* stream, GM_ADDR tempAB, GM_ADDR cOrig, GM_ADDR cOut, const GemmTilingData& tilingData);

void hgemm_scale_do(
    uint32_t numBlocks, void* stream, GM_ADDR cInOut, int32_t m, int32_t n, int32_t ldc, float betaReal);
