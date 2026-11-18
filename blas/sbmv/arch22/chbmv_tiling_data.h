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

// Scalar fallback: FP32 arithmetic on AIV, one owner per output.
// Only the exact 8 output bytes are transferred; the UB slot is DMA aligned.
constexpr uint32_t CHBMV_OUTPUT_BYTES = 32;
constexpr uint32_t CHBMV_VECTOR_ROWS = 64;
constexpr uint32_t CHBMV_VECTOR_SCRATCH_BYTES = 36864;
constexpr uint32_t CHBMV_MIRROR_TILE = 64;
constexpr uint32_t CHBMV_DIRECT_TILE = 32;
struct ChbmvTilingData {
    uint32_t n, k, band, lda, upper, numBlocks, blockRows;
    uint32_t vectorMode, aCacheBytes, xCacheBytes, vectorRows;
    int64_t incx, incy, xStart, yStart;
    float alphaReal, alphaImag, betaReal, betaImag;
};
