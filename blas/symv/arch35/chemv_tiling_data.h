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

// Register-tile row count shared by the host tiling and the RegOrdered kernel:
// one definition so the two sides cannot drift apart.
// The kernel fast path only accepts a rowsPerBlock from {16, 32, 64}; the host
// ChemvRowsPerBlock stays within that set by construction (starts at 16 and
// doubles while below CHEMV_REG_ROWS).
constexpr uint32_t CHEMV_REG_ROWS = 64U;
static_assert(CHEMV_REG_ROWS > 0U && CHEMV_REG_ROWS <= 64U, "one register lane per row");

// Single definition of the supported rowsPerBlock set, consumed by the host
// tiling gate and the kernel fail-silent guard alike. A function-like macro is
// used because the kernel consumes it inside __aicore__ code, where only
// device-callee functions are callable in the ccec device pass.
#define CHEMV_ROWS_PER_BLOCK_VALID(rowsPerBlock) \
    ((rowsPerBlock) == 16U || (rowsPerBlock) == 32U || (rowsPerBlock) == 64U)

struct ChemvTilingData {
    uint32_t n;
    uint32_t lda;
    uint32_t uplo;
    uint32_t nthreads;
    uint32_t rowsPerBlock;
    int64_t incx;
    int64_t incy;
    float alphaReal;
    float alphaImag;
    float betaReal;
    float betaImag;
};
