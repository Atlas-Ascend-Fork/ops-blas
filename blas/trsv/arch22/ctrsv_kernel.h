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
#include "kernel_operator.h"

constexpr uint32_t CTRSV_TILING_REGION_BYTES = 512U;
constexpr uint32_t CTRSV_GROUP_ROWS = 256U;
constexpr uint32_t CTRSV_WORKSPACE_ALIGNMENT = 512U;
constexpr uint32_t CTRSV_FLAG_SLOT_BYTES = 32U;

struct CtrsvTilingData {
    int32_t n, lda, incx, uplo, trans, diag;
};
static_assert(
    sizeof(CtrsvTilingData) <= CTRSV_TILING_REGION_BYTES,
    "Ctrsv tiling metadata exceeds its reserved workspace region");

struct CtrsvPreparedArgs {
    uint32_t n, lda;
    int32_t uplo;
    uint32_t blocks;
    uint32_t flags, ping, diagonalReal, diagonalImag;
};

void ctrsv_tiny_do(uint8_t* a, uint8_t* x, uint32_t n, int32_t uplo, int32_t trans, int32_t diag, void* stream);
void ctrsv_ordered_unit_do(uint8_t* a, uint8_t* x, uint32_t n, int32_t uplo, int32_t trans, void* stream);
void ctrsv_kernel_do(
    uint8_t* a, uint8_t* x, uint8_t* workspace, int32_t trans, int32_t diag, bool prepared, bool tail,
    const CtrsvPreparedArgs& args, void* stream);
