/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file ctrmv_common.h
 * \brief single-precision complex (complex64) trmv common definitions for ascend950
 */

#pragma once

#include <cstdint>

// Fused single-launch small-n path (CtrmvSmall): the whole x vector is staged
// into UB as native float2, so n is bounded by the UB cache layout and incx
// must be 1. The block is one row group per CTRMV_SMALL_THREADS_PER_ROW
// threads (column split + warp-shuffle reduction).
constexpr uint32_t CTRMV_SMALL_N_MAX = 64;
constexpr uint32_t CTRMV_SMALL_THREADS_PER_ROW = 4;
constexpr uint32_t CTRMV_SMALL_BLOCK_THREADS = CTRMV_SMALL_N_MAX * CTRMV_SMALL_THREADS_PER_ROW;

struct CtrmvTilingData {
    uint32_t numThreads;
    uint32_t rowsPerCore;
    uint32_t threadsPerRow;
    uint32_t n;
    uint32_t lda;
    uint32_t uplo;
    uint32_t trans;
    uint32_t diag;
    int64_t incx;
};

void ctrmv_arch35_kernel_do(
    const aclblasComplex* a, aclblasComplex* x, uint8_t* workspace, const CtrmvTilingData& tilingData,
    uint32_t numBlocks, void* stream);

void ctrmv_arch35_small_kernel_do(
    const aclblasComplex* a, aclblasComplex* x, const CtrmvTilingData& tilingData, void* stream);
