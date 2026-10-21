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
 * \file ctbmv_common.h
 * \brief single-precision complex (complex64) tbmv common definitions for ascend950
 */

#pragma once

#include <cstddef>
#include <cstdint>

#include "cann_ops_blas_common.h"

// 64 KiB UB budget for the cached x window. complex64 = 8 bytes / element => 8192 elements.
static constexpr uint32_t CTBMV_UB_X_ELEMS = 8192;
// For small unit-stride vectors, keeping the cached input and writing the result
// in the same kernel avoids the compute-to-workspace and copy-back launches.
static constexpr uint32_t CTBMV_SINGLE_BLOCK_UB_N = 256;
// At or below this n the kernel runs the serial Netlib path: it avoids the
// workspace and preserves the in-place update order, which wins where launch
// overhead dominates parallel row reduction. The host reads the same bound,
// because below it the kernel ignores the block count.
static constexpr uint32_t CTBMV_NETLIB_SMALL_N = 32;
static constexpr uint32_t CTBMV_WARP_SIZE = 32;
// A wide band leaves every output row a long serial chain that one thread cannot
// hide. From this band width the kernel switches strategy - one warp per row
// with the shuffle reduction, or the UB column walk on the transpose path - and
// the host spreads the shape over more cores instead of pinning it to one
// block. Host and kernel must read the same value: the host sizes the grid
// around the strategy the kernel is about to pick.
static constexpr uint32_t CTBMV_WIDE_BAND_MIN_K = 32;
// Row window for spreading a wide band over cores, and the band widths at which
// the window steps down. A wider band makes each row's chain longer, so a
// smaller window keeps enough resident warps per core to hide the band loads.
static constexpr uint32_t CTBMV_WIDE_BAND_ROWS_PER_BLOCK = 64;
static constexpr uint32_t CTBMV_WIDE_BAND_MEDIUM_MIN_K = 100;
static constexpr uint32_t CTBMV_WIDE_BAND_MEDIUM_ROWS_PER_BLOCK = 48;
static constexpr uint32_t CTBMV_WIDE_BAND_EXTREME_MIN_K = 200;
static constexpr uint32_t CTBMV_WIDE_BAND_EXTREME_ROWS_PER_BLOCK = 32;
// Small wide-band shapes fit in a single UB block, so they are spread with a
// narrower window than the table above.
static constexpr uint32_t CTBMV_SMALL_WIDEBAND_ROWS_PER_BLOCK = 16;
// Sixteen resident warps hide the GM-load and shuffle latency for k=64 rows.
static constexpr uint32_t CTBMV_WARP_REDUCTION_THREADS = 512;

struct CtbmvTilingData {
    uint32_t numThreads;
    uint32_t numBlocks;
    uint32_t rowsPerBlock;
    uint32_t useUb;
    uint32_t n;
    uint32_t k;
    uint32_t lda;
    uint32_t uplo;
    uint32_t trans;
    uint32_t diag;
    int64_t incx;
};

void ctbmv_arch35_kernel_do(
    const aclblasComplex* a, aclblasComplex* x, uint8_t* workspace, const CtbmvTilingData& tilingData,
    uint32_t numBlocks, void* stream);


