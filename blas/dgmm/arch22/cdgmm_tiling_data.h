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
 * \file cdgmm_tiling_data.h
 * \brief Tiling data structure for aclblasCdgmm (arch22).
 *        Shared by host side and kernel side to avoid layout mismatch.
 *
 *        Row-major storage: A and C are m x n row-major complex matrices.
 *        Complex elements are stored as interleaved float pairs (real, imag).
 *        Only LEFT mode is implemented; RIGHT is rejected by the Host.
 */

#pragma once

#include <cstdint>

static constexpr uint32_t CDGMM_MODE_LEFT = 0;
static constexpr uint32_t CDGMM_MODE_RIGHT = 1;

static constexpr uint32_t CDGMM_MAX_CORES = 40;

/*!
 * \brief Tiling data for Cdgmm.
 *
 * Row-split decomposition: each core handles a contiguous range of
 * rows [startRow[i], startRow[i] + rowCount[i]).  The Host balances
 * rows across at most CDGMM_MAX_CORES cores (capped by aivCoreNum and m).
 *
 * Matrices A and C are row-major.  lda/ldc are row strides (number of
 * complex elements between consecutive rows).  Complex elements are
 * stored as interleaved float pairs (real, imag).
 */
struct CdgmmTilingData {
    uint32_t mode;     // normalized: CDGMM_MODE_LEFT or CDGMM_MODE_RIGHT
    uint32_t m;        // number of rows of matrix A/C
    uint32_t n;        // number of columns of matrix A/C
    int32_t  incx;     // stride of vector x (may be negative)
    uint32_t lda;      // row stride of A (row-major, in complex elements)
    uint32_t ldc;      // row stride of C (row-major, in complex elements)

    uint32_t startRow[CDGMM_MAX_CORES];  // start row per core
    uint32_t rowCount[CDGMM_MAX_CORES];  // row count per core
};
