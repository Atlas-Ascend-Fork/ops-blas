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
 * \file sgemv_tiling_data.h
 * \brief Tiling data structure shared between host and kernel for sgemv (SIMT).
 */

#pragma once

#include <cstdint>

struct SgemvTilingData {
    uint32_t numThreads;   // threads per block
    uint32_t rowsPerBlock; // ceil(outDim / numBlocks)
    uint32_t m;            // matrix A rows
    uint32_t n;            // matrix A columns
    uint32_t lda;          // A leading dimension
    uint32_t trans;        // 0 = N, 1 = T/C
    float alpha;           // scalar alpha
    float beta;            // scalar beta
    int64_t incx;          // x stride
    int64_t incy;          // y stride
    // Slab fast-path fields (host fills them when useSlab != 0; the generic
    // path leaves them zeroed).
    uint32_t useSlab;    // 1 = column-chunked slab kernel family
    uint32_t chunkLen;   // columns per chunk
    uint32_t colChunks;  // number of column chunks
    uint32_t segShift;   // trans=T small-m: log2(row segments per column)
    uint32_t rowTiles;   // N: row tiles per chunk; T large-m: row tiles (0 = direct write-back)
    uint32_t betaIsZero; // 1 = beta == 0 (y is not read)
    uint32_t usePairs;   // 1 = float2 row-pair loads (m even and lda even)
};
