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

// uplo encoding shared by host and kernel: 0 = ACLBLAS_LOWER, 1 = ACLBLAS_UPPER.
constexpr uint32_t CTRTTP_UPLO_LOWER = 0U;
constexpr uint32_t CTRTTP_UPLO_UPPER = 1U;

// Stride of a complex64 element in 32-byte UB lanes, and the number of packed elements per lane block.
constexpr uint32_t CTRTTP_LANES_PER_COMPLEX = 2U;
constexpr uint32_t CTRTTP_BYTES_PER_COMPLEX = 8U;
constexpr uint32_t CTRTTP_COMPLEX_PER_BLOCK = 4U;
// One column is one DMA pair, so the queue depth decides how many column transfers stay in
// flight per core. Four slots hide the DMA latency of the mid-size orders, where a column is
// only a few KiB, while the 4 x 62 KiB tile still covers the longest 8192-order column.
constexpr uint32_t CTRTTP_QUEUE_DEPTH = 4U;
// Upper bound of one GM<->UB burst; 14336 complex64 elements are 112 KiB, the largest
// transfer the DataCopyPad path is exercised with elsewhere in this repository.
constexpr uint32_t CTRTTP_MAX_CHUNK_ELEMS = 14336U;

/**
 * @brief Tiling data of aclblasCtrttp.
 *
 * Every block packs the columns j = blockIdx, blockIdx + useCoreNum, ..., so the columns (and
 * therefore the bytes and the number of DMA bursts) are spread evenly over the blocks for any
 * uplo / lda. Offsets are 64-bit because n = 8192 already packs 33 558 528 elements.
 */
struct CtrttpTilingData {
    uint32_t n;             // triangular matrix order
    uint32_t lda;           // leading dimension of A (>= max(1, n))
    uint32_t uplo;          // CTRTTP_UPLO_LOWER / CTRTTP_UPLO_UPPER
    uint32_t useCoreNum;    // number of AIV cores taking part
    uint32_t tileElems;     // max complex64 elements per UB burst
    uint64_t totalPacked;   // n * (n + 1) / 2
};

