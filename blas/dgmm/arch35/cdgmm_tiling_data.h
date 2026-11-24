/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>

constexpr uint32_t CDGMM_MAX_THREADS = 1024U;
constexpr uint32_t CDGMM_MIN_THREADS = 128U;
constexpr uint32_t CDGMM_MIN_CORE_ELEMENTS = 1024U;
constexpr uint32_t CDGMM_ALIGNMENT = 16U; // 128 bytes of complex64.

struct CdgmmTilingData {
    uint64_t totalElements;
    uint64_t elementsPerBlock;
    uint32_t m;
    uint32_t n;
    uint32_t lda;
    uint32_t ldc;
    int32_t incx;
    uint32_t left;
    uint32_t compact;
    uint32_t wideIndex;
    uint32_t blockCount;
    uint32_t threadCount;
};

// Called only after argument validation and the empty-matrix return.
inline CdgmmTilingData MakeCdgmmTiling(
    bool left, uint32_t m, uint32_t n, uint32_t lda, uint32_t ldc, int32_t incx, uint32_t coreCount)
{
    CdgmmTilingData tiling{};
    tiling.m = m;
    tiling.n = n;
    tiling.lda = lda;
    tiling.ldc = ldc;
    tiling.incx = incx;
    tiling.left = left ? 1U : 0U;
    tiling.compact = (lda == m && ldc == m) ? 1U : 0U;
    tiling.totalElements = static_cast<uint64_t>(m) * n;

    const uint64_t absIncx =
        incx < 0 ? static_cast<uint64_t>(-static_cast<int64_t>(incx)) : static_cast<uint64_t>(incx);
    const uint64_t xLength = left ? m : n;
    const uint64_t matrixSpan = static_cast<uint64_t>(std::max(lda, ldc)) * n;
    const uint64_t xSpan = (xLength - 1U) * absIncx + 1U;
    // Check float-lane offsets as well as logical element counts.
    const uint64_t maxLane = std::max(matrixSpan, xSpan) * 2U - 1U;
    tiling.wideIndex = maxLane > static_cast<uint64_t>(std::numeric_limits<int32_t>::max());

    const uint64_t wanted = (tiling.totalElements - 1U) / CDGMM_MIN_CORE_ELEMENTS + 1U;
    const uint64_t available = std::min<uint32_t>(coreCount, 65535U);
    const uint64_t blocks = std::max<uint64_t>(1U, std::min(wanted, available));
    const uint64_t perBlock = (tiling.totalElements - 1U) / blocks + 1U;
    tiling.elementsPerBlock = ((perBlock - 1U) / CDGMM_ALIGNMENT + 1U) * CDGMM_ALIGNMENT;
    tiling.blockCount = static_cast<uint32_t>((tiling.totalElements - 1U) / tiling.elementsPerBlock + 1U);

    const uint32_t threadLimit = tiling.wideIndex ? 512U : CDGMM_MAX_THREADS;
    tiling.threadCount = CDGMM_MIN_THREADS;
    while (tiling.threadCount < threadLimit && tiling.threadCount < tiling.elementsPerBlock) {
        tiling.threadCount *= 2U;
    }
    return tiling;
}
