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

#include "kernel_operator.h"
#include "simt_api/common_functions.h"

#include "cdgmm_compute.h"
#include "cdgmm_kernel.h"

template <typename Index, bool Left, bool Compact>
__simt_vf__ __aicore__ LAUNCH_BOUND(CDGMM_MAX_THREADS) inline void CdgmmSimt(
    const __gm__ float* a, const __gm__ float* x, __gm__ float* c, uint64_t totalElements, uint64_t elementsPerBlock,
    uint32_t m, uint32_t n, uint32_t lda, uint32_t ldc, int32_t incx, uint32_t coreIndex)
{
    const uint64_t begin = static_cast<uint64_t>(coreIndex) * elementsPerBlock;
    const uint64_t candidateEnd = begin + elementsPerBlock;
    const uint64_t end = candidateEnd < totalElements ? candidateEnd : totalElements;
    const uint32_t xLength = Left ? m : n;
    const Index step = static_cast<Index>(blockDim.x);
    const Index first = static_cast<Index>(begin + threadIdx.x);
    Index column = first / static_cast<Index>(m);
    Index row = first - column * static_cast<Index>(m);
    const Index columnStep = step / static_cast<Index>(m);
    const Index rowStep = step - columnStep * static_cast<Index>(m);
    for (Index index = first; static_cast<uint64_t>(index) < end; index += step) {
        CdgmmComputeElementAtCoordinates<Index, Left, Compact>(index, row, column, lda, ldc, xLength, incx, a, x, c);
        row += rowStep;
        column += columnStep;
        if (row >= static_cast<Index>(m)) {
            row -= static_cast<Index>(m);
            ++column;
        }
    }
}

template <typename Index, bool Left, bool Compact>
__aicore__ inline void LaunchCdgmmVf(GM_ADDR a, GM_ADDR x, GM_ADDR c, const CdgmmTilingData& tiling)
{
    const uint32_t coreIndex = AscendC::GetBlockIdx();
    asc_vf_call<CdgmmSimt<Index, Left, Compact>>(
        dim3{tiling.threadCount, 1U, 1U}, reinterpret_cast<const __gm__ float*>(a),
        reinterpret_cast<const __gm__ float*>(x), reinterpret_cast<__gm__ float*>(c), tiling.totalElements,
        tiling.elementsPerBlock, tiling.m, tiling.n, tiling.lda, tiling.ldc, tiling.incx, coreIndex);
}

template <typename Index>
__aicore__ inline void DispatchCdgmm(GM_ADDR a, GM_ADDR x, GM_ADDR c, const CdgmmTilingData& tiling)
{
    if (tiling.left != 0U) {
        if (tiling.compact != 0U) {
            LaunchCdgmmVf<Index, true, true>(a, x, c, tiling);
        } else {
            LaunchCdgmmVf<Index, true, false>(a, x, c, tiling);
        }
    } else if (tiling.compact != 0U) {
        LaunchCdgmmVf<Index, false, true>(a, x, c, tiling);
    } else {
        LaunchCdgmmVf<Index, false, false>(a, x, c, tiling);
    }
}
