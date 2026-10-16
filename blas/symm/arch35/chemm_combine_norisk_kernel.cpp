/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "kernel_operator.h"
#include "chemm_tiling_data.h"

#include "chemm_combine_common.h"

template <bool FOUR, bool SPLIT1 = false>
__simt_vf__ __aicore__ LAUNCH_BOUND(512) inline void ChemmCombineNoRisk(
    __gm__ const float* a, __gm__ const float* b, __gm__ const float* w, __gm__ float* c, int64_t m, int64_t n,
    int64_t lda, int64_t ldb, int64_t ldc, uint32_t left, uint32_t upper, float ar, float ai, float br, float bi,
    uint32_t mp, uint32_t np, uint32_t splits, uint32_t core, uint32_t cores, __gm__ uint64_t* queue, uint64_t capacity,
    __gm__ const uint32_t* bounded, uint32_t riskProfile)
{
    const ChemmCombineInput input{w, m, n, ldc, int64_t(mp) * np, mp, splits};
    for (int64_t pos = int64_t(core) * 512 + threadIdx.x; pos < ldc * n; pos += int64_t(cores) * 512) {
        const ChemmCombinedElement element = ChemmLoadCombinedElement<FOUR, SPLIT1>(input, pos);
        if (element.row >= m)
            continue;
        float re = element.re;
        float im = element.im;

        const int64_t kdim = left ? m : n;
        const bool unsafePackedInput = bounded == nullptr || *bounded == 0;
        const bool needFallback = unsafePackedInput || kdim > kChemmRiskReferenceK;
        if (needFallback && ChemmHasCancellationRisk(element.sums, kdim, riskProfile)) {
            if (ChemmQueueFallback(pos, core, capacity, queue))
                continue;
        }
        if (!__builtin_isfinite(re) || !__builtin_isfinite(im))
            ChemmRecoverNonFiniteElement(
                a, b, left != 0U, upper != 0U, m, n, lda, ldb, element.row, element.col, re, im);
        ChemmWriteCombinedElement(c, pos, re, im, ar, ai, br, bi);
    }
}
__global__ __aicore__ void chemm_combine_norisk_kernel(
    GM_ADDR a, GM_ADDR b, GM_ADDR w, GM_ADDR c, ChemmTiling t, uint32_t mp, uint32_t np, uint32_t splits,
    uint32_t products, GM_ADDR queue, uint64_t capacity, GM_ADDR bounded, uint32_t riskProfile)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (products == 4) {
        if (splits == 1) {
            AscendC::Simt::VF_CALL<ChemmCombineNoRisk<true, true>>(
                AscendC::Simt::Dim3(512), (__gm__ const float*)a, (__gm__ const float*)b, (__gm__ const float*)w,
                (__gm__ float*)c, t.m, t.n, t.lda, t.ldb, t.ldc, t.left, t.upper, t.ar, t.ai, t.br, t.bi, mp, np,
                splits, AscendC::GetBlockIdx(), AscendC::GetBlockNum(), (__gm__ uint64_t*)queue, capacity,
                (__gm__ const uint32_t*)bounded, riskProfile);
        } else {
            AscendC::Simt::VF_CALL<ChemmCombineNoRisk<true>>(
                AscendC::Simt::Dim3(512), (__gm__ const float*)a, (__gm__ const float*)b, (__gm__ const float*)w,
                (__gm__ float*)c, t.m, t.n, t.lda, t.ldb, t.ldc, t.left, t.upper, t.ar, t.ai, t.br, t.bi, mp, np,
                splits, AscendC::GetBlockIdx(), AscendC::GetBlockNum(), (__gm__ uint64_t*)queue, capacity,
                (__gm__ const uint32_t*)bounded, riskProfile);
        }
    } else {
        if (splits == 1) {
            AscendC::Simt::VF_CALL<ChemmCombineNoRisk<false, true>>(
                AscendC::Simt::Dim3(512), (__gm__ const float*)a, (__gm__ const float*)b, (__gm__ const float*)w,
                (__gm__ float*)c, t.m, t.n, t.lda, t.ldb, t.ldc, t.left, t.upper, t.ar, t.ai, t.br, t.bi, mp, np,
                splits, AscendC::GetBlockIdx(), AscendC::GetBlockNum(), (__gm__ uint64_t*)queue, capacity,
                (__gm__ const uint32_t*)bounded, riskProfile);
        } else {
            AscendC::Simt::VF_CALL<ChemmCombineNoRisk<false>>(
                AscendC::Simt::Dim3(512), (__gm__ const float*)a, (__gm__ const float*)b, (__gm__ const float*)w,
                (__gm__ float*)c, t.m, t.n, t.lda, t.ldb, t.ldc, t.left, t.upper, t.ar, t.ai, t.br, t.bi, mp, np,
                splits, AscendC::GetBlockIdx(), AscendC::GetBlockNum(), (__gm__ uint64_t*)queue, capacity,
                (__gm__ const uint32_t*)bounded, riskProfile);
        }
    }
}

// Compact narrow-output epilogue for unit alpha and zero beta.  Output columns
// are split into enough row bands to use the available vector cores; each task
// computes its column and band once, keeping divide/modulo out of the row loop.
__simt_vf__ __aicore__ LAUNCH_BOUND(384) inline void ChemmCombineNarrowOutputSimt384(
    __gm__ const float* w, __gm__ float* c, uint32_t m, uint32_t n, uint32_t ldc, uint32_t mp, uint32_t np,
    uint32_t core, uint32_t cores)
{
#pragma clang fp contract(off)
    const uint32_t bandsPerColumn = cores >= n ? cores / n : 1U;
    const uint32_t rowsPerBand = (m + bandsPerColumn - 1) / bandsPerColumn;
    const uint32_t tasks = n * bandsPerColumn;
    const uint32_t plane = mp * np;
    for (uint32_t task = core; task < tasks; task += cores) {
        const uint32_t col = task / bandsPerColumn;
        const uint32_t band = task % bandsPerColumn;
        const uint32_t rowBegin = band * rowsPerBand;
        const uint32_t rowEnd = (rowBegin + rowsPerBand) < m ? rowBegin + rowsPerBand : m;
        const uint32_t outputBase = col * ldc;
        const uint32_t productBase = col * mp;
        for (uint32_t row = rowBegin + threadIdx.x; row < rowEnd; row += 384) {
            const uint32_t q = productBase + row;
            const float p0 = w[q];
            const float p1 = w[plane + q];
            const float p2 = w[2 * plane + q];
            const uint32_t pos = outputBase + row;
            c[2 * pos] = p0 - p1;
            c[2 * pos + 1] = (p2 - p0) - p1;
        }
    }
}

__global__ __aicore__ void chemm_combine_narrow_output_simt384_kernel(
    GM_ADDR w, GM_ADDR c, ChemmTiling t, uint32_t mp, uint32_t np)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::Simt::VF_CALL<ChemmCombineNarrowOutputSimt384>(
        AscendC::Simt::Dim3(384), (__gm__ const float*)w, (__gm__ float*)c, uint32_t(t.m), uint32_t(t.n),
        uint32_t(t.ldc), mp, np, AscendC::GetBlockIdx(), AscendC::GetBlockNum());
}

void chemm_combine_narrow_output_do(
    const void* w, void* c, ChemmTiling t, uint32_t mp, uint32_t np, uint32_t cores, void* stream)
{
    chemm_combine_narrow_output_simt384_kernel<<<cores, nullptr, stream>>>((uint8_t*)w, (uint8_t*)c, t, mp, np);
}

void chemm_combine_norisk_do(
    const void* a, const void* b, void* w, void* c, ChemmTiling t, uint32_t mp, uint32_t np, uint32_t splits,
    uint32_t products, uint32_t cores, void* stream, void* queue, uint64_t capacity, void* bounded,
    uint32_t riskProfile)
{
    chemm_combine_norisk_kernel<<<cores, nullptr, stream>>>(
        (uint8_t*)a, (uint8_t*)b, (uint8_t*)w, (uint8_t*)c, t, mp, np, splits, products, (uint8_t*)queue, capacity,
        (uint8_t*)bounded, riskProfile);
}
