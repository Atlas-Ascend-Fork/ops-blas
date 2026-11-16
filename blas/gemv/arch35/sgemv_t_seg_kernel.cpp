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
 * \file sgemv_t_seg_kernel.cpp
 * \brief Segmented trans=T slab kernels for arch35 sgemv (SIMT).
 *
 * DRAM-streaming slabs whose column count quantizes badly over the warp
 * round-robin slice every column into 2^segShift row segments; the
 * (column, segment) items are dealt round-robin so every warp stays busy.
 * Kept in its own translation unit so the code generation of the plain-slab
 * kernels is unaffected.
 */

#include <cstdint>
#include "cann_ops_blas_common.h"
#include "sgemv_slab_common.h"

// Rows of one row segment: ceil(rows / 2^segShift), rounded up to even so
// every segment start stays 8-byte aligned for float2 pairing (host
// guarantees m and lda even when the PAIR variant is selected).
__simt_callee__ __aicore__ inline uint32_t SgemvTSegRows(uint32_t rows, uint32_t segShift)
{
    uint32_t segs = 1U << segShift;
    uint32_t ceilRows = (rows + segs - 1U) >> segShift;
    return (ceilRows % 2U == 0U) ? ceilRows : ceilRows + 1U;
}

// Everything the segmented item pass needs: segment geometry, the span's
// row count and GM base, and the pointers.
struct SgemvTSegPlan {
    uint32_t items;
    uint32_t segShift;
    uint32_t segs;
    uint32_t segRows;
    uint32_t rows;
    uint32_t rBase;
    uint32_t c0;
    uint32_t lda;
    __gm__ const float* aGm;
    __ubuf__ const float* xUb;
    __ubuf__ float* part;
};

// Assemble the plan for a span of `rows` rows starting at GM row rBase
// (whole column: rows = m, rBase = 0; row tile: the tile's rows/r0).
__simt_callee__ __aicore__ inline SgemvTSegPlan SgemvTSegMakePlan(
    uint32_t nCols, uint32_t segShift, uint32_t rows, uint32_t rBase, uint32_t c0, uint32_t lda,
    __gm__ const float* aGm, __ubuf__ const float* xUb, __ubuf__ float* part)
{
    SgemvTSegPlan plan;
    plan.items = nCols << segShift; // host: items <= SGEMV_T_MAX_ITEMS
    plan.segShift = segShift;
    plan.segs = 1U << segShift;
    plan.segRows = SgemvTSegRows(rows, segShift);
    plan.rows = rows;
    plan.rBase = rBase;
    plan.c0 = c0;
    plan.lda = lda;
    plan.aGm = aGm;
    plan.xUb = xUb;
    plan.part = part;
    return plan;
}

// (column, segment) items dealt round-robin over the warps; the per-item
// partial is warp-reduced and parked in part[].
template <bool PAIR>
__simt_callee__ __aicore__ inline void SgemvTSegDotLoop(
    const SgemvTSegPlan& plan, uint32_t lane, uint32_t wid, uint32_t numWarps)
{
    for (uint32_t it = wid; it < plan.items; it += numWarps) {
        uint32_t cRel = it >> plan.segShift;
        uint32_t seg = it & (plan.segs - 1U);
        uint32_t rr = seg * plan.segRows;
        uint32_t len = (rr < plan.rows) ? ((plan.rows - rr < plan.segRows) ? (plan.rows - rr) : plan.segRows) : 0U;
        __gm__ const float* aCol = plan.aGm + static_cast<uint64_t>(plan.c0 + cRel) * plan.lda + plan.rBase + rr;
        float partial = SgemvTColumnDot<PAIR>(len, lane, aCol, plan.xUb + rr);
        partial = SgemvWarpReduce(partial);
        if (lane == 0U) {
            plan.part[it] = partial;
        }
    }
}

// Sum the row segments of column j from the per-item partials.
__simt_callee__ __aicore__ inline float SgemvTSegCombineOne(
    uint32_t j, uint32_t segShift, uint32_t segs, __ubuf__ const float* part)
{
    float s = 0.0f;
    uint32_t base = j << segShift;
    for (uint32_t sg = 0; sg < segs; ++sg) {
        s += part[base + sg];
    }
    return s;
}

// Segmented small-m T: per-item partials meet in __ubuf__ (one syncthreads
// per block) and threads combine the segments of each column before the
// direct write-back.
template <bool PAIR>
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void SgemvTSlabSeg(
    uint32_t m, uint32_t n, uint32_t lda, float alpha, float beta, uint32_t betaIsZero, int64_t incy, uint32_t chunkLen,
    uint32_t segShift, __gm__ const float* aGm, __gm__ const float* xGm, __gm__ float* yGm)
{
    __ubuf__ float xUb[SGEMV_UB_X_FLOATS];
    __ubuf__ float part[SGEMV_T_MAX_ITEMS];

    SgemvTSmallCtx ctx = SgemvTSmallPrologue(m, n, chunkLen, xGm, xUb);

    uint32_t nCols = ctx.c1 - ctx.c0;
    uint32_t segs = 1U << segShift;
    SgemvTSegPlan plan = SgemvTSegMakePlan(nCols, segShift, m, 0U, ctx.c0, lda, aGm, xUb, part);
    SgemvTSegDotLoop<PAIR>(plan, ctx.lane, ctx.wid, ctx.numWarps);
    asc_syncthreads();

    for (uint32_t j = ctx.t; j < nCols; j += blockDim.x) {
        float s = SgemvTSegCombineOne(j, segShift, segs, part);
        int64_t yIdx = SgemvStridedIdx(static_cast<int64_t>(ctx.c0 + j), static_cast<int64_t>(n), incy);
        SgemvWriteBack(yGm, yIdx, s, alpha, beta, betaIsZero);
    }
}

// Segmented large-m T: the tile's columns are cut into segments the same
// way; per-column partials (after the in-block segment combine) go to
// ws[rowTile*n + col] and the shared finalize applies alpha/beta.
template <bool PAIR>
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void SgemvTSlabTileSeg(
    uint32_t m, uint32_t n, uint32_t lda, uint32_t chunkLen, uint32_t colChunks, uint32_t segShift,
    __gm__ const float* aGm, __gm__ const float* xGm, __gm__ float* wsGm)
{
    __ubuf__ float xUb[SGEMV_T_TILE_ROWS];
    __ubuf__ float part[SGEMV_T_MAX_ITEMS];

    SgemvTTileCtx ctx;
    if (!SgemvTTilePrologue(m, n, chunkLen, colChunks, xGm, xUb, ctx)) {
        return;
    }

    uint32_t nCols = ctx.c1 - ctx.c0;
    uint32_t segs = 1U << segShift;
    SgemvTSegPlan plan = SgemvTSegMakePlan(nCols, segShift, ctx.rows, ctx.r0, ctx.c0, lda, aGm, xUb, part);
    SgemvTSegDotLoop<PAIR>(plan, ctx.lane, ctx.wid, ctx.numWarps);
    asc_syncthreads();

    for (uint32_t j = ctx.t; j < nCols; j += blockDim.x) {
        float s = SgemvTSegCombineOne(j, segShift, segs, part);
        wsGm[static_cast<uint64_t>(ctx.rowTile) * n + ctx.c0 + j] = s;
    }
}

// ==========================================================================
//  Global kernels
// ==========================================================================

template <bool PAIR>
__global__ __aicore__ void sgemv_t_slabseg_kernel(
    GM_ADDR a, GM_ADDR x, GM_ADDR y, uint32_t m, uint32_t n, uint32_t lda, float alpha, float beta, uint32_t betaIsZero,
    int64_t incy, uint32_t chunkLen, uint32_t segShift)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    auto* aGm = reinterpret_cast<__gm__ const float*>(a);
    auto* xGm = reinterpret_cast<__gm__ const float*>(x);
    auto* yGm = reinterpret_cast<__gm__ float*>(y);
    asc_vf_call<SgemvTSlabSeg<PAIR>>(
        dim3{SIMT_MAX_THREAD_NUM, 1, 1}, m, n, lda, alpha, beta, betaIsZero, incy, chunkLen, segShift, aGm, xGm, yGm);
}

template <bool PAIR>
__global__ __aicore__ void sgemv_t_tileseg_kernel(
    GM_ADDR a, GM_ADDR x, GM_ADDR workSpace, uint32_t m, uint32_t n, uint32_t lda, uint32_t chunkLen,
    uint32_t colChunks, uint32_t segShift)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    auto* aGm = reinterpret_cast<__gm__ const float*>(a);
    auto* xGm = reinterpret_cast<__gm__ const float*>(x);
    auto* wsGm = reinterpret_cast<__gm__ float*>(workSpace);
    asc_vf_call<SgemvTSlabTileSeg<PAIR>>(
        dim3{SIMT_MAX_THREAD_NUM, 1, 1}, m, n, lda, chunkLen, colChunks, segShift, aGm, xGm, wsGm);
}

// Launch entry for both segmented T variants (called from SgemvLaunchSlab).
void SgemvLaunchTSeg(
    uint8_t* a, uint8_t* x, uint8_t* y, uint8_t* workSpace, const SgemvTilingData& tiling, uint32_t numBlocks,
    void* stream)
{
    bool usePairs = tiling.usePairs != 0U;
    if (tiling.rowTiles != 0U) {
        if (usePairs) {
            sgemv_t_tileseg_kernel<true><<<numBlocks, nullptr, stream>>>(
                a, x, workSpace, tiling.m, tiling.n, tiling.lda, tiling.chunkLen, tiling.colChunks, tiling.segShift);
        } else {
            sgemv_t_tileseg_kernel<false><<<numBlocks, nullptr, stream>>>(
                a, x, workSpace, tiling.m, tiling.n, tiling.lda, tiling.chunkLen, tiling.colChunks, tiling.segShift);
        }
        return;
    }
    if (usePairs) {
        sgemv_t_slabseg_kernel<true><<<numBlocks, nullptr, stream>>>(
            a, x, y, tiling.m, tiling.n, tiling.lda, tiling.alpha, tiling.beta, tiling.betaIsZero, tiling.incy,
            tiling.chunkLen, tiling.segShift);
    } else {
        sgemv_t_slabseg_kernel<false><<<numBlocks, nullptr, stream>>>(
            a, x, y, tiling.m, tiling.n, tiling.lda, tiling.alpha, tiling.beta, tiling.betaIsZero, tiling.incy,
            tiling.chunkLen, tiling.segShift);
    }
}
