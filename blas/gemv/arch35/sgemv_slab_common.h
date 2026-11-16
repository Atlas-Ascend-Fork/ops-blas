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
 * \file sgemv_slab_common.h
 * \brief Shared constants and device helpers of the arch35 sgemv slab kernels.
 *
 * The segmented T kernels live in their own translation unit so their
 * register and __ubuf__ footprints cannot perturb the code generation of
 * the plain-slab kernels; the helpers both units need are kept here.
 */

#pragma once

#include <cstdint>
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "simt_api/math_functions.h"
#include "simt_api/vector_functions.h"
#include "sgemv_tiling_data.h"
#include "common/helper/kernel_constant.h"

// Disable implicit FMA contraction so every partial product crosses an
// explicit binary32 rounding boundary (matches the generic sgemv kernels).
#pragma clang fp contract(off)

// ==========================================================================
//  Shared constants
// ==========================================================================
static constexpr uint32_t SGEMV_UB_X_FLOATS = 16384; // 64 KB __ubuf__ x cache

// T slab geometry (one column per warp; 2048 threads = 64 warps per block).
static constexpr uint32_t SGEMV_T_WARP_WIDTH = 32;
// Upper bound of (column, segment) items per block. Coarse enough that the
// round-robin over 64 warps quantizes to at most ~1/8 of a round, fine
// enough for part[] to stay a 2 KB __ubuf__ array.
static constexpr uint32_t SGEMV_T_MAX_ITEMS = 512;
// Large-m T path: row tile whose x segment is staged in UB (32 KB).
static constexpr uint32_t SGEMV_T_TILE_ROWS = 8192;

// ==========================================================================
//  Device helpers
// ==========================================================================

// An explicit fma with an exact -0 addend is one correctly-rounded binary32
// multiplication; keeps each product rounded before the following Add.
__simt_callee__ __aicore__ inline float SgemvRoundedMul(float lhs, float rhs) { return fmaf(lhs, rhs, -0.0F); }

// Strided index of logical element i in a vector of logical length len.
__simt_callee__ __aicore__ inline int64_t SgemvStridedIdx(int64_t i, int64_t len, int64_t inc)
{
    return (inc >= 0) ? (i * inc) : ((len - 1 - i) * (-inc));
}

// y[yIdx] = alpha * acc + beta * y[yIdx]; beta == 0 overwrites without
// reading y (keeps Inf/NaN payloads in acc intact, Netlib semantics).
__simt_callee__ __aicore__ inline void SgemvWriteBack(
    __gm__ float* yGm, int64_t yIdx, float acc, float alpha, float beta, uint32_t betaIsZero)
{
    float out = (alpha == 1.0F) ? acc : SgemvRoundedMul(alpha, acc);
    if (betaIsZero != 0U) {
        yGm[yIdx] = out;
    } else if (beta == 1.0F) {
        yGm[yIdx] = out + yGm[yIdx];
    } else {
        yGm[yIdx] = out + SgemvRoundedMul(beta, yGm[yIdx]);
    }
}

__simt_callee__ __aicore__ inline float SgemvWarpReduce(float val)
{
    for (int32_t offset = SGEMV_T_WARP_WIDTH / 2; offset > 0; offset >>= 1) {
        val += asc_shfl_down(val, static_cast<uint32_t>(offset));
    }
    return val;
}

// Row-pair variant (host guarantees m and lda even): each float2 load covers
// two consecutive rows of the column; both products are separately rounded
// and added in row order.
template <bool PAIR>
__simt_callee__ __aicore__ inline float SgemvTColumnDot(
    uint32_t len, uint32_t lane, __gm__ const float* aCol, __ubuf__ const float* xUb)
{
    float acc = 0.0f;
    if constexpr (PAIR) {
        __gm__ const float2* aCol2 = reinterpret_cast<__gm__ const float2*>(aCol);
        __ubuf__ const float2* xUb2 = reinterpret_cast<__ubuf__ const float2*>(xUb);
        for (uint32_t p = lane; p < len / 2U; p += SGEMV_T_WARP_WIDTH) {
            float2 av = aCol2[p];
            float2 xv = xUb2[p];
            acc += SgemvRoundedMul(av.x, xv.x);
            acc += SgemvRoundedMul(av.y, xv.y);
        }
    } else {
        for (uint32_t row = lane; row < len; row += SGEMV_T_WARP_WIDTH) {
            acc += SgemvRoundedMul(aCol[row], xUb[row]);
        }
    }
    return acc;
}

// ==========================================================================
//  Shared kernel prologues (the plain and segmented slabs duplicate them)
// ==========================================================================

// Warp geometry and the block's column range of the small-m T slabs: one
// block per column chunk, x cached whole into __ubuf__ (block-strided
// copy, one barrier).
struct SgemvTSmallCtx {
    uint32_t t;
    uint32_t lane;
    uint32_t wid;
    uint32_t numWarps;
    uint32_t c0;
    uint32_t c1;
};

__simt_callee__ __aicore__ inline SgemvTSmallCtx SgemvTSmallPrologue(
    uint32_t m, uint32_t n, uint32_t chunkLen, __gm__ const float* xGm, __ubuf__ float* xUb)
{
    SgemvTSmallCtx ctx;
    ctx.t = threadIdx.x;
    ctx.lane = ctx.t % SGEMV_T_WARP_WIDTH;
    ctx.wid = ctx.t / SGEMV_T_WARP_WIDTH;
    ctx.numWarps = blockDim.x / SGEMV_T_WARP_WIDTH;
    for (uint32_t i = ctx.t; i < m; i += blockDim.x) {
        xUb[i] = xGm[i];
    }
    asc_syncthreads();
    ctx.c0 = blockIdx.x * chunkLen;
    ctx.c1 = ctx.c0 + chunkLen;
    if (ctx.c1 > n) {
        ctx.c1 = n;
    }
    return ctx;
}

// Warp geometry, tile row range and staged x of the large-m T slabs: the
// grid is colChunks x rowTiles and each block stages its row tile's x
// segment in __ubuf__ (one barrier). Returns false when the block's row
// tile is past m; no block-wide synchronization follows in that path, so
// threads may leave directly.
struct SgemvTTileCtx {
    uint32_t t;
    uint32_t lane;
    uint32_t wid;
    uint32_t numWarps;
    uint32_t chunk;
    uint32_t rowTile;
    uint32_t r0;
    uint32_t rows;
    uint32_t c0;
    uint32_t c1;
};

__simt_callee__ __aicore__ inline bool SgemvTTilePrologue(
    uint32_t m, uint32_t n, uint32_t chunkLen, uint32_t colChunks, __gm__ const float* xGm, __ubuf__ float* xUb,
    SgemvTTileCtx& ctx)
{
    ctx.t = threadIdx.x;
    ctx.lane = ctx.t % SGEMV_T_WARP_WIDTH;
    ctx.wid = ctx.t / SGEMV_T_WARP_WIDTH;
    ctx.numWarps = blockDim.x / SGEMV_T_WARP_WIDTH;
    ctx.chunk = blockIdx.x % colChunks;
    ctx.rowTile = blockIdx.x / colChunks;
    ctx.r0 = ctx.rowTile * SGEMV_T_TILE_ROWS;
    if (ctx.r0 >= m) {
        return false;
    }
    ctx.rows = m - ctx.r0;
    if (ctx.rows > SGEMV_T_TILE_ROWS) {
        ctx.rows = SGEMV_T_TILE_ROWS;
    }
    ctx.c0 = ctx.chunk * chunkLen;
    ctx.c1 = ctx.c0 + chunkLen;
    if (ctx.c1 > n) {
        ctx.c1 = n;
    }
    for (uint32_t i = ctx.t; i < ctx.rows; i += blockDim.x) {
        xUb[i] = xGm[ctx.r0 + i];
    }
    asc_syncthreads();
    return true;
}
