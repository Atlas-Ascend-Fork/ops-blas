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
 * \file sgemv_slab_kernel.cpp
 * \brief SGEMV slab fast paths for arch35 (DAV_3510).
 *
 * Column-chunked kernels ported from the tuned cgemv slab design, simplified
 * for real fp32:
 *  - trans=N: one row (pair) per thread walks its column chunk of A with GM
 *    loads and leaves a partial in workSpace[chunk*m + row]; a second
 *    stream-ordered launch reduces the partials and applies alpha/beta.
 *    Tall matrices switch to an MTE2 double-buffered transport feeding a
 *    SIMD-Regbase broadcast multiply.
 *  - trans=T: one warp per column streams whole columns; x is cached in UB
 *    when it fits, otherwise the matrix is cut into row tiles whose x
 *    segment is staged in UB per block (workSpace[rowTile*n + col] partials
 *    + shared finalize).
 */

#include <cstdint>
#include "cann_ops_blas_common.h"
#include "sgemv_slab_common.h"

using namespace AscendC;

// ==========================================================================
//  Constants
// ==========================================================================

// N MTE transport: one matrix tile = 16 columns x 1024 rows (64 KB).
static constexpr uint32_t SGEMV_N_MTE_ROWS = 1024;
static constexpr uint32_t SGEMV_N_MTE_COLS = 16;
static constexpr uint32_t SGEMV_N_MTE_MAX_SLAB_COLS = 128;
static constexpr uint32_t SGEMV_N_MTE_ROW_BYTES = SGEMV_N_MTE_ROWS * sizeof(float);
static constexpr uint32_t SGEMV_N_MTE_A_BYTES = SGEMV_N_MTE_COLS * SGEMV_N_MTE_ROW_BYTES;
// Keep a full vector-read suffix after the last scalar broadcast address.
static constexpr uint32_t SGEMV_N_MTE_X_BYTES = SGEMV_N_MTE_MAX_SLAB_COLS * sizeof(float) + 512U;
static_assert(2U * SGEMV_N_MTE_A_BYTES + SGEMV_N_MTE_X_BYTES + SGEMV_N_MTE_ROW_BYTES <= UB_SIZE);

// Shared workspace finalize geometry.
static constexpr uint32_t SGEMV_FIN_ROWS = 128; // outputs per finalize block (== SIMT_MIN_THREAD_NUM)
static constexpr uint32_t SGEMV_FIN_GROUPS = 8; // partial groups per output => 1024 threads per block
static constexpr uint32_t SGEMV_FIN_VEC_ROWS = 64;
static constexpr uint32_t SGEMV_FIN_VEC_MAX_CHUNKS = 64;
static constexpr uint32_t SGEMV_FIN_VEC_SLOT_FLOATS = SGEMV_FIN_VEC_ROWS;
static constexpr uint32_t SGEMV_FIN_VEC_SLOT_BYTES = SGEMV_FIN_VEC_SLOT_FLOATS * sizeof(float);
static constexpr uint32_t SGEMV_FIN_VEC_WS_BYTES = SGEMV_FIN_VEC_MAX_CHUNKS * SGEMV_FIN_VEC_SLOT_BYTES;
static constexpr uint32_t SGEMV_FIN_VEC_OUT_BYTES = SGEMV_FIN_VEC_SLOT_BYTES;

// ==========================================================================
//  N slab — MTE2 transport + SIMD broadcast compute (tall matrices)
// ==========================================================================

// One 64-row group per iteration; each lane holds one row of the group and
// accumulates the tile's columns in order (DIST_BRC_B32 broadcasts x[col]).
__simd_vf__ inline void SgemvNSlabMteTileVf(
    uint32_t cols, uint32_t activeRows, uint32_t reset, __ubuf__ float* a, __ubuf__ float* x, __ubuf__ float* acc)
{
    for (uint32_t row = 0; row < activeRows; row += 64U) {
        Reg::RegTensor<float> aV, xV, prod, accV;
        uint32_t maskCount = activeRows - row;
        Reg::MaskReg mask = Reg::UpdateMask<float>(maskCount);
        if (reset != 0U) {
            Reg::Duplicate(accV, 0.0F, mask);
        } else {
            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(accV, acc + row);
        }
        for (uint32_t col = 0; col < cols; ++col) {
            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(aV, a + col * SGEMV_N_MTE_ROWS + row);
            Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(xV, x + col);
            Reg::Mul(prod, aV, xV, mask);
            Reg::Add(accV, accV, prod, mask);
        }
        Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM>(acc + row, accV, mask);
    }
    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
}

// 2D copy of [cols, activeRows] tile into 4 KB row slots (32 B aligned pitch).
__aicore__ inline void SgemvNSlabMteCopy(
    LocalTensor<float> dst, GlobalTensor<float> src, uint32_t col, uint32_t cols, uint32_t row0, uint32_t activeRows,
    uint32_t lda)
{
    uint32_t rowBytes = activeRows * sizeof(float);
    uint32_t alignedRowBytes = (rowBytes + 31U) & ~31U;
    uint32_t rightPad = (alignedRowBytes - rowBytes) / sizeof(float);
    uint32_t dstStrideBlocks = (SGEMV_N_MTE_ROW_BYTES - alignedRowBytes) / 32U;
    uint32_t srcStrideBytes = (lda - activeRows) * sizeof(float);
    uint64_t offset = static_cast<uint64_t>(col) * lda + row0;
    DataCopyPad(
        dst, src[offset], DataCopyExtParams{static_cast<uint16_t>(cols), rowBytes, srcStrideBytes, dstStrideBlocks, 0},
        DataCopyPadExtParams<float>{true, 0, static_cast<uint8_t>(rightPad), 0.0F});
}

struct SgemvNSlabMteTile {
    uint32_t row0;
    uint32_t rows;
    uint32_t col0;
    uint32_t cols;
    uint32_t lda;
};

__aicore__ inline void SgemvNSlabMtePrefetch(
    LocalTensor<float> aLocal, GlobalTensor<float> aGm, const SgemvNSlabMteTile& shape, uint32_t tile)
{
    uint32_t nextBuffer = (tile & 1U) ^ 1U;
    if (tile != 0U) {
        // Reusing a matrix slot must follow its previous vector reads.
        WaitFlag<HardEvent::V_MTE2>(nextBuffer);
    }
    uint32_t nextOffset = (tile + 1U) * SGEMV_N_MTE_COLS;
    uint32_t nextCols = shape.cols - nextOffset;
    if (nextCols > SGEMV_N_MTE_COLS) {
        nextCols = SGEMV_N_MTE_COLS;
    }
    SgemvNSlabMteCopy(
        aLocal[nextBuffer * SGEMV_N_MTE_A_BYTES / sizeof(float)], aGm, shape.col0 + nextOffset, nextCols, shape.row0,
        shape.rows, shape.lda);
    SetFlag<HardEvent::MTE2_V>(nextBuffer);
}

__aicore__ inline void SgemvNSlabMteComputeTiles(
    GlobalTensor<float> aGm, GlobalTensor<float> xGm, LocalTensor<float> aLocal, LocalTensor<float> xLocal,
    LocalTensor<float> accLocal, const SgemvNSlabMteTile& shape)
{
    uint32_t firstCols = shape.cols < SGEMV_N_MTE_COLS ? shape.cols : SGEMV_N_MTE_COLS;
    uint32_t xBytes = shape.cols * sizeof(float);
    DataCopyPad(
        xLocal, xGm[shape.col0], DataCopyExtParams{1, xBytes, 0, 0, 0}, DataCopyPadExtParams<float>{false, 0, 0, 0});
    SgemvNSlabMteCopy(aLocal, aGm, shape.col0, firstCols, shape.row0, shape.rows, shape.lda);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    uint32_t tiles = (shape.cols + SGEMV_N_MTE_COLS - 1U) / SGEMV_N_MTE_COLS;
    for (uint32_t tile = 0; tile < tiles; ++tile) {
        uint32_t buffer = tile & 1U;
        uint32_t offset = tile * SGEMV_N_MTE_COLS;
        uint32_t tileCols = shape.cols - offset;
        if (tileCols > SGEMV_N_MTE_COLS) {
            tileCols = SGEMV_N_MTE_COLS;
        }
        if (tile + 1U < tiles) {
            SgemvNSlabMtePrefetch(aLocal, aGm, shape, tile);
        }
        WaitFlag<HardEvent::MTE2_V>(buffer);
        asc_vf_call<SgemvNSlabMteTileVf>(
            tileCols, shape.rows, tile == 0U ? 1U : 0U,
            reinterpret_cast<__ubuf__ float*>(aLocal.GetPhyAddr()) + buffer * SGEMV_N_MTE_A_BYTES / sizeof(float),
            reinterpret_cast<__ubuf__ float*>(xLocal.GetPhyAddr()) + offset,
            reinterpret_cast<__ubuf__ float*>(accLocal.GetPhyAddr()));
        if (tile + 2U < tiles) {
            SetFlag<HardEvent::V_MTE2>(buffer);
        }
    }
}

__aicore__ inline void SgemvNSlabMteStore(
    GlobalTensor<float> wsGm, LocalTensor<float> accLocal, uint32_t chunk, uint32_t m, const SgemvNSlabMteTile& shape)
{
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    uint32_t outputBytes = shape.rows * sizeof(float);
    DataCopyPad(
        wsGm[static_cast<uint64_t>(chunk) * m + shape.row0], accLocal, DataCopyExtParams{1, outputBytes, 0, 0, 0});
    SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
}

__aicore__ inline void SgemvNSlabMteRun(
    GM_ADDR a, GM_ADDR x, GM_ADDR workSpace, uint32_t m, uint32_t n, uint32_t lda, uint32_t chunkLen,
    uint32_t colChunks, TPipe& pipe)
{
    if (colChunks == 0U) {
        return;
    }
    uint32_t block = GetBlockIdx();
    uint32_t chunk = block % colChunks;
    uint32_t row0 = (block / colChunks) * SGEMV_N_MTE_ROWS;
    if (row0 >= m) {
        return;
    }
    GlobalTensor<float> aGm, xGm, wsGm;
    aGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a));
    xGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x));
    wsGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workSpace));
    TBuf<QuePosition::VECCALC> aBuf, xBuf, accBuf;
    pipe.InitBuffer(aBuf, 2U * SGEMV_N_MTE_A_BYTES);
    pipe.InitBuffer(xBuf, SGEMV_N_MTE_X_BYTES);
    pipe.InitBuffer(accBuf, SGEMV_N_MTE_ROW_BYTES);
    LocalTensor<float> aLocal = aBuf.Get<float>();
    LocalTensor<float> xLocal = xBuf.Get<float>();
    LocalTensor<float> accLocal = accBuf.Get<float>();

    SgemvNSlabMteTile shape{row0, m - row0, chunk * chunkLen, 0, lda};
    if (shape.rows > SGEMV_N_MTE_ROWS) {
        shape.rows = SGEMV_N_MTE_ROWS;
    }
    shape.cols = n - shape.col0;
    if (shape.cols > chunkLen) {
        shape.cols = chunkLen;
    }
    SgemvNSlabMteComputeTiles(aGm, xGm, aLocal, xLocal, accLocal, shape);
    SgemvNSlabMteStore(wsGm, accLocal, chunk, m, shape);
}

// ==========================================================================
//  N slab — SIMT compute (one row pair or one row per thread)
// ==========================================================================

// Block decomposition shared by the N-slab SIMT compute kernels: (chunk,
// tile) from the block index and the chunk's column range clipped to n.
struct SgemvNBlock {
    uint32_t bdim;
    uint32_t t;
    uint32_t chunk;
    uint32_t tile;
    uint32_t c0;
    uint32_t c1;
};

__simt_callee__ __aicore__ inline SgemvNBlock SgemvNBlockInfo(uint32_t n, uint32_t chunkLen, uint32_t colChunks)
{
    SgemvNBlock blk;
    blk.bdim = blockDim.x;
    blk.t = threadIdx.x;
    blk.chunk = 0U;
    blk.tile = 0U;
    blk.c0 = 0U;
    blk.c1 = 0U;
    // host derives colChunks = ceil(n / chunkLen) with n >= 1, so it is never
    // 0 on a live launch; the guard keeps the divisors provably non-zero and
    // degrades to an empty column range (c0 == c1) if it ever is.
    if (colChunks != 0U) {
        blk.chunk = blockIdx.x % colChunks;
        blk.tile = blockIdx.x / colChunks;
        blk.c0 = blk.chunk * chunkLen;
        blk.c1 = blk.c0 + chunkLen;
        if (blk.c1 > n) {
            blk.c1 = n;
        }
    }
    return blk;
}

// Row-pair variant: thread t owns rows tile*2*bdim + 2t and +2t+1, so each
// float2 load covers one column of both rows (host guarantees m and lda
// even). x is read straight from GM (one uniform float per column; it stays
// cache-resident). Partial per row -> ws[chunk*m + row]. UNROLL batches four
// independent GM loads per thread — a win only when the grid is too small to
// fill the machine's warp slots (small m); at full occupancy it oversaturates
// the load queues. The additions stay in column order either way
// (bit-identical to the scalar recurrence).
template <bool UNROLL>
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void SgemvNSlabCompute2(
    uint32_t m, uint32_t n, uint32_t lda, uint32_t chunkLen, uint32_t colChunks, __gm__ const float* aGm,
    __gm__ const float* xGm, __gm__ float* wsGm)
{
    SgemvNBlock blk = SgemvNBlockInfo(n, chunkLen, colChunks);
    uint32_t row = blk.tile * (blk.bdim * 2U) + blk.t * 2U;
    if (row >= m) {
        return; // no block-level synchronization follows: idle threads may leave
    }

    __gm__ const float2* a2 = reinterpret_cast<__gm__ const float2*>(aGm);
    uint32_t lda2 = lda / 2U;
    uint32_t base = row / 2U + blk.c0 * lda2;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    uint32_t c = blk.c0;
    if constexpr (UNROLL) {
        for (; c + 4U <= blk.c1; c += 4U) {
            float2 av0 = a2[base];
            float2 av1 = a2[base + lda2];
            float2 av2 = a2[base + 2U * lda2];
            float2 av3 = a2[base + 3U * lda2];
            float xv0 = xGm[c];
            float xv1 = xGm[c + 1U];
            float xv2 = xGm[c + 2U];
            float xv3 = xGm[c + 3U];
            acc0 += SgemvRoundedMul(av0.x, xv0);
            acc1 += SgemvRoundedMul(av0.y, xv0);
            acc0 += SgemvRoundedMul(av1.x, xv1);
            acc1 += SgemvRoundedMul(av1.y, xv1);
            acc0 += SgemvRoundedMul(av2.x, xv2);
            acc1 += SgemvRoundedMul(av2.y, xv2);
            acc0 += SgemvRoundedMul(av3.x, xv3);
            acc1 += SgemvRoundedMul(av3.y, xv3);
            base += 4U * lda2;
        }
    }
    for (; c < blk.c1; ++c) {
        float2 av = a2[base];
        float xv = xGm[c];
        acc0 += SgemvRoundedMul(av.x, xv);
        acc1 += SgemvRoundedMul(av.y, xv);
        base += lda2;
    }

    __gm__ float* myWs = wsGm + static_cast<uint64_t>(blk.chunk) * m;
    myWs[row] = acc0;
    myWs[row + 1U] = acc1;
}

// Single-row variant (odd m or odd lda): 4 B loads, otherwise identical.
template <bool UNROLL>
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void SgemvNSlabCompute1(
    uint32_t m, uint32_t n, uint32_t lda, uint32_t chunkLen, uint32_t colChunks, __gm__ const float* aGm,
    __gm__ const float* xGm, __gm__ float* wsGm)
{
    SgemvNBlock blk = SgemvNBlockInfo(n, chunkLen, colChunks);
    uint32_t row = blk.tile * blk.bdim + blk.t;
    if (row >= m) {
        return;
    }

    uint32_t colBase = blk.c0 * lda + row;
    float acc = 0.0f;
    uint32_t c = blk.c0;
    if constexpr (UNROLL) {
        for (; c + 4U <= blk.c1; c += 4U) {
            float av0 = aGm[colBase];
            float av1 = aGm[colBase + lda];
            float av2 = aGm[colBase + 2U * lda];
            float av3 = aGm[colBase + 3U * lda];
            float xv0 = xGm[c];
            float xv1 = xGm[c + 1U];
            float xv2 = xGm[c + 2U];
            float xv3 = xGm[c + 3U];
            acc += SgemvRoundedMul(av0, xv0);
            acc += SgemvRoundedMul(av1, xv1);
            acc += SgemvRoundedMul(av2, xv2);
            acc += SgemvRoundedMul(av3, xv3);
            colBase += 4U * lda;
        }
    }
    for (; c < blk.c1; ++c) {
        acc += SgemvRoundedMul(aGm[colBase], xGm[c]);
        colBase += lda;
    }

    wsGm[static_cast<uint64_t>(blk.chunk) * m + row] = acc;
}

// ==========================================================================
//  Shared workspace finalize (cross-chunk reduction + alpha/beta write-back)
// ==========================================================================

// Block b owns FIN_ROWS outputs; thread t handles output (t % FIN_ROWS) and
// every FIN_GROUPS-th partial starting at (t / FIN_ROWS). The FIN_GROUPS
// partial sums per output meet in UB. Works for both N (partials over column
// chunks, outLen = m) and large-m T (partials over row tiles, outLen = n).
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void SgemvSlabFinalize(
    uint32_t outLen, uint32_t numParts, float alpha, float beta, uint32_t betaIsZero, int64_t incy,
    __gm__ const float* wsGm, __gm__ float* yGm)
{
    __ubuf__ float red[SGEMV_FIN_GROUPS * SGEMV_FIN_ROWS];
    int64_t len64 = static_cast<int64_t>(outLen);
    uint32_t t = threadIdx.x;
    uint32_t rowLocal = t % SGEMV_FIN_ROWS;
    uint32_t grp = t / SGEMV_FIN_ROWS;

    uint32_t out = blockIdx.x * SGEMV_FIN_ROWS + rowLocal;

    float s = 0.0f;
    if (out < outLen) {
        // Partial j of this output lives at ws[j*outLen + out]; this thread sums j = grp, grp+G, ...
        uint32_t idx = grp * outLen + out;
        for (uint32_t j = grp; j < numParts; j += SGEMV_FIN_GROUPS) {
            s += wsGm[idx];
            idx += SGEMV_FIN_GROUPS * outLen;
        }
    }
    red[t] = s;
    asc_syncthreads();

    if (grp == 0 && out < outLen) {
        for (uint32_t g = 1; g < SGEMV_FIN_GROUPS; ++g) {
            s += red[g * SGEMV_FIN_ROWS + rowLocal];
        }
        int64_t yIdx = SgemvStridedIdx(static_cast<int64_t>(out), len64, incy);
        SgemvWriteBack(yGm, yIdx, s, alpha, beta, betaIsZero);
    }
}

// MTE2+SIMD finalize for the alpha==1 / beta==0 / incy==1 fast case. A block
// owns exactly one SIMD register of outputs; at most 64 partial slices are
// packed into fixed 256-byte UB slots. The chunk association (g, g+8, ...;
// then group 0 adds groups 1..7 in order) is intentionally identical to
// SgemvSlabFinalize.
__simd_vf__ inline void SgemvFinalizeMteSimd64Vf(
    uint32_t numChunks, uint32_t activeRows, __ubuf__ float* ws, __ubuf__ float* out)
{
    Reg::RegTensor<float> partial, sum;
    uint32_t maskCount = activeRows;
    Reg::MaskReg mask = Reg::UpdateMask<float>(maskCount);

    for (uint16_t group = 0; group < SGEMV_FIN_GROUPS; ++group) {
        Reg::Duplicate(sum, 0.0F, mask);
        for (uint16_t chunk = group; chunk < static_cast<uint16_t>(numChunks); chunk += SGEMV_FIN_GROUPS) {
            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(partial, ws + chunk * SGEMV_FIN_VEC_SLOT_FLOATS);
            Reg::Add(sum, sum, partial, mask);
        }
        Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM>(ws + group * SGEMV_FIN_VEC_SLOT_FLOATS, sum, mask);
    }
    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();

    Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(sum, ws);
    for (uint16_t group = 1; group < SGEMV_FIN_GROUPS; ++group) {
        Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(partial, ws + group * SGEMV_FIN_VEC_SLOT_FLOATS);
        Reg::Add(sum, sum, partial, mask);
    }
    Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM>(out, sum, mask);
    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
}

__aicore__ inline void SgemvFinalizeMteSimd64Run(
    GM_ADDR y, GM_ADDR workSpace, uint32_t outLen, uint32_t numChunks, TPipe& pipe)
{
    uint32_t row0 = GetBlockIdx() * SGEMV_FIN_VEC_ROWS;
    if (row0 >= outLen) {
        return;
    }
    uint32_t activeRows = outLen - row0;
    if (activeRows > SGEMV_FIN_VEC_ROWS) {
        activeRows = SGEMV_FIN_VEC_ROWS;
    }
    GlobalTensor<float> wsGm, yGm;
    wsGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workSpace));
    yGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y));

    TBuf<QuePosition::VECCALC> wsBuf, outBuf;
    pipe.InitBuffer(wsBuf, SGEMV_FIN_VEC_WS_BYTES);
    pipe.InitBuffer(outBuf, SGEMV_FIN_VEC_OUT_BYTES);
    LocalTensor<float> wsLocal = wsBuf.Get<float>();
    LocalTensor<float> outLocal = outBuf.Get<float>();

    uint32_t rowBytes = activeRows * sizeof(float);
    uint32_t alignedRowBytes = (rowBytes + 31U) & ~31U;
    uint32_t rightPad = (alignedRowBytes - rowBytes) / sizeof(float);
    uint32_t dstStrideBlocks = (SGEMV_FIN_VEC_SLOT_BYTES - alignedRowBytes) / 32U;
    uint32_t srcStrideBytes = (outLen - activeRows) * sizeof(float);
    DataCopyPad(
        wsLocal, wsGm[row0],
        DataCopyExtParams{static_cast<uint16_t>(numChunks), rowBytes, srcStrideBytes, dstStrideBlocks, 0},
        DataCopyPadExtParams<float>{true, 0, static_cast<uint8_t>(rightPad), 0.0F});
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

    asc_vf_call<SgemvFinalizeMteSimd64Vf>(
        numChunks, activeRows, reinterpret_cast<__ubuf__ float*>(wsLocal.GetPhyAddr()),
        reinterpret_cast<__ubuf__ float*>(outLocal.GetPhyAddr()));

    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    DataCopyPad(yGm[row0], outLocal, DataCopyExtParams{1, rowBytes, 0, 0, 0});
    SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
}

// ==========================================================================
//  T slab — one warp per column, x cached in UB (m <= 16K)
// ==========================================================================

template <bool PAIR>
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void SgemvTSlab(
    uint32_t m, uint32_t n, uint32_t lda, float alpha, float beta, uint32_t betaIsZero, int64_t incy, uint32_t chunkLen,
    __gm__ const float* aGm, __gm__ const float* xGm, __gm__ float* yGm)
{
    __ubuf__ float xUb[SGEMV_UB_X_FLOATS];

    SgemvTSmallCtx ctx = SgemvTSmallPrologue(m, n, chunkLen, xGm, xUb);

    int64_t n64 = static_cast<int64_t>(n);
    for (uint32_t c = ctx.c0 + ctx.wid; c < ctx.c1; c += ctx.numWarps) {
        float partial = SgemvTColumnDot<PAIR>(m, ctx.lane, aGm + static_cast<uint64_t>(c) * lda, xUb);
        partial = SgemvWarpReduce(partial);
        if (ctx.lane == 0U) {
            int64_t yIdx = SgemvStridedIdx(static_cast<int64_t>(c), n64, incy);
            SgemvWriteBack(yGm, yIdx, partial, alpha, beta, betaIsZero);
        }
    }
}

// ==========================================================================
//  T slab — large m (x longer than the UB cache): grid = colChunks x rowTiles,
//  each block stages its row tile's x segment in UB and warps own whole
//  columns of the tile; partials go to ws[rowTile*n + col] and the shared
//  finalize combines them.
// ==========================================================================

template <bool PAIR>
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void SgemvTSlabTile(
    uint32_t m, uint32_t n, uint32_t lda, uint32_t chunkLen, uint32_t colChunks, __gm__ const float* aGm,
    __gm__ const float* xGm, __gm__ float* wsGm)
{
    __ubuf__ float xUb[SGEMV_T_TILE_ROWS];

    SgemvTTileCtx ctx;
    if (!SgemvTTilePrologue(m, n, chunkLen, colChunks, xGm, xUb, ctx)) {
        return;
    }

    for (uint32_t c = ctx.c0 + ctx.wid; c < ctx.c1; c += ctx.numWarps) {
        float partial = SgemvTColumnDot<PAIR>(ctx.rows, ctx.lane, aGm + static_cast<uint64_t>(c) * lda + ctx.r0, xUb);
        partial = SgemvWarpReduce(partial);
        if (ctx.lane == 0U) {
            wsGm[static_cast<uint64_t>(ctx.rowTile) * n + c] = partial;
        }
    }
}

// ==========================================================================
//  Global kernels
// ==========================================================================

__global__ __aicore__ void sgemv_n_mte_compute_kernel(
    GM_ADDR a, GM_ADDR x, GM_ADDR workSpace, uint32_t m, uint32_t n, uint32_t lda, uint32_t chunkLen,
    uint32_t colChunks)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    SgemvNSlabMteRun(a, x, workSpace, m, n, lda, chunkLen, colChunks, pipe);
}

template <bool UNROLL>
__global__ __aicore__ void sgemv_n_compute2_kernel(
    GM_ADDR a, GM_ADDR x, GM_ADDR workSpace, uint32_t numThreads, uint32_t m, uint32_t n, uint32_t lda,
    uint32_t chunkLen, uint32_t colChunks)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    auto* aGm = reinterpret_cast<__gm__ const float*>(a);
    auto* xGm = reinterpret_cast<__gm__ const float*>(x);
    auto* wsGm = reinterpret_cast<__gm__ float*>(workSpace);
    asc_vf_call<SgemvNSlabCompute2<UNROLL>>(dim3{numThreads, 1, 1}, m, n, lda, chunkLen, colChunks, aGm, xGm, wsGm);
}

template <bool UNROLL>
__global__ __aicore__ void sgemv_n_compute1_kernel(
    GM_ADDR a, GM_ADDR x, GM_ADDR workSpace, uint32_t numThreads, uint32_t m, uint32_t n, uint32_t lda,
    uint32_t chunkLen, uint32_t colChunks)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    auto* aGm = reinterpret_cast<__gm__ const float*>(a);
    auto* xGm = reinterpret_cast<__gm__ const float*>(x);
    auto* wsGm = reinterpret_cast<__gm__ float*>(workSpace);
    asc_vf_call<SgemvNSlabCompute1<UNROLL>>(dim3{numThreads, 1, 1}, m, n, lda, chunkLen, colChunks, aGm, xGm, wsGm);
}

__global__ __aicore__ void sgemv_finalize_kernel(
    GM_ADDR y, GM_ADDR workSpace, uint32_t outLen, uint32_t numParts, float alpha, float beta, uint32_t betaIsZero,
    int64_t incy)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    auto* wsGm = reinterpret_cast<__gm__ const float*>(workSpace);
    auto* yGm = reinterpret_cast<__gm__ float*>(y);
    asc_vf_call<SgemvSlabFinalize>(
        dim3{SGEMV_FIN_ROWS * SGEMV_FIN_GROUPS, 1, 1}, outLen, numParts, alpha, beta, betaIsZero, incy, wsGm, yGm);
}

__global__ __aicore__ void sgemv_finalize_mte_simd64_kernel(
    GM_ADDR y, GM_ADDR workSpace, uint32_t outLen, uint32_t numChunks)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    SgemvFinalizeMteSimd64Run(y, workSpace, outLen, numChunks, pipe);
}

template <bool PAIR>
__global__ __aicore__ void sgemv_t_slab_kernel(
    GM_ADDR a, GM_ADDR x, GM_ADDR y, uint32_t m, uint32_t n, uint32_t lda, float alpha, float beta, uint32_t betaIsZero,
    int64_t incy, uint32_t chunkLen)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    auto* aGm = reinterpret_cast<__gm__ const float*>(a);
    auto* xGm = reinterpret_cast<__gm__ const float*>(x);
    auto* yGm = reinterpret_cast<__gm__ float*>(y);
    asc_vf_call<SgemvTSlab<PAIR>>(
        dim3{SIMT_MAX_THREAD_NUM, 1, 1}, m, n, lda, alpha, beta, betaIsZero, incy, chunkLen, aGm, xGm, yGm);
}

template <bool PAIR>
__global__ __aicore__ void sgemv_t_tile_kernel(
    GM_ADDR a, GM_ADDR x, GM_ADDR workSpace, uint32_t m, uint32_t n, uint32_t lda, uint32_t chunkLen,
    uint32_t colChunks)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    auto* aGm = reinterpret_cast<__gm__ const float*>(a);
    auto* xGm = reinterpret_cast<__gm__ const float*>(x);
    auto* wsGm = reinterpret_cast<__gm__ float*>(workSpace);
    asc_vf_call<SgemvTSlabTile<PAIR>>(dim3{SIMT_MAX_THREAD_NUM, 1, 1}, m, n, lda, chunkLen, colChunks, aGm, xGm, wsGm);
}

// ==========================================================================
//  Launch dispatch
// ==========================================================================

// Segmented T kernels (sgemv_t_seg_kernel.cpp, separate translation unit so
// their code generation cannot perturb the plain-slab kernels above).
void SgemvLaunchTSeg(
    uint8_t* a, uint8_t* x, uint8_t* y, uint8_t* workSpace, const SgemvTilingData& tiling, uint32_t numBlocks,
    void* stream);

static void SgemvLaunchNCompute(
    uint8_t* a, uint8_t* x, uint8_t* y, uint8_t* workSpace, const SgemvTilingData& tiling, uint32_t numBlocks,
    void* stream)
{
    // Short slabs in tall matrices can replace repeated SIMT GM loads with
    // MTE tiles while retaining the existing column recurrence. A starved
    // grid (m too small to fill the warp slots) is latency-bound, so the
    // DMA transport wins there as well.
    bool useTallShortSlab =
        tiling.m >= SIMT_MAX_THREAD_NUM && tiling.m > tiling.n && tiling.chunkLen <= 2U * SGEMV_N_MTE_COLS;
    bool useMteTransport = tiling.lda % 4U != 0U || useTallShortSlab || tiling.numThreads < SIMT_MAX_THREAD_NUM;
    if (tiling.m >= SGEMV_N_MTE_ROWS && useMteTransport && tiling.chunkLen >= SGEMV_N_MTE_COLS &&
        tiling.chunkLen <= SGEMV_N_MTE_MAX_SLAB_COLS) {
        uint32_t mteBlocks = tiling.colChunks * ((tiling.m + SGEMV_N_MTE_ROWS - 1U) / SGEMV_N_MTE_ROWS);
        sgemv_n_mte_compute_kernel<<<mteBlocks, nullptr, stream>>>(
            a, x, workSpace, tiling.m, tiling.n, tiling.lda, tiling.chunkLen, tiling.colChunks);
        return;
    }
    // Unroll only when the grid cannot fill the machine's warp slots (small
    // m): the four in-flight loads per thread substitute for missing warps.
    bool useUnroll = tiling.numThreads < SIMT_MAX_THREAD_NUM;
    if (tiling.usePairs != 0U) {
        if (useUnroll) {
            sgemv_n_compute2_kernel<true><<<numBlocks, nullptr, stream>>>(
                a, x, workSpace, tiling.numThreads, tiling.m, tiling.n, tiling.lda, tiling.chunkLen, tiling.colChunks);
        } else {
            sgemv_n_compute2_kernel<false><<<numBlocks, nullptr, stream>>>(
                a, x, workSpace, tiling.numThreads, tiling.m, tiling.n, tiling.lda, tiling.chunkLen, tiling.colChunks);
        }
    } else {
        if (useUnroll) {
            sgemv_n_compute1_kernel<true><<<numBlocks, nullptr, stream>>>(
                a, x, workSpace, tiling.numThreads, tiling.m, tiling.n, tiling.lda, tiling.chunkLen, tiling.colChunks);
        } else {
            sgemv_n_compute1_kernel<false><<<numBlocks, nullptr, stream>>>(
                a, x, workSpace, tiling.numThreads, tiling.m, tiling.n, tiling.lda, tiling.chunkLen, tiling.colChunks);
        }
    }
}

static void SgemvLaunchNFinalize(
    uint8_t* a, uint8_t* x, uint8_t* y, uint8_t* workSpace, const SgemvTilingData& tiling, uint32_t numBlocks,
    void* stream)
{
    bool useMteSimd64 = tiling.m > SIMT_MAX_THREAD_NUM && tiling.alpha == 1.0F && tiling.betaIsZero != 0U &&
                        tiling.incy == 1 && tiling.colChunks != 0U && tiling.colChunks <= SGEMV_FIN_VEC_MAX_CHUNKS;
    if (useMteSimd64) {
        uint32_t finBlocks = (tiling.m + SGEMV_FIN_VEC_ROWS - 1U) / SGEMV_FIN_VEC_ROWS;
        sgemv_finalize_mte_simd64_kernel<<<finBlocks, nullptr, stream>>>(y, workSpace, tiling.m, tiling.colChunks);
    } else {
        uint32_t finBlocks = (tiling.m + SGEMV_FIN_ROWS - 1U) / SGEMV_FIN_ROWS;
        sgemv_finalize_kernel<<<finBlocks, nullptr, stream>>>(
            y, workSpace, tiling.m, tiling.colChunks, tiling.alpha, tiling.beta, tiling.betaIsZero, tiling.incy);
    }
}

void SgemvLaunchSlab(
    uint8_t* a, uint8_t* x, uint8_t* y, uint8_t* workSpace, const SgemvTilingData& tiling, uint32_t numBlocks,
    void* stream)
{
    if (tiling.trans == 0U) {
        SgemvLaunchNCompute(a, x, y, workSpace, tiling, numBlocks, stream);
        SgemvLaunchNFinalize(a, x, y, workSpace, tiling, numBlocks, stream);
        return;
    }

    bool usePairs = tiling.usePairs != 0U;
    if (tiling.rowTiles != 0U) {
        // Large-m T: row-tiled blocks + workspace partials + shared finalize.
        bool useTileSeg = tiling.segShift != 0U && (tiling.chunkLen << tiling.segShift) <= SGEMV_T_MAX_ITEMS;
        if (useTileSeg) {
            SgemvLaunchTSeg(a, x, y, workSpace, tiling, numBlocks, stream);
        } else if (usePairs) {
            sgemv_t_tile_kernel<true><<<numBlocks, nullptr, stream>>>(
                a, x, workSpace, tiling.m, tiling.n, tiling.lda, tiling.chunkLen, tiling.colChunks);
        } else {
            sgemv_t_tile_kernel<false><<<numBlocks, nullptr, stream>>>(
                a, x, workSpace, tiling.m, tiling.n, tiling.lda, tiling.chunkLen, tiling.colChunks);
        }
        uint32_t finBlocks = (tiling.n + SGEMV_FIN_ROWS - 1U) / SGEMV_FIN_ROWS;
        sgemv_finalize_kernel<<<finBlocks, nullptr, stream>>>(
            y, workSpace, tiling.n, tiling.rowTiles, tiling.alpha, tiling.beta, tiling.betaIsZero, tiling.incy);
        return;
    }

    // x fits in UB: direct write-back, no workspace.
    if (tiling.segShift != 0U) {
        SgemvLaunchTSeg(a, x, y, workSpace, tiling, numBlocks, stream);
    } else if (usePairs) {
        sgemv_t_slab_kernel<true><<<numBlocks, nullptr, stream>>>(
            a, x, y, tiling.m, tiling.n, tiling.lda, tiling.alpha, tiling.beta, tiling.betaIsZero, tiling.incy,
            tiling.chunkLen);
    } else {
        sgemv_t_slab_kernel<false><<<numBlocks, nullptr, stream>>>(
            a, x, y, tiling.m, tiling.n, tiling.lda, tiling.alpha, tiling.beta, tiling.betaIsZero, tiling.incy,
            tiling.chunkLen);
    }
}
