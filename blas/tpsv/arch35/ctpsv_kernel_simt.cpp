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
 * \file ctpsv_kernel_simt.cpp
 * \brief SIMT VF kernel for ctpsv (n >= CTPSV_SIMT_THRESHOLD), multi-block.
 *
 * The solve is blocked into tiles of CTPSV_TILE rows and distributed over numBlocks AI cores:
 *   1) block 0 stages the diagonal CTPSV_TILE x CTPSV_TILE block into UB and solves it with a
 *      single thread (the only inherently sequential part);
 *   2) the AscendC SyncAll() cross-core barrier publishes block 0's results to every core;
 *   3) every core reloads the CTPSV_TILE unknowns from GM and applies them to its own share of
 *      the remaining rows in a fully parallel rank-CTPSV_TILE update;
 *   4) a second SyncAll() separates the update from the next tile.
 * This keeps the barrier count at O(n / CTPSV_TILE) and spreads the O(n^2) update across cores.
 *
 * Cross-core synchronisation uses the framework SyncAll() (see blas/dotex/arch35, blas/nrm2/arch35)
 * instead of a hand-rolled global-memory spin barrier, so the tile loop lives in the __global__
 * kernel and each phase is a separate asc_vf_call.
 *
 * The hot loops are written as short as possible: the packed-matrix index is carried incrementally
 * (one addition per element) instead of being recomputed from the quadratic packed formula.
 */

#include <cstdint>
#include "cann_ops_blas_common.h"
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "ctpsv_tiling_data.h"
#include "common/helper/kernel_constant.h"
#include "ctpsv_kernel_utils.h"

using namespace AscendC;

// Number of rows solved by the sequential panel step (single block).
static constexpr uint32_t CTPSV_TILE = 16;

// ==========================================================================
//  Complex (float32 pairs) helpers
// ==========================================================================

__simt_callee__ inline float CtpsvSimtAbsf(float v) { return (v < 0.0f) ? -v : v; }

// Complex division following the Smith algorithm used by libgcc's __divsc3, so that
// the kernel result matches the complex division performed by the reference BLAS golden.
// Defined by the shared CTPSV_DEFINE_CDIV macro (see ctpsv_kernel_utils.h).
CTPSV_DEFINE_CDIV(__simt_callee__, CtpsvSimtCdiv, CtpsvSimtAbsf)

// ==========================================================================
//  Indexing helpers
// ==========================================================================

// Packed index of the matrix element that multiplies x[col] in the equation of x[row].
template <bool UPLO_IS_UPPER, bool TRANS_IS_NO_TRANS>
__simt_callee__ inline uint32_t CtpsvSimtApIdx(uint32_t row, uint32_t col, uint32_t n)
{
    if constexpr (!UPLO_IS_UPPER) {
        if constexpr (TRANS_IS_NO_TRANS) {
            return CtpsvPackedLowerIdxSimt(row, col, n);
        } else {
            return CtpsvPackedLowerIdxSimt(col, row, n);
        }
    } else {
        if constexpr (TRANS_IS_NO_TRANS) {
            return CtpsvPackedUpperIdxSimt(row, col);
        } else {
            return CtpsvPackedUpperIdxSimt(col, row);
        }
    }
}

__simt_callee__ inline uint32_t CtpsvSimtXOff(uint32_t i, uint32_t n, int64_t incx)
{
    if (incx >= 0) {
        return i * static_cast<uint32_t>(incx);
    }
    return (n - 1U - i) * static_cast<uint32_t>(-incx);
}

// ==========================================================================
//  Block staging (runs inside the panel VF on block 0)
// ==========================================================================

template <bool UPLO_IS_UPPER, bool TRANS_IS_NO_TRANS>
__simt_callee__ inline void CtpsvSimtStage(
    uint32_t bStart, uint32_t bEnd, uint32_t n, int64_t incx, __gm__ const float* apGm, __gm__ volatile float* xGm,
    __ubuf__ float* apBlock, __ubuf__ float* xBlock)
{
    constexpr uint32_t T = CTPSV_TILE;
    for (uint32_t k = threadIdx.x; k < T * T; k += blockDim.x) {
        const uint32_t li = k / T;
        const uint32_t lj = k % T;
        const uint32_t i = bStart + li;
        const uint32_t j = bStart + lj;
        if (i < n && j < n) {
            const uint32_t apIdx = CtpsvSimtApIdx<UPLO_IS_UPPER, TRANS_IS_NO_TRANS>(i, j, n);
            apBlock[2U * (li * T + lj)] = apGm[2U * apIdx];
            apBlock[2U * (li * T + lj) + 1U] = apGm[2U * apIdx + 1U];
        }
    }
    const uint32_t tileRows = bEnd - bStart;
    for (uint32_t li = threadIdx.x; li < tileRows; li += blockDim.x) {
        const uint32_t xOff = CtpsvSimtXOff(bStart + li, n, incx);
        xBlock[2U * li] = xGm[2U * xOff];
        xBlock[2U * li + 1U] = xGm[2U * xOff + 1U];
    }
}

// Reload the (already solved) CTPSV_TILE unknowns of the current tile from GM. Runs inside the
// update VF on every core so that all cores share block 0's panel results.
__simt_callee__ inline void CtpsvSimtReloadX(
    uint32_t bStart, uint32_t bEnd, uint32_t n, int64_t incx, __gm__ volatile float* xGm, __ubuf__ float* xBlock)
{
    const uint32_t tileRows = bEnd - bStart;
    for (uint32_t li = threadIdx.x; li < tileRows; li += blockDim.x) {
        const uint32_t xOff = CtpsvSimtXOff(bStart + li, n, incx);
        xBlock[2U * li] = xGm[2U * xOff];
        xBlock[2U * li + 1U] = xGm[2U * xOff + 1U];
    }
}

// ==========================================================================
//  Panel step — sequential triangular solve of one CTPSV_TILE x CTPSV_TILE block
// ==========================================================================

// Solve one row i of the current tile: accumulate the already-solved off-diagonal terms over
// columns [jBegin, jEnd), then divide by the diagonal (unless unit diagonal) and write the result
// back to UB and GM. Shared by the forward and backward panel phases, which differ only in the
// traversal order and the column range.
template <bool CONJ, bool DIAG_IS_UNIT>
__simt_callee__ inline void CtpsvSimtSolveRow(
    uint32_t bStart, uint32_t i, uint32_t jBegin, uint32_t jEnd, uint32_t n, int64_t incx, __gm__ volatile float* xGm,
    __ubuf__ const float* apBlock, __ubuf__ float* xBlock)
{
    constexpr uint32_t T = CTPSV_TILE;
    const uint32_t li = i - bStart;
    __ubuf__ const float* apRow = &apBlock[2U * (li * T)];
    float sr = xBlock[2U * li];
    float si = xBlock[2U * li + 1U];

    for (uint32_t j = jBegin; j < jEnd; ++j) {
        const uint32_t lj = j - bStart;
        float aReal = apRow[2U * lj];
        float aImag = apRow[2U * lj + 1U];
        if constexpr (CONJ) {
            aImag = -aImag;
        }
        const float xjReal = xBlock[2U * lj];
        const float xjImag = xBlock[2U * lj + 1U];
        sr -= aReal * xjReal - aImag * xjImag;
        si -= aReal * xjImag + aImag * xjReal;
    }

    if constexpr (!DIAG_IS_UNIT) {
        float dReal = apRow[2U * li];
        float dImag = apRow[2U * li + 1U];
        if constexpr (CONJ) {
            dImag = -dImag;
        }
        float outReal;
        float outImag;
        CtpsvSimtCdiv(sr, si, dReal, dImag, outReal, outImag);
        sr = outReal;
        si = outImag;
    }
    xBlock[2U * li] = sr;
    xBlock[2U * li + 1U] = si;
    const uint32_t xOff = CtpsvSimtXOff(i, n, incx);
    xGm[2U * xOff] = sr;
    xGm[2U * xOff + 1U] = si;
}

template <bool CONJ, bool DIAG_IS_UNIT>
__simt_callee__ inline void CtpsvSimtPanelForward(
    uint32_t bStart, uint32_t bEnd, uint32_t n, int64_t incx, __gm__ volatile float* xGm, __ubuf__ const float* apBlock,
    __ubuf__ float* xBlock)
{
    if (threadIdx.x != 0) {
        return;
    }
    for (uint32_t i = bStart; i < bEnd; ++i) {
        CtpsvSimtSolveRow<CONJ, DIAG_IS_UNIT>(bStart, i, bStart, i, n, incx, xGm, apBlock, xBlock);
    }
}

template <bool CONJ, bool DIAG_IS_UNIT>
__simt_callee__ inline void CtpsvSimtPanelBackward(
    uint32_t bStart, uint32_t bEnd, uint32_t n, int64_t incx, __gm__ volatile float* xGm, __ubuf__ const float* apBlock,
    __ubuf__ float* xBlock)
{
    if (threadIdx.x != 0) {
        return;
    }
    for (uint32_t i = bEnd; i-- > bStart;) {
        CtpsvSimtSolveRow<CONJ, DIAG_IS_UNIT>(bStart, i, i + 1U, bEnd, n, incx, xGm, apBlock, xBlock);
    }
}

// ==========================================================================
//  Trailing rank-CTPSV_TILE update
// ==========================================================================

// Rows [rowFrom, rowTo) of x are updated by the CTPSV_TILE unknowns of the current block; the
// row range is split evenly across the numBlocks blocks. The packed index is carried along the
// tile instead of being recomputed: for the non-transposed layouts it is i + C(j) with
// C(j + 1) - C(j) = n - 1 - j (lower) or j + 1 (upper); for the transposed layouts it is
// j + C(i), i.e. it advances by one per column.
// Initialize the packed index carried along a tile for row i, together with its per-column delta
// and the delta's step, for the four uplo/trans layout combinations.
template <bool UPLO_IS_UPPER, bool TRANS_IS_NO_TRANS>
__simt_callee__ inline void CtpsvSimtInitApIdx(
    uint32_t i, uint32_t bStart, uint32_t n, uint32_t& apIdx, int32_t& apDelta, int32_t& apDeltaStep)
{
    if constexpr (TRANS_IS_NO_TRANS) {
        if constexpr (!UPLO_IS_UPPER) {
            apIdx = i + (2U * n - bStart - 1U) * bStart / 2U;
            apDelta = static_cast<int32_t>(n - 1U - bStart);
            apDeltaStep = -1;
        } else {
            apIdx = i + bStart * (bStart + 1U) / 2U;
            apDelta = static_cast<int32_t>(bStart + 1U);
            apDeltaStep = 1;
        }
    } else {
        if constexpr (!UPLO_IS_UPPER) {
            apIdx = bStart + (2U * n - i - 1U) * i / 2U;
        } else {
            apIdx = bStart + i * (i + 1U) / 2U;
        }
        apDelta = 1;
        apDeltaStep = 0;
    }
}

template <bool UPLO_IS_UPPER, bool TRANS_IS_NO_TRANS, bool CONJ>
__simt_callee__ inline void CtpsvSimtUpdateRows(
    uint32_t bStart, uint32_t bEnd, uint32_t rowFrom, uint32_t rowTo, uint32_t n, int64_t incx,
    __gm__ const float* apGm, __gm__ volatile float* xGm, __ubuf__ const float* xBlock, uint32_t numBlocks)
{
    const uint32_t totalRows = rowTo - rowFrom;
    const uint32_t perBlock = totalRows / numBlocks;
    const uint32_t rem = totalRows % numBlocks;
    const uint32_t myStart = rowFrom + blockIdx.x * perBlock + (blockIdx.x < rem ? blockIdx.x : rem);
    const uint32_t myCount = perBlock + (blockIdx.x < rem ? 1U : 0U);
    const uint32_t myEnd = myStart + myCount;

    for (uint32_t i = myStart + threadIdx.x; i < myEnd; i += blockDim.x) {
        const uint32_t xOff = CtpsvSimtXOff(i, n, incx);
        float accReal = xGm[2U * xOff];
        float accImag = xGm[2U * xOff + 1U];

        uint32_t apIdx;
        int32_t apDelta;
        int32_t apDeltaStep;
        CtpsvSimtInitApIdx<UPLO_IS_UPPER, TRANS_IS_NO_TRANS>(i, bStart, n, apIdx, apDelta, apDeltaStep);

        for (uint32_t j = bStart; j < bEnd; ++j) {
            float aReal = apGm[2U * apIdx];
            float aImag = apGm[2U * apIdx + 1U];
            if constexpr (CONJ) {
                aImag = -aImag;
            }
            const uint32_t lj = j - bStart;
            const float xjReal = xBlock[2U * lj];
            const float xjImag = xBlock[2U * lj + 1U];
            accReal -= aReal * xjReal - aImag * xjImag;
            accImag -= aReal * xjImag + aImag * xjReal;
            apIdx += static_cast<uint32_t>(apDelta);
            apDelta += apDeltaStep;
        }

        xGm[2U * xOff] = accReal;
        xGm[2U * xOff + 1U] = accImag;
    }
}

// ==========================================================================
//  SIMT VF phases — one asc_vf_call per phase, SyncAll() in between
// ==========================================================================

// Panel phase (block 0 only): stage the T×T block + T unknowns into UB, then solve with thread 0.
template <bool UPLO_IS_UPPER, bool TRANS_IS_NO_TRANS, bool CONJ, bool DIAG_IS_UNIT>
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void CtpsvSimtPanel(
    uint32_t bStart, uint32_t bEnd, uint32_t n, int64_t incx, __gm__ const float* apGm, __gm__ volatile float* xGm)
{
    constexpr uint32_t T = CTPSV_TILE;
    __ubuf__ float apBlock[T * T * 2U];
    __ubuf__ float xBlock[T * 2U];
    constexpr bool kForward = (!UPLO_IS_UPPER && TRANS_IS_NO_TRANS) || (UPLO_IS_UPPER && !TRANS_IS_NO_TRANS);

    CtpsvSimtStage<UPLO_IS_UPPER, TRANS_IS_NO_TRANS>(bStart, bEnd, n, incx, apGm, xGm, apBlock, xBlock);
    asc_syncthreads();
    if constexpr (kForward) {
        CtpsvSimtPanelForward<CONJ, DIAG_IS_UNIT>(bStart, bEnd, n, incx, xGm, apBlock, xBlock);
    } else {
        CtpsvSimtPanelBackward<CONJ, DIAG_IS_UNIT>(bStart, bEnd, n, incx, xGm, apBlock, xBlock);
    }
}

// Update phase (all blocks): reload the T solved unknowns into UB, then apply the rank-T update.
template <bool UPLO_IS_UPPER, bool TRANS_IS_NO_TRANS, bool CONJ>
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void CtpsvSimtUpdate(
    uint32_t bStart, uint32_t bEnd, uint32_t rowFrom, uint32_t rowTo, uint32_t n, int64_t incx,
    __gm__ const float* apGm, __gm__ volatile float* xGm, uint32_t numBlocks)
{
    constexpr uint32_t T = CTPSV_TILE;
    __ubuf__ float xBlock[T * 2U];
    CtpsvSimtReloadX(bStart, bEnd, n, incx, xGm, xBlock);
    asc_syncthreads();
    CtpsvSimtUpdateRows<UPLO_IS_UPPER, TRANS_IS_NO_TRANS, CONJ>(
        bStart, bEnd, rowFrom, rowTo, n, incx, apGm, xGm, xBlock, numBlocks);
}

// ==========================================================================
//  SIMT kernel dispatcher — tile loop lives here, SyncAll between phases
// ==========================================================================

template <bool UPLO_IS_UPPER, bool TRANS_IS_NO_TRANS, bool CONJ, bool DIAG_IS_UNIT>
__aicore__ inline void CtpsvSimtRun(
    uint32_t n, int64_t incx, __gm__ const float* apGm, __gm__ volatile float* xGm, uint32_t numBlocks,
    uint32_t numThreads)
{
    constexpr uint32_t T = CTPSV_TILE;
    constexpr bool kForward = (!UPLO_IS_UPPER && TRANS_IS_NO_TRANS) || (UPLO_IS_UPPER && !TRANS_IS_NO_TRANS);
    const dim3 vfDim{numThreads, 1, 1};

    if constexpr (kForward) {
        for (uint32_t bStart = 0; bStart < n; bStart += T) {
            const uint32_t bEnd = ((bStart + T) < n) ? (bStart + T) : n;
            if (GetBlockIdx() == 0) {
                asc_vf_call<CtpsvSimtPanel<UPLO_IS_UPPER, TRANS_IS_NO_TRANS, CONJ, DIAG_IS_UNIT>>(
                    vfDim, bStart, bEnd, n, incx, apGm, xGm);
            }
            SyncAll();
            asc_vf_call<CtpsvSimtUpdate<UPLO_IS_UPPER, TRANS_IS_NO_TRANS, CONJ>>(
                vfDim, bStart, bEnd, bEnd, n, n, incx, apGm, xGm, numBlocks);
            SyncAll();
        }
    } else {
        for (uint32_t bEnd = n; bEnd > 0;) {
            const uint32_t bStart = (bEnd > T) ? (bEnd - T) : 0U;
            if (GetBlockIdx() == 0) {
                asc_vf_call<CtpsvSimtPanel<UPLO_IS_UPPER, TRANS_IS_NO_TRANS, CONJ, DIAG_IS_UNIT>>(
                    vfDim, bStart, bEnd, n, incx, apGm, xGm);
            }
            SyncAll();
            asc_vf_call<CtpsvSimtUpdate<UPLO_IS_UPPER, TRANS_IS_NO_TRANS, CONJ>>(
                vfDim, bStart, bEnd, 0U, bStart, n, incx, apGm, xGm, numBlocks);
            SyncAll();
            bEnd = bStart;
        }
    }
}

#define CTPSV_SIMT_LAUNCH(uploVal, transIsNoTrans, conjVal)                             \
    do {                                                                                \
        if (tiling.diag == ACLBLAS_NON_UNIT) {                                          \
            CtpsvSimtRun<uploVal, transIsNoTrans, conjVal, false>(                      \
                tiling.n, tiling.incx, apGm, xGm, tiling.numBlocks, tiling.numThreads); \
        } else {                                                                        \
            CtpsvSimtRun<uploVal, transIsNoTrans, conjVal, true>(                       \
                tiling.n, tiling.incx, apGm, xGm, tiling.numBlocks, tiling.numThreads); \
        }                                                                               \
    } while (0)

#define CTPSV_SIMT_BY_TRANS(uploVal)                  \
    do {                                              \
        if (tiling.trans == ACLBLAS_OP_N) {           \
            CTPSV_SIMT_LAUNCH(uploVal, true, false);  \
        } else if (tiling.trans == ACLBLAS_OP_T) {    \
            CTPSV_SIMT_LAUNCH(uploVal, false, false); \
        } else {                                      \
            CTPSV_SIMT_LAUNCH(uploVal, false, true);  \
        }                                             \
    } while (0)

__global__ __aicore__ void ctpsv_simt_kernel(CtpsvTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    auto* apGm = reinterpret_cast<__gm__ float*>(tiling.ap);
    auto* xGm = reinterpret_cast<__gm__ volatile float*>(tiling.x);

    if (tiling.uplo == ACLBLAS_LOWER) {
        CTPSV_SIMT_BY_TRANS(false);
    } else {
        CTPSV_SIMT_BY_TRANS(true);
    }
}

#undef CTPSV_SIMT_BY_TRANS
#undef CTPSV_SIMT_LAUNCH

// ==========================================================================
//  SIMT kernel launch wrapper — called from ctpsv_kernel_do
// ==========================================================================

void ctpsv_simt_kernel_do(const CtpsvTilingData& tiling, void* stream)
{
    ctpsv_simt_kernel<<<tiling.numBlocks, nullptr, stream>>>(tiling);
}
