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
 * \file ctrmm_kernel.cpp
 * \brief CTRMM Kernel implementation for ascend950 (DAV_3510) — M2 correctness baseline.
 *
 * Single AIV SIMT kernel computing, for every output element,
 *   side = LEFT :  C(i,j) = alpha * sum_k op(A)(i,k) * B(k,j)   (ord(A) = m)
 *   side = RIGHT:  C(i,j) = alpha * sum_k B(i,k) * op(A)(k,j)   (ord(A) = n)
 * with the classic 4-multiply complex MAC (re += ar*br - ai*bi; im += ar*bi + ai*br),
 * k ascending (Netlib-compatible accumulation order), structural zeros of the
 * non-referenced triangle skipped (additive zero, never reads GM), diag = UNIT
 * contributing 1+0i without reading A's diagonal slot (NaN/Inf safe).
 *
 * All 24 enum combinations (side2 x uplo2 x trans3 x diag2) are dispatched to
 * compile-time template instantiations — no runtime enum branching inside the
 * compute loop.
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * Expert-kernel structure notes:
 *   (1) Tiling structure: CtrmmTilingData (ctrmm_tiling_data.h) — currently a
 *       flat row-partition for this naive pass. The three-phase expert kernel
 *       will need the strmm-style triplet (mirror/gemm/scale tilings) plus a
 *       per-row-block kLo/kHi field for tile-level triangular truncation.
 *   (2) Kernel entry contract: ctrmm_kernel.h — kernel_do(...GM pointers,
 *       tiling, numBlocks, stream); by-value tiling (F1 discipline), stream
 *       from the handle; host does not sync.
 *   (3) Data path (this file): A/B read directly from GM column-major
 *       (element idx -> float offset 2*idx for complex64); C written.
 *       B==C aliasing: the naive and pipeline kernels DO have a cross-element
 *       read-after-write hazard (element (i,j) reads B(k,j) that thread (k,j)
 *       overwrites); the host side (ctrmm_host.cpp, C==B branch) stages B
 *       through a workspace D2D copy (or pipeline shadow) before kernel launch.
 *       Only the TINY kernel's internal read-all→syncthreads→write-all pattern
 *       is self-safe against B==C aliasing without host intervention.
 *   (4) Three-phase wiring reference (strmm, same directory):
 *       strmm_mirror_kernel (AIV SIMT, triangular mirror + UNIT diag=1 +
 *       transpose-on-read for trans=T/C) -> strmm_gemm_kernel (AIC tensor_api,
 *       Te::Copy GM2L1/L12L0A/L12L0B + Mmad + CopyL0C2GM, host does the
 *       column-major m<->n swap + side flip) -> strmm_scale_kernel (AIV SIMT,
 *       alpha scaling into C). cherk (blas/herk/arch35/) additionally shows
 *       the all-device complex deinterleave -> 4x real GEMM -> combine shape
 *       that the 3M expert kernel generalizes.
 * ─────────────────────────────────────────────────────────────────────────────
 */

#include <cstdint>

#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "tensor_api/tensor.h"
#include "cann_ops_blas_common.h"
#include "common/helper/kernel_constant.h"
#define KERNEL_UTILS_LITE
#include "common/helper/kernel_utils.h"
#include "ctrmm_kernel.h"

namespace te = AscendC::Te;

constexpr uint16_t PIPE_FLAG = 0;

// ================================================================================
// Shared SIMT accumulation helpers — the single source of truth for the inner
// k-loop of every SIMT compute path (naive / tiny / tiny-k32). Each template is
// instantiated per flag combination, so `if constexpr` here folds exactly as it
// did inline: no runtime branch, no added state. The arithmetic is preserved
// operation-for-operation in the original order, which is what keeps all three
// paths bit-identical to each other and to the recorded golden baselines.
// ================================================================================

// Kahan-compensated float accumulation of one complex term (double is not
// allowed in aicore functions on this platform). Guarded: once a non-finite
// term or accumulator state appears (Inf/NaN-bearing inputs), Kahan's
// compensation arithmetic itself produces NaN (Inf-Inf) and would poison
// positions that must propagate like a plain sum; fall back to plain
// accumulation there. (x - x == 0) holds only for finite x.
__simt_callee__ __aicore__ inline void CtrmmKahanPlane(float t, float& sum, float& cmp)
{
    const bool finite = ((t - t) == 0.0f) && ((sum - sum) == 0.0f) && ((cmp - cmp) == 0.0f);
    if (finite) {
        const float y = t - cmp;
        const float t2 = sum + y;
        cmp = (t2 - sum) - y;
        sum = t2;
    } else {
        sum = sum + t;
    }
}

// Complex Kahan absorb of one term (real then imaginary, same order as inline).
__simt_callee__ __aicore__ inline void CtrmmKahanAccum(
    float tRe, float tIm, float& sumRe, float& cmpRe, float& sumIm, float& cmpIm)
{
    CtrmmKahanPlane(tRe, sumRe, cmpRe);
    CtrmmKahanPlane(tIm, sumIm, cmpIm);
}

// Row/column of one B element for the op(A) product: with side = Left the term
// is B(k,j) @ [j*ldb + k]; with side = Right it is B(i,k) @ [k*ldb + i].
template <bool SIDE_IS_LEFT>
__simt_callee__ __aicore__ inline int64_t CtrmmBOffset(int64_t i64, int64_t j64, int64_t k64, int64_t ldb64)
{
    return SIDE_IS_LEFT ? (j64 * ldb64 + k64) : (k64 * ldb64 + i64);
}

// Structural-triangle membership and float offset of op(A)(.,k) — for
// side = Left the element is op(A)(i,k), for side = Right op(A)(k,j).
// Returns false when the element is a structural zero of the triangle that the
// operation does not reference (the caller skips the term).
template <bool SIDE_IS_LEFT, bool UPLO_IS_UPPER, bool TRANS_IS_T>
__simt_callee__ __aicore__ inline bool CtrmmATriOffset(
    uint32_t i, uint32_t j, uint32_t k, int64_t i64, int64_t j64, int64_t k64, int64_t lda64, int64_t& aIdx)
{
    const bool upper = UPLO_IS_UPPER;
    if constexpr (SIDE_IS_LEFT) {
        // op(A)(i,k): N -> A(i,k) @ [k*lda+i]; T/C -> A(k,i) @ [i*lda+k]
        if constexpr (TRANS_IS_T) {
            aIdx = i64 * lda64 + k64;
            return upper ? (k <= i) : (k >= i);
        } else {
            aIdx = k64 * lda64 + i64;
            return upper ? (i <= k) : (i >= k);
        }
    } else {
        // op(A)(k,j): N -> A(k,j) @ [j*lda+k]; T/C -> A(j,k) @ [k*lda+j]
        if constexpr (TRANS_IS_T) {
            aIdx = k64 * lda64 + j64;
            return upper ? (j <= k) : (j >= k);
        } else {
            aIdx = j64 * lda64 + k64;
            return upper ? (k <= j) : (k >= j);
        }
    }
}

// Read A[k] (with the trans=C conjugate) and B, form the complex product and
// absorb it into the Kahan accumulators. The triangle test is the caller's:
// both call sites (CtrmmDotProduct, CtrmmTinyK32Step) reach this only after
// CtrmmATriOffset said the element is inside the referenced triangle.
template <bool SIDE_IS_LEFT, bool CONJ>
__simt_callee__ __aicore__ inline void CtrmmAccumABProduct(
    int64_t aIdx, int64_t i64, int64_t j64, int64_t k64, int64_t ldb64, const __gm__ float* __restrict aGm,
    const __gm__ float* __restrict bGm, float& sumRe, float& cmpRe, float& sumIm, float& cmpIm)
{
    const float ar = aGm[2 * aIdx];
    float ai = aGm[2 * aIdx + 1];
    if constexpr (CONJ) {
        ai = -ai; // trans = C: conjugate transpose
    }
    const int64_t bIdx = CtrmmBOffset<SIDE_IS_LEFT>(i64, j64, k64, ldb64);
    const float br = bGm[2 * bIdx];
    const float bi = bGm[2 * bIdx + 1];
    // Classic 4M complex multiply-add.
    const float tRe = ar * br - ai * bi;
    const float tIm = ar * bi + ai * br;
    // Kahan absorb — guarded against Inf/NaN, see CtrmmKahanPlane.
    CtrmmKahanAccum(tRe, tIm, sumRe, cmpRe, sumIm, cmpIm);
}

// One output element of C: the whole Kahan-compensated dot product of op(A)
// with B, returned as the compensated pair (sum, cmp) for the caller to finish
// with alpha. Body is the former per-path inner loop, unchanged in operation
// order, so every SIMT path produces identical bits.
template <bool SIDE_IS_LEFT, bool UPLO_IS_UPPER, bool TRANS_IS_T, bool CONJ, bool DIAG_IS_UNIT>
__simt_callee__ __aicore__ inline void CtrmmDotProduct(
    uint32_t i, uint32_t j, uint32_t ordA, int64_t lda64, int64_t ldb64, const __gm__ float* __restrict aGm,
    const __gm__ float* __restrict bGm, float& sumRe, float& cmpRe, float& sumIm, float& cmpIm)
{
    const uint32_t iAbs = i;
    const uint32_t jAbs = j;
    const int64_t i64 = static_cast<int64_t>(iAbs);
    const int64_t j64 = static_cast<int64_t>(jAbs);
    sumRe = 0.0f;
    cmpRe = 0.0f;
    sumIm = 0.0f;
    cmpIm = 0.0f;
    for (uint32_t k = 0; k < ordA; ++k) {
        const int64_t k64 = static_cast<int64_t>(k);
        // Diagonal position of op(A) in this dot product.
        const bool isDiag = SIDE_IS_LEFT ? (k == iAbs) : (k == jAbs);
        float tRe;
        float tIm;
        if constexpr (DIAG_IS_UNIT) {
            if (isDiag) {
                // op(A) diagonal is 1+0i; A's slot is NOT read.
                const int64_t bIdx = CtrmmBOffset<SIDE_IS_LEFT>(i64, j64, k64, ldb64);
                tRe = bGm[2 * bIdx];
                tIm = bGm[2 * bIdx + 1];
                // Kahan absorb (1+0i)*B term — same finite guard.
                CtrmmKahanAccum(tRe, tIm, sumRe, cmpRe, sumIm, cmpIm);
                continue;
            }
        }
        // Structural-triangle membership of the op(A) element.
        int64_t aIdx;
        if (!CtrmmATriOffset<SIDE_IS_LEFT, UPLO_IS_UPPER, TRANS_IS_T>(iAbs, jAbs, k, i64, j64, k64, lda64, aIdx)) {
            continue; // structural zero of the non-referenced triangle
        }
        CtrmmAccumABProduct<SIDE_IS_LEFT, CONJ>(aIdx, i64, j64, k64, ldb64, aGm, bGm, sumRe, cmpRe, sumIm, cmpIm);
    }
}

// Finish one element: C = alpha * (sum + cmp), column-major complex64.
__simt_callee__ __aicore__ inline void CtrmmWriteOutput(
    __gm__ float* __restrict cGm, int64_t i64, int64_t j64, int64_t ldc64, float alphaRe, float alphaIm, float sumRe,
    float cmpRe, float sumIm, float cmpIm)
{
    const float totRe = sumRe + cmpRe;
    const float totIm = sumIm + cmpIm;
    const float oRe = alphaRe * totRe - alphaIm * totIm;
    const float oIm = alphaRe * totIm + alphaIm * totRe;
    const int64_t cIdx = j64 * ldc64 + i64;
    cGm[2 * cIdx] = oRe;
    cGm[2 * cIdx + 1] = oIm;
}

// One full output element: dot product + alpha, written to C. Shared by the
// naive (row-strided) and tiny (flattened grid-stride) paths so their loop
// bodies stay a single source of truth.
template <bool SIDE_IS_LEFT, bool UPLO_IS_UPPER, bool TRANS_IS_T, bool CONJ, bool DIAG_IS_UNIT>
__simt_callee__ __aicore__ inline void CtrmmComputeOne(
    uint32_t i, uint32_t j, uint32_t ordA, int64_t lda64, int64_t ldb64, int64_t ldc64, float alphaRe, float alphaIm,
    const __gm__ float* __restrict aGm, const __gm__ float* __restrict bGm, __gm__ float* __restrict cGm)
{
    const int64_t i64 = static_cast<int64_t>(i);
    const int64_t j64 = static_cast<int64_t>(j);
    float sumRe;
    float cmpRe;
    float sumIm;
    float cmpIm;
    CtrmmDotProduct<SIDE_IS_LEFT, UPLO_IS_UPPER, TRANS_IS_T, CONJ, DIAG_IS_UNIT>(
        i, j, ordA, lda64, ldb64, aGm, bGm, sumRe, cmpRe, sumIm, cmpIm);
    CtrmmWriteOutput(cGm, i64, j64, ldc64, alphaRe, alphaIm, sumRe, cmpRe, sumIm, cmpIm);
}

// ================================================================================
// Naive compute: one SIMT thread strides over output rows [rowStart, rowEnd);
// for each row it walks all n columns and the full k range of op(A).
// Template flags fully resolve the 24-enum matrix at compile time:
//   SIDE_IS_LEFT / UPLO_IS_UPPER / TRANS_IS_T (T or C) / CONJ (C only) / DIAG_IS_UNIT
// ================================================================================

template <bool SIDE_IS_LEFT, bool UPLO_IS_UPPER, bool TRANS_IS_T, bool CONJ, bool DIAG_IS_UNIT>
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void CtrmmNaiveCompute(
    uint32_t m, uint32_t n, uint32_t lda, uint32_t ldb, uint32_t ldc, float alphaRe, float alphaIm,
    const __gm__ float* __restrict aGm, const __gm__ float* __restrict bGm, __gm__ float* __restrict cGm,
    uint32_t rowStart, uint32_t rowEnd)
{
    const int64_t lda64 = static_cast<int64_t>(lda);
    const int64_t ldb64 = static_cast<int64_t>(ldb);
    const int64_t ldc64 = static_cast<int64_t>(ldc);
    // Column-major complex64: element (r,c) of an ld-strided matrix sits at
    // float offset 2*(c*ld + r); real part first, imaginary part second.
    const uint32_t ordA = SIDE_IS_LEFT ? m : n;

    // Kahan-compensated float accumulation (double is not allowed in aicore
    // functions on this platform). A plain sequential float sum drifts far from
    // the golden's accumulation profile on heavy-cancellation elements; the
    // compensation keeps this side near-exact so the residual against the
    // float32 golden is the golden's own rounding. Performance is a non-goal.
    for (uint32_t i = rowStart + threadIdx.x; i < rowEnd; i += blockDim.x) {
        for (uint32_t j = 0; j < n; ++j) {
            CtrmmComputeOne<SIDE_IS_LEFT, UPLO_IS_UPPER, TRANS_IS_T, CONJ, DIAG_IS_UNIT>(
                i, j, ordA, lda64, ldb64, ldc64, alphaRe, alphaIm, aGm, bGm, cGm);
        }
    }
}

// ── Nested dispatch: 24 compile-time instantiations, no runtime branching ──

template <bool SIDE_IS_LEFT, bool UPLO_IS_UPPER, bool TRANS_IS_T, bool CONJ>
__aicore__ inline void CtrmmDispatchDiag(
    bool diagUnit, uint32_t m, uint32_t n, uint32_t lda, uint32_t ldb, uint32_t ldc, float alphaRe, float alphaIm,
    const __gm__ float* aGm, const __gm__ float* bGm, __gm__ float* cGm, uint32_t rowStart, uint32_t rowEnd,
    uint32_t nthreads)
{
    if (diagUnit) {
        asc_vf_call<CtrmmNaiveCompute<SIDE_IS_LEFT, UPLO_IS_UPPER, TRANS_IS_T, CONJ, true>>(
            dim3{nthreads, 1, 1}, m, n, lda, ldb, ldc, alphaRe, alphaIm, aGm, bGm, cGm, rowStart, rowEnd);
    } else {
        asc_vf_call<CtrmmNaiveCompute<SIDE_IS_LEFT, UPLO_IS_UPPER, TRANS_IS_T, CONJ, false>>(
            dim3{nthreads, 1, 1}, m, n, lda, ldb, ldc, alphaRe, alphaIm, aGm, bGm, cGm, rowStart, rowEnd);
    }
}

template <bool SIDE_IS_LEFT, bool UPLO_IS_UPPER, bool TRANS_IS_T>
__aicore__ inline void CtrmmDispatchConj(
    bool conj, bool diagUnit, uint32_t m, uint32_t n, uint32_t lda, uint32_t ldb, uint32_t ldc, float alphaRe,
    float alphaIm, const __gm__ float* aGm, const __gm__ float* bGm, __gm__ float* cGm, uint32_t rowStart,
    uint32_t rowEnd, uint32_t nthreads)
{
    if (conj) {
        CtrmmDispatchDiag<SIDE_IS_LEFT, UPLO_IS_UPPER, TRANS_IS_T, true>(
            diagUnit, m, n, lda, ldb, ldc, alphaRe, alphaIm, aGm, bGm, cGm, rowStart, rowEnd, nthreads);
    } else {
        CtrmmDispatchDiag<SIDE_IS_LEFT, UPLO_IS_UPPER, TRANS_IS_T, false>(
            diagUnit, m, n, lda, ldb, ldc, alphaRe, alphaIm, aGm, bGm, cGm, rowStart, rowEnd, nthreads);
    }
}

template <bool SIDE_IS_LEFT, bool UPLO_IS_UPPER>
__aicore__ inline void CtrmmDispatchTrans(
    uint32_t transMode, bool diagUnit, uint32_t m, uint32_t n, uint32_t lda, uint32_t ldb, uint32_t ldc, float alphaRe,
    float alphaIm, const __gm__ float* aGm, const __gm__ float* bGm, __gm__ float* cGm, uint32_t rowStart,
    uint32_t rowEnd, uint32_t nthreads)
{
    if (transMode == ACLBLAS_OP_N) {
        CtrmmDispatchConj<SIDE_IS_LEFT, UPLO_IS_UPPER, false>(
            false, diagUnit, m, n, lda, ldb, ldc, alphaRe, alphaIm, aGm, bGm, cGm, rowStart, rowEnd, nthreads);
    } else if (transMode == ACLBLAS_OP_T) {
        CtrmmDispatchConj<SIDE_IS_LEFT, UPLO_IS_UPPER, true>(
            false, diagUnit, m, n, lda, ldb, ldc, alphaRe, alphaIm, aGm, bGm, cGm, rowStart, rowEnd, nthreads);
    } else { // ACLBLAS_OP_C
        CtrmmDispatchConj<SIDE_IS_LEFT, UPLO_IS_UPPER, true>(
            true, diagUnit, m, n, lda, ldb, ldc, alphaRe, alphaIm, aGm, bGm, cGm, rowStart, rowEnd, nthreads);
    }
}

template <bool SIDE_IS_LEFT>
__aicore__ inline void CtrmmDispatchUplo(
    bool uploUpper, uint32_t transMode, bool diagUnit, uint32_t m, uint32_t n, uint32_t lda, uint32_t ldb, uint32_t ldc,
    float alphaRe, float alphaIm, const __gm__ float* aGm, const __gm__ float* bGm, __gm__ float* cGm,
    uint32_t rowStart, uint32_t rowEnd, uint32_t nthreads)
{
    if (uploUpper) {
        CtrmmDispatchTrans<SIDE_IS_LEFT, true>(
            transMode, diagUnit, m, n, lda, ldb, ldc, alphaRe, alphaIm, aGm, bGm, cGm, rowStart, rowEnd, nthreads);
    } else {
        CtrmmDispatchTrans<SIDE_IS_LEFT, false>(
            transMode, diagUnit, m, n, lda, ldb, ldc, alphaRe, alphaIm, aGm, bGm, cGm, rowStart, rowEnd, nthreads);
    }
}

__global__ __aicore__ void ctrmm_naive_kernel(
    const __gm__ uint8_t* gmA, const __gm__ uint8_t* gmB, __gm__ uint8_t* gmC, const CtrmmTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    auto* aGm = reinterpret_cast<const __gm__ float* __restrict>(gmA);
    auto* bGm = reinterpret_cast<const __gm__ float* __restrict>(gmB);
    auto* cGm = reinterpret_cast<__gm__ float* __restrict>(gmC);

    const int32_t blkIdx = AscendC::GetBlockIdx();
    const uint32_t rowStart = static_cast<uint32_t>(blkIdx) * tiling.rowsPerCore;
    const uint32_t rowEnd = Min<uint32_t>(rowStart + tiling.rowsPerCore, tiling.m);
    if (rowStart >= rowEnd) {
        return;
    }

    const bool uploUpper = (tiling.uploMode == ACLBLAS_UPPER);
    const bool diagUnit = (tiling.diagMode == ACLBLAS_UNIT);

    if (tiling.sideMode == ACLBLAS_SIDE_LEFT) {
        CtrmmDispatchUplo<true>(
            uploUpper, tiling.transMode, diagUnit, tiling.m, tiling.n, tiling.lda, tiling.ldb, tiling.ldc,
            tiling.alphaReal, tiling.alphaImag, aGm, bGm, cGm, rowStart, rowEnd, tiling.nthreads);
    } else {
        CtrmmDispatchUplo<false>(
            uploUpper, tiling.transMode, diagUnit, tiling.m, tiling.n, tiling.lda, tiling.ldb, tiling.ldc,
            tiling.alphaReal, tiling.alphaImag, aGm, bGm, cGm, rowStart, rowEnd, tiling.nthreads);
    }
}

void ctrmm_naive_kernel_do(
    const uint8_t* gmA, const uint8_t* gmB, uint8_t* gmC, const CtrmmTilingData& tiling, uint32_t numBlocks,
    void* stream)
{
    ctrmm_naive_kernel<<<numBlocks, nullptr, stream>>>(gmA, gmB, gmC, tiling);
}

// ================================================================================
// TINY fast path — single-block flattened (i,j) kernel for m,n <= 32.
//
// The naive row-per-core kernel is memory-latency-bound at tiny sizes (one
// active thread per row walks a serial k-loop of strided scalar GM loads:
// 32x32 measures ~277us). This variant keeps the naive inner loop VERBATIM
// (Kahan-compensated accumulation, structural-triangle skip, UNIT diagonal
// no-read, trans=C conjugate — identical numerics) but distributes outputs
// flattened over (i,j) with grid-stride, stages results in registers, and
// writes C only after asc_syncthreads(): every A/B read completes before any
// C write, which makes the in-place B==C case deterministically safe (the
// naive kernel relies on scheduling luck for the same hazard).
// ================================================================================
template <bool SIDE_IS_LEFT, bool UPLO_IS_UPPER, bool TRANS_IS_T, bool CONJ, bool DIAG_IS_UNIT>
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void CtrmmTinyCompute(
    uint32_t m, uint32_t n, uint32_t lda, uint32_t ldb, uint32_t ldc, float alphaRe, float alphaIm,
    const __gm__ float* __restrict aGm, const __gm__ float* __restrict bGm, __gm__ float* __restrict cGm)
{
    const uint32_t total = m * n;
    const int64_t lda64 = static_cast<int64_t>(lda);
    const int64_t ldb64 = static_cast<int64_t>(ldb);
    const int64_t ldc64 = static_cast<int64_t>(ldc);
    const uint32_t ordA = SIDE_IS_LEFT ? m : n;

    // Pass 1: compute every assigned output into registers (GM reads only).
    for (uint32_t idx = threadIdx.x; idx < total; idx += blockDim.x) {
        const uint32_t i = idx / n;
        const uint32_t j = idx % n;
        CtrmmComputeOne<SIDE_IS_LEFT, UPLO_IS_UPPER, TRANS_IS_T, CONJ, DIAG_IS_UNIT>(
            i, j, ordA, lda64, ldb64, ldc64, alphaRe, alphaIm, aGm, bGm, cGm);
    }
}

// ── Tiny nested dispatch (mirrors the naive chain; single block, no row split) ──

template <bool SIDE_IS_LEFT, bool UPLO_IS_UPPER, bool TRANS_IS_T, bool CONJ>
__aicore__ inline void CtrmmTinyDispatchDiag(
    bool diagUnit, uint32_t m, uint32_t n, uint32_t lda, uint32_t ldb, uint32_t ldc, float alphaRe, float alphaIm,
    const __gm__ float* aGm, const __gm__ float* bGm, __gm__ float* cGm, uint32_t nthreads)
{
    if (diagUnit) {
        asc_vf_call<CtrmmTinyCompute<SIDE_IS_LEFT, UPLO_IS_UPPER, TRANS_IS_T, CONJ, true>>(
            dim3{nthreads, 1, 1}, m, n, lda, ldb, ldc, alphaRe, alphaIm, aGm, bGm, cGm);
    } else {
        asc_vf_call<CtrmmTinyCompute<SIDE_IS_LEFT, UPLO_IS_UPPER, TRANS_IS_T, CONJ, false>>(
            dim3{nthreads, 1, 1}, m, n, lda, ldb, ldc, alphaRe, alphaIm, aGm, bGm, cGm);
    }
}

template <bool SIDE_IS_LEFT, bool UPLO_IS_UPPER, bool TRANS_IS_T>
__aicore__ inline void CtrmmTinyDispatchConj(
    bool conj, bool diagUnit, uint32_t m, uint32_t n, uint32_t lda, uint32_t ldb, uint32_t ldc, float alphaRe,
    float alphaIm, const __gm__ float* aGm, const __gm__ float* bGm, __gm__ float* cGm, uint32_t nthreads)
{
    if (conj) {
        CtrmmTinyDispatchDiag<SIDE_IS_LEFT, UPLO_IS_UPPER, TRANS_IS_T, true>(
            diagUnit, m, n, lda, ldb, ldc, alphaRe, alphaIm, aGm, bGm, cGm, nthreads);
    } else {
        CtrmmTinyDispatchDiag<SIDE_IS_LEFT, UPLO_IS_UPPER, TRANS_IS_T, false>(
            diagUnit, m, n, lda, ldb, ldc, alphaRe, alphaIm, aGm, bGm, cGm, nthreads);
    }
}

template <bool SIDE_IS_LEFT, bool UPLO_IS_UPPER>
__aicore__ inline void CtrmmTinyDispatchTrans(
    uint32_t transMode, bool diagUnit, uint32_t m, uint32_t n, uint32_t lda, uint32_t ldb, uint32_t ldc, float alphaRe,
    float alphaIm, const __gm__ float* aGm, const __gm__ float* bGm, __gm__ float* cGm, uint32_t nthreads)
{
    if (transMode == ACLBLAS_OP_N) {
        CtrmmTinyDispatchConj<SIDE_IS_LEFT, UPLO_IS_UPPER, false>(
            false, diagUnit, m, n, lda, ldb, ldc, alphaRe, alphaIm, aGm, bGm, cGm, nthreads);
    } else if (transMode == ACLBLAS_OP_T) {
        CtrmmTinyDispatchConj<SIDE_IS_LEFT, UPLO_IS_UPPER, true>(
            false, diagUnit, m, n, lda, ldb, ldc, alphaRe, alphaIm, aGm, bGm, cGm, nthreads);
    } else {
        CtrmmTinyDispatchConj<SIDE_IS_LEFT, UPLO_IS_UPPER, true>(
            true, diagUnit, m, n, lda, ldb, ldc, alphaRe, alphaIm, aGm, bGm, cGm, nthreads);
    }
}

template <bool SIDE_IS_LEFT>
__aicore__ inline void CtrmmTinyDispatchUplo(
    bool uploUpper, uint32_t transMode, bool diagUnit, uint32_t m, uint32_t n, uint32_t lda, uint32_t ldb, uint32_t ldc,
    float alphaRe, float alphaIm, const __gm__ float* aGm, const __gm__ float* bGm, __gm__ float* cGm,
    uint32_t nthreads)
{
    if (uploUpper) {
        CtrmmTinyDispatchTrans<SIDE_IS_LEFT, true>(
            transMode, diagUnit, m, n, lda, ldb, ldc, alphaRe, alphaIm, aGm, bGm, cGm, nthreads);
    } else {
        CtrmmTinyDispatchTrans<SIDE_IS_LEFT, false>(
            transMode, diagUnit, m, n, lda, ldb, ldc, alphaRe, alphaIm, aGm, bGm, cGm, nthreads);
    }
}

__global__ __aicore__ void ctrmm_tiny_kernel(
    const __gm__ uint8_t* gmA, const __gm__ uint8_t* gmB, __gm__ uint8_t* gmC, const CtrmmTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    auto* aGm = reinterpret_cast<const __gm__ float* __restrict>(gmA);
    auto* bGm = reinterpret_cast<const __gm__ float* __restrict>(gmB);
    auto* cGm = reinterpret_cast<__gm__ float* __restrict>(gmC);
    const bool uploUpper = (tiling.uploMode == ACLBLAS_UPPER);
    const bool diagUnit = (tiling.diagMode == ACLBLAS_UNIT);
    if (tiling.sideMode == ACLBLAS_SIDE_LEFT) {
        CtrmmTinyDispatchUplo<true>(
            uploUpper, tiling.transMode, diagUnit, tiling.m, tiling.n, tiling.lda, tiling.ldb, tiling.ldc,
            tiling.alphaReal, tiling.alphaImag, aGm, bGm, cGm, tiling.nthreads);
    } else {
        CtrmmTinyDispatchUplo<false>(
            uploUpper, tiling.transMode, diagUnit, tiling.m, tiling.n, tiling.lda, tiling.ldb, tiling.ldc,
            tiling.alphaReal, tiling.alphaImag, aGm, bGm, cGm, tiling.nthreads);
    }
}

void ctrmm_tiny_kernel_do(
    const uint8_t* gmA, const uint8_t* gmB, uint8_t* gmC, const CtrmmTilingData& tiling, void* stream)
{
    ctrmm_tiny_kernel<<<1, nullptr, stream>>>(gmA, gmB, gmC, tiling);
}

// ---- TINY K=32 specialization: compile-time-unrolled k loop. The generic
// path walks ordA GM loads per output serially (latency chain ~28us at
// 32x32); with a literal bound the unroller + inliner emits named registers
// and the scheduler batches the independent __restrict GM loads ahead of the
// Kahan MAC chain (identical order, identical numerics).
template <bool SIDE_IS_LEFT, bool UPLO_IS_UPPER, bool TRANS_IS_T, bool CONJ, bool DIAG_IS_UNIT>
__simt_callee__ __aicore__ inline void CtrmmTinyK32Step(
    uint32_t k, uint32_t i, uint32_t j, int64_t lda64, int64_t ldb64, int64_t i64, int64_t j64,
    const __gm__ float* __restrict aGm, const __gm__ float* __restrict bGm, float& sumRe, float& cmpRe, float& sumIm,
    float& cmpIm)
{
    const int64_t k64 = static_cast<int64_t>(k);
    const bool isDiag = SIDE_IS_LEFT ? (k == i) : (k == j);
    float tRe;
    float tIm;
    if (DIAG_IS_UNIT && isDiag) {
        const int64_t bIdx = CtrmmBOffset<SIDE_IS_LEFT>(i64, j64, k64, ldb64);
        tRe = bGm[2 * bIdx];
        tIm = bGm[2 * bIdx + 1];
        CtrmmKahanAccum(tRe, tIm, sumRe, cmpRe, sumIm, cmpIm);
        return;
    }
    int64_t aIdx;
    if (!CtrmmATriOffset<SIDE_IS_LEFT, UPLO_IS_UPPER, TRANS_IS_T>(i, j, k, i64, j64, k64, lda64, aIdx)) {
        return;
    }
    CtrmmAccumABProduct<SIDE_IS_LEFT, CONJ>(aIdx, i64, j64, k64, ldb64, aGm, bGm, sumRe, cmpRe, sumIm, cmpIm);
}

template <bool SIDE_IS_LEFT, bool UPLO_IS_UPPER, bool TRANS_IS_T, bool CONJ, bool DIAG_IS_UNIT>
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void CtrmmTinyK32Compute(
    uint32_t m, uint32_t n, uint32_t lda, uint32_t ldb, uint32_t ldc, float alphaRe, float alphaIm,
    const __gm__ float* __restrict aGm, const __gm__ float* __restrict bGm, __gm__ float* __restrict cGm)
{
    const uint32_t total = m * n;
    const int64_t lda64 = static_cast<int64_t>(lda);
    const int64_t ldb64 = static_cast<int64_t>(ldb);
    const int64_t ldc64 = static_cast<int64_t>(ldc);
    (void)lda64;
    for (uint32_t idx = threadIdx.x; idx < total; idx += blockDim.x) {
        const uint32_t i = idx / n;
        const uint32_t j = idx % n;
        const int64_t i64 = static_cast<int64_t>(i);
        const int64_t j64 = static_cast<int64_t>(j);
        float sumRe = 0.0f, cmpRe = 0.0f, sumIm = 0.0f, cmpIm = 0.0f;
#pragma unroll
        for (uint32_t k = 0; k < 32; ++k) {
            CtrmmTinyK32Step<SIDE_IS_LEFT, UPLO_IS_UPPER, TRANS_IS_T, CONJ, DIAG_IS_UNIT>(
                k, i, j, lda64, ldb64, i64, j64, aGm, bGm, sumRe, cmpRe, sumIm, cmpIm);
        }
        CtrmmWriteOutput(cGm, i64, j64, ldc64, alphaRe, alphaIm, sumRe, cmpRe, sumIm, cmpIm);
    }
}

template <bool SIDE_IS_LEFT, bool UPLO_IS_UPPER, bool TRANS_IS_T, bool CONJ>
__aicore__ inline void CtrmmTinyK32DispatchDiag(
    bool diagUnit, uint32_t m, uint32_t n, uint32_t lda, uint32_t ldb, uint32_t ldc, float alphaRe, float alphaIm,
    const __gm__ float* aGm, const __gm__ float* bGm, __gm__ float* cGm, uint32_t nthreads)
{
    if (diagUnit) {
        asc_vf_call<CtrmmTinyK32Compute<SIDE_IS_LEFT, UPLO_IS_UPPER, TRANS_IS_T, CONJ, true>>(
            dim3{nthreads, 1, 1}, m, n, lda, ldb, ldc, alphaRe, alphaIm, aGm, bGm, cGm);
    } else {
        asc_vf_call<CtrmmTinyK32Compute<SIDE_IS_LEFT, UPLO_IS_UPPER, TRANS_IS_T, CONJ, false>>(
            dim3{nthreads, 1, 1}, m, n, lda, ldb, ldc, alphaRe, alphaIm, aGm, bGm, cGm);
    }
}
template <bool SIDE_IS_LEFT, bool UPLO_IS_UPPER, bool TRANS_IS_T>
__aicore__ inline void CtrmmTinyK32DispatchConj(
    bool conj, bool diagUnit, uint32_t m, uint32_t n, uint32_t lda, uint32_t ldb, uint32_t ldc, float alphaRe,
    float alphaIm, const __gm__ float* aGm, const __gm__ float* bGm, __gm__ float* cGm, uint32_t nthreads)
{
    if (conj)
        CtrmmTinyK32DispatchDiag<SIDE_IS_LEFT, UPLO_IS_UPPER, TRANS_IS_T, true>(
            diagUnit, m, n, lda, ldb, ldc, alphaRe, alphaIm, aGm, bGm, cGm, nthreads);
    else
        CtrmmTinyK32DispatchDiag<SIDE_IS_LEFT, UPLO_IS_UPPER, TRANS_IS_T, false>(
            diagUnit, m, n, lda, ldb, ldc, alphaRe, alphaIm, aGm, bGm, cGm, nthreads);
}
template <bool SIDE_IS_LEFT, bool UPLO_IS_UPPER>
__aicore__ inline void CtrmmTinyK32DispatchTrans(
    uint32_t transMode, bool diagUnit, uint32_t m, uint32_t n, uint32_t lda, uint32_t ldb, uint32_t ldc, float alphaRe,
    float alphaIm, const __gm__ float* aGm, const __gm__ float* bGm, __gm__ float* cGm, uint32_t nthreads)
{
    if (transMode == ACLBLAS_OP_N)
        CtrmmTinyK32DispatchConj<SIDE_IS_LEFT, UPLO_IS_UPPER, false>(
            false, diagUnit, m, n, lda, ldb, ldc, alphaRe, alphaIm, aGm, bGm, cGm, nthreads);
    else if (transMode == ACLBLAS_OP_T)
        CtrmmTinyK32DispatchConj<SIDE_IS_LEFT, UPLO_IS_UPPER, true>(
            false, diagUnit, m, n, lda, ldb, ldc, alphaRe, alphaIm, aGm, bGm, cGm, nthreads);
    else
        CtrmmTinyK32DispatchConj<SIDE_IS_LEFT, UPLO_IS_UPPER, true>(
            true, diagUnit, m, n, lda, ldb, ldc, alphaRe, alphaIm, aGm, bGm, cGm, nthreads);
}
template <bool SIDE_IS_LEFT>
__aicore__ inline void CtrmmTinyK32DispatchUplo(
    bool uploUpper, uint32_t transMode, bool diagUnit, uint32_t m, uint32_t n, uint32_t lda, uint32_t ldb, uint32_t ldc,
    float alphaRe, float alphaIm, const __gm__ float* aGm, const __gm__ float* bGm, __gm__ float* cGm,
    uint32_t nthreads)
{
    if (uploUpper)
        CtrmmTinyK32DispatchTrans<SIDE_IS_LEFT, true>(
            transMode, diagUnit, m, n, lda, ldb, ldc, alphaRe, alphaIm, aGm, bGm, cGm, nthreads);
    else
        CtrmmTinyK32DispatchTrans<SIDE_IS_LEFT, false>(
            transMode, diagUnit, m, n, lda, ldb, ldc, alphaRe, alphaIm, aGm, bGm, cGm, nthreads);
}

__global__ __aicore__ void ctrmm_tiny_k32_kernel(
    const __gm__ uint8_t* gmA, const __gm__ uint8_t* gmB, __gm__ uint8_t* gmC, const CtrmmTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    auto* aGm = reinterpret_cast<const __gm__ float* __restrict>(gmA);
    auto* bGm = reinterpret_cast<const __gm__ float* __restrict>(gmB);
    auto* cGm = reinterpret_cast<__gm__ float* __restrict>(gmC);
    const bool uploUpper = (tiling.uploMode == ACLBLAS_UPPER);
    const bool diagUnit = (tiling.diagMode == ACLBLAS_UNIT);
    if (tiling.sideMode == ACLBLAS_SIDE_LEFT) {
        CtrmmTinyK32DispatchUplo<true>(
            uploUpper, tiling.transMode, diagUnit, tiling.m, tiling.n, tiling.lda, tiling.ldb, tiling.ldc,
            tiling.alphaReal, tiling.alphaImag, aGm, bGm, cGm, tiling.nthreads);
    } else {
        CtrmmTinyK32DispatchUplo<false>(
            uploUpper, tiling.transMode, diagUnit, tiling.m, tiling.n, tiling.lda, tiling.ldb, tiling.ldc,
            tiling.alphaReal, tiling.alphaImag, aGm, bGm, cGm, tiling.nthreads);
    }
}

void ctrmm_tiny_k32_kernel_do(
    const uint8_t* gmA, const uint8_t* gmB, uint8_t* gmC, const CtrmmTilingData& tiling, void* stream)
{
    ctrmm_tiny_k32_kernel<<<1, nullptr, stream>>>(gmA, gmB, gmC, tiling);
}

// ================================================================================
// M3 4M all-device pipeline — Phase-A (split/mirror) and Phase-C (combine).
// Phase-B is the tensor_api cube kernel driven from the host (ctrmm_host.cpp).
// ================================================================================

// A-mirror element: materialize op(A)(row, col) with the non-referenced
// triangle zeroed, UNIT diagonal written as 1+0i (A's slot never read), and
// trans = C negating the imaginary part. Same index math as the naive kernel.
#include "ctrmm_split_mirror.h"

__global__ __aicore__ void ctrmm_split_mirror_kernel(
    const __gm__ uint8_t* gmA, const __gm__ uint8_t* gmB, __gm__ uint8_t* gmAr, __gm__ uint8_t* gmAi,
    __gm__ uint8_t* gmBr, __gm__ uint8_t* gmBi, __gm__ uint8_t* gmAsum, __gm__ uint8_t* gmBsum,
    const CtrmmSplitMirrorTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    auto* aGm = reinterpret_cast<const __gm__ float* __restrict>(gmA);
    auto* bGm = reinterpret_cast<const __gm__ float* __restrict>(gmB);
    auto* arGm = reinterpret_cast<__gm__ float* __restrict>(gmAr);
    auto* aiGm = reinterpret_cast<__gm__ float* __restrict>(gmAi);
    auto* brGm = reinterpret_cast<__gm__ float* __restrict>(gmBr);
    auto* biGm = reinterpret_cast<__gm__ float* __restrict>(gmBi);
    auto* asumGm = reinterpret_cast<__gm__ float* __restrict>(gmAsum);
    auto* bsumGm = reinterpret_cast<__gm__ float* __restrict>(gmBsum);

    const int32_t blkIdx = AscendC::GetBlockIdx();
    const uint32_t rowStartA = static_cast<uint32_t>(blkIdx) * tiling.rowsPerCoreA;
    const uint32_t rowEndA = Min<uint32_t>(rowStartA + tiling.rowsPerCoreA, tiling.dimA);
    const uint32_t rowStartB = static_cast<uint32_t>(blkIdx) * tiling.rowsPerCoreB;
    const uint32_t rowEndB = Min<uint32_t>(rowStartB + tiling.rowsPerCoreB, tiling.m);
    if (rowStartA >= rowEndA && rowStartB >= rowEndB) {
        return;
    }

    const bool uploUpper = (tiling.uploMode == ACLBLAS_UPPER);
    const bool diagUnit = (tiling.diagMode == ACLBLAS_UNIT);
    if (uploUpper) {
        CtrmmDispatchSplitTrans<true>(
            tiling.transMode, diagUnit, tiling.dimA, tiling.m, tiling.n, tiling.lda, tiling.ldb, aGm, bGm, arGm, aiGm,
            brGm, biGm, asumGm, bsumGm, rowStartA, rowEndA, rowStartB, rowEndB, tiling.nthreads, tiling.skipB);
    } else {
        CtrmmDispatchSplitTrans<false>(
            tiling.transMode, diagUnit, tiling.dimA, tiling.m, tiling.n, tiling.lda, tiling.ldb, aGm, bGm, arGm, aiGm,
            brGm, biGm, asumGm, bsumGm, rowStartA, rowEndA, rowStartB, rowEndB, tiling.nthreads, tiling.skipB);
    }
}

void ctrmm_split_mirror_kernel_do(
    const uint8_t* gmA, const uint8_t* gmB, uint8_t* gmAr, uint8_t* gmAi, uint8_t* gmBr, uint8_t* gmBi, uint8_t* gmAsum,
    uint8_t* gmBsum, const CtrmmSplitMirrorTilingData& tiling, uint32_t numBlocks, void* stream)
{
    ctrmm_split_mirror_kernel<<<numBlocks, nullptr, stream>>>(gmA, gmB, gmAr, gmAi, gmBr, gmBi, gmAsum, gmBsum, tiling);
}

// ================================================================================
// Phase-B alternative: strmm-lineage cube kernel (L0 lever, 方案 L0 路线 A).
// Ported from strmm_gemm_kernel (same directory) which measures 5.1-5.7 real
// TMAC/s on this device — 2.5-2.8x the GEMM-module path. Host drives it with
// the strmm recipe: tiling built on swapped (n, m) + flipped side, lda/ldb
// unswapped; temp stores the product readable as P(i,j) @ [j*stride + i].
// ================================================================================

#include "ctrmm_phaseb_tile.h"

// One real plane product through the strmm-lineage pipeline. Kept as a plain
// __aicore__ function so the merged entry point can run several products in a
// single launch (the L1/flag lifecycle below is per-product, exactly as it was
// per-launch before — only the launch cost disappears).
__aicore__ inline void CtrmmPhaseBRunProduct(
    __gm__ uint8_t* gmA, const __gm__ uint8_t* gmB, __gm__ uint8_t* gmTemp, const CtrmmPhaseBTilingData& tiling)
{
    uint32_t K = (tiling.sideMode == ACLBLAS_SIDE_LEFT) ? tiling.m : tiling.n;
    const __gm__ uint8_t* gmLeft = (tiling.sideMode == ACLBLAS_SIDE_LEFT) ? gmA : gmB;
    const __gm__ uint8_t* gmRight = (tiling.sideMode == ACLBLAS_SIDE_LEFT) ? gmB : gmA;
    uint64_t leftLd = (tiling.sideMode == ACLBLAS_SIDE_LEFT) ? tiling.lda : tiling.ldb;
    uint64_t leftRows = tiling.m;
    uint64_t rightLd = (tiling.sideMode == ACLBLAS_SIDE_LEFT) ? tiling.ldb : tiling.lda;
    uint64_t rightRows = K;
    auto gmLeftTensor = te::MakeTensor(
        te::MakeMemPtr<te::Location::GM>(reinterpret_cast<__gm__ float*>(const_cast<__gm__ uint8_t*>(gmLeft))),
        te::MakeFrameLayout<te::NDExtLayoutPtn>(leftRows, leftLd));
    auto gmRightTensor = te::MakeTensor(
        te::MakeMemPtr<te::Location::GM>(reinterpret_cast<__gm__ float*>(const_cast<__gm__ uint8_t*>(gmRight))),
        te::MakeFrameLayout<te::NDExtLayoutPtn>(rightRows, rightLd));
    auto gmTempTensor = te::MakeTensor(
        te::MakeMemPtr<te::Location::GM>(reinterpret_cast<__gm__ float*>(gmTemp)),
        te::MakeFrameLayout<te::NDExtLayoutPtn>(tiling.m, tiling.tempRowStride));
    uint32_t divM = CeilDiv<uint32_t>(tiling.m, tiling.singleCoreM);
    uint32_t divN = CeilDiv<uint32_t>(tiling.n, tiling.singleCoreN);
    uint64_t totalTiles = static_cast<uint64_t>(divM) * static_cast<uint64_t>(divN);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(0);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(1);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(0);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(1);
    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(PIPE_FLAG);
    uint64_t l0PingPong = 0;
    uint64_t abL1LoopCnt = 0;
    for (uint64_t tileIdx = AscendC::GetBlockIdx(); tileIdx < totalTiles; tileIdx += AscendC::GetBlockNum()) {
        uint64_t coreIdxM = tileIdx / divN;
        uint64_t coreIdxN = tileIdx % divN;
        if (coreIdxM % 2 == 1) {
            coreIdxN = divN - 1 - coreIdxN;
        }
        uint32_t mStart = coreIdxM * tiling.singleCoreM;
        uint32_t nStart = coreIdxN * tiling.singleCoreN;
        ctrmm_phaseb::ProcessTile(
            gmLeftTensor, gmRightTensor, gmTempTensor, K, tiling.tileKChunk, ctrmm_phaseb::PB_BASE_K, tiling.clipMode,
            mStart, Min<uint32_t>(mStart + tiling.singleCoreM, tiling.m), nStart,
            Min<uint32_t>(nStart + tiling.singleCoreN, tiling.n), tiling.tileM, tiling.tileN, abL1LoopCnt, l0PingPong);
    }
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(1);
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(0);
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(1);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(PIPE_FLAG);
}

__global__ __aicore__ void ctrmm_phaseb_kernel(
    __gm__ uint8_t* gmA, const __gm__ uint8_t* gmB, __gm__ uint8_t* gmTemp, const CtrmmPhaseBTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    CtrmmPhaseBRunProduct(gmA, gmB, gmTemp, tiling);
}

void ctrmm_phaseb_kernel_do(
    uint8_t* gmA, const uint8_t* gmB, uint8_t* gmTemp, const CtrmmPhaseBTilingData& tiling, uint32_t numBlocks,
    void* stream)
{
    ctrmm_phaseb_kernel<<<numBlocks, nullptr, stream>>>(gmA, gmB, gmTemp, tiling);
}

// Merged entry point: three products (P1 = A0*B0, P2 = A1*B1, P3 = A2*B2) in ONE
// launch. Textually identical work to three ctrmm_phaseb_kernel launches; only
// the per-launch fixed cost (~15-20us at 512^2) is removed. The scheduling
// model (singleCoreM/N, tileM/N, L1 double buffering, flag lifecycle) is
// untouched on purpose — earlier experiments showed changing it hurt performance.
__global__ __aicore__ void ctrmm_phaseb3_kernel(
    __gm__ uint8_t* gmA0, __gm__ uint8_t* gmA1, __gm__ uint8_t* gmA2, const __gm__ uint8_t* gmB0,
    const __gm__ uint8_t* gmB1, const __gm__ uint8_t* gmB2, __gm__ uint8_t* gmT0, __gm__ uint8_t* gmT1,
    __gm__ uint8_t* gmT2, const CtrmmPhaseBTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    CtrmmPhaseBRunProduct(gmA0, gmB0, gmT0, tiling);
    CtrmmPhaseBRunProduct(gmA1, gmB1, gmT1, tiling);
    CtrmmPhaseBRunProduct(gmA2, gmB2, gmT2, tiling);
}

void ctrmm_phaseb3_kernel_do(
    uint8_t* gmA0, uint8_t* gmA1, uint8_t* gmA2, const uint8_t* gmB0, const uint8_t* gmB1, const uint8_t* gmB2,
    uint8_t* gmT0, uint8_t* gmT1, uint8_t* gmT2, const CtrmmPhaseBTilingData& tiling, uint32_t numBlocks, void* stream)
{
    ctrmm_phaseb3_kernel<<<numBlocks, nullptr, stream>>>(gmA0, gmA1, gmA2, gmB0, gmB1, gmB2, gmT0, gmT1, gmT2, tiling);
}

// ================================================================================
// Phase-C alternative: tensor_api (TLA) combine — L5 lever, cherk combine
// lineage. Same math as the SIMT combine, but the plane traffic moves through
// UB with 2D te::Copy (block DMA) instead of element-wise SIMT scalar loads,
// and the complex interleaving uses the Interleave vector intrinsic.
//   re = alphaRe*pr - alphaIm*pi ;  im = alphaRe*pi + alphaIm*pr
//   pr = P1-P2 ;  pi = P3-P1-P2   (Gauss 3M)
// ================================================================================

// Explicit using-declarations for the TLA section (this file mixes SIMT and
// tensor_api code, so no blanket `using namespace AscendC`).
using AscendC::GlobalTensor;
using AscendC::HardEvent;
using AscendC::Interleave;
using AscendC::LocalTensor;
using AscendC::PipeBarrier;
using AscendC::SetFlag;
using AscendC::TBuf;
using AscendC::TPipe;
using AscendC::TPosition;
using AscendC::WaitFlag;

namespace ctrmm_tla {

constexpr uint32_t TLA_BLOCK = 64; // rows/cols per UB tile (cherk SCALE_BLOCK)
// UB budget guard (raised in review as "TLA_BLOCK has no static assertion"):
// the hungriest of the three tensor_api kernels is the combine kernel, which
// allocates 7 plane-sized buffers (P1..P3 + re|im output + packed complex
// staging); the two split kernels allocate 5. With TLA_BLOCK = 64 each plane is
// 16 KB, so combine needs 112 KB and the splits 80 KB. The AIV UB on this
// platform is ~192 KB, which bounds TLA_BLOCK to 83 for the combine kernel;
// bumping TLA_BLOCK to 128 would need 448 KB and cannot fit. The assertions
// below turn that implicit budget into a compile-time contract.
constexpr uint32_t TLA_UB_BYTES = 192 * 1024; // AIV UB per core (conservative)
constexpr uint32_t TLA_PLANE_BYTES = TLA_BLOCK * TLA_BLOCK * sizeof(float);
static_assert(
    7 * TLA_PLANE_BYTES <= TLA_UB_BYTES, "combine kernel needs 7 plane-sized UB buffers; TLA_BLOCK too large for UB");
static_assert(
    5 * TLA_PLANE_BYTES <= TLA_UB_BYTES, "split kernels need 5 plane-sized UB buffers; TLA_BLOCK too large for UB");
constexpr uint32_t TLA_ALIGN = 8; // 32B float alignment

// 1D UB tensor view over a LocalTensor region (cherk-local helper re-declared
// here so the two kernels stay self-contained).
template <typename T>
__aicore__ inline auto MakeUbTensor1D(const LocalTensor<T>& lt, uint32_t offset, uint32_t count)
{
    return te::MakeTensor(
        te::MakeMemPtr<te::Location::UB, T>(lt.GetPhyAddr() + static_cast<uint64_t>(offset) * sizeof(T)),
        te::MakeFrameLayout<te::NDLayoutPtn>(static_cast<uint64_t>(1), static_cast<uint64_t>(count)));
}

template <typename UbTensor>
__aicore__ inline void LoadPlaneBlock(
    __gm__ float* gmPlane, UbTensor ubPlane, uint32_t jBase, uint32_t iBase, uint32_t cols, uint32_t rows,
    uint32_t jExtent, uint32_t stride, uint32_t ubStride)
{
    // Plane P(i,j) lives at [j*stride + i]. Read as a row-major (NDExt) view:
    // dim0 = jExtent (rows of the view), dim1 = stride (row stride).
    auto gmT = te::MakeTensor(
        te::MakeMemPtr<te::Location::GM>(gmPlane),
        te::MakeFrameLayout<te::NDExtLayoutPtn>(static_cast<uint64_t>(jExtent), static_cast<uint64_t>(stride)));
    auto gmBlock = gmT.Slice(
        te::MakeCoord(static_cast<uint64_t>(jBase), static_cast<uint64_t>(iBase)),
        te::MakeShape(static_cast<uint64_t>(cols), static_cast<uint64_t>(rows)));
    te::Copy(te::MakeCopy(te::CopyGM2UB{}), ubPlane, gmBlock);
}

// Complex tile GM -> UB. The source is viewed as (jExtent x ldGm*2) floats and
// sliced (cols x rows*2) at (jBase, iBase*2); the destination is a flat
// interleaved [re,im,...] stream whose row stride is padded to ubStride*2. The
// MTE2->V barrier is part of the copy contract, as it was at both call sites.
template <typename UbTensor>
__aicore__ inline void LoadComplexTile(
    __gm__ float* gmCplx, UbTensor ubIn, uint32_t jBase, uint32_t iBase, uint32_t cols, uint32_t rows, uint32_t jExtent,
    uint32_t ldGm, uint32_t ubStride)
{
    auto gmT = te::MakeTensor(
        te::MakeMemPtr<te::Location::GM>(gmCplx),
        te::MakeFrameLayout<te::NDExtLayoutPtn>(static_cast<uint64_t>(jExtent), static_cast<uint64_t>(ldGm * 2)));
    auto gmBlock = gmT.Slice(
        te::MakeCoord(static_cast<uint64_t>(jBase), static_cast<uint64_t>(iBase) * 2),
        te::MakeShape(static_cast<uint64_t>(cols), static_cast<uint64_t>(rows) * 2));
    auto ubT = te::MakeTensor(
        te::MakeMemPtr<te::Location::UB, float>(ubIn.GetPhyAddr()),
        te::MakeFrameLayout<te::NDExtLayoutPtn>(static_cast<uint64_t>(cols), static_cast<uint64_t>(ubStride * 2)));
    te::Copy(te::MakeCopy(te::CopyGM2UB{}), ubT, gmBlock);
    SetFlag<HardEvent::MTE2_V>(0);
    WaitFlag<HardEvent::MTE2_V>(0);
}

// Deinterleave the packed [re,im,...] stream and form the 3M sum plane
// sum = re + im in UB. The float count passed to DeInterleave is 2x the plane
// element count (the interleaved source length); passing the plane count
// deinterleaves only half the stream and mis-pairs — see the note at the
// previous call sites. Issues the V->MTE3 barrier both callers relied on.
template <typename UbTensor>
__aicore__ inline void SumPlanes(UbTensor ubSum, UbTensor ubRe, UbTensor ubIm, uint32_t cnt)
{
    auto reT = MakeUbTensor1D(ubRe, 0U, cnt);
    auto imT = MakeUbTensor1D(ubIm, 0U, cnt);
    auto sumT = MakeUbTensor1D(ubSum, 0U, cnt);
    te::Transform<te::Inst::MulScalar>(sumT, reT, 1.0f); // sum = re
    te::Transform<te::Inst::Axpy>(sumT, imT, 1.0f);      // sum = re + im
    PipeBarrier<PIPE_ALL>();
    SetFlag<HardEvent::V_MTE3>(0);
    WaitFlag<HardEvent::V_MTE3>(0);
}

template <typename UbTensor>
__aicore__ inline void SplitComplexTile(UbTensor ubRe, UbTensor ubIm, UbTensor ubSum, UbTensor ubIn, uint32_t cnt)
{
    DeInterleave(ubRe, ubIm, ubIn, static_cast<int32_t>(2 * cnt));
    PipeBarrier<PIPE_ALL>();
    SumPlanes(ubSum, ubRe, ubIm, cnt);
}

// One float plane UB -> GM, plane layout (jExtent x stride), tile staged at
// (jBase, iBase). Written out explicitly rather than through a device lambda:
// lambdas capturing te:: tensors do not resolve on this target.
template <typename UbTensor>
__aicore__ inline void StorePlaneBlock(
    __gm__ float* gmPlane, UbTensor ubPlane, uint32_t jBase, uint32_t iBase, uint32_t cols, uint32_t rows,
    uint32_t jExtent, uint32_t stride, uint32_t ubStride)
{
    auto gmT = te::MakeTensor(
        te::MakeMemPtr<te::Location::GM>(gmPlane),
        te::MakeFrameLayout<te::NDExtLayoutPtn>(static_cast<uint64_t>(jExtent), static_cast<uint64_t>(stride)));
    auto gmBlock = gmT.Slice(
        te::MakeCoord(static_cast<uint64_t>(jBase), static_cast<uint64_t>(iBase)),
        te::MakeShape(static_cast<uint64_t>(cols), static_cast<uint64_t>(rows)));
    auto ubT = te::MakeTensor(
        te::MakeMemPtr<te::Location::UB, float>(ubPlane.GetPhyAddr()),
        te::MakeFrameLayout<te::NDExtLayoutPtn>(static_cast<uint64_t>(cols), static_cast<uint64_t>(ubStride)));
    te::Copy(te::MakeCopy(te::CopyUB2GM{}), gmBlock, ubT);
}

// Store the three 3M planes (br, bi, bsum) of one tile. The enclosing kernels
// issue their own MTE3->V barrier after the group; this only covers the copies.
template <typename UbTensor>
__aicore__ inline void StoreThreePlanes(
    __gm__ float* gmRe, __gm__ float* gmIm, __gm__ float* gmSum, UbTensor ubRe, UbTensor ubIm, UbTensor ubSum,
    uint32_t jBase, uint32_t iBase, uint32_t cols, uint32_t rows, uint32_t jExtent, uint32_t stride, uint32_t ubStride)
{
    StorePlaneBlock(gmRe, ubRe, jBase, iBase, cols, rows, jExtent, stride, ubStride);
    StorePlaneBlock(gmIm, ubIm, jBase, iBase, cols, rows, jExtent, stride, ubStride);
    StorePlaneBlock(gmSum, ubSum, jBase, iBase, cols, rows, jExtent, stride, ubStride);
}

// Gauss 3M combine for one tile: pi = P3-P1-P2, pr = P1-P2, then apply the
// complex alpha and interleave into packed complex pairs. Extracted verbatim;
// the ORDER MATTERS note explains why pi must be computed first.
// All buffers arrive as already-located tensors: the caller performs the two
// GetWithOffset splits (non-const member), so this helper holds no TBuf and
// every parameter is a plain const reference.
__aicore__ inline void CtrmmCombine3MTile(
    const LocalTensor<float>& p1Ub, const LocalTensor<float>& p2Ub, const LocalTensor<float>& p3Ub,
    const LocalTensor<float>& reUb, const LocalTensor<float>& imUb, const LocalTensor<float>& packLo,
    const LocalTensor<float>& packHi, uint32_t cnt, float alphaRe, float alphaIm)
{
    // ORDER MATTERS: pi is computed FIRST while P1/P2 are intact.
    // (The first version computed pr = P1-P2 in place on p1Ub and then used p1Ub
    // as P1 for pi, yielding pi = P3 - P1 — real channel correct, imaginary
    // wrong.)
    auto piT = MakeUbTensor1D(p3Ub, 0U, cnt);
    auto p1T1 = MakeUbTensor1D(p1Ub, 0U, cnt);
    auto p2T2 = MakeUbTensor1D(p2Ub, 0U, cnt);
    te::Transform<te::Inst::Axpy>(piT, p1T1, -1.0f); // pi = P3 - P1
    te::Transform<te::Inst::Axpy>(piT, p2T2, -1.0f); // pi = P3 - P1 - P2
    auto prT = MakeUbTensor1D(p1Ub, 0U, cnt);
    auto p2T1 = MakeUbTensor1D(p2Ub, 0U, cnt);
    te::Transform<te::Inst::Axpy>(prT, p2T1, -1.0f); // pr = P1 - P2
    PipeBarrier<PIPE_ALL>();

    // re = alphaRe*pr - alphaIm*pi ; im = alphaRe*pi + alphaIm*pr
    // outUb holds [re(cnt) | im(cnt)] then [interleaved complex 2*cnt].
    auto reT = MakeUbTensor1D(reUb, 0U, cnt);
    auto imT = MakeUbTensor1D(imUb, 0U, cnt);
    auto prT2 = MakeUbTensor1D(p1Ub, 0U, cnt);
    auto piT2 = MakeUbTensor1D(p3Ub, 0U, cnt);
    te::Transform<te::Inst::MulScalar>(reT, prT2, alphaRe);
    te::Transform<te::Inst::Axpy>(reT, piT2, -alphaIm);
    te::Transform<te::Inst::MulScalar>(imT, piT2, alphaRe);
    te::Transform<te::Inst::Axpy>(imT, prT2, alphaIm);
    PipeBarrier<PIPE_ALL>();

    // Interleave re/im into complex pairs. The destination must not alias the
    // sources; the caller supplies the API's low/high destination pair.
    Interleave(packLo, packHi, reUb, imUb, static_cast<int32_t>(cnt));
    PipeBarrier<PIPE_ALL>();
    SetFlag<HardEvent::V_MTE3>(0);
    WaitFlag<HardEvent::V_MTE3>(0);
}

// Store one packed complex tile (UB -> GM), C viewed as (n x ldc*2) floats.
__aicore__ inline void StoreComplexTile(
    __gm__ uint8_t* gmC, const LocalTensor<float>& packUb, uint32_t jBase, uint32_t iBase, uint32_t cols, uint32_t rows,
    uint32_t n, uint32_t ldc, uint32_t ubStride)
{
    auto cGmT = te::MakeTensor(
        te::MakeMemPtr<te::Location::GM>(reinterpret_cast<__gm__ float*>(gmC)),
        te::MakeFrameLayout<te::NDExtLayoutPtn>(static_cast<uint64_t>(n), static_cast<uint64_t>(ldc * 2)));
    auto cBlock = cGmT.Slice(
        te::MakeCoord(static_cast<uint64_t>(jBase), static_cast<uint64_t>(iBase) * 2),
        te::MakeShape(static_cast<uint64_t>(cols), static_cast<uint64_t>(rows) * 2));
    auto ubOut = te::MakeTensor(
        te::MakeMemPtr<te::Location::UB, float>(packUb.GetPhyAddr()),
        te::MakeFrameLayout<te::NDExtLayoutPtn>(static_cast<uint64_t>(cols), static_cast<uint64_t>(ubStride * 2)));
    te::Copy(te::MakeCopy(te::CopyUB2GM{}), cBlock, ubOut);
    SetFlag<HardEvent::MTE3_V>(0);
    WaitFlag<HardEvent::MTE3_V>(0);
}

// Load the P1..P3 tiles of one combine block into UB and join the MTE2->V
// barrier once. Extracted verbatim from the combine kernel body.
__aicore__ inline void LoadThreePlaneBlocks(
    __gm__ float* gmP1, __gm__ float* gmP2, __gm__ float* gmP3, const LocalTensor<float>& p1Ub,
    const LocalTensor<float>& p2Ub, const LocalTensor<float>& p3Ub, uint32_t jBase, uint32_t iBase, uint32_t cols,
    uint32_t rows, uint32_t n, uint32_t stride, uint32_t ubStride)
{
    auto ub1 = te::MakeTensor(
        te::MakeMemPtr<te::Location::UB, float>(p1Ub.GetPhyAddr()),
        te::MakeFrameLayout<te::NDExtLayoutPtn>(static_cast<uint64_t>(cols), static_cast<uint64_t>(ubStride)));
    auto ub2 = te::MakeTensor(
        te::MakeMemPtr<te::Location::UB, float>(p2Ub.GetPhyAddr()),
        te::MakeFrameLayout<te::NDExtLayoutPtn>(static_cast<uint64_t>(cols), static_cast<uint64_t>(ubStride)));
    auto ub3 = te::MakeTensor(
        te::MakeMemPtr<te::Location::UB, float>(p3Ub.GetPhyAddr()),
        te::MakeFrameLayout<te::NDExtLayoutPtn>(static_cast<uint64_t>(cols), static_cast<uint64_t>(ubStride)));
    LoadPlaneBlock(gmP1, ub1, jBase, iBase, cols, rows, n, stride, ubStride);
    LoadPlaneBlock(gmP2, ub2, jBase, iBase, cols, rows, n, stride, ubStride);
    LoadPlaneBlock(gmP3, ub3, jBase, iBase, cols, rows, n, stride, ubStride);
    SetFlag<HardEvent::MTE2_V>(0);
    WaitFlag<HardEvent::MTE2_V>(0);
}

// Zero one column's complement range [zeroStart, zeroStart+zeroCount) inside the
// padded UB tile: scalar head/tail plus the 32B-aligned middle span. Split out of
// CtrmmMaskTriangleTile so each stage stays small; the write order and the
// three barrier-free passes are unchanged (the caller barriers once after all
// columns).
__aicore__ inline void CtrmmZeroColumnRange(
    const LocalTensor<float>& reUb, const LocalTensor<float>& imUb, uint32_t c, uint32_t ubStride, uint32_t zeroStart,
    uint32_t zeroCount, uint32_t* midZeros, uint32_t* midCounts, uint32_t& midCount)
{
    // Scalar (PIPE_S) parts of the range: head and tail <= 7
    // elements each, so that every Duplicate below starts on a
    // 32B-aligned UB offset. All scalar writes for every column
    // are issued in this pass and separated from the vector
    // passes by barriers (mixing S and V writes on the same
    // buffer inside one loop loses the scalar writes).
    const uint32_t base = c * ubStride;
    const uint32_t zs = base + zeroStart;
    const uint32_t ze = zs + zeroCount;
    const uint32_t alignedS = ((zs + TLA_ALIGN - 1) / TLA_ALIGN) * TLA_ALIGN;
    const uint32_t alignedE = (ze / TLA_ALIGN) * TLA_ALIGN;
    const uint32_t headEnd = Min<uint32_t>(alignedS, ze);
    for (uint32_t o = zs; o < headEnd; ++o) {
        reUb.SetValue(static_cast<int32_t>(o), 0.0f);
        imUb.SetValue(static_cast<int32_t>(o), 0.0f);
    }
    if (alignedE > alignedS) {
        const uint32_t midStart = alignedS;
        const uint32_t midEnd = alignedE;
        midZeros[midCount] = midStart;
        midCounts[midCount] = midEnd - midStart;
        midCount++;
    }
    const uint32_t tailStart = (alignedE > alignedS) ? alignedE : alignedS;
    for (uint32_t o = tailStart; o < ze; ++o) {
        reUb.SetValue(static_cast<int32_t>(o), 0.0f);
        imUb.SetValue(static_cast<int32_t>(o), 0.0f);
    }
}

// Phase-A triangle masking for one tile: zero the complement of the stored
// triangle in the re/im planes (scalar path, see the notes inside), then write
// the UNIT diagonal as 1+0i. Barriers are part of the contract.
__aicore__ inline void CtrmmMaskTriangleTile(
    const LocalTensor<float>& reUb, const LocalTensor<float>& imUb, uint32_t jBase, uint32_t iBase, uint32_t cols,
    uint32_t rows, uint32_t ubStride, bool planeLower, bool diagUnit)
{
    // Per-column triangle masking (row interval): vector Duplicates
    // first, then the UNIT diagonal scalars — the two phases are split
    // so V->S->V ordering is explicit (mixed scalar/vector writes in one
    // loop is what the earlier EXECUTION_FAILED pointed at).
    uint32_t midZeros[64];
    uint32_t midCounts[64];
    uint32_t midCount = 0;
    for (uint32_t c = 0; c < cols; ++c) {
        const uint32_t absJ = jBase + c;
        uint32_t zeroStart;
        uint32_t zeroCount;
        if (planeLower) {
            zeroStart = 0;
            zeroCount = (absJ > iBase) ? Min<uint32_t>(absJ - iBase, rows) : 0;
        } else {
            const uint32_t keepEnd = (absJ >= iBase) ? Min<uint32_t>(absJ - iBase + 1, rows) : 0;
            zeroStart = keepEnd;
            zeroCount = rows - keepEnd;
        }
        if (zeroCount > 0) {
            CtrmmZeroColumnRange(reUb, imUb, c, ubStride, zeroStart, zeroCount, midZeros, midCounts, midCount);
        }
    }
    PipeBarrier<PIPE_ALL>();
    // Triangle masking: scalar zeroing of each column's complement
    // range. This is the only mask implementation kept: the vector
    // Duplicate variants wrote past an offset LocalTensor's range
    // (measured 8248 / 8240 outliers on TC_SQ_044), and a vectorized
    // te::Transform variant was measured to bring no speedup (44-47us vs
    // 44-45us Phase-A at 512^2, within noise), so it was removed rather
    // than kept as an unused runtime path. The experiment record lives in
    // the repository history and the implementation notes.
    for (uint32_t k = 0; k < midCount; ++k) {
        for (uint32_t o = 0; o < midCounts[k]; ++o) {
            reUb.SetValue(static_cast<int32_t>(midZeros[k] + o), 0.0f);
            imUb.SetValue(static_cast<int32_t>(midZeros[k] + o), 0.0f);
        }
    }
    PipeBarrier<PIPE_ALL>();
    if (diagUnit) {
        for (uint32_t c = 0; c < cols; ++c) {
            const uint32_t absJ = jBase + c;
            if (absJ >= iBase && absJ < iBase + rows) {
                const uint32_t diagLocal = absJ - iBase;
                reUb.SetValue(static_cast<int32_t>(c * ubStride + diagLocal), 1.0f);
                imUb.SetValue(static_cast<int32_t>(c * ubStride + diagLocal), 0.0f);
            }
        }
        PipeBarrier<PIPE_ALL>();
    }
}

// Phase-A fused kernel, A-side mirror (trans = N): per-column triangle masking
// with a row interval (UPPER keeps rows [0..absJ], LOWER keeps [absJ..rows)),
// UNIT diagonal written as 1+0i. Extracted verbatim from the fused kernel body;
// the barrier sequence and the scalar-mask ordering are unchanged.
__aicore__ inline void CtrmmSplitFusedMirrorA(
    __gm__ float* aGm, __gm__ float* arGm, __gm__ float* aiGm, __gm__ float* asumGm, const LocalTensor<float>& inUb,
    const LocalTensor<float>& reUb, const LocalTensor<float>& imUb, const LocalTensor<float>& sumUb, uint32_t dimA,
    uint32_t lda, uint32_t aStart, uint32_t aEnd, bool planeLower, bool diagUnit, bool transIsN)
{
    if (!transIsN) {
        return;
    }
    {
        for (uint32_t jBase = 0; jBase < dimA; jBase += TLA_BLOCK) {
            const uint32_t cols = Min<uint32_t>(TLA_BLOCK, dimA - jBase);
            for (uint32_t iBase = aStart; iBase < aEnd; iBase += TLA_BLOCK) {
                const uint32_t rows = Min<uint32_t>(TLA_BLOCK, aEnd - iBase);
                const uint32_t ubStride = RoundUp<uint32_t>(rows, TLA_ALIGN);
                const uint32_t cnt = ubStride * cols;

                LoadComplexTile(aGm, inUb, jBase, iBase, cols, rows, dimA, lda, ubStride);
                DeInterleave(reUb, imUb, inUb, static_cast<int32_t>(2 * cnt));
                PipeBarrier<PIPE_ALL>();

                CtrmmMaskTriangleTile(reUb, imUb, jBase, iBase, cols, rows, ubStride, planeLower, diagUnit);
                SumPlanes(sumUb, reUb, imUb, cnt);
                StoreThreePlanes(arGm, aiGm, asumGm, reUb, imUb, sumUb, jBase, iBase, cols, rows, dimA, lda, ubStride);
                SetFlag<HardEvent::MTE3_V>(0);
                WaitFlag<HardEvent::MTE3_V>(0);
            }
        }
    }
}

// Phase-A fused kernel, B-side split (rows on the i axis; thin shapes use full
// height column bands so tiles stay TLA_BLOCK-sized). Extracted verbatim.
__aicore__ inline void CtrmmSplitFusedSplitB(
    __gm__ float* bGm, __gm__ float* brGm, __gm__ float* biGm, __gm__ float* bsumGm, const LocalTensor<float>& inUb,
    const LocalTensor<float>& reUb, const LocalTensor<float>& imUb, const LocalTensor<float>& sumUb, uint32_t m,
    uint32_t n, uint32_t ldb, uint32_t rowsPerCoreB, uint32_t colsPerCoreB, uint32_t blkUnique)
{
    // ---- B-side split (rows on the i axis; thin shapes: column bands) ----
    const uint32_t bRowsPerCore = rowsPerCoreB;
    uint32_t jLo = 0;
    uint32_t jHi = n;
    uint32_t iLo = 0;
    uint32_t iHi = 0;
    if (colsPerCoreB > 0) {
        // Column-band mode (host sets this only for thin trans=N shapes):
        // each block owns a full-height column band -> proper TLA_BLOCK tiles
        // instead of fragmented <TLA_BLOCK-row slivers.
        iLo = 0;
        iHi = m;
        jLo = blkUnique * colsPerCoreB;
        jHi = Min<uint32_t>(jLo + colsPerCoreB, n);
    } else {
        iLo = blkUnique * bRowsPerCore;
        iHi = Min<uint32_t>(iLo + bRowsPerCore, m);
    }
    for (uint32_t jBase = jLo; jBase < jHi; jBase += TLA_BLOCK) {
        const uint32_t cols = Min<uint32_t>(TLA_BLOCK, jHi - jBase);
        for (uint32_t iBase = iLo; iBase < iHi; iBase += TLA_BLOCK) {
            const uint32_t rows = Min<uint32_t>(TLA_BLOCK, iHi - iBase);
            const uint32_t ubStride = RoundUp<uint32_t>(rows, TLA_ALIGN);
            const uint32_t cnt = ubStride * cols;
            LoadComplexTile(bGm, inUb, jBase, iBase, cols, rows, n, ldb, ubStride);
            SplitComplexTile(reUb, imUb, sumUb, inUb, cnt);
            StoreThreePlanes(brGm, biGm, bsumGm, reUb, imUb, sumUb, jBase, iBase, cols, rows, n, ldb, ubStride);
            SetFlag<HardEvent::MTE3_V>(0);
            WaitFlag<HardEvent::MTE3_V>(0);
        }
    }
}

} // namespace ctrmm_tla

// gmP4 is reserved by the 4-plane workspace layout the host allocates
// ([P1|P2|P3|P4|...]); the Gauss 3M combine reads only P1..P3, so P4 is
// deliberately unused here and costs no UB.
__global__ __aicore__ void ctrmm_combine_tla_kernel(
    const __gm__ uint8_t* gmP1, const __gm__ uint8_t* gmP2, const __gm__ uint8_t* gmP3, const __gm__ uint8_t* gmP4,
    __gm__ uint8_t* gmC, const CtrmmCombineTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    using namespace ctrmm_tla;

    TBuf<TPosition::VECCALC> bufP1;
    TBuf<TPosition::VECCALC> bufP2;
    TBuf<TPosition::VECCALC> bufP3;
    TBuf<TPosition::VECCALC> bufOut;
    TBuf<TPosition::VECCALC> bufPack;
    TPipe pipe;
    constexpr uint32_t PLANE_FLOATS = TLA_BLOCK * TLA_BLOCK;
    pipe.InitBuffer(bufP1, PLANE_FLOATS * sizeof(float));
    pipe.InitBuffer(bufP2, PLANE_FLOATS * sizeof(float));
    pipe.InitBuffer(bufP3, PLANE_FLOATS * sizeof(float));
    pipe.InitBuffer(bufOut, 2 * PLANE_FLOATS * sizeof(float));  // re | im
    pipe.InitBuffer(bufPack, 2 * PLANE_FLOATS * sizeof(float)); // interleaved complex

    const uint32_t m = tiling.m;
    const uint32_t n = tiling.n;
    const uint32_t ldc = tiling.ldc;
    const uint32_t stride = tiling.tempRowStride;
    const float alphaRe = tiling.alphaReal;
    const float alphaIm = tiling.alphaImag;

    const int32_t blkIdx = AscendC::GetBlockIdx();
    const uint32_t rowStart = static_cast<uint32_t>(blkIdx) * tiling.rowsPerCore;
    const uint32_t rowEnd = Min<uint32_t>(rowStart + tiling.rowsPerCore, m);
    if (rowStart >= rowEnd) {
        return;
    }

    LocalTensor<float> p1Ub = bufP1.Get<float>();
    LocalTensor<float> p2Ub = bufP2.Get<float>();
    LocalTensor<float> p3Ub = bufP3.Get<float>();
    LocalTensor<float> outUb = bufOut.Get<float>();
    LocalTensor<float> packUb = bufPack.Get<float>();

    for (uint32_t jBase = 0; jBase < n; jBase += TLA_BLOCK) {
        const uint32_t cols = Min<uint32_t>(TLA_BLOCK, n - jBase);
        for (uint32_t iBase = rowStart; iBase < rowEnd; iBase += TLA_BLOCK) {
            const uint32_t rows = Min<uint32_t>(TLA_BLOCK, rowEnd - iBase);
            const uint32_t ubStride = RoundUp<uint32_t>(rows, TLA_ALIGN);
            const uint32_t cnt = ubStride * cols;

            LoadThreePlaneBlocks(
                reinterpret_cast<__gm__ float*>(const_cast<__gm__ uint8_t*>(gmP1)),
                reinterpret_cast<__gm__ float*>(const_cast<__gm__ uint8_t*>(gmP2)),
                reinterpret_cast<__gm__ float*>(const_cast<__gm__ uint8_t*>(gmP3)), p1Ub, p2Ub, p3Ub, jBase, iBase,
                cols, rows, n, stride, ubStride);

            // NOTE: LocalTensor::operator[] is not a plain offset view — use
            // GetWithOffset (cherk precedent) for the second halves, otherwise
            // re/im (and packLo/packHi) alias and only the real channel is correct.
            LocalTensor<float> imUb = bufOut.GetWithOffset<float>(PLANE_FLOATS, PLANE_FLOATS * sizeof(float));
            LocalTensor<float> packHi = bufPack.GetWithOffset<float>(cnt, cnt * sizeof(float));
            CtrmmCombine3MTile(p1Ub, p2Ub, p3Ub, outUb, imUb, packUb, packHi, cnt, alphaRe, alphaIm);

            StoreComplexTile(gmC, packUb, jBase, iBase, cols, rows, n, ldc, ubStride);
        }
    }
}

void ctrmm_combine_tla_kernel_do(
    const uint8_t* gmP1, const uint8_t* gmP2, const uint8_t* gmP3, const uint8_t* gmP4, uint8_t* gmC,
    const CtrmmCombineTilingData& tiling, uint32_t numBlocks, void* stream)
{
    ctrmm_combine_tla_kernel<<<numBlocks, nullptr, stream>>>(gmP1, gmP2, gmP3, gmP4, gmC, tiling);
}

// ================================================================================
// Phase-A (B-side) tensor_api variant: complex B -> Br/Bi planes + Bsum
// pre-combined plane, via UB block DMA + DeInterleave/Add (cherk deinterleave
// lineage). The A-side mirror keeps its SIMT form (triangle mask + trans +
// conj + UNIT diag); this kernel covers the B-side traffic, which dominates
// the phase for thin/wide shapes and matches the A-side volume for squares.
// ================================================================================

__global__ __aicore__ void ctrmm_split_b_tla_kernel(
    const __gm__ uint8_t* gmB, __gm__ uint8_t* gmBr, __gm__ uint8_t* gmBi, __gm__ uint8_t* gmBsum,
    const CtrmmSplitMirrorTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    using namespace ctrmm_tla;

    const uint32_t m = tiling.m;
    const uint32_t n = tiling.n;
    const uint32_t ldb = tiling.ldb;
    const uint32_t bRowsPerCore = tiling.rowsPerCoreB;
    __gm__ float* bGm = reinterpret_cast<__gm__ float*>(const_cast<__gm__ uint8_t*>(gmB));
    __gm__ float* brGm = reinterpret_cast<__gm__ float*>(gmBr);
    __gm__ float* biGm = reinterpret_cast<__gm__ float*>(gmBi);
    __gm__ float* bsumGm = reinterpret_cast<__gm__ float*>(gmBsum);

    TBuf<TPosition::VECCALC> bufIn;
    TBuf<TPosition::VECCALC> bufBr;
    TBuf<TPosition::VECCALC> bufBi;
    TBuf<TPosition::VECCALC> bufBs;
    TPipe pipe;
    constexpr uint32_t PLANE_FLOATS = TLA_BLOCK * TLA_BLOCK;
    pipe.InitBuffer(bufIn, 2 * PLANE_FLOATS * sizeof(float)); // complex tile
    pipe.InitBuffer(bufBr, PLANE_FLOATS * sizeof(float));
    pipe.InitBuffer(bufBi, PLANE_FLOATS * sizeof(float));
    pipe.InitBuffer(bufBs, PLANE_FLOATS * sizeof(float));

    LocalTensor<float> inUb = bufIn.Get<float>();
    LocalTensor<float> brUb = bufBr.Get<float>();
    LocalTensor<float> biUb = bufBi.Get<float>();
    LocalTensor<float> bsUb = bufBs.Get<float>();

    const int32_t blkIdx = AscendC::GetBlockIdx();
    const uint32_t rowStart = static_cast<uint32_t>(blkIdx) * bRowsPerCore;
    const uint32_t rowEnd = Min<uint32_t>(rowStart + bRowsPerCore, m);
    if (rowStart >= rowEnd) {
        return;
    }

    for (uint32_t jBase = 0; jBase < n; jBase += TLA_BLOCK) {
        const uint32_t cols = Min<uint32_t>(TLA_BLOCK, n - jBase);
        for (uint32_t iBase = rowStart; iBase < rowEnd; iBase += TLA_BLOCK) {
            const uint32_t rows = Min<uint32_t>(TLA_BLOCK, rowEnd - iBase);
            // Complex-tile UB row stride: RoundUp(rows, 8) float pairs, i.e. a
            // contiguous padded stream (cherk's ubCplxStride) — DeInterleave
            // consumes the whole padded buffer as one flat interleaved stream.
            const uint32_t ubStride = RoundUp<uint32_t>(rows, TLA_ALIGN);
            const uint32_t cnt = ubStride * cols;

            // GM complex tile -> UB: view B as (n x ldb*2) floats, slice
            // (cols x rows*2) at (jBase, iBase*2).
            LoadComplexTile(bGm, inUb, jBase, iBase, cols, rows, n, ldb, ubStride);

            // Deinterleave [re,im,re,im,...] -> Br, Bi planes, then form
            // bs = br + bi in UB.
            SplitComplexTile(brUb, biUb, bsUb, inUb, cnt);

            // Store three planes (UB -> GM), layout (n x ldb).
            StoreThreePlanes(brGm, biGm, bsumGm, brUb, biUb, bsUb, jBase, iBase, cols, rows, n, ldb, ubStride);
            SetFlag<HardEvent::MTE3_V>(0);
            WaitFlag<HardEvent::MTE3_V>(0);
        }
    }
}

void ctrmm_split_b_tla_kernel_do(
    const uint8_t* gmB, uint8_t* gmBr, uint8_t* gmBi, uint8_t* gmBsum, const CtrmmSplitMirrorTilingData& tiling,
    uint32_t numBlocks, void* stream)
{
    ctrmm_split_b_tla_kernel<<<numBlocks, nullptr, stream>>>(gmB, gmBr, gmBi, gmBsum, tiling);
}

// ================================================================================
// Phase-A fused tensor_api kernel: A-side mirror (trans = N) + B-side split,
// single launch. A-mirror masking uses per-column interval zeroing (the
// triangle is a row interval per column: UPPER keeps rows [0..j], LOWER keeps
// [j..rows)), UNIT diagonal written as 1+0i. trans = T/C still uses the SIMT
// A-mirror (transposed GM read + conj), only the B side runs here.
// ================================================================================

__global__ __aicore__ void ctrmm_split_fused_tla_kernel(
    const __gm__ uint8_t* gmA, const __gm__ uint8_t* gmB, __gm__ uint8_t* gmAr, __gm__ uint8_t* gmAi,
    __gm__ uint8_t* gmAsum, __gm__ uint8_t* gmBr, __gm__ uint8_t* gmBi, __gm__ uint8_t* gmBsum,
    const CtrmmSplitMirrorTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    using namespace ctrmm_tla;

    const uint32_t dimA = tiling.dimA;
    const uint32_t m = tiling.m;
    const uint32_t n = tiling.n;
    const uint32_t lda = tiling.lda;
    const uint32_t ldb = tiling.ldb;
    const bool planeLower = (tiling.uploMode == ACLBLAS_LOWER);
    const bool diagUnit = (tiling.diagMode == ACLBLAS_UNIT);
    __gm__ float* aGm = reinterpret_cast<__gm__ float*>(const_cast<__gm__ uint8_t*>(gmA));
    __gm__ float* bGm = reinterpret_cast<__gm__ float*>(const_cast<__gm__ uint8_t*>(gmB));
    __gm__ float* arGm = reinterpret_cast<__gm__ float*>(gmAr);
    __gm__ float* aiGm = reinterpret_cast<__gm__ float*>(gmAi);
    __gm__ float* asumGm = reinterpret_cast<__gm__ float*>(gmAsum);
    __gm__ float* brGm = reinterpret_cast<__gm__ float*>(gmBr);
    __gm__ float* biGm = reinterpret_cast<__gm__ float*>(gmBi);
    __gm__ float* bsumGm = reinterpret_cast<__gm__ float*>(gmBsum);

    TBuf<TPosition::VECCALC> bufIn;
    TBuf<TPosition::VECCALC> bufRe;
    TBuf<TPosition::VECCALC> bufIm;
    TBuf<TPosition::VECCALC> bufSum;
    TPipe pipe;
    constexpr uint32_t PLANE_FLOATS = TLA_BLOCK * TLA_BLOCK;
    pipe.InitBuffer(bufIn, 2 * PLANE_FLOATS * sizeof(float));
    pipe.InitBuffer(bufRe, PLANE_FLOATS * sizeof(float));
    pipe.InitBuffer(bufIm, PLANE_FLOATS * sizeof(float));
    pipe.InitBuffer(bufSum, PLANE_FLOATS * sizeof(float));
    LocalTensor<float> inUb = bufIn.Get<float>();
    LocalTensor<float> reUb = bufRe.Get<float>();
    LocalTensor<float> imUb = bufIm.Get<float>();
    LocalTensor<float> sumUb = bufSum.Get<float>();

    const int32_t blkIdx = AscendC::GetBlockIdx();

    // ---- A-side mirror (rows on the i axis), one launch covers A and B ----
    // trans = N only; T/C uses the transposed variant below.
    const uint32_t aRowsPerCore = tiling.rowsPerCoreA;
    const uint32_t aStart = static_cast<uint32_t>(blkIdx) * aRowsPerCore;
    const uint32_t aEnd = Min<uint32_t>(aStart + aRowsPerCore, dimA);
    CtrmmSplitFusedMirrorA(
        aGm, arGm, aiGm, asumGm, inUb, reUb, imUb, sumUb, dimA, lda, aStart, aEnd, planeLower, diagUnit,
        tiling.transMode == ACLBLAS_OP_N);

    CtrmmSplitFusedSplitB(
        bGm, brGm, biGm, bsumGm, inUb, reUb, imUb, sumUb, m, n, ldb, tiling.rowsPerCoreB, tiling.colsPerCoreB,
        static_cast<uint32_t>(blkIdx));
}

void ctrmm_split_fused_tla_kernel_do(
    const uint8_t* gmA, const uint8_t* gmB, uint8_t* gmAr, uint8_t* gmAi, uint8_t* gmAsum, uint8_t* gmBr, uint8_t* gmBi,
    uint8_t* gmBsum, const CtrmmSplitMirrorTilingData& tiling, uint32_t numBlocks, void* stream)
{
    ctrmm_split_fused_tla_kernel<<<numBlocks, nullptr, stream>>>(
        gmA, gmB, gmAr, gmAi, gmAsum, gmBr, gmBi, gmBsum, tiling);
}
