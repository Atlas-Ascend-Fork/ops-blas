/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file cgbmv_kernel.cpp
 * \brief Single-precision complex general banded matrix-vector multiply (SIMT).
 *
 * Complex values are stored as interleaved (real, imag) float pairs. Band storage: A(i, j) lives at
 * band row (ku + i - j) of column j, i.e. float offset 2 * ((ku + i - j) + lda * j).
 */

#include <cstdint>
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "cgbmv_kernel.h"
#include "common/helper/kernel_constant.h"

namespace {

constexpr int64_t FLOATS_PER_COMPLEX = 2; // complex element k: real at 2k, imag at 2k + 1

// Physical element index of logical vector element idx (BLAS semantics for negative increments).
__simt_callee__ __aicore__ inline int64_t VecIndex(uint32_t idx, uint32_t len, int64_t inc)
{
    return (inc >= 0) ? static_cast<int64_t>(idx) * inc : static_cast<int64_t>(len - 1U - idx) * (-inc);
}

struct CgbmvAcc {
    float r;
    float i;
};

__simt_callee__ __aicore__ inline uint32_t FloatBits(float value)
{
    union {
        float value;
        uint32_t bits;
    } repr{};
    repr.value = value;
    return repr.bits;
}

__simt_callee__ __aicore__ inline bool IsFiniteFloat(float value)
{
    return (FloatBits(value) & 0x7fffffffU) < 0x7f800000U;
}

__simt_callee__ __aicore__ inline bool IsInfiniteFloat(float value)
{
    return (FloatBits(value) & 0x7fffffffU) == 0x7f800000U;
}

__simt_callee__ __aicore__ inline bool IsNanFloat(float value)
{
    return (FloatBits(value) & 0x7fffffffU) > 0x7f800000U;
}

__simt_callee__ __aicore__ inline float SignedZeroOrOne(float source, bool one)
{
    // arch35 SIMT has no fcopysign lowering. Form the IEEE-754 zero/unit bits directly, including -0.
    union {
        float value;
        uint32_t bits;
    } output{};
    output.bits = (FloatBits(source) & 0x80000000U) | (one ? 0x3f800000U : 0U);
    return output.value;
}

// Scalar complex multiplication in the exceptional path follows the C complex infinity-recovery rule
// used by Netlib's compiler runtime. In particular, (1+0i)*(Inf+Inf*i) must not become (NaN+NaN*i).
__simt_callee__ __aicore__ inline CgbmvAcc ReferenceComplexMultiply(CgbmvAcc a, CgbmvAcc b)
{
    // Separate products prevent contraction from changing Inf/NaN propagation in this fallback.
    volatile float rr = a.r * b.r;
    volatile float ii = a.i * b.i;
    volatile float ri = a.r * b.i;
    volatile float ir = a.i * b.r;
    CgbmvAcc out{rr - ii, ri + ir};
    if (!IsNanFloat(out.r) || !IsNanFloat(out.i)) {
        return out;
    }
    bool aInf = IsInfiniteFloat(a.r) || IsInfiniteFloat(a.i);
    bool bInf = IsInfiniteFloat(b.r) || IsInfiniteFloat(b.i);
    bool productInf = IsInfiniteFloat(rr) || IsInfiniteFloat(ii) || IsInfiniteFloat(ri) || IsInfiniteFloat(ir);
    if (!aInf && !bInf && !productInf) {
        return out;
    }
    if (aInf) {
        a = {SignedZeroOrOne(a.r, IsInfiniteFloat(a.r)), SignedZeroOrOne(a.i, IsInfiniteFloat(a.i))};
    } else {
        a = {IsNanFloat(a.r) ? SignedZeroOrOne(a.r, false) : a.r, IsNanFloat(a.i) ? SignedZeroOrOne(a.i, false) : a.i};
    }
    if (bInf) {
        b = {SignedZeroOrOne(b.r, IsInfiniteFloat(b.r)), SignedZeroOrOne(b.i, IsInfiniteFloat(b.i))};
    } else {
        b = {IsNanFloat(b.r) ? SignedZeroOrOne(b.r, false) : b.r, IsNanFloat(b.i) ? SignedZeroOrOne(b.i, false) : b.i};
    }
    constexpr float infinity = __builtin_inff();
    out = {infinity * (a.r * b.r - a.i * b.i), infinity * (a.r * b.i + a.i * b.r)};
    return out;
}

// beta * y as the reference forms it before any product is added: beta == 0 gives 0 without reading y.
__simt_callee__ __aicore__ inline CgbmvAcc ReferenceBetaY(
    __gm__ const float* yp, float betaR, float betaI, uint32_t betaMode)
{
    if (betaMode == CGBMV_BETA_ZERO) {
        return {0.0f, 0.0f};
    }
    CgbmvAcc y{yp[0], yp[1]};
    return (betaMode == CGBMV_BETA_ONE) ? y : ReferenceComplexMultiply({betaR, betaI}, y);
}

// y[k] = out + beta * y[k]; beta == 0 never reads y, beta == 1 adds y without a multiplication. If the plain
// beta * y leaves the result non-finite (y holds Inf/NaN), beta * y is formed again with ReferenceComplexMultiply,
// so that e.g. (2+0i)*(Inf+Inf*i) stays infinite as in the reference instead of becoming NaN.
__simt_callee__ __aicore__ inline void StoreY(
    __gm__ float* yGm, int64_t k, float outR, float outI, float betaR, float betaI, uint32_t betaMode)
{
    __gm__ float* yp = yGm + FLOATS_PER_COMPLEX * k;
    if (betaMode == CGBMV_BETA_ZERO) {
        yp[0] = outR;
        yp[1] = outI;
        return;
    }
    float yr = yp[0];
    float yi = yp[1];
    if (betaMode == CGBMV_BETA_ONE) {
        yp[0] = yr + outR;
        yp[1] = yi + outI;
        return;
    }
    float resR = (betaR * yr - betaI * yi) + outR;
    float resI = (betaR * yi + betaI * yr) + outI;
    if (!IsFiniteFloat(resR) || !IsFiniteFloat(resI)) {
        CgbmvAcc by = ReferenceComplexMultiply({betaR, betaI}, {yr, yi});
        resR = by.r + outR;
        resI = by.i + outI;
    }
    yp[0] = resR;
    yp[1] = resI;
}

// Exceptional N rows are recomputed in reference order, starting with beta*y. A parallel partial sum can
// overflow even when cancellation with beta*y keeps every sequential update and the final result finite.
// This path runs only after a non-finite partial-sum reduction; normal rows retain their parallel arithmetic.
__simt_callee__ __aicore__ inline void StoreNResult(
    uint32_t row, uint32_t m, uint32_t n, uint32_t kl, uint32_t ku, uint32_t lda, int64_t incx, int64_t incy,
    float alphaR, float alphaI, float betaR, float betaI, uint32_t betaMode, float sumR, float sumI,
    __gm__ const float* aGm, __gm__ const float* xGm, __gm__ float* yGm)
{
    int64_t yk = VecIndex(row, m, incy);
    if (IsFiniteFloat(sumR) && IsFiniteFloat(sumI)) {
        StoreY(yGm, yk, sumR, sumI, betaR, betaI, betaMode);
        return;
    }
    CgbmvAcc acc = ReferenceBetaY(yGm + FLOATS_PER_COMPLEX * yk, betaR, betaI, betaMode);
    uint32_t begin = row > kl ? row - kl : 0U;
    // 64-bit intermediate: in uint32 this sum only stays in range because of the kl <= m - 1 / ku <= n - 1
    // bounds, and a wrapped value would pass the clamp below as a too-small end.
    uint64_t endWide = static_cast<uint64_t>(row) + ku + 1U;
    uint32_t end = (endWide < n) ? static_cast<uint32_t>(endWide) : n;
    for (uint32_t col = begin; col < end; ++col) {
        int64_t xk = FLOATS_PER_COMPLEX * VecIndex(col, n, incx);
        float xr = xGm[xk];
        float xi = xGm[xk + 1];
        CgbmvAcc scaled = ReferenceComplexMultiply({alphaR, alphaI}, {xr, xi});
        int64_t ak = FLOATS_PER_COMPLEX * (static_cast<int64_t>(ku) + row - col + static_cast<int64_t>(lda) * col);
        float ar = aGm[ak];
        float ai = aGm[ak + 1];
        CgbmvAcc term = ReferenceComplexMultiply(scaled, {ar, ai});
        acc.r += term.r;
        acc.i += term.i;
    }
    yGm[FLOATS_PER_COMPLEX * yk] = acc.r;
    yGm[FLOATS_PER_COMPLEX * yk + 1] = acc.i;
}

// trans N: columns [begin, end) that one slice of a 32-row tile walks. The tile's window is
// [tileBase - kl, tileBase + 32 + ku) clipped to [0, n); an empty window gives every slice an empty range.
struct CgbmvColRange {
    uint32_t begin;
    uint32_t end;
};

__simt_callee__ __aicore__ inline CgbmvColRange SliceColumns(
    uint32_t tileBase, uint32_t slice, uint32_t slices, uint32_t n, uint32_t kl, uint32_t ku)
{
    uint32_t winBegin = (tileBase > kl) ? (tileBase - kl) : 0U;
    uint64_t winEndWide = static_cast<uint64_t>(tileBase) + CGBMV_WARP_SIZE + ku;
    uint32_t winEnd = (winEndWide < n) ? static_cast<uint32_t>(winEndWide) : n;
    // slices is the host-side tiling.slices, always in [SIMT_MIN_THREAD_NUM / CGBMV_WARP_SIZE,
    // CGBMV_N_MAX_THREADS / CGBMV_WARP_SIZE]; the zero test keeps the division below safe on its own.
    if (winBegin >= winEnd || slices == 0U) {
        return {0U, 0U};
    }
    uint32_t chunk = (winEnd - winBegin + slices - 1U) / slices;
    uint32_t begin = winBegin + slice * chunk;
    uint32_t end = (begin + chunk < winEnd) ? (begin + chunk) : winEnd;
    return {begin, end};
}

// trans N, long slices: sum of A(row, col) * (alpha * x(col)) over cols, skipping columns outside row's band; x is
// scaled by alpha first as in the reference, so A * x alone can not overflow where the reference stays finite. Every
// lane of a tile walks the same column sequence, so one step reads consecutive band rows of one column (coalesced).
__simt_callee__ __aicore__ inline CgbmvAcc BandRowDot(
    uint32_t row, CgbmvColRange cols, uint32_t kl, uint32_t ku, uint32_t lda, uint32_t n, int64_t incx, float alphaR,
    float alphaI, __gm__ const float* aGm, __gm__ const float* xGm)
{
    CgbmvAcc acc{0.0f, 0.0f};
    if (cols.begin >= cols.end) {
        return acc;
    }
    // d = ku + row - col is the band row of A(row, col); it lies in the band iff 0 <= d <= kl + ku. A negative d
    // wraps to a large unsigned value, so a single comparison covers both sides. Moving to the next column
    // decrements d, advances A by lda - 1 elements and x by incx (for either sign of incx).
    uint32_t d = ku + row - cols.begin;
    const uint32_t bandLast = kl + ku;
    int64_t aOff =
        FLOATS_PER_COMPLEX * (static_cast<int64_t>(ku) + row - cols.begin + static_cast<int64_t>(lda) * cols.begin);
    int64_t xOff = FLOATS_PER_COMPLEX * VecIndex(cols.begin, n, incx);
    const int64_t aStep = FLOATS_PER_COMPLEX * (static_cast<int64_t>(lda) - 1);
    const int64_t xStep = FLOATS_PER_COMPLEX * incx;
    for (uint32_t col = cols.begin; col < cols.end; ++col) {
        if (d <= bandLast) {
            float ar = aGm[aOff];
            float ai = aGm[aOff + 1];
            float xr = xGm[xOff];
            float xi = xGm[xOff + 1];
            float tr = alphaR * xr - alphaI * xi;
            float ti = alphaR * xi + alphaI * xr;
            acc.r += ar * tr - ai * ti;
            acc.i += ar * ti + ai * tr;
        }
        --d;
        aOff += aStep;
        xOff += xStep;
    }
    return acc;
}

// y[row] = beta * y[row] for logical rows start, start + stride, ... below len; A and x are not read and
// beta == 1 leaves y untouched.
__simt_callee__ __aicore__ inline void ScaleRows(
    uint32_t start, uint32_t stride, uint32_t len, int64_t incy, float betaR, float betaI, uint32_t betaMode,
    __gm__ float* yGm)
{
    if (betaMode == CGBMV_BETA_ONE) {
        return;
    }
    for (uint32_t row = start; row < len; row += stride) {
        StoreY(yGm, VecIndex(row, len, incy), 0.0f, 0.0f, betaR, betaI, betaMode);
    }
}

// ==========================================================================
//  alpha == 0: y = beta * y (A and x are not read)
// ==========================================================================
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void CgbmvScaleSimt(
    uint32_t blk, uint32_t nblk, uint32_t len, int64_t incy, float betaR, float betaI, uint32_t betaMode,
    __gm__ float* yGm)
{
    ScaleRows(blk * blockDim.x + threadIdx.x, nblk * blockDim.x, len, incy, betaR, betaI, betaMode, yGm);
}

// ==========================================================================
//  trans = N: rows [0, min(m, n + kl)) are cut into 32-row tiles that the blocks take in turn. All `slices`
//  warps of a block walk the tile's column window, each taking one contiguous part of it; lane l of every warp
//  handles row tileBase + l. Terms are alpha * x scaled first (reference order); partial sums meet in UB and are
//  added in slice order.
//
//  STEPPED picks the inner loop: BandRowDot's stepped-offset walk for windows of at least
//  CGBMV_N_STEPPED_MIN_COLS columns, or a direct per-column address computation for shorter ones. Both stay
//  inline - moved into a `__simt_callee__` the short loop measured about 10% slower on such shapes on 950PR.
//  Tiling, UB reduction and alpha order are identical, so the two share one body.
// ==========================================================================
template <bool STEPPED>
__simt_vf__ __aicore__ LAUNCH_BOUND(CGBMV_N_MAX_THREADS) inline void CgbmvNSimt(
    uint32_t blk, uint32_t nblk, uint32_t slices, uint32_t m, uint32_t n, uint32_t kl, uint32_t ku, uint32_t lda,
    float alphaR, float alphaI, float betaR, float betaI, uint32_t betaMode, int64_t incx, int64_t incy,
    __gm__ const float* aGm, __gm__ const float* xGm, __gm__ float* yGm)
{
    __ubuf__ float partR[CGBMV_N_MAX_THREADS];
    __ubuf__ float partI[CGBMV_N_MAX_THREADS];
    uint32_t lane = threadIdx.x & (CGBMV_WARP_SIZE - 1U);
    uint32_t slice = threadIdx.x / CGBMV_WARP_SIZE;
    uint64_t bandEnd = static_cast<uint64_t>(n) + kl;
    uint32_t bandRows = (m < bandEnd) ? m : static_cast<uint32_t>(bandEnd);
    uint32_t tiles = (bandRows + CGBMV_WARP_SIZE - 1U) / CGBMV_WARP_SIZE;
    for (uint32_t tile = blk; tile < tiles; tile += nblk) {
        uint32_t row = tile * CGBMV_WARP_SIZE + lane;
        CgbmvColRange cols = SliceColumns(tile * CGBMV_WARP_SIZE, slice, slices, n, kl, ku);
        CgbmvAcc acc{0.0f, 0.0f};
        if (STEPPED) {
            if (row < m) {
                acc = BandRowDot(row, cols, kl, ku, lda, n, incx, alphaR, alphaI, aGm, xGm);
            }
        } else {
            // Same band test as BandRowDot: d = ku + row - col lies in the band iff d <= kl + ku, and a
            // negative d wraps to a large unsigned value, so one comparison covers both sides without any
            // sum that could exceed uint32.
            const uint32_t bandLast = kl + ku;
            for (uint32_t col = cols.begin; col < cols.end; ++col) {
                uint32_t d = ku + row - col;
                if (row >= m || d > bandLast) {
                    continue;
                }
                int64_t xk = FLOATS_PER_COMPLEX * VecIndex(col, n, incx);
                int64_t ak = FLOATS_PER_COMPLEX * (static_cast<int64_t>(d) + static_cast<int64_t>(lda) * col);
                float tr = alphaR * xGm[xk] - alphaI * xGm[xk + 1];
                float ti = alphaR * xGm[xk + 1] + alphaI * xGm[xk];
                acc.r += aGm[ak] * tr - aGm[ak + 1] * ti;
                acc.i += aGm[ak] * ti + aGm[ak + 1] * tr;
            }
        }
        partR[threadIdx.x] = acc.r;
        partI[threadIdx.x] = acc.i;
        asc_syncthreads();
        // The stepped loop leaves every row from n + kl on to ScaleRows; the short loop writes the last tile's
        // out-of-band rows itself (their sum is 0, since alpha only multiplies in-band terms).
        if (slice == 0U && row < (STEPPED ? bandRows : m)) {
            CgbmvAcc sum{0.0f, 0.0f};
            for (uint32_t s = 0; s < slices; ++s) {
                sum.r += partR[s * CGBMV_WARP_SIZE + lane];
                sum.i += partI[s * CGBMV_WARP_SIZE + lane];
            }
            StoreNResult(
                row, m, n, kl, ku, lda, incx, incy, alphaR, alphaI, betaR, betaI, betaMode, sum.r, sum.i, aGm, xGm,
                yGm);
        }
        asc_syncthreads();
    }
    uint32_t scaleFrom = (STEPPED ? bandRows : tiles * CGBMV_WARP_SIZE) + blk * blockDim.x + threadIdx.x;
    ScaleRows(scaleFrom, nblk * blockDim.x, m, incy, betaR, betaI, betaMode, yGm);
}

// trans T/C: y[col] = alpha * acc + beta * y[col] for the reduced column sum acc. The lanes add their terms in a
// different order from the reference, so a partial sum can overflow (or Inf * 0 can give NaN) where the reference
// stays finite or infinite. Such a column is recomputed in reference order: temp accumulates A(i, col) * x(i)
// (conj(A) for C) over ascending i, then y = beta * y + alpha * temp. Finite results keep the parallel sum.
//
// Only this column recomputation is noinline, keeping its serial loop outside the T/C fast path. N retains its
// row recomputation inline; N and SCALE use StoreY for beta * y handling. StoreTResult forms and checks the final
// value directly, without StoreY's separate recovery branch.
template <bool CONJ>
__simt_callee__ __aicore__ __attribute__((noinline)) void RecomputeTColumn(
    uint32_t col, uint32_t rBegin, uint32_t rEnd, uint32_t m, uint32_t ku, uint32_t lda, int64_t incx, float alphaR,
    float alphaI, float betaR, float betaI, uint32_t betaMode, __gm__ const float* aGm, __gm__ const float* xGm,
    __gm__ float* yp)
{
    CgbmvAcc temp{0.0f, 0.0f};
    int64_t colOffset = static_cast<int64_t>(lda) * col;
    for (uint32_t r = rBegin; r < rEnd; ++r) {
        int64_t ak = FLOATS_PER_COMPLEX * (static_cast<int64_t>(r) + colOffset);
        int64_t xk = FLOATS_PER_COMPLEX * VecIndex(col + r - ku, m, incx);
        float ai = CONJ ? -aGm[ak + 1] : aGm[ak + 1];
        CgbmvAcc term = ReferenceComplexMultiply({aGm[ak], ai}, {xGm[xk], xGm[xk + 1]});
        temp.r += term.r;
        temp.i += term.i;
    }
    CgbmvAcc out = ReferenceComplexMultiply({alphaR, alphaI}, temp);
    CgbmvAcc by = ReferenceBetaY(yp, betaR, betaI, betaMode);
    yp[0] = by.r + out.r;
    yp[1] = by.i + out.i;
}

// Stores T/C column `col` from the reduced lane sums. The result is formed here (alpha * acc, then beta * y) and
// checked once: a non-finite result goes to RecomputeTColumn, including when only beta * y or the final addition
// is non-finite. This avoids inlining StoreY's separate beta * y fallback into CgbmvTSimt.
template <bool CONJ>
__simt_callee__ __aicore__ inline void StoreTResult(
    uint32_t col, uint32_t rBegin, uint32_t rEnd, uint32_t m, uint32_t n, uint32_t ku, uint32_t lda, int64_t incx,
    int64_t incy, float alphaR, float alphaI, float betaR, float betaI, uint32_t betaMode, float accR, float accI,
    __gm__ const float* aGm, __gm__ const float* xGm, __gm__ float* yGm)
{
    __gm__ float* yp = yGm + FLOATS_PER_COMPLEX * VecIndex(col, n, incy);
    float resR = alphaR * accR - alphaI * accI;
    float resI = alphaR * accI + alphaI * accR;
    if (betaMode != CGBMV_BETA_ZERO) {
        float yr = yp[0];
        float yi = yp[1];
        if (betaMode == CGBMV_BETA_ONE) {
            resR += yr;
            resI += yi;
        } else {
            resR += betaR * yr - betaI * yi;
            resI += betaR * yi + betaI * yr;
        }
    }
    if (IsFiniteFloat(resR) && IsFiniteFloat(resI)) {
        yp[0] = resR;
        yp[1] = resI;
        return;
    }
    RecomputeTColumn<CONJ>(col, rBegin, rEnd, m, ku, lda, incx, alphaR, alphaI, betaR, betaI, betaMode, aGm, xGm, yp);
}

// ==========================================================================
//  trans = T / C: a group of `group` lanes (power of 2, <= 32) reduces one band column. The outer
//  column loop is warp-uniform, so every lane takes part in the shuffles after its (per-lane) row loop.
//  Lane 0 stores the column through StoreTResult, which redoes non-finite columns in reference order.
// ==========================================================================
template <bool CONJ>
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void CgbmvTSimt(
    uint32_t blk, uint32_t nblk, uint32_t m, uint32_t n, uint32_t kl, uint32_t ku, uint32_t lda, float alphaR,
    float alphaI, float betaR, float betaI, uint32_t betaMode, int64_t incx, int64_t incy, uint32_t group,
    __gm__ const float* aGm, __gm__ const float* xGm, __gm__ float* yGm)
{
    uint32_t lane = threadIdx.x & (group - 1U);
    uint32_t colsPerWarp = CGBMV_WARP_SIZE / group;
    uint32_t gtid = blk * blockDim.x + threadIdx.x;
    uint32_t warpStride = (nblk * blockDim.x / CGBMV_WARP_SIZE) * colsPerWarp;
    // 64-bit intermediates: kl + ku + 1 and m + ku both act as clamp bounds below, where a wrapped value
    // would cut rEnd short instead of being caught.
    uint64_t bandRows = static_cast<uint64_t>(kl) + ku + 1U;
    uint64_t rowsPlusKu = static_cast<uint64_t>(m) + ku;
    for (uint32_t base = (gtid / CGBMV_WARP_SIZE) * colsPerWarp; base < n; base += warpStride) {
        uint32_t col = base + (gtid & (CGBMV_WARP_SIZE - 1U)) / group;
        // band row r maps to matrix row i = col - ku + r; valid rows satisfy 0 <= i < m
        uint32_t rBegin = (col < ku) ? (ku - col) : 0U;
        uint64_t rEndWide = (rowsPlusKu > col) ? (rowsPlusKu - col) : 0U;
        rEndWide = (rEndWide < bandRows) ? rEndWide : bandRows;
        uint32_t rEnd = (col < n) ? static_cast<uint32_t>(rEndWide) : 0U;
        float accR = 0.0f;
        float accI = 0.0f;
        int64_t colOffset = static_cast<int64_t>(lda) * col;
        for (uint32_t r = rBegin + lane; r < rEnd; r += group) {
            int64_t ak = FLOATS_PER_COMPLEX * (static_cast<int64_t>(r) + colOffset);
            int64_t xk = FLOATS_PER_COMPLEX * VecIndex(col + r - ku, m, incx);
            float ar = aGm[ak];
            float ai = CONJ ? -aGm[ak + 1] : aGm[ak + 1];
            float xr = xGm[xk];
            float xi = xGm[xk + 1];
            accR += ar * xr - ai * xi;
            accI += ar * xi + ai * xr;
        }
        for (uint32_t off = group >> 1U; off > 0U; off >>= 1U) {
            accR += asc_shfl_down(accR, off, static_cast<int32_t>(group));
            accI += asc_shfl_down(accI, off, static_cast<int32_t>(group));
        }
        if (lane == 0U && col < n) {
            StoreTResult<CONJ>(
                col, rBegin, rEnd, m, n, ku, lda, incx, incy, alphaR, alphaI, betaR, betaI, betaMode, accR, accI, aGm,
                xGm, yGm);
        }
    }
}

struct CgbmvScalars {
    float alphaR;
    float alphaI;
    float betaR;
    float betaI;
    uint32_t mode;
    uint32_t betaMode;
};

// alpha/beta come from the tiling (host scalars) or from GM (device scalars). For device scalars the host could
// not evaluate them, so mode/betaMode are derived here with the same rules the host applies to host scalars.
__aicore__ inline CgbmvScalars LoadScalars(GM_ADDR alphaGm, GM_ADDR betaGm, const CgbmvTilingData& tiling)
{
    CgbmvScalars sc{tiling.alphaR, tiling.alphaI, tiling.betaR, tiling.betaI, tiling.mode, tiling.betaMode};
    if (tiling.alphaIsDevice != 0U) {
        auto* ap = reinterpret_cast<__gm__ const float*>(alphaGm);
        sc.alphaR = ap[0];
        sc.alphaI = ap[1];
        sc.mode = (sc.alphaR == 0.0f && sc.alphaI == 0.0f) ? CGBMV_MODE_SCALE : CGBMV_MODE_GBMV;
    }
    if (tiling.betaIsDevice != 0U) {
        auto* bp = reinterpret_cast<__gm__ const float*>(betaGm);
        sc.betaR = bp[0];
        sc.betaI = bp[1];
        if (sc.betaR == 0.0f && sc.betaI == 0.0f) {
            sc.betaMode = CGBMV_BETA_ZERO;
        } else {
            sc.betaMode = (sc.betaR == 1.0f && sc.betaI == 0.0f) ? CGBMV_BETA_ONE : CGBMV_BETA_GENERAL;
        }
    }
    return sc;
}

} // namespace

extern "C" __global__ __aicore__ void cgbmv_kernel(
    GM_ADDR a, GM_ADDR x, GM_ADDR y, GM_ADDR alphaGm, GM_ADDR betaGm, const CgbmvTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    CgbmvScalars sc = LoadScalars(alphaGm, betaGm, tiling);
    if (sc.mode == CGBMV_MODE_SCALE && sc.betaMode == CGBMV_BETA_ONE) {
        return; // alpha == 0 and beta == 1: y is left untouched (reachable only with device scalars)
    }
    auto* aGm = reinterpret_cast<__gm__ const float*>(a);
    auto* xGm = reinterpret_cast<__gm__ const float*>(x);
    auto* yGm = reinterpret_cast<__gm__ float*>(y);
    dim3 threads{tiling.numThreads, 1, 1};
    // Core index/count are passed explicitly: blockIdx/gridDim inside the VF do not identify the AIV core.
    uint32_t blk = static_cast<uint32_t>(AscendC::GetBlockIdx());
    uint32_t nblk = static_cast<uint32_t>(AscendC::GetBlockNum());

    if (sc.mode == CGBMV_MODE_SCALE) {
        uint32_t len = (tiling.trans == CGBMV_TRANS_N) ? tiling.m : tiling.n;
        asc_vf_call<CgbmvScaleSimt>(threads, blk, nblk, len, tiling.incy, sc.betaR, sc.betaI, sc.betaMode, yGm);
        return;
    }
    if (tiling.trans == CGBMV_TRANS_N) {
        if (tiling.nStepped != 0U) {
            asc_vf_call<CgbmvNSimt<true>>(
                threads, blk, nblk, tiling.slices, tiling.m, tiling.n, tiling.kl, tiling.ku, tiling.lda, sc.alphaR,
                sc.alphaI, sc.betaR, sc.betaI, sc.betaMode, tiling.incx, tiling.incy, aGm, xGm, yGm);
        } else {
            asc_vf_call<CgbmvNSimt<false>>(
                threads, blk, nblk, tiling.slices, tiling.m, tiling.n, tiling.kl, tiling.ku, tiling.lda, sc.alphaR,
                sc.alphaI, sc.betaR, sc.betaI, sc.betaMode, tiling.incx, tiling.incy, aGm, xGm, yGm);
        }
    } else if (tiling.trans == CGBMV_TRANS_T) {
        asc_vf_call<CgbmvTSimt<false>>(
            threads, blk, nblk, tiling.m, tiling.n, tiling.kl, tiling.ku, tiling.lda, sc.alphaR, sc.alphaI, sc.betaR,
            sc.betaI, sc.betaMode, tiling.incx, tiling.incy, tiling.groupSize, aGm, xGm, yGm);
    } else {
        asc_vf_call<CgbmvTSimt<true>>(
            threads, blk, nblk, tiling.m, tiling.n, tiling.kl, tiling.ku, tiling.lda, sc.alphaR, sc.alphaI, sc.betaR,
            sc.betaI, sc.betaMode, tiling.incx, tiling.incy, tiling.groupSize, aGm, xGm, yGm);
    }
}

void cgbmv_kernel_do(
    uint8_t* a, uint8_t* x, uint8_t* y, uint8_t* alphaGm, uint8_t* betaGm, uint32_t numBlocks,
    const CgbmvTilingData& tiling, void* stream)
{
    cgbmv_kernel<<<numBlocks, nullptr, stream>>>(a, x, y, alphaGm, betaGm, tiling);
}
