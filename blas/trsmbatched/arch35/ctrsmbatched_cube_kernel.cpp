/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#define CTRSM_BATCHED_DEVICE
#include "ctrsmbatched_kernel.h"
#include "tensor_api/tensor.h"

// Workspace-free large NON_UNIT panels.
namespace CtrsmBatchedVector {
constexpr uint32_t PANEL = 96;
constexpr uint32_t RHS = 64;
constexpr uint32_t THREADS = 512;
constexpr uint32_t DIAG_SCALE_OFFSET = PANEL * PANEL * 2;
constexpr uint32_t A_FLOATS = DIAG_SCALE_OFFSET + PANEL;
// One extra 32-byte block distributes column-major packing across UB banks.
constexpr uint32_t B_STRIDE = RHS * 2 + 8;
constexpr uint32_t B_FLOATS = PANEL * B_STRIDE;
// Reserve the remaining 8 KiB for the compiler's SIMT stack.
constexpr uint32_t STORAGE_FLOATS = 53248;
static __ubuf__ float sharedStorage[STORAGE_FLOATS];

__simt_callee__ CTRSM_INLINE inline uint64_t BOffset(uint32_t row, uint32_t rhs, CtrsmBatchedTilingData t)
{
    return t.side == ACLBLAS_SIDE_LEFT ? row + static_cast<uint64_t>(rhs) * t.ldb :
                                         rhs + static_cast<uint64_t>(row) * t.ldb;
}

__simt_callee__ CTRSM_INLINE inline aclblasComplex NormalizeDiagonal(aclblasComplex v, __ubuf__ float* diagonalScale)
{
    const float realAbs = v.real < 0 ? -v.real : v.real;
    const float imagAbs = v.imag < 0 ? -v.imag : v.imag;
    const float scale = realAbs > imagAbs ? realAbs : imagAbs;
    *diagonalScale = scale;
    v = CtrsmBatched::Div({1, 0}, {v.real / scale, v.imag / scale});
    return v;
}

template <bool DIAGONAL>
__simt_callee__ CTRSM_INLINE inline void LoadTileA(
    GM_ADDR a, GM_ADDR b, CtrsmBatchedTilingData t, uint64_t job, uint32_t start, uint32_t next, uint32_t rows,
    uint32_t size, __ubuf__ float* aa, __ubuf__ float* out)
{
    const bool left = t.side == ACLBLAS_SIDE_LEFT;
    const uint32_t k = left ? t.m : t.n;
    const bool upper = (t.uplo == ACLBLAS_UPPER) != (t.trans != ACLBLAS_OP_N);
    const bool forward = left ? !upper : upper;
    const auto ap = reinterpret_cast<const __gm__ aclblasComplex*>(a);
    const bool aColumnMajor = left == (t.trans == ACLBLAS_OP_N);
    for (uint32_t q = threadIdx.x; q < rows * size; q += THREADS) {
        const uint32_t r = aColumnMajor ? q % rows : q / size;
        const uint32_t c = aColumnMajor ? q / rows : q % size;
        if (r < rows && c < size && (!DIAGONAL || c <= r)) {
            const uint32_t row = forward ? next + r : k - 1 - next - r;
            const uint32_t col = forward ? start + c : k - 1 - start - c;
            aclblasComplex v;
            if (DIAGONAL && r == c && t.diag == ACLBLAS_UNIT) {
                v = {1, 0};
            } else {
                v = left ? CtrsmBatched::LoadA(ap, row, col, t) : CtrsmBatched::LoadA(ap, col, row, t);
                if (DIAGONAL && r == c) {
                    // Inverting the original diagonal can underflow or overflow.
                    // Cache its normalized inverse and scale the RHS before use.
                    v = NormalizeDiagonal(v, aa + DIAG_SCALE_OFFSET + r);
                }
            }
            aa[(r * PANEL + c) * 2] = v.real;
            aa[(r * PANEL + c) * 2 + 1] = v.imag;
        }
    }
}

template <bool DIAGONAL>
__simt_callee__ CTRSM_INLINE inline void LoadTileB(
    GM_ADDR a, GM_ADDR b, CtrsmBatchedTilingData t, uint64_t job, uint32_t start, uint32_t next, uint32_t rows,
    uint32_t size, __ubuf__ float* aa, __ubuf__ float* out)
{
    const bool left = t.side == ACLBLAS_SIDE_LEFT;
    const uint32_t k = left ? t.m : t.n, rhsCount = left ? t.n : t.m;
    const uint32_t tiles = (rhsCount + RHS - 1) / RHS;
    const uint32_t first = (job % tiles) * RHS;
    const uint32_t count = rhsCount - first < RHS ? rhsCount - first : RHS;
    const bool upper = (t.uplo == ACLBLAS_UPPER) != (t.trans != ACLBLAS_OP_N);
    const bool forward = left ? !upper : upper;
    const auto ap = reinterpret_cast<const __gm__ aclblasComplex*>(a);
    const auto bp = reinterpret_cast<__gm__ aclblasComplex*>(b);
    const bool aColumnMajor = left == (t.trans == ACLBLAS_OP_N);
    for (uint32_t q = threadIdx.x; q < rows * count; q += THREADS) {
        const uint32_t r = left ? q % rows : q / count;
        const uint32_t rhs = left ? q / rows : q % count;
        const uint32_t row = forward ? next + r : k - 1 - next - r;
        const uint64_t off = BOffset(row, first + rhs, t);
        aclblasComplex v = {bp[off].real, bp[off].imag};
        // The first panel touches each B element once before any subtraction.
        if (start == 0 && (t.alpha.real != 1 || t.alpha.imag != 0))
            v = CtrsmBatched::ScaleInput(t.alpha, v);
        out[r * B_STRIDE + rhs] = v.real;
        out[r * B_STRIDE + RHS + rhs] = v.imag;
    }
}

template <bool DIAGONAL>
__simt_callee__ CTRSM_INLINE inline void LoadTileBody(
    GM_ADDR a, GM_ADDR b, CtrsmBatchedTilingData t, uint64_t job, uint32_t start, uint32_t next, uint32_t rows,
    uint32_t size, __ubuf__ float* aa, __ubuf__ float* out)
{
    LoadTileA<DIAGONAL>(a, b, t, job, start, next, rows, size, aa, out);
    LoadTileB<DIAGONAL>(a, b, t, job, start, next, rows, size, aa, out);
}

__simt_callee__ CTRSM_INLINE inline void StoreTileBody(
    GM_ADDR b, CtrsmBatchedTilingData t, uint64_t job, uint32_t next, uint32_t rows, __ubuf__ float* in)
{
    const bool left = t.side == ACLBLAS_SIDE_LEFT;
    const uint32_t k = left ? t.m : t.n, rhsCount = left ? t.n : t.m;
    const uint32_t tiles = (rhsCount + RHS - 1) / RHS;
    const uint32_t first = (job % tiles) * RHS;
    const uint32_t count = rhsCount - first < RHS ? rhsCount - first : RHS;
    const bool upper = (t.uplo == ACLBLAS_UPPER) != (t.trans != ACLBLAS_OP_N);
    const bool forward = left ? !upper : upper;
    const auto bp = reinterpret_cast<__gm__ aclblasComplex*>(b);
    for (uint32_t q = threadIdx.x; q < rows * count; q += THREADS) {
        const uint32_t r = left ? q % rows : q / count;
        const uint32_t rhs = left ? q / rows : q % count;
        const uint32_t row = forward ? next + r : k - 1 - next - r;
        const uint64_t off = BOffset(row, first + rhs, t);
        bp[off].real = in[r * B_STRIDE + rhs];
        bp[off].imag = in[r * B_STRIDE + RHS + rhs];
    }
}

template <bool DIAGONAL>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREADS) inline void PrepareTile(
    GM_ADDR a, GM_ADDR b, CtrsmBatchedTilingData t, uint64_t job, uint32_t previous, uint32_t previousRows,
    __ubuf__ float* previousData, uint32_t start, uint32_t next, uint32_t rows, uint32_t size, __ubuf__ float* aa,
    __ubuf__ float* out)
{
    if (previousRows != 0) {
        StoreTileBody(b, t, job, previous, previousRows, previousData);
        // All reads of the old tile must complete before reusing its UB buffer.
        asc_syncthreads();
    }
    LoadTileBody<DIAGONAL>(a, b, t, job, start, next, rows, size, aa, out);
}

__simt_vf__ __aicore__ LAUNCH_BOUND(THREADS) inline void StoreTile(
    GM_ADDR b, CtrsmBatchedTilingData t, uint64_t job, uint32_t next, uint32_t rows, __ubuf__ float* in)
{
    StoreTileBody(b, t, job, next, rows, in);
}

template <bool DIAGONAL>
__aicore__ inline void Compute(
    __ubuf__ float* aa, __ubuf__ float* xx, __ubuf__ float* out, uint16_t rows, uint16_t size, uint32_t count,
    bool unit)
{
    using namespace AscendC::MicroAPI;
    const uint16_t chunks = static_cast<uint16_t>((count + 63) / 64);
    __VEC_SCOPE__
    {
        RegTensor<float> vr, vi, xr, xi, ar, ai, rr, ii, ri, ir, pr, pi, scale;
        for (uint16_t r = 0; r < rows; ++r) {
            for (uint16_t chunk = 0; chunk < chunks; ++chunk) {
                const uint32_t laneStart = chunk * 64;
                uint32_t remain = count - laneStart;
                auto mask = UpdateMask<float>(remain);
                DataCopy(vr, out + r * B_STRIDE + laneStart);
                DataCopy(vi, out + r * B_STRIDE + RHS + laneStart);
                const uint16_t limit = DIAGONAL ? r : size;
                for (uint16_t c = 0; c < limit; ++c) {
                    DataCopy(xr, xx + c * B_STRIDE + laneStart);
                    DataCopy(xi, xx + c * B_STRIDE + RHS + laneStart);
                    DataCopy<float, LoadDist::DIST_BRC_B32>(ar, aa + (r * PANEL + c) * 2);
                    DataCopy<float, LoadDist::DIST_BRC_B32>(ai, aa + (r * PANEL + c) * 2 + 1);
                    Mul(rr, ar, xr, mask);
                    Mul(ii, ai, xi, mask);
                    Mul(ri, ar, xi, mask);
                    Mul(ir, ai, xr, mask);
                    Sub(pr, rr, ii, mask);
                    Add(pi, ri, ir, mask);
                    Sub(vr, vr, pr, mask);
                    Sub(vi, vi, pi, mask);
                }
                if (DIAGONAL && !unit) {
                    DataCopy<float, LoadDist::DIST_BRC_B32>(scale, aa + DIAG_SCALE_OFFSET + r);
                    Div(vr, vr, scale, mask);
                    Div(vi, vi, scale, mask);
                    DataCopy<float, LoadDist::DIST_BRC_B32>(ar, aa + (r * PANEL + r) * 2);
                    DataCopy<float, LoadDist::DIST_BRC_B32>(ai, aa + (r * PANEL + r) * 2 + 1);
                    Mul(rr, ar, vr, mask);
                    Mul(ii, ai, vi, mask);
                    Mul(ri, ar, vi, mask);
                    Mul(ir, ai, vr, mask);
                    Sub(vr, rr, ii, mask);
                    Add(vi, ri, ir, mask);
                }
                DataCopy(out + r * B_STRIDE + laneStart, vr, mask);
                DataCopy(out + r * B_STRIDE + RHS + laneStart, vi, mask);
            }
        }
    }
}

__aicore__ inline void Run(GM_ADDR a, GM_ADDR b, CtrsmBatchedTilingData t)
{
    using namespace AscendC;
    auto storage = CtrsmBatchedVector::sharedStorage;
    auto aa = storage;
    auto xx = aa + A_FLOATS;
    auto yy = xx + B_FLOATS;
    const uint32_t k = t.side == ACLBLAS_SIDE_LEFT ? t.m : t.n;
    const uint32_t rhsCount = t.side == ACLBLAS_SIDE_LEFT ? t.n : t.m;
    const uint32_t tiles = (rhsCount + RHS - 1) / RHS;
    for (uint64_t job = GetBlockIdx(); job < static_cast<uint64_t>(tiles) * t.batchCount; job += GetBlockNum()) {
        const uint64_t batch = job / tiles;
        const auto matrixA = reinterpret_cast<GM_ADDR>(reinterpret_cast<__gm__ uint64_t*>(a)[batch]);
        const auto matrixB = reinterpret_cast<GM_ADDR>(reinterpret_cast<__gm__ uint64_t*>(b)[batch]);
        const uint32_t first = (job % tiles) * RHS;
        const uint32_t count = rhsCount - first < RHS ? rhsCount - first : RHS;
        uint32_t previous = 0, previousRows = 0;
        __ubuf__ float* previousData = xx;
        for (uint32_t start = 0; start < k; start += PANEL) {
            const uint32_t size = k - start < PANEL ? k - start : PANEL;
            asc_vf_call<PrepareTile<true>>(
                dim3{THREADS, 1, 1}, matrixA, matrixB, t, job, previous, previousRows, previousData, start, start, size,
                size, aa, xx);
            PipeBarrier<PIPE_ALL>();
            Compute<true>(aa, xx, xx, size, size, count, t.diag == ACLBLAS_UNIT);
            PipeBarrier<PIPE_ALL>();
            previous = start;
            previousRows = size;
            previousData = xx;
            for (uint32_t next = start + size; next < k; next += PANEL) {
                const uint32_t rows = k - next < PANEL ? k - next : PANEL;
                asc_vf_call<PrepareTile<false>>(
                    dim3{THREADS, 1, 1}, matrixA, matrixB, t, job, previous, previousRows, previousData, start, next,
                    rows, size, aa, yy);
                PipeBarrier<PIPE_ALL>();
                Compute<false>(aa, xx, yy, rows, size, count, true);
                PipeBarrier<PIPE_ALL>();
                previous = next;
                previousRows = rows;
                previousData = yy;
            }
        }
        asc_vf_call<StoreTile>(dim3{THREADS, 1, 1}, matrixB, t, job, previous, previousRows, previousData);
        PipeBarrier<PIPE_ALL>();
    }
}
} // namespace CtrsmBatchedVector

// Cube workspace layout and matrix packing.
namespace CtrsmBatchedCubeIO {
using namespace AscendC;
constexpr uint32_t THREADS = 512;
// One 32-byte block of complex padding spreads transpose reads across UB banks.
constexpr uint32_t INPUT_PADDING = 4;

__simt_callee__ CTRSM_INLINE inline float InversePackingScale(float scale)
{
    // Preserve signed infinity for a zero diagonal without dividing by zero.
    if (scale == 0.0f) {
        return CtrsmBatched::CopySign(__builtin_inff(), scale);
    }
    return __fdividef(1.0f, scale);
}

template <uint32_t TILE>
__simt_callee__ CTRSM_INLINE inline void LoadPackingScales(
    const __gm__ aclblasComplex* ap, CtrsmBatchedTilingData t, uint32_t firstRow, uint32_t k, bool forward,
    __ubuf__ float* scales, __ubuf__ float* inverseScales)
{
    if (threadIdx.x < TILE) {
        const uint32_t nr = firstRow + threadIdx.x;
        float scale = 1;
        if (nr < k) {
            const uint32_t row = forward ? nr : k - 1 - nr;
            const auto d = CtrsmBatched::LoadA(ap, row, row, t);
            const float ar = d.real < 0 ? -d.real : d.real, ai = d.imag < 0 ? -d.imag : d.imag;
            scale = ar > ai ? ar : ai;
        }
        scales[threadIdx.x] = scale;
        inverseScales[threadIdx.x] = InversePackingScale(scale);
    }
    asc_syncthreads();
}

__simt_callee__ CTRSM_INLINE inline aclblasComplex ScalePackedValue(aclblasComplex v, float scale, float inverseScale)
{
    if (scale >= 1.0e-15f && scale <= 1.0e15f) {
        v.real *= inverseScale;
        v.imag *= inverseScale;
    } else if (scale != 0.0f) {
        v.real /= scale;
        v.imag /= scale;
    } else {
        // inverseScale is signed infinity; retain the original Inf/NaN propagation.
        v.real *= inverseScale;
        v.imag *= inverseScale;
    }
    return v;
}

// Map logical solve coordinates to the padded tile staged by DMA.
template <uint32_t TILE>
__simt_callee__ CTRSM_INLINE inline aclblasComplex LoadStagedA(
    const __ubuf__ aclblasComplex* raw, CtrsmBatchedTilingData t, uint32_t firstRow, uint32_t firstCol, uint32_t r,
    uint32_t c)
{
    const bool left = t.side == ACLBLAS_SIDE_LEFT;
    const uint32_t k = left ? t.m : t.n;
    const bool upper = (t.uplo == ACLBLAS_UPPER) != (t.trans != ACLBLAS_OP_N);
    const bool forward = left ? !upper : upper, columnMajor = left == (t.trans == ACLBLAS_OP_N);
    const uint32_t rows = k - firstRow < TILE ? k - firstRow : TILE, cols = k - firstCol < TILE ? k - firstCol : TILE;
    const uint32_t rr = forward ? r : rows - 1 - r, cc = forward ? c : cols - 1 - c;
    const uint32_t off = columnMajor ? rr + cc * (TILE + INPUT_PADDING) : cc + rr * (TILE + INPUT_PADDING);
    aclblasComplex v{raw[off].real, raw[off].imag};
    if (t.trans == ACLBLAS_OP_C)
        v.imag = -v.imag;
    return v;
}

template <uint32_t TILE, bool STAGED = false>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREADS) inline void PackA(
    GM_ADDR a, CtrsmBatchedTilingData t, uint32_t firstRow, uint32_t firstCol, __ubuf__ float* out)
{
    const bool left = t.side == ACLBLAS_SIDE_LEFT;
    const uint32_t k = left ? t.m : t.n;
    const bool upper = (t.uplo == ACLBLAS_UPPER) != (t.trans != ACLBLAS_OP_N);
    const bool forward = left ? !upper : upper, columnMajor = left == (t.trans == ACLBLAS_OP_N);
    const auto ap = reinterpret_cast<const __gm__ aclblasComplex*>(a);
    constexpr uint32_t PAIR = 4 * TILE + 8;
    auto scales = out + TILE * PAIR;
    auto inverseScales = scales + TILE;
    auto raw = reinterpret_cast<__ubuf__ aclblasComplex*>(inverseScales + TILE);
    LoadPackingScales<TILE>(ap, t, firstRow, k, forward, scales, inverseScales);
    for (uint32_t q = threadIdx.x; q < TILE * TILE; q += THREADS) {
        // Four rows per warp balance source reads and expanded-matrix stores.
        const uint32_t r = columnMajor ? (STAGED ? q / (4 * TILE) * 4 + q % 4 : q % TILE) : q / TILE;
        const uint32_t c = columnMajor ? (STAGED ? q / 4 % TILE : q / TILE) : q % TILE;
        const uint32_t nr = firstRow + r, nc = firstCol + c;
        // Solve and update only consume the stored lower triangle.
        if (nr >= k || nc > nr)
            continue;
        aclblasComplex v{0, 0};
        const uint32_t row = forward ? nr : k - 1 - nr, col = forward ? nc : k - 1 - nc;
        if constexpr (STAGED)
            v = LoadStagedA<TILE>(raw, t, firstRow, firstCol, r, c);
        else
            v = left ? CtrsmBatched::LoadA(ap, row, col, t) : CtrsmBatched::LoadA(ap, col, row, t);
        v = ScalePackedValue(v, scales[r], inverseScales[r]);
        // Cube computes the negative update, then Fixpipe adds it directly to Y.
        out[r * PAIR + 2 * c] = -v.real;
        out[r * PAIR + 2 * c + 1] = v.imag;
        out[r * PAIR + 2 * TILE + 2 * c] = -v.imag;
        out[r * PAIR + 2 * TILE + 2 * c + 1] = -v.real;
    }
}

template <uint32_t TILE, uint32_t COLS = TILE, bool STAGED = false>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREADS) inline void PackY(
    GM_ADDR a, GM_ADDR b, CtrsmBatchedTilingData t, uint32_t firstRow, uint32_t firstRhs, __ubuf__ float* out)
{
    const bool left = t.side == ACLBLAS_SIDE_LEFT;
    const uint32_t k = left ? t.m : t.n, rhs = left ? t.n : t.m;
    const bool upper = (t.uplo == ACLBLAS_UPPER) != (t.trans != ACLBLAS_OP_N);
    const bool forward = left ? !upper : upper;
    const auto ap = reinterpret_cast<const __gm__ aclblasComplex*>(a);
    const auto bp = reinterpret_cast<const __gm__ aclblasComplex*>(b);
    constexpr uint32_t PAIR = 2 * COLS + 8;
    auto scales = out + TILE * PAIR;
    auto inverseScales = scales + TILE;
    auto raw = reinterpret_cast<__ubuf__ aclblasComplex*>(inverseScales + TILE);
    LoadPackingScales<TILE>(ap, t, firstRow, k, forward, scales, inverseScales);
    for (uint32_t q = threadIdx.x; q < TILE * COLS; q += THREADS) {
        const uint32_t r = left ? q / (4 * COLS) * 4 + q % 4 : q / COLS;
        const uint32_t c = left ? q / 4 % COLS : q % COLS;
        const uint32_t nr = firstRow + r, nc = firstRhs + c;
        aclblasComplex v{0, 0};
        if (nr < k && nc < rhs) {
            const uint32_t row = forward ? nr : k - 1 - nr;
            const uint64_t off = CtrsmBatchedVector::BOffset(row, nc, t);
            if constexpr (STAGED) {
                const uint32_t rows = k - firstRow < TILE ? k - firstRow : TILE, rr = forward ? r : rows - 1 - r;
                const uint32_t pos = left ? c * (TILE + INPUT_PADDING) + rr : rr * (COLS + INPUT_PADDING) + c;
                v = {raw[pos].real, raw[pos].imag};
            } else
                v = {bp[off].real, bp[off].imag};
            v = ScalePackedValue(v, scales[r], inverseScales[r]);
            if (t.alpha.real != 1 || t.alpha.imag != 0)
                v = CtrsmBatched::ScaleInput(t.alpha, v);
        }
        out[r * PAIR + c] = v.real;
        out[r * PAIR + COLS + c] = v.imag;
    }
}

struct CubePacking {
    CtrsmBatchedTilingData t;
    uint32_t k, rhs, kp, rp, kt, ykt, rt;
    uint64_t cBatch, yBatch;
    bool left, forward, columnMajor;
    __gm__ float* c;
    __gm__ float* y;
    __ubuf__ float* ub;
};

template <uint32_t TILE, uint32_t ROWS, uint32_t COLS>
__aicore__ inline void PackATiles(GM_ADDR a, GM_ADDR b, const CubePacking& ctx)
{
    constexpr bool STAGED_A =
        TILE * (4 * TILE + 8) + 2 * TILE + 2 * TILE * (TILE + INPUT_PADDING) <= CtrsmBatchedVector::STORAGE_FLOATS;
    for (uint64_t job = GetBlockIdx(); job < static_cast<uint64_t>(ctx.t.batchCount) * ctx.kt * ctx.kt;
         job += GetBlockNum()) {
        const uint64_t batch = job / (ctx.kt * ctx.kt);
        const uint32_t tile = job % (ctx.kt * ctx.kt), row = tile / ctx.kt * TILE, col = tile % ctx.kt * TILE;
        if (col > row)
            continue;
        const uint32_t rows = ctx.kp - row < TILE ? ctx.kp - row : TILE,
                       cols = ctx.kp - col < TILE ? ctx.kp - col : TILE;
        const auto ap = reinterpret_cast<GM_ADDR>(reinterpret_cast<__gm__ uint64_t*>(a)[batch]);
        if constexpr (STAGED_A) {
            const uint32_t nr = ctx.k - row < TILE ? ctx.k - row : TILE, nc = ctx.k - col < TILE ? ctx.k - col : TILE;
            uint32_t rr = ctx.forward ? row : ctx.k - row - nr, cc = ctx.forward ? col : ctx.k - col - nc;
            if (!ctx.columnMajor) {
                const auto temp = rr;
                rr = cc;
                cc = temp;
            }
            auto raw = ctx.ub + TILE * (4 * TILE + 8) + 2 * TILE;
            asc_copy_gm2ub_align(
                raw, reinterpret_cast<__gm__ float*>(ap) + 2 * (rr + static_cast<uint64_t>(cc) * ctx.t.lda),
                ctx.columnMajor ? nc : nr, (ctx.columnMajor ? nr : nc) * 8, 0, 0, false, 0,
                static_cast<uint64_t>(ctx.t.lda) * 8, (TILE + INPUT_PADDING) * 8);
            PipeBarrier<PIPE_ALL>();
        }
        asc_vf_call<PackA<TILE, STAGED_A>>(dim3{THREADS, 1, 1}, ap, ctx.t, row, col, ctx.ub);
        PipeBarrier<PIPE_ALL>();
        auto destination = ctx.c + batch * ctx.cBatch + 2ULL * row * (2 * ctx.kp) + 2 * col;
        asc_copy_ub2gm_align(
            destination, ctx.ub, rows, 2 * cols * sizeof(float), 0, 4ULL * ctx.kp * sizeof(float),
            (4 * TILE + 8) * sizeof(float));
        asc_copy_ub2gm_align(
            destination + 2 * ctx.kp, ctx.ub + 2 * TILE, rows, 2 * cols * sizeof(float), 0,
            4ULL * ctx.kp * sizeof(float), (4 * TILE + 8) * sizeof(float));
        PipeBarrier<PIPE_ALL>();
    }
}

template <uint32_t TILE, uint32_t ROWS, uint32_t COLS>
__aicore__ inline void PackYTiles(GM_ADDR a, GM_ADDR b, const CubePacking& ctx)
{
    constexpr bool STAGED_B =
        ROWS * (2 * COLS + 8) + 2 * ROWS + 2 * ROWS * COLS + 2 * INPUT_PADDING * (ROWS > COLS ? ROWS : COLS) <=
        CtrsmBatchedVector::STORAGE_FLOATS;
    for (uint64_t job = GetBlockIdx(); job < static_cast<uint64_t>(ctx.t.batchCount) * ctx.ykt * ctx.rt;
         job += GetBlockNum()) {
        const uint64_t batch = job / (ctx.ykt * ctx.rt);
        const uint32_t tile = job % (ctx.ykt * ctx.rt), row = tile / ctx.rt * ROWS, col = tile % ctx.rt * COLS;
        const uint32_t rows = ctx.kp - row < ROWS ? ctx.kp - row : ROWS,
                       cols = ctx.rp - col < COLS ? ctx.rp - col : COLS;
        const auto bp = reinterpret_cast<GM_ADDR>(reinterpret_cast<__gm__ uint64_t*>(b)[batch]);
        const auto ap = reinterpret_cast<GM_ADDR>(reinterpret_cast<__gm__ uint64_t*>(a)[batch]);
        if constexpr (STAGED_B) {
            const uint32_t nr = ctx.k - row < ROWS ? ctx.k - row : ROWS,
                           nc = ctx.rhs - col < COLS ? ctx.rhs - col : COLS;
            const uint32_t rr = ctx.forward ? row : ctx.k - row - nr;
            const uint64_t off =
                ctx.left ? rr + static_cast<uint64_t>(col) * ctx.t.ldb : col + static_cast<uint64_t>(rr) * ctx.t.ldb;
            auto raw = ctx.ub + ROWS * (2 * COLS + 8) + 2 * ROWS;
            asc_copy_gm2ub_align(
                raw, reinterpret_cast<__gm__ float*>(bp) + 2 * off, ctx.left ? nc : nr, (ctx.left ? nr : nc) * 8, 0, 0,
                false, 0, static_cast<uint64_t>(ctx.t.ldb) * 8, ((ctx.left ? ROWS : COLS) + INPUT_PADDING) * 8);
            PipeBarrier<PIPE_ALL>();
        }
        asc_vf_call<PackY<ROWS, COLS, STAGED_B>>(dim3{THREADS, 1, 1}, ap, bp, ctx.t, row, col, ctx.ub);
        PipeBarrier<PIPE_ALL>();
        auto destination = ctx.y + batch * ctx.yBatch + 2ULL * row * ctx.rp + col;
        asc_copy_ub2gm_align(
            destination, ctx.ub, rows, cols * sizeof(float), 0, 2ULL * ctx.rp * sizeof(float),
            (2 * COLS + 8) * sizeof(float));
        asc_copy_ub2gm_align(
            destination + ctx.rp, ctx.ub + COLS, rows, cols * sizeof(float), 0, 2ULL * ctx.rp * sizeof(float),
            (2 * COLS + 8) * sizeof(float));
        PipeBarrier<PIPE_ALL>();
    }
}

template <uint32_t TILE, uint32_t ROWS = TILE, uint32_t COLS = TILE>
__aicore__ inline void Pack(GM_ADDR a, GM_ADDR b, GM_ADDR workspace, CtrsmBatchedTilingData t, uint32_t kp, uint32_t rp)
{
    static_assert(TILE * (4 * TILE + 8) + 2 * TILE <= CtrsmBatchedVector::STORAGE_FLOATS);
    const bool left = t.side == ACLBLAS_SIDE_LEFT;
    const uint32_t k = left ? t.m : t.n, rhs = left ? t.n : t.m;
    const bool upper = (t.uplo == ACLBLAS_UPPER) != (t.trans != ACLBLAS_OP_N), forward = left ? !upper : upper;
    const bool columnMajor = left == (t.trans == ACLBLAS_OP_N);
    const uint32_t kt = (kp + TILE - 1) / TILE, ykt = (kp + ROWS - 1) / ROWS, rt = (rp + COLS - 1) / COLS;
    static_assert(ROWS * (2 * COLS + 8) + 2 * ROWS <= CtrsmBatchedVector::A_FLOATS + 2 * CtrsmBatchedVector::B_FLOATS);
    const uint64_t cBatch = 4ULL * kp * kp, yBatch = 2ULL * kp * rp;
    auto c = reinterpret_cast<__gm__ float*>(workspace), y = c + cBatch * t.batchCount;
    auto storage = CtrsmBatchedVector::sharedStorage;
    auto ub = storage;
    const CubePacking ctx{t, k, rhs, kp, rp, kt, ykt, rt, cBatch, yBatch, left, forward, columnMajor, c, y, ub};
    PackATiles<TILE, ROWS, COLS>(a, b, ctx);
    PackYTiles<TILE, ROWS, COLS>(a, b, ctx);
}

template <uint32_t TILE, uint32_t COLS = TILE>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREADS) inline void UnpackY(
    bool left, bool forward, uint32_t rows, uint32_t cols, __ubuf__ float* in, __ubuf__ float* out)
{
    constexpr uint32_t L_STRIDE = TILE * 2 + 8, R_STRIDE = COLS * 2 + 8;
    constexpr uint32_t Y_STRIDE = COLS + 8;
    for (uint32_t q = threadIdx.x; q < TILE * COLS; q += THREADS) {
        const uint32_t r = left ? q / (4 * COLS) * 4 + q % 4 : q / COLS;
        const uint32_t c = left ? q / 4 % COLS : q % COLS;
        if (r >= rows || c >= cols)
            continue;
        const uint32_t srcRow = forward ? r : rows - 1 - r;
        const uint32_t dest = left ? c * L_STRIDE + 2 * r : r * R_STRIDE + 2 * c;
        out[dest] = in[(2 * srcRow) * Y_STRIDE + c];
        out[dest + 1] = in[(2 * srcRow + 1) * Y_STRIDE + c];
    }
}

template <uint32_t TILE, uint32_t COLS = TILE>
__aicore__ inline void Unpack(GM_ADDR b, GM_ADDR workspace, CtrsmBatchedTilingData t, uint32_t kp, uint32_t rp)
{
    constexpr uint32_t L_STRIDE = TILE * 2 + 8, R_STRIDE = COLS * 2 + 8;
    constexpr uint32_t Y_STRIDE = COLS + 8;
    static_assert(
        2 * TILE * Y_STRIDE + (COLS * L_STRIDE > TILE * R_STRIDE ? COLS * L_STRIDE : TILE * R_STRIDE) <=
        CtrsmBatchedVector::A_FLOATS + 2 * CtrsmBatchedVector::B_FLOATS);
    const bool left = t.side == ACLBLAS_SIDE_LEFT;
    const uint32_t k = left ? t.m : t.n, rhs = left ? t.n : t.m;
    const bool upper = (t.uplo == ACLBLAS_UPPER) != (t.trans != ACLBLAS_OP_N);
    const bool forward = left ? !upper : upper;
    const uint32_t kt = (k + TILE - 1) / TILE, rt = (rhs + COLS - 1) / COLS;
    auto y = reinterpret_cast<__gm__ float*>(workspace) + 4ULL * kp * kp * t.batchCount;
    auto storage = CtrsmBatchedVector::sharedStorage;
    auto in = storage, out = in + 2 * TILE * Y_STRIDE;
    for (uint64_t job = GetBlockIdx(); job < static_cast<uint64_t>(t.batchCount) * kt * rt; job += GetBlockNum()) {
        const uint64_t batch = job / (kt * rt);
        const uint32_t tile = job % (kt * rt), row = tile / rt * TILE, col = tile % rt * COLS;
        const uint32_t rows = k - row < TILE ? k - row : TILE, cols = rhs - col < COLS ? rhs - col : COLS;
        asc_copy_gm2ub_align(
            in, y + batch * 2ULL * kp * rp + 2ULL * row * rp + col, 2 * rows, cols * sizeof(float), 0, 0, false, 0,
            static_cast<uint64_t>(rp) * sizeof(float), Y_STRIDE * sizeof(float));
        PipeBarrier<PIPE_ALL>();
        asc_vf_call<UnpackY<TILE, COLS>>(dim3{THREADS, 1, 1}, left, forward, rows, cols, in, out);
        PipeBarrier<PIPE_ALL>();
        const uint32_t physicalRow = forward ? row : k - row - rows;
        auto bp = reinterpret_cast<__gm__ float*>(reinterpret_cast<__gm__ uint64_t*>(b)[batch]);
        const uint64_t off =
            left ? physicalRow + static_cast<uint64_t>(col) * t.ldb : col + static_cast<uint64_t>(physicalRow) * t.ldb;
        asc_copy_ub2gm_align(
            bp + 2 * off, out, left ? cols : rows, (left ? rows : cols) * 2 * sizeof(float), 0,
            static_cast<uint64_t>(t.ldb) * 2 * sizeof(float), (left ? L_STRIDE : R_STRIDE) * sizeof(float));
        PipeBarrier<PIPE_ALL>();
    }
}
} // namespace CtrsmBatchedCubeIO

// Cube diagonal solves and trailing updates (NON_UNIT only).
namespace CtrsmBatchedCube {
using namespace AscendC;
namespace te = AscendC::Te;
constexpr uint32_t THREADS = 128;
constexpr uint32_t PANEL = 64;
constexpr uint32_t INVERSE_STRIDE = 2 * PANEL + 8;
constexpr uint32_t RHS = CtrsmBatchedVector::RHS;
constexpr uint32_t TILE_M = 128;
constexpr uint32_t TILE_N = 128;
// In-place solve: a tile must load the whole panel before overwriting any of it.
static_assert(TILE_M >= 2 * PANEL && PANEL <= RHS);
static_assert(TILE_M * 2 * PANEL * sizeof(float) <= 65536);
static_assert(TILE_N * 2 * PANEL * sizeof(float) <= 65536);
static_assert(2 * PANEL * INVERSE_STRIDE <= CtrsmBatchedVector::A_FLOATS);

extern "C" __global__ __aicore__ void ctrsm_cube_pack(
    GM_ADDR a, GM_ADDR b, GM_ADDR workspace, CtrsmBatchedTilingData t, uint32_t kp, uint32_t rp)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    const uint32_t k = t.side == ACLBLAS_SIDE_LEFT ? t.m : t.n;
    const uint32_t rhs = t.side == ACLBLAS_SIDE_LEFT ? t.n : t.m;
    if (k <= 16) {
        CtrsmBatchedCubeIO::Pack<32, 16, 256>(a, b, workspace, t, kp, rp);
        return;
    }
    if (k <= 24) {
        CtrsmBatchedCubeIO::Pack<32, 24, 256>(a, b, workspace, t, kp, rp);
        return;
    }
    if (k <= 32) {
        CtrsmBatchedCubeIO::Pack<32, 32, 256>(a, b, workspace, t, kp, rp);
        return;
    }
    if (k <= 48) {
        CtrsmBatchedCubeIO::Pack<64, 48, 128>(a, b, workspace, t, kp, rp);
        return;
    }
    if (k <= 128) {
        CtrsmBatchedCubeIO::Pack<64, 64, 128>(a, b, workspace, t, kp, rp);
        return;
    }
    if (k <= 256 && rhs <= 64) {
        CtrsmBatchedCubeIO::Pack<64, 96, 64>(a, b, workspace, t, kp, rp);
        return;
    }
    if (rhs <= 64) {
        CtrsmBatchedCubeIO::Pack<96, 96, 64>(a, b, workspace, t, kp, rp);
        return;
    }
    if ((t.side == ACLBLAS_SIDE_LEFT) == (t.trans == ACLBLAS_OP_N))
        CtrsmBatchedCubeIO::Pack<88, 88, 96>(a, b, workspace, t, kp, rp);
    else if (k > 512 && rhs >= 256)
        CtrsmBatchedCubeIO::Pack<64, 64, 128>(a, b, workspace, t, kp, rp);
    else
        CtrsmBatchedCubeIO::Pack<96>(a, b, workspace, t, kp, rp);
}

extern "C" __global__ __aicore__ void ctrsm_cube_unpack(
    GM_ADDR b, GM_ADDR workspace, CtrsmBatchedTilingData t, uint32_t kp, uint32_t rp)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    const uint32_t k = t.side == ACLBLAS_SIDE_LEFT ? t.m : t.n;
    const uint32_t rhs = t.side == ACLBLAS_SIDE_LEFT ? t.n : t.m;
    if (k <= 16) {
        CtrsmBatchedCubeIO::Unpack<16, 256>(b, workspace, t, kp, rp);
        return;
    }
    if (k <= 24) {
        CtrsmBatchedCubeIO::Unpack<24, 256>(b, workspace, t, kp, rp);
        return;
    }
    if (k <= 32) {
        CtrsmBatchedCubeIO::Unpack<32, 256>(b, workspace, t, kp, rp);
        return;
    }
    if (k <= 48) {
        CtrsmBatchedCubeIO::Unpack<48, 128>(b, workspace, t, kp, rp);
        return;
    }
    if (k <= 128) {
        CtrsmBatchedCubeIO::Unpack<64, 128>(b, workspace, t, kp, rp);
        return;
    }
    if (rhs <= 64) {
        CtrsmBatchedCubeIO::Unpack<96, 64>(b, workspace, t, kp, rp);
        return;
    }
    if ((t.side == ACLBLAS_SIDE_LEFT) == (t.trans == ACLBLAS_OP_N))
        CtrsmBatchedCubeIO::Unpack<88, 96>(b, workspace, t, kp, rp);
    else if (k > 512 && rhs >= 256)
        CtrsmBatchedCubeIO::Unpack<64, 128>(b, workspace, t, kp, rp);
    else
        CtrsmBatchedCubeIO::Unpack<96>(b, workspace, t, kp, rp);
}

__simt_vf__ __aicore__ LAUNCH_BOUND(THREADS) inline void LoadInverse(
    GM_ADDR matrix, uint32_t kp, uint32_t start, uint32_t size, bool unit, __ubuf__ float* aa, __ubuf__ float* yy)
{
    auto cp = reinterpret_cast<__gm__ float*>(matrix);
    for (uint32_t q = threadIdx.x; q < size * size; q += THREADS) {
        const uint32_t r = q / size, c = q % size;
        if (c > r)
            continue;
        const uint64_t off = 2ULL * (start + r) * (2 * kp) + 2 * (start + c);
        aclblasComplex v{-cp[off], -cp[off + 2 * kp]};
        if (r == c && !unit) {
            const float ar = v.real < 0 ? -v.real : v.real, ai = v.imag < 0 ? -v.imag : v.imag;
            const float scale = ar > ai ? ar : ai;
            aa[CtrsmBatchedVector::DIAG_SCALE_OFFSET + r] = scale;
            v = CtrsmBatched::Div({1, 0}, {v.real / scale, v.imag / scale});
        }
        aa[(r * CtrsmBatchedVector::PANEL + c) * 2] = v.real;
        aa[(r * CtrsmBatchedVector::PANEL + c) * 2 + 1] = v.imag;
    }
    for (uint32_t q = threadIdx.x; q < size * RHS; q += THREADS) {
        const uint32_t r = q / RHS, c = q % RHS;
        yy[r * CtrsmBatchedVector::B_STRIDE + c] = r == c ? 1 : 0;
        yy[r * CtrsmBatchedVector::B_STRIDE + RHS + c] = 0;
    }
}

__simt_vf__ __aicore__
LAUNCH_BOUND(THREADS) inline void PackInverse(uint32_t size, __ubuf__ float* yy, __ubuf__ float* out)
{
    for (uint32_t q = threadIdx.x; q < size * size; q += THREADS) {
        const uint32_t r = q / size, c = q % size;
        const float real = yy[r * CtrsmBatchedVector::B_STRIDE + c];
        const float imag = yy[r * CtrsmBatchedVector::B_STRIDE + RHS + c];
        out[(2 * r) * INVERSE_STRIDE + 2 * c] = real;
        out[(2 * r) * INVERSE_STRIDE + 2 * c + 1] = -imag;
        out[(2 * r + 1) * INVERSE_STRIDE + 2 * c] = imag;
        out[(2 * r + 1) * INVERSE_STRIDE + 2 * c + 1] = real;
    }
}

extern "C" __global__ __aicore__ void ctrsm_cube_inverse(
    GM_ADDR workspace, CtrsmBatchedTilingData t, uint32_t kp, uint32_t rp, uint32_t panel)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    const uint32_t k = t.side == ACLBLAS_SIDE_LEFT ? t.m : t.n, panels = (k + panel - 1) / panel;
    auto storage = CtrsmBatchedVector::sharedStorage;
    auto aa = storage, yy = aa + CtrsmBatchedVector::A_FLOATS;
    for (uint64_t job = GetBlockIdx(); job < static_cast<uint64_t>(t.batchCount) * panels; job += GetBlockNum()) {
        const uint64_t batch = job / panels;
        const uint32_t start = job % panels * panel, size = k - start < panel ? k - start : panel;
        auto matrix = reinterpret_cast<__gm__ float*>(workspace) + batch * 4ULL * kp * kp;
        asc_vf_call<LoadInverse>(
            dim3{THREADS, 1, 1}, reinterpret_cast<GM_ADDR>(matrix), kp, start, size, t.diag == ACLBLAS_UNIT, aa, yy);
        PipeBarrier<PIPE_ALL>();
        // Keep a full vector of identity columns, including the last partial panel.
        // The fixed vector width also covers every column of a partial panel.
        // Keep the runtime diagonal argument: replacing it with constant false
        // changes CANN 9.1's vector division lowering and loses NON_UNIT accuracy.
        CtrsmBatchedVector::Compute<true>(aa, yy, yy, size, size, RHS, t.diag == ACLBLAS_UNIT);
        PipeBarrier<PIPE_ALL>();
        asc_vf_call<PackInverse>(dim3{THREADS, 1, 1}, size, yy, aa);
        PipeBarrier<PIPE_ALL>();
        asc_copy_ub2gm_align(
            matrix + 2ULL * start * (2 * kp) + 2 * start, aa, 2 * size, 2 * size * sizeof(float), 0,
            2ULL * kp * sizeof(float), INVERSE_STRIDE * sizeof(float));
        PipeBarrier<PIPE_ALL>();
    }
}

template <typename A, typename B, typename C, typename Q>
__aicore__ inline void MultiplyCubeTile(
    A& a0, B& b0, C& c0, Q& qg, uint32_t row, uint32_t nc, uint32_t mm, uint32_t nn, uint32_t reduction)
{
    SetFlag<HardEvent::MTE1_M>(0);
    WaitFlag<HardEvent::MTE1_M>(0);
    te::Mmad(
        te::MmadAtom<te::MmadTraits<te::MmadOperation>>{}.with(te::MmadParams{
            static_cast<uint16_t>(mm), static_cast<uint16_t>(nn), static_cast<uint16_t>(reduction), 3, true}),
        c0, a0, b0);
    SetFlag<HardEvent::M_MTE1>(0);
    SetFlag<HardEvent::M_FIX>(0);
    WaitFlag<HardEvent::M_FIX>(0);
    te::MakeCopy(te::CopyL0C2GM{})
        .Call(qg.Slice(te::MakeCoord(row, nc), te::MakeShape(mm, nn)), c0, te::FixpipeParams{3});
    SetFlag<HardEvent::FIX_M>(0);
}

// Independent output tiles use two L1 slots. L0 and Fixpipe remain serialized.
constexpr uint32_t L1_MATRIX_BYTES = 65536;
struct CubeTile {
    uint64_t batch;
    uint32_t row, col, rows, cols;
};

struct CubeWork {
    __gm__ float* c;
    __gm__ float* y;
    uint64_t cBatch, yBatch, jobs;
    uint32_t kp, rp, start, reduction, firstRow, rowCount, mt, nt;
};

template <uint32_t TM, uint32_t TN>
__aicore__ inline CubeTile CubeTileAt(uint64_t job, const CubeWork& work)
{
    const uint32_t tile = job % (work.mt * work.nt), row = tile / work.nt * TM, col = tile % work.nt * TN;
    const uint32_t rows = work.rowCount - row < TM ? work.rowCount - row : TM;
    const uint32_t cols = work.rp - col < TN ? work.rp - col : TN;
    return {job / (work.mt * work.nt), work.firstRow + row, col, rows, cols};
}

template <uint32_t SLOT>
__aicore__ inline void LoadCubeTile(const CubeWork& work, const CubeTile& tile)
{
    WaitFlag<HardEvent::MTE1_MTE2>(SLOT);
    auto cg = te::MakeTensor(
        te::MakeMemPtr<te::Location::GM>(work.c + tile.batch * work.cBatch),
        te::MakeFrameLayout<te::NDExtLayoutPtn>(2 * work.kp, 2 * work.kp));
    auto yg = te::MakeTensor(
        te::MakeMemPtr<te::Location::GM>(work.y + tile.batch * work.yBatch),
        te::MakeFrameLayout<te::NDExtLayoutPtn>(2 * work.kp, work.rp));
    auto ca = te::MakeTensor(
        te::MakeMemPtr<te::Location::L1, float>(2 * SLOT * L1_MATRIX_BYTES),
        te::MakeFrameLayout<te::NZLayoutPtn, Std::Int<8>>(tile.rows, work.reduction));
    auto yb = te::MakeTensor(
        te::MakeMemPtr<te::Location::L1, float>((2 * SLOT + 1) * L1_MATRIX_BYTES),
        te::MakeFrameLayout<te::ZNLayoutPtn, Std::Int<8>>(work.reduction, tile.cols));
    te::Copy(
        te::MakeCopy(te::CopyGM2L1{}), ca,
        cg.Slice(te::MakeCoord(tile.row, 2 * work.start), te::MakeShape(tile.rows, work.reduction)));
    te::Copy(
        te::MakeCopy(te::CopyGM2L1{}), yb,
        yg.Slice(te::MakeCoord(2 * work.start, tile.col), te::MakeShape(work.reduction, tile.cols)));
    SetFlag<HardEvent::MTE2_MTE1>(SLOT);
}

template <uint32_t SLOT>
__aicore__ inline void ComputeLoadedCubeTile(const CubeWork& work, const CubeTile& tile)
{
    WaitFlag<HardEvent::M_MTE1>(0);
    WaitFlag<HardEvent::FIX_M>(0);
    WaitFlag<HardEvent::MTE2_MTE1>(SLOT);
    auto ca = te::MakeTensor(
        te::MakeMemPtr<te::Location::L1, float>(2 * SLOT * L1_MATRIX_BYTES),
        te::MakeFrameLayout<te::NZLayoutPtn, Std::Int<8>>(tile.rows, work.reduction));
    auto yb = te::MakeTensor(
        te::MakeMemPtr<te::Location::L1, float>((2 * SLOT + 1) * L1_MATRIX_BYTES),
        te::MakeFrameLayout<te::ZNLayoutPtn, Std::Int<8>>(work.reduction, tile.cols));
    auto a0 = te::MakeTensor(
        te::MakeMemPtr<te::Location::L0A, float>(0),
        te::MakeFrameLayout<te::NZLayoutPtn, Std::Int<8>>(tile.rows, work.reduction));
    auto b0 = te::MakeTensor(
        te::MakeMemPtr<te::Location::L0B, float>(0),
        te::MakeFrameLayout<te::ZNLayoutPtn, Std::Int<8>>(work.reduction, tile.cols));
    auto c0 = te::MakeTensor(
        te::MakeMemPtr<te::Location::L0C, float>(0),
        te::MakeFrameLayout<te::NZLayoutPtn, Std::Int<16>>(tile.rows, tile.cols));
    auto qg = te::MakeTensor(
        te::MakeMemPtr<te::Location::GM>(work.y + tile.batch * work.yBatch),
        te::MakeFrameLayout<te::NDExtLayoutPtn>(2 * work.kp, work.rp));
    te::Copy(te::MakeCopy(te::CopyL12L0A{}), a0, ca);
    te::Copy(te::MakeCopy(te::CopyL12L0B{}), b0, yb);
    // Return this L1 slot only after both L0 copies have completed.
    SetFlag<HardEvent::MTE1_MTE2>(SLOT);
    MultiplyCubeTile(a0, b0, c0, qg, tile.row, tile.col, tile.rows, tile.cols, work.reduction);
}

template <uint32_t TM, uint32_t TN>
__aicore__ inline void RunCubeTiles(const CubeWork& work)
{
    const uint64_t step = GetBlockNum(), first = GetBlockIdx();
    if (first < work.jobs)
        LoadCubeTile<0>(work, CubeTileAt<TM, TN>(first, work));
    bool second = false;
    for (uint64_t job = first; job < work.jobs; job += step) {
        if (job + step < work.jobs) {
            if (second)
                LoadCubeTile<0>(work, CubeTileAt<TM, TN>(job + step, work));
            else
                LoadCubeTile<1>(work, CubeTileAt<TM, TN>(job + step, work));
        }
        if (second)
            ComputeLoadedCubeTile<1>(work, CubeTileAt<TM, TN>(job, work));
        else
            ComputeLoadedCubeTile<0>(work, CubeTileAt<TM, TN>(job, work));
        second = !second;
    }
}

template <bool SOLVE, uint32_t TM = TILE_M, uint32_t TN = TILE_N>
__aicore__ inline void CubeCompute(
    GM_ADDR workspace, CtrsmBatchedTilingData t, uint32_t kp, uint32_t rp, uint32_t start, uint32_t size)
{
    InitSocState();
    const uint32_t k = t.side == ACLBLAS_SIDE_LEFT ? t.m : t.n;
    // Scaling by a nontrivial alpha can amplify HF32 input rounding beyond the
    // component-wise absolute tolerance. Keep full FP32 whenever alpha's L1
    // norm can increase either output component (also covers non-finite alpha).
    const float alphaReal = t.alpha.real < 0 ? -t.alpha.real : t.alpha.real;
    const float alphaImag = t.alpha.imag < 0 ? -t.alpha.imag : t.alpha.imag;
    SetHF32Mode((!SOLVE || k >= 128) && alphaReal + alphaImag <= 1.0f);
    const uint32_t m = SOLVE ? 2 * size : 2 * (k - start - size), n = rp, reduction = 2 * size;
    const uint32_t mt = (m + TM - 1) / TM, nt = (n + TN - 1) / TN;
    const uint64_t cBatch = 4ULL * kp * kp, yBatch = 2ULL * kp * rp;
    auto c = reinterpret_cast<__gm__ float*>(workspace);
    auto y = c + cBatch * t.batchCount;
    if constexpr (SOLVE)
        SetAtomicNone();
    else
        SetAtomicAdd<float>();
    SetFlag<HardEvent::MTE1_MTE2>(0);
    SetFlag<HardEvent::M_MTE1>(0);
    SetFlag<HardEvent::FIX_M>(0);
    SetFlag<HardEvent::MTE1_MTE2>(1);
    const CubeWork work{c,  y,  cBatch, yBatch,    static_cast<uint64_t>(mt) * nt * t.batchCount,
                        kp, rp, start,  reduction, 2 * (SOLVE ? start : start + size),
                        m,  mt, nt};
    RunCubeTiles<TM, TN>(work);
    WaitFlag<HardEvent::MTE1_MTE2>(0);
    WaitFlag<HardEvent::MTE1_MTE2>(1);
    WaitFlag<HardEvent::M_MTE1>(0);
    WaitFlag<HardEvent::FIX_M>(0);
    SetAtomicNone();
    SetHF32Mode(false);
}

extern "C" __global__ __aicore__ void ctrsm_cube_update(
    GM_ADDR workspace, CtrsmBatchedTilingData t, uint32_t kp, uint32_t rp, uint32_t start, uint32_t size)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    if (size <= 32)
        CubeCompute<false, 256, 256>(workspace, t, kp, rp, start, size);
    else if (size <= 48)
        CubeCompute<false, 160, 160>(workspace, t, kp, rp, start, size);
    else
        CubeCompute<false>(workspace, t, kp, rp, start, size);
}

extern "C" __global__ __aicore__ void ctrsm_cube_solve(
    GM_ADDR workspace, CtrsmBatchedTilingData t, uint32_t kp, uint32_t rp, uint32_t start, uint32_t size)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    if (size <= 32)
        CubeCompute<true, 256, 256>(workspace, t, kp, rp, start, size);
    else if (size <= 48)
        CubeCompute<true, 160, 160>(workspace, t, kp, rp, start, size);
    else
        CubeCompute<true>(workspace, t, kp, rp, start, size);
}
} // namespace CtrsmBatchedCube

void ctrsm_batched_grouped_inverse_do(
    uint8_t* workspace, const CtrsmBatchedTilingData& t, uint32_t kp, uint32_t blocks, void* stream);

void ctrsm_batched_cube_do(
    uint8_t* a, uint8_t* b, uint8_t* workspace, const CtrsmBatchedTilingData& t, uint32_t kp, uint32_t rp, uint32_t aiv,
    uint32_t aic, void* stream)
{
    using namespace CtrsmBatchedCube;
    const uint32_t k = t.side == ACLBLAS_SIDE_LEFT ? t.m : t.n;
    const uint32_t panel = k > 32 && k <= 64 && t.batchCount <= 64 ? 64 : (k <= 512 ? 32 : PANEL);
    ctrsm_cube_pack<<<aiv, nullptr, stream>>>(a, b, workspace, t, kp, rp);
    if (panel == 32)
        ctrsm_batched_grouped_inverse_do(workspace, t, kp, aiv, stream);
    else
        ctrsm_cube_inverse<<<aiv, nullptr, stream>>>(workspace, t, kp, rp, panel);
    for (uint32_t start = 0; start < k; start += panel) {
        const uint32_t size = k - start < panel ? k - start : panel;
        ctrsm_cube_solve<<<aic, nullptr, stream>>>(workspace, t, kp, rp, start, size);
        if (start + size < k) {
            ctrsm_cube_update<<<aic, nullptr, stream>>>(workspace, t, kp, rp, start, size);
        }
    }
    ctrsm_cube_unpack<<<aiv, nullptr, stream>>>(b, workspace, t, kp, rp);
}

extern "C" __global__ __aicore__ void ctrsm_batched_vector_kernel(GM_ADDR a, GM_ADDR b, CtrsmBatchedTilingData t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    CtrsmBatchedVector::Run(a, b, t);
}

void ctrsm_batched_vector_do(uint8_t* a, uint8_t* b, const CtrsmBatchedTilingData& t, uint32_t blocks, void* stream)
{
    ctrsm_batched_vector_kernel<<<blocks, nullptr, stream>>>(a, b, t);
}
