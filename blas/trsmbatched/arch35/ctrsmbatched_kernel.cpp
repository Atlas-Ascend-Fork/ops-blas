/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// This source is compiled normally for small/ordered kernels. Other entry
// families reuse its private templates without instantiating these kernels.
#define CTRSM_BATCHED_DEVICE
#include "ctrsmbatched_kernel.h"

namespace CtrsmBatched {
// Diagonal systems have no dependency updates. Divide directly with power-of-two
// scaling instead of forming a reciprocal that can underflow (or a norm that
// can overflow). Inspect only the stored triangle before writing any output.
template <typename APtr, typename BPtr>
__simt_callee__ CTRSM_INLINE inline bool SolveDiagonalRhs(APtr a, BPtr b, uint32_t rhs, CtrsmBatchedTilingData t)
{
    if (t.diag == ACLBLAS_UNIT)
        return false;
    const auto firstDiagonal = LoadA(a, 0, 0, t);
    const int exponent = ScaleExponent(firstDiagonal);
    if (exponent >= -60 && exponent <= 60)
        return false;
    const bool left = t.side == ACLBLAS_SIDE_LEFT;
    const uint32_t k = left ? t.m : t.n;
    for (uint32_t col = 0; col < k; ++col) {
        const uint32_t begin = t.uplo == ACLBLAS_UPPER ? 0 : col + 1;
        const uint32_t end = t.uplo == ACLBLAS_UPPER ? col : k;
        for (uint32_t row = begin; row < end; ++row) {
            const uint64_t off = row + static_cast<uint64_t>(col) * t.lda;
            if (a[off].real != 0 || a[off].imag != 0)
                return false;
        }
    }
    const uint64_t base = left ? static_cast<uint64_t>(rhs) * t.ldb : rhs;
    const uint64_t stride = left ? 1 : t.ldb;
    for (uint32_t i = 0; i < k; ++i) {
        const uint64_t off = base + i * stride;
        aclblasComplex value{b[off].real, b[off].imag};
        if (t.alpha.real != 1 || t.alpha.imag != 0)
            value = ScaleInput(t.alpha, value);
        value = OrderedDiv(value, LoadA(a, i, i, t));
        b[off].real = value.real;
        b[off].imag = value.imag;
    }
    return true;
}

struct SolveRange {
    uint32_t begin, end;
};

struct OrderedSolve {
    CtrsmBatchedTilingData t;
    uint32_t rhs, k, width;
    uint64_t base, stride;
    bool left, forward;
};

__simt_callee__ CTRSM_INLINE inline SolveRange NextPanel(uint32_t& cursor, const OrderedSolve& solve)
{
    if (solve.forward) {
        const uint32_t begin = cursor;
        cursor = solve.k < begin + 192 ? solve.k : begin + 192;
        return {begin, cursor};
    }
    const uint32_t end = cursor;
    cursor = solve.left ? (end > 192 ? end - 192 : 0) : ((end - 1) / 192) * 192;
    return {cursor, end};
}

__simt_callee__ CTRSM_INLINE inline SolveRange NextBlock(uint32_t& cursor, SolveRange panel, const OrderedSolve& solve)
{
    if (solve.forward) {
        const uint32_t begin = cursor;
        cursor += BlockWidth(begin - panel.begin, panel.end - panel.begin, solve.width);
        return {begin, cursor};
    }
    const uint32_t end = cursor;
    const uint32_t remainder = (end - panel.begin) % solve.width;
    cursor -= remainder ? (remainder & (0U - remainder)) : solve.width;
    return {cursor, end};
}

template <typename BPtr>
__simt_callee__ CTRSM_INLINE inline void ScaleRhs(BPtr b, const OrderedSolve& solve)
{
    if (solve.t.alpha.real == 1 && solve.t.alpha.imag == 0)
        return;
    for (uint32_t r = 0; r < solve.k; ++r) {
        const uint64_t off = solve.base + r * solve.stride;
        const auto v = ScaleInput(solve.t.alpha, {b[off].real, b[off].imag});
        b[off].real = v.real;
        b[off].imag = v.imag;
    }
}

template <typename APtr, typename BPtr>
__simt_callee__ CTRSM_INLINE inline void SubtractRange(
    APtr a, BPtr b, const OrderedSolve& solve, uint32_t row, SolveRange range, uint32_t height)
{
    const uint64_t off = solve.base + row * solve.stride;
    const auto sum = BlockDot(a, b, solve.rhs, solve.t, row, range.begin, range.end, height);
    const auto value = BlockSubtract({b[off].real, b[off].imag}, sum);
    b[off].real = value.real;
    b[off].imag = value.imag;
}

template <typename BPtr>
__simt_callee__ CTRSM_INLINE inline void PropagateNan(
    BPtr b, const OrderedSolve& solve, uint32_t row, aclblasComplex value)
{
    const uint32_t begin = solve.forward ? row + 1 : 0, end = solve.forward ? solve.k : row;
    for (uint32_t q = begin; q < end; ++q) {
        b[solve.base + q * solve.stride].real = value.real;
        b[solve.base + q * solve.stride].imag = value.imag;
    }
}

template <typename APtr, typename BPtr>
__simt_callee__ CTRSM_INLINE inline void UpdateBlock(
    APtr a, BPtr b, const OrderedSolve& solve, SolveRange block, uint32_t step, aclblasComplex value)
{
    const uint32_t row = solve.forward ? block.begin + step : block.end - 1 - step;
    for (uint32_t q = step + 1; q < block.end - block.begin; ++q) {
        const uint32_t target = solve.forward ? block.begin + q : block.end - 1 - q;
        const auto av = solve.left ? LoadA(a, target, row, solve.t) : LoadA(a, row, target, solve.t);
        const auto product = FusedMul(value, av);
        b[solve.base + target * solve.stride].real -= product.real;
        b[solve.base + target * solve.stride].imag -= product.imag;
    }
}

template <typename APtr, typename BPtr>
__simt_callee__ CTRSM_INLINE inline bool SolveBlock(APtr a, BPtr b, const OrderedSolve& solve, SolveRange block)
{
    for (uint32_t step = 0; step < block.end - block.begin; ++step) {
        const uint32_t row = solve.forward ? block.begin + step : block.end - 1 - step;
        const uint64_t off = solve.base + row * solve.stride;
        aclblasComplex value{b[off].real, b[off].imag}, inverse{1, 0};
        if (solve.t.diag != ACLBLAS_UNIT)
            inverse = BlockInverse(LoadA(a, row, row, solve.t));
        value = solve.left ? FusedMul(inverse, value) : FusedMul(value, inverse);
        b[off].real = value.real;
        b[off].imag = value.imag;
        if (__builtin_isnan(value.real) && __builtin_isnan(value.imag)) {
            PropagateNan(b, solve, row, value);
            return false;
        }
        UpdateBlock(a, b, solve, block, step, value);
    }
    return true;
}

template <typename APtr, typename BPtr>
__simt_callee__ CTRSM_INLINE inline bool SolvePanel(APtr a, BPtr b, const OrderedSolve& solve, SolveRange panel)
{
    uint32_t cursor = solve.forward ? panel.begin : panel.end;
    while (solve.forward ? cursor < panel.end : cursor > panel.begin) {
        const auto block = NextBlock(cursor, panel, solve);
        const uint32_t height = solve.left ? block.end - block.begin : BlockWidth(solve.rhs, solve.t.m, 8);
        const SolveRange previous{solve.forward ? panel.begin : block.end, solve.forward ? block.begin : panel.end};
        if (previous.end > previous.begin) {
            for (uint32_t row = block.begin; row < block.end; ++row)
                SubtractRange(a, b, solve, row, previous, height);
        }
        if (!SolveBlock(a, b, solve, block))
            return false;
    }
    return true;
}

template <typename APtr, typename BPtr>
__simt_callee__ CTRSM_INLINE inline void UpdatePanel(APtr a, BPtr b, const OrderedSolve& solve, SolveRange panel)
{
    const uint32_t begin = solve.forward ? panel.end : 0, end = solve.forward ? solve.k : panel.begin;
    for (uint32_t row = begin; row < end; ++row) {
        const uint32_t height =
            solve.left ? BlockWidth(row - begin, end - begin, 8) : BlockWidth(solve.rhs, solve.t.m, 8);
        SubtractRange(a, b, solve, row, panel, height);
    }
}

template <typename APtr, typename BPtr>
__simt_callee__ CTRSM_INLINE inline void SolveOrderedRhs(APtr a, BPtr b, uint32_t rhs, CtrsmBatchedTilingData t)
{
    if (SolveDiagonalRhs(a, b, rhs, t))
        return;
    const bool left = t.side == ACLBLAS_SIDE_LEFT;
    const bool upper = (t.uplo == ACLBLAS_UPPER) != (t.trans != ACLBLAS_OP_N);
    const OrderedSolve solve{
        t,
        rhs,
        left ? t.m : t.n,
        left ? 8U : 2U,
        left ? static_cast<uint64_t>(rhs) * t.ldb : rhs,
        left ? 1U : t.ldb,
        left,
        left ? !upper : upper};
    ScaleRhs(b, solve);
    uint32_t cursor = solve.forward ? 0 : solve.k;
    while (solve.forward ? cursor < solve.k : cursor > 0) {
        const auto panel = NextPanel(cursor, solve);
        if (!SolvePanel(a, b, solve, panel))
            return;
        UpdatePanel(a, b, solve, panel);
    }
}

// One lane owns one independent RHS: a B column for LEFT, a B row for RIGHT.
// No lane consumes another lane's writes, including across different AIV blocks.
__simt_callee__ CTRSM_INLINE inline void SolveRhs(
    const __gm__ aclblasComplex* a, __gm__ aclblasComplex* b, uint32_t rhs, CtrsmBatchedTilingData t)
{
    const bool left = t.side == ACLBLAS_SIDE_LEFT;
    const uint32_t k = left ? t.m : t.n;
    const uint64_t base = left ? static_cast<uint64_t>(rhs) * t.ldb : rhs;
    const uint64_t stride = left ? 1 : t.ldb;
    // Do not read A, its pointer array, or the old B when alpha is zero.
    if (t.alpha.real == 0.0f && t.alpha.imag == 0.0f) {
        for (uint32_t i = 0; i < k; ++i) {
            b[base + i * stride].real = 0.0f;
            b[base + i * stride].imag = 0.0f;
        }
        return;
    }
    SolveOrderedRhs(a, b, rhs, t);
}
} // namespace CtrsmBatched

#ifndef CTRSM_BATCHED_PORTABLE_CHECK
namespace CtrsmBatchedSimd {
using namespace AscendC;
using namespace AscendC::Reg;
struct ChunkValues {
    RegTensor<float> r0, i0, r1, i1, r2, i2, r3, i3;
};

struct ChunkTails {
    MaskReg t0, t1, t2, t3;
};

template <uint32_t WIDTH, uint32_t CHUNKS>
__simd_callee__ inline void LoadChunks(ChunkValues& v, __ubuf__ float* bb, uint16_t row)
{
    constexpr uint32_t STRIDE = 2 * WIDTH + 8;
    if constexpr (CHUNKS > 0) {
        DataCopy(v.r0, bb + row * STRIDE + 0);
        DataCopy(v.i0, bb + row * STRIDE + WIDTH + 0);
    }
    if constexpr (CHUNKS > 1) {
        DataCopy(v.r1, bb + row * STRIDE + 64);
        DataCopy(v.i1, bb + row * STRIDE + WIDTH + 64);
    }
    if constexpr (CHUNKS > 2) {
        DataCopy(v.r2, bb + row * STRIDE + 128);
        DataCopy(v.i2, bb + row * STRIDE + WIDTH + 128);
    }
    if constexpr (CHUNKS > 3) {
        DataCopy(v.r3, bb + row * STRIDE + 192);
        DataCopy(v.i3, bb + row * STRIDE + WIDTH + 192);
    }
}

template <uint32_t WIDTH, uint32_t CHUNKS>
__simd_callee__ inline void StoreChunks(ChunkValues& v, __ubuf__ float* bb, uint16_t row, MaskReg& mask)
{
    constexpr uint32_t STRIDE = 2 * WIDTH + 8;
    if constexpr (CHUNKS > 0) {
        DataCopy(bb + row * STRIDE + 0, v.r0, mask);
        DataCopy(bb + row * STRIDE + WIDTH + 0, v.i0, mask);
    }
    if constexpr (CHUNKS > 1) {
        DataCopy(bb + row * STRIDE + 64, v.r1, mask);
        DataCopy(bb + row * STRIDE + WIDTH + 64, v.i1, mask);
    }
    if constexpr (CHUNKS > 2) {
        DataCopy(bb + row * STRIDE + 128, v.r2, mask);
        DataCopy(bb + row * STRIDE + WIDTH + 128, v.i2, mask);
    }
    if constexpr (CHUNKS > 3) {
        DataCopy(bb + row * STRIDE + 192, v.r3, mask);
        DataCopy(bb + row * STRIDE + WIDTH + 192, v.i3, mask);
    }
}

__simd_callee__ inline void SubtractProduct(
    RegTensor<float>& real, RegTensor<float>& imag, RegTensor<float>& ar, RegTensor<float>& ai, RegTensor<float>& xr,
    RegTensor<float>& xi, MaskReg& mask)
{
    RegTensor<float> pr, pi;
    Mul(pr, ai, xi, mask);
    Muls(pr, pr, -1.0f, mask);
    MulAddDst(pr, ar, xr, mask);
    Mul(pi, ar, xi, mask);
    MulAddDst(pi, ai, xr, mask);
    Sub(real, real, pr, mask);
    Sub(imag, imag, pi, mask);
}

__simd_callee__ inline void UnitIdentity(RegTensor<float>& real, RegTensor<float>& imag, MaskReg& mask)
{
    RegTensor<float> zeroReal, zeroImag;
    Muls(zeroReal, imag, 0.0f, mask);
    Muls(zeroImag, real, 0.0f, mask);
    Sub(real, real, zeroReal, mask);
    Add(imag, imag, zeroImag, mask);
}

template <uint32_t CHUNKS>
__simd_callee__ inline void UnitChunks(ChunkValues& v, MaskReg& mask)
{
    if constexpr (CHUNKS > 0)
        UnitIdentity(v.r0, v.i0, mask);
    if constexpr (CHUNKS > 1)
        UnitIdentity(v.r1, v.i1, mask);
    if constexpr (CHUNKS > 2)
        UnitIdentity(v.r2, v.i2, mask);
    if constexpr (CHUNKS > 3)
        UnitIdentity(v.r3, v.i3, mask);
}

} // namespace CtrsmBatchedSimd
namespace CtrsmBatchedBlockedVector {
template <uint32_t WIDTH>
__aicore__ inline void Run(
    __ubuf__ float* aa, __ubuf__ float* bb, uint16_t k, CtrsmBatchedTilingData t, uint32_t firstRhs, uint16_t count);
}
namespace CtrsmBatchedSmallVector {
using namespace AscendC::Reg;
using namespace CtrsmBatchedSimd;
constexpr uint32_t THREADS = 256;
// Leave room for the compiler's SIMT stack within the 224 KiB shared-memory
// limit.
constexpr uint32_t STORAGE_FLOATS = 53248;

__simt_callee__ inline bool Unsafe(float v) { return !__builtin_isfinite(v) || v > 1.0e15f || v < -1.0e15f; }

template <uint32_t MAX_K, uint32_t GROUPS, uint32_t WIDTH, uint32_t TRANSFER_THREADS = 256>
struct Staging {
    static constexpr uint32_t BASE =
        GROUPS * MAX_K * (MAX_K + 1) + MAX_K * (2 * WIDTH + 8) + WIDTH + (TRANSFER_THREADS + 8) / 8 * 8;
    static constexpr bool B = MAX_K <= 64 && BASE + 2 * MAX_K * WIDTH <= 50944;
    static constexpr bool A = B && BASE + 2 * MAX_K * WIDTH + 2 * GROUPS * MAX_K * MAX_K <= 50944;
};

template <uint32_t MAX_K, bool STAGED>
__simt_callee__ inline aclblasComplex LoadInputA(
    const __gm__ aclblasComplex* ap, __ubuf__ aclblasComplex* raw, uint32_t row, uint32_t col, CtrsmBatchedTilingData t)
{
    if constexpr (STAGED) {
        const uint32_t off = t.trans == ACLBLAS_OP_N ? row + col * MAX_K : col + row * MAX_K;
        aclblasComplex v{raw[off].real, raw[off].imag};
        if (t.trans == ACLBLAS_OP_C)
            v.imag = -v.imag;
        return v;
    }
    return CtrsmBatched::LoadA(ap, row, col, t);
}

template <uint32_t MAX_K, uint32_t GROUPS, uint32_t FIXED_K>
struct IndexLayout {
    static constexpr uint32_t K = MAX_K <= 8 ? 8 : (MAX_K <= 16 ? 16 : (MAX_K <= 32 ? 32 : 64));
    static constexpr bool PADDED = GROUPS == 1 && MAX_K <= 32 && (FIXED_K == 0 || K != MAX_K);
};

template <
    uint32_t MAX_K, uint32_t GROUPS, uint32_t WIDTH, uint32_t FIXED_K, uint32_t FIXED_RHS, uint32_t TRANSFER_THREADS>
struct TransferState {
    CtrsmBatchedTilingData t;
    __ubuf__ aclblasComplex* raw;
    __ubuf__ aclblasComplex* rawA;
    const __gm__ aclblasComplex* ap;
    __gm__ aclblasComplex* bp;
    uint64_t batch;
    uint32_t group, lane, originalK, rhsCount, k, tiles, tile, first, count;
    bool left, forward, valid, unsafe;
};

template <uint32_t MODE, bool ALPHA_ONE>
__simt_callee__ CTRSM_INLINE inline void SetTransferMode(CtrsmBatchedTilingData& t)
{
    if constexpr (ALPHA_ONE) {
        t.alpha.real = 1.0f;
        t.alpha.imag = 0.0f;
    }
    if constexpr (MODE == 1) {
        t.side = ACLBLAS_SIDE_LEFT;
        t.trans = ACLBLAS_OP_N;
        t.diag = ACLBLAS_NON_UNIT;
    }
    if constexpr (MODE == 2) {
        t.side = ACLBLAS_SIDE_LEFT;
        t.trans = ACLBLAS_OP_C;
        t.diag = ACLBLAS_UNIT;
    }
    if constexpr (MODE == 3) {
        t.side = ACLBLAS_SIDE_RIGHT;
        t.trans = ACLBLAS_OP_N;
        t.diag = ACLBLAS_UNIT;
    }
    if constexpr (MODE == 4) {
        t.side = ACLBLAS_SIDE_RIGHT;
        t.trans = ACLBLAS_OP_C;
        t.diag = ACLBLAS_UNIT;
    }
}

template <
    uint32_t MAX_K, uint32_t GROUPS, uint32_t WIDTH, uint32_t FIXED_K, uint32_t FIXED_RHS, uint32_t TRANSFER_THREADS>
__simt_callee__ CTRSM_INLINE inline void SetTransferJob(
    TransferState<MAX_K, GROUPS, WIDTH, FIXED_K, FIXED_RHS, TRANSFER_THREADS>& ctx, GM_ADDR a, GM_ADDR b, uint64_t job)
{
    constexpr uint32_t RHS = WIDTH / GROUPS, GROUP_THREADS = TRANSFER_THREADS / GROUPS;
    if (job <= 0xffffffffULL) {
        ctx.batch = static_cast<uint32_t>(job) / ctx.tiles;
        ctx.tile = static_cast<uint32_t>(job) % ctx.tiles;
    } else {
        ctx.batch = job / ctx.tiles;
        ctx.tile = job % ctx.tiles;
    }
    ctx.first = ctx.tile * RHS, ctx.count = ctx.rhsCount - ctx.first < RHS ? ctx.rhsCount - ctx.first : RHS;
    const bool upper = (ctx.t.uplo == ACLBLAS_UPPER) != (ctx.t.trans != ACLBLAS_OP_N);
    ctx.forward = ctx.left ? !upper : upper;
    ctx.ap = nullptr;
    ctx.bp = nullptr;
    if (ctx.valid) {
        ctx.ap = reinterpret_cast<const __gm__ aclblasComplex*>(reinterpret_cast<__gm__ uint64_t*>(a)[ctx.batch]);
        ctx.bp = reinterpret_cast<__gm__ aclblasComplex*>(reinterpret_cast<__gm__ uint64_t*>(b)[ctx.batch]);
    }
}

template <
    uint32_t MAX_K, uint32_t GROUPS, uint32_t WIDTH, uint32_t FIXED_K, uint32_t FIXED_RHS, uint32_t TRANSFER_THREADS>
__simt_callee__ CTRSM_INLINE inline TransferState<MAX_K, GROUPS, WIDTH, FIXED_K, FIXED_RHS, TRANSFER_THREADS>
MakeTransferState(
    GM_ADDR a, GM_ADDR b, CtrsmBatchedTilingData t, uint64_t base, __ubuf__ uint32_t* bad, uint32_t rowsLimit)
{
    constexpr uint32_t RHS = WIDTH / GROUPS, GROUP_THREADS = TRANSFER_THREADS / GROUPS;
    TransferState<MAX_K, GROUPS, WIDTH, FIXED_K, FIXED_RHS, TRANSFER_THREADS> ctx{};
    ctx.t = t;
    ctx.raw = reinterpret_cast<__ubuf__ aclblasComplex*>(bad + (TRANSFER_THREADS + 8) / 8 * 8);
    ctx.group = threadIdx.x / GROUP_THREADS, ctx.lane = threadIdx.x % GROUP_THREADS;
    ctx.rawA = ctx.raw + MAX_K * WIDTH + ctx.group * MAX_K * MAX_K;
    ctx.left = t.side == ACLBLAS_SIDE_LEFT;
    ctx.originalK = FIXED_K ? FIXED_K : (ctx.left ? t.m : t.n);
    ctx.rhsCount = FIXED_RHS ? FIXED_RHS : (ctx.left ? t.n : t.m);
    ctx.k = FIXED_K ? FIXED_K : (ctx.originalK < rowsLimit ? ctx.originalK : rowsLimit);
    ctx.tiles = (ctx.rhsCount + RHS - 1) / RHS;
    ctx.valid = (base + ctx.group) < static_cast<uint64_t>(ctx.tiles) * t.batchCount;
    SetTransferJob(ctx, a, b, base + ctx.group);
    ctx.unsafe = Unsafe(t.alpha.real) || Unsafe(t.alpha.imag);
    return ctx;
}

template <
    uint32_t MAX_K, uint32_t GROUPS, uint32_t WIDTH, uint32_t FIXED_K, uint32_t FIXED_RHS, uint32_t TRANSFER_THREADS>
__simt_callee__ CTRSM_INLINE inline void LoadTransferDiagonal(
    TransferState<MAX_K, GROUPS, WIDTH, FIXED_K, FIXED_RHS, TRANSFER_THREADS>& ctx, __ubuf__ float* aa,
    __ubuf__ float* bb, __ubuf__ uint32_t* bad)
{
    constexpr uint32_t RHS = WIDTH / GROUPS, GROUP_THREADS = TRANSFER_THREADS / GROUPS;
    constexpr uint32_t A_SIZE = MAX_K * (MAX_K + 1);
    constexpr bool PREFETCH_A = Staging<MAX_K, GROUPS, WIDTH, TRANSFER_THREADS>::A;
    for (uint32_t r = ctx.lane; r < ctx.k; r += GROUP_THREADS) {
        aclblasComplex v{1, 0};
        if (ctx.valid && ctx.t.diag != ACLBLAS_UNIT) {
            const uint32_t row = ctx.forward ? r : ctx.originalK - 1 - r;
            v = LoadInputA<MAX_K, PREFETCH_A>(ctx.ap, ctx.rawA, row, row, ctx.t);
            ctx.unsafe |= Unsafe(v.real) || Unsafe(v.imag);
            const float ar = v.real < 0 ? -v.real : v.real, ai = v.imag < 0 ? -v.imag : v.imag;
            const float scale = ar > ai ? ar : ai;
            ctx.unsafe |= !(scale >= 1.0e-15f);
            const float norm = v.real * v.real + v.imag * v.imag;
            const float inverse = __fdividef(1.0f, norm);
            v = {v.real * inverse, -v.imag * inverse};
        }
        const uint32_t dst =
            GROUPS >= 4 ? (r * (r + 1) + 2 * r) * GROUPS + 2 * ctx.group : ctx.group * A_SIZE + r * (r + 1) + 2 * r;
        aa[dst] = v.real;
        aa[dst + 1] = v.imag;
    }
}

template <
    uint32_t MAX_K, uint32_t GROUPS, uint32_t WIDTH, uint32_t FIXED_K, uint32_t FIXED_RHS, uint32_t TRANSFER_THREADS>
__simt_callee__ CTRSM_INLINE inline void LoadTransferTriangle(
    TransferState<MAX_K, GROUPS, WIDTH, FIXED_K, FIXED_RHS, TRANSFER_THREADS>& ctx, __ubuf__ float* aa,
    __ubuf__ float* bb, __ubuf__ uint32_t* bad)
{
    constexpr uint32_t RHS = WIDTH / GROUPS, GROUP_THREADS = TRANSFER_THREADS / GROUPS;
    constexpr uint32_t A_SIZE = MAX_K * (MAX_K + 1);
    constexpr bool PREFETCH_A = Staging<MAX_K, GROUPS, WIDTH, TRANSFER_THREADS>::A;
    constexpr uint32_t INDEX_K = IndexLayout<MAX_K, GROUPS, FIXED_K>::K;
    constexpr bool PAD_INDEX = IndexLayout<MAX_K, GROUPS, FIXED_K>::PADDED;
    const uint32_t indexK = PAD_INDEX ? INDEX_K : ctx.k;
    for (uint32_t q = ctx.lane; q < ctx.k * indexK; q += GROUP_THREADS) {
        const bool columnMajor = ctx.left == (ctx.t.trans == ACLBLAS_OP_N);
        const uint32_t r = columnMajor ? q % indexK : q / indexK, c = columnMajor ? q / indexK : q % indexK;
        if (r >= ctx.k || c >= ctx.k || r <= c)
            continue;
        const uint32_t row = ctx.forward ? r : ctx.originalK - 1 - r, col = ctx.forward ? c : ctx.originalK - 1 - c;
        aclblasComplex v{0, 0};
        if (ctx.valid)
            v = ctx.left ? LoadInputA<MAX_K, PREFETCH_A>(ctx.ap, ctx.rawA, row, col, ctx.t) :
                           LoadInputA<MAX_K, PREFETCH_A>(ctx.ap, ctx.rawA, col, row, ctx.t);
        ctx.unsafe |= Unsafe(v.real) || Unsafe(v.imag);
        const uint32_t dst =
            GROUPS >= 4 ? (r * (r + 1) + 2 * c) * GROUPS + 2 * ctx.group : ctx.group * A_SIZE + r * (r + 1) + 2 * c;
        aa[dst] = v.real;
        aa[dst + 1] = v.imag;
    }
}

template <
    uint32_t MAX_K, uint32_t GROUPS, uint32_t WIDTH, uint32_t FIXED_K, uint32_t FIXED_RHS, uint32_t TRANSFER_THREADS>
__simt_callee__ CTRSM_INLINE inline aclblasComplex ReadTransferRhs(
    TransferState<MAX_K, GROUPS, WIDTH, FIXED_K, FIXED_RHS, TRANSFER_THREADS>& ctx, uint32_t step, uint32_t rhs,
    uint32_t row)
{
    constexpr uint32_t RHS = WIDTH / GROUPS, GROUP_THREADS = TRANSFER_THREADS / GROUPS;
    constexpr bool PREFETCH = Staging<MAX_K, GROUPS, WIDTH, TRANSFER_THREADS>::B;
    aclblasComplex v;
    const uint64_t src = ctx.left ? row + static_cast<uint64_t>(ctx.first + rhs) * ctx.t.ldb :
                                    ctx.first + rhs + static_cast<uint64_t>(row) * ctx.t.ldb;
    if constexpr (PREFETCH) {
        const uint32_t rr = ctx.forward ? step : ctx.k - 1 - step;
        const uint32_t off = ctx.group * MAX_K * RHS + (ctx.left ? rhs * MAX_K + rr : rr * RHS + rhs);
        v = {ctx.raw[off].real, ctx.raw[off].imag};
    } else {
        v = {ctx.bp[src].real, ctx.bp[src].imag};
    }
    return v;
}

template <
    uint32_t MAX_K, uint32_t GROUPS, uint32_t WIDTH, uint32_t FIXED_K, uint32_t FIXED_RHS, uint32_t TRANSFER_THREADS>
__simt_callee__ CTRSM_INLINE inline void LoadTransferRhs(
    TransferState<MAX_K, GROUPS, WIDTH, FIXED_K, FIXED_RHS, TRANSFER_THREADS>& ctx, __ubuf__ float* aa,
    __ubuf__ float* bb, __ubuf__ uint32_t* bad)
{
    constexpr uint32_t B_STRIDE = 2 * WIDTH + 8;
    constexpr uint32_t RHS = WIDTH / GROUPS, GROUP_THREADS = TRANSFER_THREADS / GROUPS;
    constexpr uint32_t INDEX_K = IndexLayout<MAX_K, GROUPS, FIXED_K>::K;
    constexpr bool PAD_INDEX = IndexLayout<MAX_K, GROUPS, FIXED_K>::PADDED;
    const uint32_t indexRows = PAD_INDEX && ctx.left ? INDEX_K : ctx.k;
    for (uint32_t q = ctx.lane; q < indexRows * RHS; q += GROUP_THREADS) {
        const uint32_t step = ctx.left ? q % indexRows : q / RHS, rhs = ctx.left ? q / indexRows : q % RHS;
        if (step >= ctx.k)
            continue;
        const uint32_t row = ctx.forward ? step : ctx.originalK - 1 - step;
        aclblasComplex v{0, 0};
        if (ctx.valid && rhs < ctx.count) {
            v = ReadTransferRhs(ctx, step, rhs, row);
            ctx.unsafe |= Unsafe(v.real) || Unsafe(v.imag);
            if (ctx.t.alpha.real != 1 || ctx.t.alpha.imag != 0)
                v = CtrsmBatched::ScaleInput(ctx.t.alpha, v);
        }
        bb[step * B_STRIDE + ctx.group * RHS + rhs] = v.real;
        bb[step * B_STRIDE + WIDTH + ctx.group * RHS + rhs] = v.imag;
    }
}

template <
    uint32_t MAX_K, uint32_t GROUPS, uint32_t WIDTH, uint32_t FIXED_K, uint32_t FIXED_RHS, uint32_t TRANSFER_THREADS,
    bool FINAL>
__simt_callee__ CTRSM_INLINE inline void CheckTransferOutput(
    TransferState<MAX_K, GROUPS, WIDTH, FIXED_K, FIXED_RHS, TRANSFER_THREADS>& ctx, __ubuf__ float* aa,
    __ubuf__ float* bb, __ubuf__ uint32_t* bad)
{
    constexpr uint32_t B_STRIDE = 2 * WIDTH + 8;
    if constexpr (FINAL) {
        if (threadIdx.x == 0)
            bad[TRANSFER_THREADS] = 0;
        asc_syncthreads();
    } else {
        ctx.unsafe |= bad[threadIdx.x] != 0;
        uint32_t severity = ctx.unsafe ? 2 : 0;
        for (uint32_t q = threadIdx.x; q < ctx.k * WIDTH; q += TRANSFER_THREADS) {
            const uint32_t row = q / WIDTH, rhs = q % WIDTH;
            const float real = bb[row * B_STRIDE + rhs], imag = bb[row * B_STRIDE + WIDTH + rhs];
            if (Unsafe(real) || Unsafe(imag)) {
                severity = 2;
            }
        }
        asc_syncthreads();
        const auto warpBad = asc_reduce_max(severity);
        if (threadIdx.x % warpSize == 0)
            bad[threadIdx.x / warpSize] = warpBad;
        asc_syncthreads();
        if (threadIdx.x == 0) {
            uint32_t any = 0;
            for (uint32_t i = 0; i < TRANSFER_THREADS / warpSize; ++i)
                any = bad[i] > any ? bad[i] : any;
            bad[TRANSFER_THREADS] = any;
        }
        asc_syncthreads();
    }
}

template <
    uint32_t MAX_K, uint32_t GROUPS, uint32_t WIDTH, uint32_t FIXED_K, uint32_t FIXED_RHS, uint32_t TRANSFER_THREADS>
__simt_callee__ CTRSM_INLINE inline void CompleteTransferRhs(
    TransferState<MAX_K, GROUPS, WIDTH, FIXED_K, FIXED_RHS, TRANSFER_THREADS>& ctx, __ubuf__ float* aa,
    __ubuf__ float* bb, __ubuf__ uint32_t* bad)
{
    constexpr uint32_t B_STRIDE = 2 * WIDTH + 8;
    constexpr uint32_t RHS = WIDTH / GROUPS, GROUP_THREADS = TRANSFER_THREADS / GROUPS;
    if (ctx.valid && ctx.lane < ctx.count) {
        bool complete = true;
        if (ctx.originalK > ctx.k) {
            const uint32_t slot = (ctx.k - 1) * B_STRIDE + ctx.group * RHS + ctx.lane;
            complete = __builtin_isnan(bb[slot]) && __builtin_isnan(bb[slot + WIDTH]);
        }
        bad[ctx.group * RHS + ctx.lane] = complete ? 1 : 0;
        if (!complete)
            CtrsmBatched::SolveOrderedRhs(ctx.ap, ctx.bp, ctx.first + ctx.lane, ctx.t);
    }
}

template <
    uint32_t MAX_K, uint32_t GROUPS, uint32_t WIDTH, uint32_t FIXED_K, uint32_t FIXED_RHS, uint32_t TRANSFER_THREADS>
__simt_callee__ CTRSM_INLINE inline void StoreTransferElement(
    TransferState<MAX_K, GROUPS, WIDTH, FIXED_K, FIXED_RHS, TRANSFER_THREADS>& ctx, __ubuf__ float* aa,
    __ubuf__ float* bb, __ubuf__ uint32_t* bad, uint32_t step, uint32_t rhs)
{
    constexpr uint32_t B_STRIDE = 2 * WIDTH + 8;
    constexpr uint32_t RHS = WIDTH / GROUPS, GROUP_THREADS = TRANSFER_THREADS / GROUPS;
    constexpr bool PREFETCH = Staging<MAX_K, GROUPS, WIDTH, TRANSFER_THREADS>::B;
    const uint32_t cachedStep = step < ctx.k ? step : ctx.k - 1;
    const uint32_t row = ctx.forward ? step : ctx.originalK - 1 - step;
    const uint64_t dst = ctx.left ? row + static_cast<uint64_t>(ctx.first + rhs) * ctx.t.ldb :
                                    ctx.first + rhs + static_cast<uint64_t>(row) * ctx.t.ldb;
    const float real = bb[cachedStep * B_STRIDE + ctx.group * RHS + rhs];
    const float imag = bb[cachedStep * B_STRIDE + WIDTH + ctx.group * RHS + rhs];
    if constexpr (PREFETCH) {
        const uint32_t off = ctx.group * MAX_K * RHS + (ctx.left ? rhs * MAX_K + row : row * RHS + rhs);
        ctx.raw[off].real = real;
        ctx.raw[off].imag = imag;
    } else {
        ctx.bp[dst].real = real;
        ctx.bp[dst].imag = imag;
    }
}

template <
    uint32_t MAX_K, uint32_t GROUPS, uint32_t WIDTH, uint32_t FIXED_K, uint32_t FIXED_RHS, uint32_t TRANSFER_THREADS>
__simt_callee__ CTRSM_INLINE inline void StoreTransferRows(
    TransferState<MAX_K, GROUPS, WIDTH, FIXED_K, FIXED_RHS, TRANSFER_THREADS>& ctx, __ubuf__ float* aa,
    __ubuf__ float* bb, __ubuf__ uint32_t* bad)
{
    constexpr uint32_t RHS = WIDTH / GROUPS, GROUP_THREADS = TRANSFER_THREADS / GROUPS;
    for (uint32_t step = ctx.lane / 32; ctx.valid && step < ctx.originalK; step += GROUP_THREADS / 32) {
        for (uint32_t rhs = ctx.lane % 32; rhs < ctx.count; rhs += 32) {
            if (step >= ctx.originalK)
                continue;
            if (!bad[ctx.group * RHS + rhs])
                continue;
            StoreTransferElement(ctx, aa, bb, bad, step, rhs);
        }
    }
}

template <
    uint32_t MAX_K, uint32_t GROUPS, uint32_t WIDTH, uint32_t FIXED_K, uint32_t FIXED_RHS, uint32_t TRANSFER_THREADS>
__simt_callee__ CTRSM_INLINE inline void StoreTransferElements(
    TransferState<MAX_K, GROUPS, WIDTH, FIXED_K, FIXED_RHS, TRANSFER_THREADS>& ctx, __ubuf__ float* aa,
    __ubuf__ float* bb, __ubuf__ uint32_t* bad)
{
    constexpr uint32_t RHS = WIDTH / GROUPS, GROUP_THREADS = TRANSFER_THREADS / GROUPS;
    // Host dispatch bounds ctx.originalK to 2048 and ctx.count to WIDTH (<=256).
    // Keep ctx.tile indexing in 32 bits; addresses remain 64-bit below.
    constexpr uint32_t INDEX_K = IndexLayout<MAX_K, GROUPS, FIXED_K>::K;
    const uint32_t indexRows =
        IndexLayout<MAX_K, GROUPS, FIXED_K>::PADDED && ctx.left && ctx.count >= 64 ? INDEX_K : ctx.originalK;
    if (GROUPS == 1 && !ctx.left) {
        StoreTransferRows(ctx, aa, bb, bad);
    } else {
        for (uint32_t q = ctx.lane; ctx.valid && q < indexRows * ctx.count; q += GROUP_THREADS) {
            const uint32_t step = ctx.left ? q % indexRows : q / ctx.count,
                           rhs = ctx.left ? q / indexRows : q % ctx.count;
            if (step >= ctx.originalK)
                continue;
            if (!bad[ctx.group * RHS + rhs])
                continue;
            StoreTransferElement(ctx, aa, bb, bad, step, rhs);
        }
    }
}

template <
    uint32_t MAX_K, uint32_t GROUPS, uint32_t WIDTH, uint32_t FIXED_K, uint32_t FIXED_RHS, uint32_t TRANSFER_THREADS,
    bool FINAL>
__simt_callee__ CTRSM_INLINE inline void StoreTransfer(
    TransferState<MAX_K, GROUPS, WIDTH, FIXED_K, FIXED_RHS, TRANSFER_THREADS>& ctx, __ubuf__ float* aa,
    __ubuf__ float* bb, __ubuf__ uint32_t* bad)
{
    CheckTransferOutput<MAX_K, GROUPS, WIDTH, FIXED_K, FIXED_RHS, TRANSFER_THREADS, FINAL>(ctx, aa, bb, bad);
    if (!FINAL && bad[TRANSFER_THREADS] == 2) {
        if (ctx.valid && ctx.lane < ctx.count)
            CtrsmBatched::SolveOrderedRhs(ctx.ap, ctx.bp, ctx.first + ctx.lane, ctx.t);
    } else {
        CompleteTransferRhs(ctx, aa, bb, bad);
        asc_syncthreads();
        StoreTransferElements(ctx, aa, bb, bad);
    }
}

template <
    uint32_t MAX_K, uint32_t GROUPS, uint32_t WIDTH, bool STORE, bool FINAL = false, uint32_t FIXED_K = 0,
    uint32_t FIXED_RHS = 0, uint32_t MODE = 0, uint32_t TRANSFER_THREADS = 256, bool ALPHA_ONE = false>
__simt_vf__ __aicore__ LAUNCH_BOUND(TRANSFER_THREADS) inline void Transfer(
    GM_ADDR a, GM_ADDR b, CtrsmBatchedTilingData t, uint64_t base, __ubuf__ float* aa, __ubuf__ float* bb,
    __ubuf__ uint32_t* indices, __ubuf__ uint32_t* bad, uint32_t rowsLimit = MAX_K)
{
    SetTransferMode<MODE, ALPHA_ONE>(t);
    auto ctx =
        MakeTransferState<MAX_K, GROUPS, WIDTH, FIXED_K, FIXED_RHS, TRANSFER_THREADS>(a, b, t, base, bad, rowsLimit);
    if constexpr (!STORE) {
        constexpr uint32_t RHS = WIDTH / GROUPS, A_SIZE = MAX_K * (MAX_K + 1);
        if (threadIdx.x < WIDTH)
            indices[threadIdx.x] = (threadIdx.x / RHS) * (GROUPS >= 4 ? 2 : A_SIZE);
        LoadTransferDiagonal(ctx, aa, bb, bad);
        LoadTransferTriangle(ctx, aa, bb, bad);
        LoadTransferRhs(ctx, aa, bb, bad);
        bad[threadIdx.x] = ctx.unsafe ? 1 : 0;
    } else {
        StoreTransfer<MAX_K, GROUPS, WIDTH, FIXED_K, FIXED_RHS, TRANSFER_THREADS, FINAL>(ctx, aa, bb, bad);
    }
}

// The NON_UNIT reciprocal path keeps four independently rounded products.
// Exceptional inputs/outputs use the ordered scalar fallback instead.
__simd_callee__ inline void ComplexProduct(
    AscendC::Reg::RegTensor<float>& pr, AscendC::Reg::RegTensor<float>& pi, AscendC::Reg::RegTensor<float>& ar,
    AscendC::Reg::RegTensor<float>& ai, AscendC::Reg::RegTensor<float>& xr, AscendC::Reg::RegTensor<float>& xi,
    AscendC::Reg::MaskReg& mask)
{
    using namespace AscendC::Reg;
    RegTensor<float> rr, ii, ri, ir;
    Mul(rr, ar, xr, mask);
    Mul(ii, ai, xi, mask);
    Mul(ri, ar, xi, mask);
    Mul(ir, ai, xr, mask);
    Sub(pr, rr, ii, mask);
    Add(pi, ri, ir, mask);
}

template <uint32_t GROUPS>
__simd_callee__ inline void LoadCoefficient(
    AscendC::Reg::RegTensor<float>& ar, AscendC::Reg::RegTensor<float>& ai, __ubuf__ float* aa, uint32_t offset,
    AscendC::Reg::RegTensor<uint32_t>& index, AscendC::Reg::RegTensor<uint32_t>& imagIndex, AscendC::Reg::MaskReg& mask)
{
    using namespace AscendC::Reg;
    if constexpr (GROUPS == 1) {
        DataCopy<float, LoadDist::DIST_BRC_B32>(ar, aa + offset);
        DataCopy<float, LoadDist::DIST_BRC_B32>(ai, aa + offset + 1);
    } else if constexpr (GROUPS >= 4) {
        RegTensor<float> packed;
        DataCopy(packed, aa + offset * GROUPS);
        Gather(ar, packed, index);
        Gather(ai, packed, imagIndex);
    } else {
        Gather(ar, aa + offset, index, mask);
        Gather(ai, aa + offset + 1, index, mask);
    }
}

__simd_callee__ inline void SubtractRoundedProduct(
    RegTensor<float>& real, RegTensor<float>& imag, RegTensor<float>& ar, RegTensor<float>& ai, RegTensor<float>& xr,
    RegTensor<float>& xi, MaskReg& mask)
{
    RegTensor<float> pr, pi;
    ComplexProduct(pr, pi, ar, ai, xr, xi, mask);
    Sub(real, real, pr, mask);
    Sub(imag, imag, pi, mask);
}

__simd_callee__ inline void MultiplyRounded(
    RegTensor<float>& real, RegTensor<float>& imag, RegTensor<float>& ar, RegTensor<float>& ai, MaskReg& mask)
{
    RegTensor<float> pr, pi;
    ComplexProduct(pr, pi, ar, ai, real, imag, mask);
    Move(real, pr);
    Move(imag, pi);
}

template <uint32_t WIDTH, uint32_t CHUNKS>
__simd_callee__ inline void SubtractMultipleColumn(
    ChunkValues& v, __ubuf__ float* aa, __ubuf__ float* bb, uint16_t r, uint16_t c, MaskReg& mask)
{
    constexpr uint32_t STRIDE = 2 * WIDTH + 8;
    RegTensor<float> ar, ai, xr, xi;
    DataCopy<float, LoadDist::DIST_BRC_B32>(ar, aa + r * (r + 1) + 2 * c);
    DataCopy<float, LoadDist::DIST_BRC_B32>(ai, aa + r * (r + 1) + 2 * c + 1);
    if constexpr (CHUNKS > 0) {
        DataCopy(xr, bb + c * STRIDE + 0);
        DataCopy(xi, bb + c * STRIDE + WIDTH + 0);
        SubtractRoundedProduct(v.r0, v.i0, ar, ai, xr, xi, mask);
    }
    if constexpr (CHUNKS > 1) {
        DataCopy(xr, bb + c * STRIDE + 64);
        DataCopy(xi, bb + c * STRIDE + WIDTH + 64);
        SubtractRoundedProduct(v.r1, v.i1, ar, ai, xr, xi, mask);
    }
    if constexpr (CHUNKS > 2) {
        DataCopy(xr, bb + c * STRIDE + 128);
        DataCopy(xi, bb + c * STRIDE + WIDTH + 128);
        SubtractRoundedProduct(v.r2, v.i2, ar, ai, xr, xi, mask);
    }
    if constexpr (CHUNKS > 3) {
        DataCopy(xr, bb + c * STRIDE + 192);
        DataCopy(xi, bb + c * STRIDE + WIDTH + 192);
        SubtractRoundedProduct(v.r3, v.i3, ar, ai, xr, xi, mask);
    }
}

template <uint32_t CHUNKS>
__simd_callee__ inline void SubtractMultipleLast(
    ChunkValues& v, ChunkValues& last, __ubuf__ float* aa, uint16_t r, MaskReg& mask)
{
    const uint16_t c = r - 1;
    RegTensor<float> ar, ai;
    DataCopy<float, LoadDist::DIST_BRC_B32>(ar, aa + r * (r + 1) + 2 * c);
    DataCopy<float, LoadDist::DIST_BRC_B32>(ai, aa + r * (r + 1) + 2 * c + 1);
    if constexpr (CHUNKS > 0)
        SubtractRoundedProduct(v.r0, v.i0, ar, ai, last.r0, last.i0, mask);
    if constexpr (CHUNKS > 1)
        SubtractRoundedProduct(v.r1, v.i1, ar, ai, last.r1, last.i1, mask);
    if constexpr (CHUNKS > 2)
        SubtractRoundedProduct(v.r2, v.i2, ar, ai, last.r2, last.i2, mask);
    if constexpr (CHUNKS > 3)
        SubtractRoundedProduct(v.r3, v.i3, ar, ai, last.r3, last.i3, mask);
}

template <uint32_t CHUNKS>
__simd_callee__ inline void SolveMultipleDiagonal(ChunkValues& v, __ubuf__ float* aa, uint16_t r, MaskReg& mask)
{
    RegTensor<float> ar, ai;
    DataCopy<float, LoadDist::DIST_BRC_B32>(ar, aa + r * (r + 1) + 2 * r);
    DataCopy<float, LoadDist::DIST_BRC_B32>(ai, aa + r * (r + 1) + 2 * r + 1);
    if constexpr (CHUNKS > 0)
        MultiplyRounded(v.r0, v.i0, ar, ai, mask);
    if constexpr (CHUNKS > 1)
        MultiplyRounded(v.r1, v.i1, ar, ai, mask);
    if constexpr (CHUNKS > 2)
        MultiplyRounded(v.r2, v.i2, ar, ai, mask);
    if constexpr (CHUNKS > 3)
        MultiplyRounded(v.r3, v.i3, ar, ai, mask);
}

template <uint32_t CHUNKS>
__simd_callee__ inline void RememberChunks(ChunkValues& last, ChunkValues& v)
{
    if constexpr (CHUNKS > 0) {
        Move(last.r0, v.r0);
        Move(last.i0, v.i0);
    }
    if constexpr (CHUNKS > 1) {
        Move(last.r1, v.r1);
        Move(last.i1, v.i1);
    }
    if constexpr (CHUNKS > 2) {
        Move(last.r2, v.r2);
        Move(last.i2, v.i2);
    }
    if constexpr (CHUNKS > 3) {
        Move(last.r3, v.r3);
        Move(last.i3, v.i3);
    }
}

template <uint32_t MAX_K, uint32_t WIDTH, uint32_t CHUNKS, uint32_t TAG>
__aicore__ inline void ComputeMultiple(__ubuf__ float* aa, __ubuf__ float* bb, uint16_t k, uint16_t first)
{
    __VEC_SCOPE__
    {
        auto mask = CreateMask<float>();
        ChunkValues v, last;
        if (first)
            LoadChunks<WIDTH, CHUNKS>(last, bb, first - 1);
        for (uint16_t r = first; r < k; ++r) {
            LoadChunks<WIDTH, CHUNKS>(v, bb, r);
            const uint16_t limit = r == 0 ? 0 : r - 1;
            for (uint16_t c = 0; c < limit; ++c)
                SubtractMultipleColumn<WIDTH, CHUNKS>(v, aa, bb, r, c, mask);
            if (r)
                SubtractMultipleLast<CHUNKS>(v, last, aa, r, mask);
            SolveMultipleDiagonal<CHUNKS>(v, aa, r, mask);
            RememberChunks<CHUNKS>(last, v);
            StoreChunks<WIDTH, CHUNKS>(v, bb, r, mask);
            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
        }
    }
}

template <uint32_t GROUPS>
__simd_callee__ inline void SubtractGroupedRow(
    __ubuf__ float* aa, __ubuf__ float* bb, uint16_t r, RegTensor<uint32_t>& index0, RegTensor<uint32_t>& index1,
    RegTensor<uint32_t>& imag0, RegTensor<uint32_t>& imag1, RegTensor<float>& v0r, RegTensor<float>& v0i,
    RegTensor<float>& v1r, RegTensor<float>& v1i, RegTensor<float>& last0r, RegTensor<float>& last0i,
    RegTensor<float>& last1r, RegTensor<float>& last1i, MaskReg& mask)
{
    constexpr uint32_t STRIDE = 264;
    RegTensor<float> packed, ar, ai, xr, xi, rr, ii, ri, ir, pr, pi;
    const uint16_t limit = r == 0 ? 0 : r - 1;
    for (uint16_t ic = 0; ic < limit; ++ic) {
        const uint16_t c = ic;
        DataCopy(packed, aa + (r * (r + 1) + 2 * c) * GROUPS);
        Gather(ar, packed, index0);
        Gather(ai, packed, imag0);
        DataCopy(xr, bb + c * STRIDE + 0);
        DataCopy(xi, bb + c * STRIDE + 128);
        ComplexProduct(pr, pi, ar, ai, xr, xi, mask);
        Sub(v0r, v0r, pr, mask);
        Sub(v0i, v0i, pi, mask);
        Gather(ar, packed, index1);
        Gather(ai, packed, imag1);
        DataCopy(xr, bb + c * STRIDE + 64);
        DataCopy(xi, bb + c * STRIDE + 192);
        ComplexProduct(pr, pi, ar, ai, xr, xi, mask);
        Sub(v1r, v1r, pr, mask);
        Sub(v1i, v1i, pi, mask);
    }
    if (r) {
        const uint16_t c = r - 1;
        DataCopy(packed, aa + (r * (r + 1) + 2 * c) * GROUPS);
        Gather(ar, packed, index0);
        Gather(ai, packed, imag0);
        ComplexProduct(pr, pi, ar, ai, last0r, last0i, mask);
        Sub(v0r, v0r, pr, mask);
        Sub(v0i, v0i, pi, mask);
        Gather(ar, packed, index1);
        Gather(ai, packed, imag1);
        ComplexProduct(pr, pi, ar, ai, last1r, last1i, mask);
        Sub(v1r, v1r, pr, mask);
        Sub(v1i, v1i, pi, mask);
    }
}

template <uint32_t GROUPS>
__simd_callee__ inline void SolveGroupedDiagonal(
    __ubuf__ float* aa, __ubuf__ float* bb, uint16_t r, RegTensor<uint32_t>& index0, RegTensor<uint32_t>& index1,
    RegTensor<uint32_t>& imag0, RegTensor<uint32_t>& imag1, RegTensor<float>& v0r, RegTensor<float>& v0i,
    RegTensor<float>& v1r, RegTensor<float>& v1i, RegTensor<float>& last0r, RegTensor<float>& last0i,
    RegTensor<float>& last1r, RegTensor<float>& last1i, MaskReg& mask)
{
    constexpr uint32_t STRIDE = 264;
    RegTensor<float> packed, ar, ai, xr, xi, rr, ii, ri, ir, pr, pi;
    const uint16_t c = r;
    DataCopy(packed, aa + (r * (r + 1) + 2 * c) * GROUPS);
    Gather(ar, packed, index0);
    Gather(ai, packed, imag0);
    Mul(rr, ar, v0r, mask);
    Mul(ii, ai, v0i, mask);
    Mul(ri, ar, v0i, mask);
    Mul(ir, ai, v0r, mask);
    Sub(v0r, rr, ii, mask);
    Add(v0i, ri, ir, mask);
    Gather(ar, packed, index1);
    Gather(ai, packed, imag1);
    Mul(rr, ar, v1r, mask);
    Mul(ii, ai, v1i, mask);
    Mul(ri, ar, v1i, mask);
    Mul(ir, ai, v1r, mask);
    Sub(v1r, rr, ii, mask);
    Add(v1i, ri, ir, mask);
}

template <uint32_t GROUPS, uint32_t TAG>
__aicore__ inline void ComputeGroupedPair(
    __ubuf__ float* aa, __ubuf__ float* bb, __ubuf__ uint32_t* indices, uint16_t k, uint16_t first)
{
    using namespace AscendC::MicroAPI;
    constexpr uint32_t STRIDE = 264;
    __VEC_SCOPE__
    {
        auto mask = CreateMask<float>();
        RegTensor<uint32_t> index0, index1, imag0, imag1;
        DataCopy(index0, indices);
        DataCopy(index1, indices + 64);
        Adds(imag0, index0, 1U, mask);
        Adds(imag1, index1, 1U, mask);
        RegTensor<float> v0r, v0i, v1r, v1i, last0r, last0i, last1r, last1i;
        if (first) {
            DataCopy(last0r, bb + (first - 1) * STRIDE);
            DataCopy(last0i, bb + (first - 1) * STRIDE + 128);
            DataCopy(last1r, bb + (first - 1) * STRIDE + 64);
            DataCopy(last1i, bb + (first - 1) * STRIDE + 192);
        }
        for (uint16_t r = first; r < k; ++r) {
            DataCopy(v0r, bb + r * STRIDE);
            DataCopy(v0i, bb + r * STRIDE + 128);
            DataCopy(v1r, bb + r * STRIDE + 64);
            DataCopy(v1i, bb + r * STRIDE + 192);

            SubtractGroupedRow<GROUPS>(
                aa, bb, r, index0, index1, imag0, imag1, v0r, v0i, v1r, v1i, last0r, last0i, last1r, last1i, mask);
            SolveGroupedDiagonal<GROUPS>(
                aa, bb, r, index0, index1, imag0, imag1, v0r, v0i, v1r, v1i, last0r, last0i, last1r, last1i, mask);
            Move(last0r, v0r);
            Move(last0i, v0i);
            DataCopy(bb + r * STRIDE + 0, v0r, mask);
            DataCopy(bb + r * STRIDE + 128, v0i, mask);
            Move(last1r, v1r);
            Move(last1i, v1i);
            DataCopy(bb + r * STRIDE + 64, v1r, mask);
            DataCopy(bb + r * STRIDE + 192, v1i, mask);
            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
        }
    }
}

template <uint32_t GROUPS, uint32_t WIDTH>
__simd_callee__ inline void SolveOrderedChunk(
    __ubuf__ float* aa, __ubuf__ float* bb, uint16_t k, uint16_t first, uint32_t laneStart, RegTensor<uint32_t>& index,
    RegTensor<uint32_t>& imagIndex, RegTensor<float>& lastReal, RegTensor<float>& lastImag, MaskReg& mask)
{
    constexpr uint32_t B_STRIDE = 2 * WIDTH + 8;
    RegTensor<float> vr, vi, xr, xi, ar, ai, pr, pi;
    for (uint16_t r = first; r < k; ++r) {
        uint32_t remain = 64;
        mask = UpdateMask<float>(remain);
        DataCopy(vr, bb + r * B_STRIDE + laneStart);
        DataCopy(vi, bb + r * B_STRIDE + WIDTH + laneStart);

        const uint16_t limit = r == 0 ? 0 : r - 1;
        for (uint16_t indexC = 0; indexC < limit; ++indexC) {
            const uint16_t c = indexC;
            DataCopy(xr, bb + c * B_STRIDE + laneStart);
            DataCopy(xi, bb + c * B_STRIDE + WIDTH + laneStart);
            LoadCoefficient<GROUPS>(ar, ai, aa, (r * (r + 1) + 2 * c), index, imagIndex, mask);
            ComplexProduct(pr, pi, ar, ai, xr, xi, mask);
            Sub(vr, vr, pr, mask);
            Sub(vi, vi, pi, mask);
        }
        if (r != 0) {
            LoadCoefficient<GROUPS>(ar, ai, aa, (r * (r + 1) + 2 * (r - 1)), index, imagIndex, mask);
            ComplexProduct(pr, pi, ar, ai, lastReal, lastImag, mask);
            Sub(vr, vr, pr, mask);
            Sub(vi, vi, pi, mask);
        }

        LoadCoefficient<GROUPS>(ar, ai, aa, (r * (r + 1) + 2 * r), index, imagIndex, mask);
        MultiplyRounded(vr, vi, ar, ai, mask);

        Move(lastReal, vr);
        Move(lastImag, vi);
        DataCopy(bb + r * B_STRIDE + laneStart, vr, mask);
        DataCopy(bb + r * B_STRIDE + WIDTH + laneStart, vi, mask);
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    }
}

template <uint32_t GROUPS, uint32_t WIDTH>
__aicore__ inline void ComputeOrderedChunks(
    __ubuf__ float* aa, __ubuf__ float* bb, __ubuf__ uint32_t* indices, uint16_t k, uint16_t first)
{
    using namespace AscendC::MicroAPI;
    constexpr uint32_t B_STRIDE = 2 * WIDTH + 8;
    const uint16_t chunks = WIDTH / 64;
    __VEC_SCOPE__
    {
        auto mask = CreateMask<float>();
        RegTensor<uint32_t> index, imagIndex;
        RegTensor<float> lastReal, lastImag;
        for (uint16_t chunk = 0; chunk < chunks; ++chunk) {
            const uint32_t laneStart = chunk * 64;
            DataCopy(index, indices + laneStart);
            Adds(imagIndex, index, 1U, mask);
            if (first != 0) {
                DataCopy(lastReal, bb + (first - 1) * B_STRIDE + laneStart);
                DataCopy(lastImag, bb + (first - 1) * B_STRIDE + WIDTH + laneStart);
            } else {
                Duplicate(lastReal, 0.0f, mask);
                Duplicate(lastImag, 0.0f, mask);
            }
            SolveOrderedChunk<GROUPS, WIDTH>(aa, bb, k, first, laneStart, index, imagIndex, lastReal, lastImag, mask);
        }
    }
}

template <uint32_t MAX_K, uint32_t GROUPS, uint32_t WIDTH, uint32_t TAG = 0>
__aicore__ inline void ComputeOrdered(
    __ubuf__ float* aa, __ubuf__ float* bb, __ubuf__ uint32_t* indices, uint16_t k, uint16_t first = 0,
    uint16_t activeRhs = WIDTH)
{
    if constexpr (GROUPS == 1 && (WIDTH == 128 || (WIDTH == 256 && MAX_K <= 40))) {
        if constexpr (WIDTH == 256) {
            if (activeRhs > 192) {
                ComputeMultiple<MAX_K, WIDTH, 4, TAG>(aa, bb, k, first);
                return;
            }
            if (activeRhs > 128) {
                ComputeMultiple<MAX_K, WIDTH, 3, TAG>(aa, bb, k, first);
                return;
            }
        }
        if (activeRhs > 64)
            ComputeMultiple<MAX_K, WIDTH, 2, TAG>(aa, bb, k, first);
        else
            ComputeMultiple<MAX_K, WIDTH, 1, TAG>(aa, bb, k, first);
    } else if constexpr (GROUPS >= 4 && WIDTH == 128) {
        ComputeGroupedPair<GROUPS, TAG>(aa, bb, indices, k, first);
    } else {
        ComputeOrderedChunks<GROUPS, WIDTH>(aa, bb, indices, k, first);
    }
}

template <uint32_t MAX_K, uint32_t GROUPS, uint32_t WIDTH, uint32_t TAG = 0>
__aicore__ inline void Compute(
    __ubuf__ float* aa, __ubuf__ float* bb, __ubuf__ uint32_t* indices, uint16_t k, uint16_t first = 0,
    uint16_t activeRhs = WIDTH)
{
    ComputeOrdered<MAX_K, GROUPS, WIDTH, TAG>(aa, bb, indices, k, first, activeRhs);
}

// Keep pivots in registers for the small square path. Coefficient offsets are
// compile-time constants in the packed lower triangle: row*(row+1) + 2*column.
template <uint32_t GROUPS, uint32_t OFFSET>
__simd_callee__ inline void FixedProduct(
    AscendC::Reg::RegTensor<float>& real, AscendC::Reg::RegTensor<float>& imag, __ubuf__ float* aa,
    AscendC::Reg::RegTensor<uint32_t>& index, AscendC::Reg::RegTensor<uint32_t>& imagIndex,
    AscendC::Reg::RegTensor<float>& xr, AscendC::Reg::RegTensor<float>& xi, AscendC::Reg::MaskReg& mask)
{
    using namespace AscendC::Reg;
    RegTensor<float> ar, ai, rr, ii, ri, ir;
    LoadCoefficient<GROUPS>(ar, ai, aa, OFFSET, index, imagIndex, mask);
    Mul(rr, ar, xr, mask);
    Mul(ii, ai, xi, mask);
    Mul(ri, ar, xi, mask);
    Mul(ir, ai, xr, mask);
    Sub(real, rr, ii, mask);
    Add(imag, ri, ir, mask);
}

template <uint32_t GROUPS, uint32_t WIDTH>
__simd_callee__ inline void SolveFixedRow0(
    __ubuf__ float* aa, __ubuf__ float* bb, uint32_t laneStart, RegTensor<uint32_t>& index,
    RegTensor<uint32_t>& imagIndex, RegTensor<float>& x0r, RegTensor<float>& x0i, MaskReg& mask)
{
    constexpr uint32_t B_STRIDE = 2 * WIDTH + 8;
    RegTensor<float> pr, pi;
    DataCopy(x0r, bb + 0 * B_STRIDE + laneStart);
    DataCopy(x0i, bb + 0 * B_STRIDE + WIDTH + laneStart);
    FixedProduct<GROUPS, 0>(pr, pi, aa, index, imagIndex, x0r, x0i, mask);
    Move(x0r, pr);
    Move(x0i, pi);
    DataCopy(bb + 0 * B_STRIDE + laneStart, x0r, mask);
    DataCopy(bb + 0 * B_STRIDE + WIDTH + laneStart, x0i, mask);
}

template <uint32_t GROUPS, uint32_t WIDTH>
__simd_callee__ inline void SolveFixedRow1(
    __ubuf__ float* aa, __ubuf__ float* bb, uint32_t laneStart, RegTensor<uint32_t>& index,
    RegTensor<uint32_t>& imagIndex, RegTensor<float>& x0r, RegTensor<float>& x0i, RegTensor<float>& x1r,
    RegTensor<float>& x1i, MaskReg& mask)
{
    constexpr uint32_t B_STRIDE = 2 * WIDTH + 8;
    RegTensor<float> pr, pi;
    DataCopy(x1r, bb + 1 * B_STRIDE + laneStart);
    DataCopy(x1i, bb + 1 * B_STRIDE + WIDTH + laneStart);
    FixedProduct<GROUPS, 2>(pr, pi, aa, index, imagIndex, x0r, x0i, mask);
    Sub(x1r, x1r, pr, mask);
    Sub(x1i, x1i, pi, mask);
    FixedProduct<GROUPS, 4>(pr, pi, aa, index, imagIndex, x1r, x1i, mask);
    Move(x1r, pr);
    Move(x1i, pi);
    DataCopy(bb + 1 * B_STRIDE + laneStart, x1r, mask);
    DataCopy(bb + 1 * B_STRIDE + WIDTH + laneStart, x1i, mask);
}

template <uint32_t GROUPS, uint32_t WIDTH>
__simd_callee__ inline void SolveFixedRow2(
    __ubuf__ float* aa, __ubuf__ float* bb, uint32_t laneStart, RegTensor<uint32_t>& index,
    RegTensor<uint32_t>& imagIndex, RegTensor<float>& x0r, RegTensor<float>& x0i, RegTensor<float>& x1r,
    RegTensor<float>& x1i, RegTensor<float>& x2r, RegTensor<float>& x2i, MaskReg& mask)
{
    constexpr uint32_t B_STRIDE = 2 * WIDTH + 8;
    RegTensor<float> pr, pi;
    DataCopy(x2r, bb + 2 * B_STRIDE + laneStart);
    DataCopy(x2i, bb + 2 * B_STRIDE + WIDTH + laneStart);
    FixedProduct<GROUPS, 6>(pr, pi, aa, index, imagIndex, x0r, x0i, mask);
    Sub(x2r, x2r, pr, mask);
    Sub(x2i, x2i, pi, mask);
    FixedProduct<GROUPS, 8>(pr, pi, aa, index, imagIndex, x1r, x1i, mask);
    Sub(x2r, x2r, pr, mask);
    Sub(x2i, x2i, pi, mask);
    FixedProduct<GROUPS, 10>(pr, pi, aa, index, imagIndex, x2r, x2i, mask);
    Move(x2r, pr);
    Move(x2i, pi);
    DataCopy(bb + 2 * B_STRIDE + laneStart, x2r, mask);
    DataCopy(bb + 2 * B_STRIDE + WIDTH + laneStart, x2i, mask);
}

template <uint32_t GROUPS, uint32_t WIDTH>
__simd_callee__ inline void SolveFixedRow3(
    __ubuf__ float* aa, __ubuf__ float* bb, uint32_t laneStart, RegTensor<uint32_t>& index,
    RegTensor<uint32_t>& imagIndex, RegTensor<float>& x0r, RegTensor<float>& x0i, RegTensor<float>& x1r,
    RegTensor<float>& x1i, RegTensor<float>& x2r, RegTensor<float>& x2i, RegTensor<float>& x3r, RegTensor<float>& x3i,
    MaskReg& mask)
{
    constexpr uint32_t B_STRIDE = 2 * WIDTH + 8;
    RegTensor<float> pr, pi;
    DataCopy(x3r, bb + 3 * B_STRIDE + laneStart);
    DataCopy(x3i, bb + 3 * B_STRIDE + WIDTH + laneStart);
    FixedProduct<GROUPS, 12>(pr, pi, aa, index, imagIndex, x0r, x0i, mask);
    Sub(x3r, x3r, pr, mask);
    Sub(x3i, x3i, pi, mask);
    FixedProduct<GROUPS, 14>(pr, pi, aa, index, imagIndex, x1r, x1i, mask);
    Sub(x3r, x3r, pr, mask);
    Sub(x3i, x3i, pi, mask);
    FixedProduct<GROUPS, 16>(pr, pi, aa, index, imagIndex, x2r, x2i, mask);
    Sub(x3r, x3r, pr, mask);
    Sub(x3i, x3i, pi, mask);
    FixedProduct<GROUPS, 18>(pr, pi, aa, index, imagIndex, x3r, x3i, mask);
    Move(x3r, pr);
    Move(x3i, pi);
    DataCopy(bb + 3 * B_STRIDE + laneStart, x3r, mask);
    DataCopy(bb + 3 * B_STRIDE + WIDTH + laneStart, x3i, mask);
}

template <uint32_t GROUPS, uint32_t WIDTH>
__simd_callee__ inline void SolveFixedRow4(
    __ubuf__ float* aa, __ubuf__ float* bb, uint32_t laneStart, RegTensor<uint32_t>& index,
    RegTensor<uint32_t>& imagIndex, RegTensor<float>& x0r, RegTensor<float>& x0i, RegTensor<float>& x1r,
    RegTensor<float>& x1i, RegTensor<float>& x2r, RegTensor<float>& x2i, RegTensor<float>& x3r, RegTensor<float>& x3i,
    RegTensor<float>& x4r, RegTensor<float>& x4i, MaskReg& mask)
{
    constexpr uint32_t B_STRIDE = 2 * WIDTH + 8;
    RegTensor<float> pr, pi;
    DataCopy(x4r, bb + 4 * B_STRIDE + laneStart);
    DataCopy(x4i, bb + 4 * B_STRIDE + WIDTH + laneStart);
    FixedProduct<GROUPS, 20>(pr, pi, aa, index, imagIndex, x0r, x0i, mask);
    Sub(x4r, x4r, pr, mask);
    Sub(x4i, x4i, pi, mask);
    FixedProduct<GROUPS, 22>(pr, pi, aa, index, imagIndex, x1r, x1i, mask);
    Sub(x4r, x4r, pr, mask);
    Sub(x4i, x4i, pi, mask);
    FixedProduct<GROUPS, 24>(pr, pi, aa, index, imagIndex, x2r, x2i, mask);
    Sub(x4r, x4r, pr, mask);
    Sub(x4i, x4i, pi, mask);
    FixedProduct<GROUPS, 26>(pr, pi, aa, index, imagIndex, x3r, x3i, mask);
    Sub(x4r, x4r, pr, mask);
    Sub(x4i, x4i, pi, mask);
    FixedProduct<GROUPS, 28>(pr, pi, aa, index, imagIndex, x4r, x4i, mask);
    Move(x4r, pr);
    Move(x4i, pi);
    DataCopy(bb + 4 * B_STRIDE + laneStart, x4r, mask);
    DataCopy(bb + 4 * B_STRIDE + WIDTH + laneStart, x4i, mask);
}

template <uint32_t GROUPS, uint32_t WIDTH>
__simd_callee__ inline void SolveFixedRow5(
    __ubuf__ float* aa, __ubuf__ float* bb, uint32_t laneStart, RegTensor<uint32_t>& index,
    RegTensor<uint32_t>& imagIndex, RegTensor<float>& x0r, RegTensor<float>& x0i, RegTensor<float>& x1r,
    RegTensor<float>& x1i, RegTensor<float>& x2r, RegTensor<float>& x2i, RegTensor<float>& x3r, RegTensor<float>& x3i,
    RegTensor<float>& x4r, RegTensor<float>& x4i, RegTensor<float>& x5r, RegTensor<float>& x5i, MaskReg& mask)
{
    constexpr uint32_t B_STRIDE = 2 * WIDTH + 8;
    RegTensor<float> pr, pi;
    DataCopy(x5r, bb + 5 * B_STRIDE + laneStart);
    DataCopy(x5i, bb + 5 * B_STRIDE + WIDTH + laneStart);
    FixedProduct<GROUPS, 30>(pr, pi, aa, index, imagIndex, x0r, x0i, mask);
    Sub(x5r, x5r, pr, mask);
    Sub(x5i, x5i, pi, mask);
    FixedProduct<GROUPS, 32>(pr, pi, aa, index, imagIndex, x1r, x1i, mask);
    Sub(x5r, x5r, pr, mask);
    Sub(x5i, x5i, pi, mask);
    FixedProduct<GROUPS, 34>(pr, pi, aa, index, imagIndex, x2r, x2i, mask);
    Sub(x5r, x5r, pr, mask);
    Sub(x5i, x5i, pi, mask);
    FixedProduct<GROUPS, 36>(pr, pi, aa, index, imagIndex, x3r, x3i, mask);
    Sub(x5r, x5r, pr, mask);
    Sub(x5i, x5i, pi, mask);
    FixedProduct<GROUPS, 38>(pr, pi, aa, index, imagIndex, x4r, x4i, mask);
    Sub(x5r, x5r, pr, mask);
    Sub(x5i, x5i, pi, mask);
    FixedProduct<GROUPS, 40>(pr, pi, aa, index, imagIndex, x5r, x5i, mask);
    Move(x5r, pr);
    Move(x5i, pi);
    DataCopy(bb + 5 * B_STRIDE + laneStart, x5r, mask);
    DataCopy(bb + 5 * B_STRIDE + WIDTH + laneStart, x5i, mask);
}

template <uint32_t GROUPS, uint32_t WIDTH>
__simd_callee__ inline void SolveFixedRow6(
    __ubuf__ float* aa, __ubuf__ float* bb, uint32_t laneStart, RegTensor<uint32_t>& index,
    RegTensor<uint32_t>& imagIndex, RegTensor<float>& x0r, RegTensor<float>& x0i, RegTensor<float>& x1r,
    RegTensor<float>& x1i, RegTensor<float>& x2r, RegTensor<float>& x2i, RegTensor<float>& x3r, RegTensor<float>& x3i,
    RegTensor<float>& x4r, RegTensor<float>& x4i, RegTensor<float>& x5r, RegTensor<float>& x5i, RegTensor<float>& x6r,
    RegTensor<float>& x6i, MaskReg& mask)
{
    constexpr uint32_t B_STRIDE = 2 * WIDTH + 8;
    RegTensor<float> pr, pi;
    DataCopy(x6r, bb + 6 * B_STRIDE + laneStart);
    DataCopy(x6i, bb + 6 * B_STRIDE + WIDTH + laneStart);
    FixedProduct<GROUPS, 42>(pr, pi, aa, index, imagIndex, x0r, x0i, mask);
    Sub(x6r, x6r, pr, mask);
    Sub(x6i, x6i, pi, mask);
    FixedProduct<GROUPS, 44>(pr, pi, aa, index, imagIndex, x1r, x1i, mask);
    Sub(x6r, x6r, pr, mask);
    Sub(x6i, x6i, pi, mask);
    FixedProduct<GROUPS, 46>(pr, pi, aa, index, imagIndex, x2r, x2i, mask);
    Sub(x6r, x6r, pr, mask);
    Sub(x6i, x6i, pi, mask);
    FixedProduct<GROUPS, 48>(pr, pi, aa, index, imagIndex, x3r, x3i, mask);
    Sub(x6r, x6r, pr, mask);
    Sub(x6i, x6i, pi, mask);
    FixedProduct<GROUPS, 50>(pr, pi, aa, index, imagIndex, x4r, x4i, mask);
    Sub(x6r, x6r, pr, mask);
    Sub(x6i, x6i, pi, mask);
    FixedProduct<GROUPS, 52>(pr, pi, aa, index, imagIndex, x5r, x5i, mask);
    Sub(x6r, x6r, pr, mask);
    Sub(x6i, x6i, pi, mask);
    FixedProduct<GROUPS, 54>(pr, pi, aa, index, imagIndex, x6r, x6i, mask);
    Move(x6r, pr);
    Move(x6i, pi);
    DataCopy(bb + 6 * B_STRIDE + laneStart, x6r, mask);
    DataCopy(bb + 6 * B_STRIDE + WIDTH + laneStart, x6i, mask);
}

template <uint32_t GROUPS, uint32_t WIDTH>
__simd_callee__ inline void SolveFixedRow7(
    __ubuf__ float* aa, __ubuf__ float* bb, uint32_t laneStart, RegTensor<uint32_t>& index,
    RegTensor<uint32_t>& imagIndex, RegTensor<float>& x0r, RegTensor<float>& x0i, RegTensor<float>& x1r,
    RegTensor<float>& x1i, RegTensor<float>& x2r, RegTensor<float>& x2i, RegTensor<float>& x3r, RegTensor<float>& x3i,
    RegTensor<float>& x4r, RegTensor<float>& x4i, RegTensor<float>& x5r, RegTensor<float>& x5i, RegTensor<float>& x6r,
    RegTensor<float>& x6i, RegTensor<float>& x7r, RegTensor<float>& x7i, MaskReg& mask)
{
    constexpr uint32_t B_STRIDE = 2 * WIDTH + 8;
    RegTensor<float> pr, pi;
    DataCopy(x7r, bb + 7 * B_STRIDE + laneStart);
    DataCopy(x7i, bb + 7 * B_STRIDE + WIDTH + laneStart);
    FixedProduct<GROUPS, 56>(pr, pi, aa, index, imagIndex, x0r, x0i, mask);
    Sub(x7r, x7r, pr, mask);
    Sub(x7i, x7i, pi, mask);
    FixedProduct<GROUPS, 58>(pr, pi, aa, index, imagIndex, x1r, x1i, mask);
    Sub(x7r, x7r, pr, mask);
    Sub(x7i, x7i, pi, mask);
    FixedProduct<GROUPS, 60>(pr, pi, aa, index, imagIndex, x2r, x2i, mask);
    Sub(x7r, x7r, pr, mask);
    Sub(x7i, x7i, pi, mask);
    FixedProduct<GROUPS, 62>(pr, pi, aa, index, imagIndex, x3r, x3i, mask);
    Sub(x7r, x7r, pr, mask);
    Sub(x7i, x7i, pi, mask);
    FixedProduct<GROUPS, 64>(pr, pi, aa, index, imagIndex, x4r, x4i, mask);
    Sub(x7r, x7r, pr, mask);
    Sub(x7i, x7i, pi, mask);
    FixedProduct<GROUPS, 66>(pr, pi, aa, index, imagIndex, x5r, x5i, mask);
    Sub(x7r, x7r, pr, mask);
    Sub(x7i, x7i, pi, mask);
    FixedProduct<GROUPS, 68>(pr, pi, aa, index, imagIndex, x6r, x6i, mask);
    Sub(x7r, x7r, pr, mask);
    Sub(x7i, x7i, pi, mask);
    FixedProduct<GROUPS, 70>(pr, pi, aa, index, imagIndex, x7r, x7i, mask);
    Move(x7r, pr);
    Move(x7i, pi);
    DataCopy(bb + 7 * B_STRIDE + laneStart, x7r, mask);
    DataCopy(bb + 7 * B_STRIDE + WIDTH + laneStart, x7i, mask);
}

template <uint32_t MAX_K, uint32_t GROUPS, uint32_t WIDTH, uint32_t TAG = 0>
__aicore__ inline void ComputeFixed8(__ubuf__ float* aa, __ubuf__ float* bb, __ubuf__ uint32_t* indices)
{
    using namespace AscendC::MicroAPI;
    constexpr uint32_t B_STRIDE = 2 * WIDTH + 8;
    const uint16_t chunks = WIDTH / 64;
    __VEC_SCOPE__
    {
        auto mask = CreateMask<float>();
        RegTensor<uint32_t> index, imagIndex;
        RegTensor<float> x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i, x4r, x4i, x5r, x5i, x6r, x6i, x7r, x7i;
        for (uint16_t chunk = 0; chunk < chunks; ++chunk) {
            const uint32_t laneStart = chunk * 64;
            DataCopy(index, indices + laneStart);
            Adds(imagIndex, index, 1U, mask);
            SolveFixedRow0<GROUPS, WIDTH>(aa, bb, laneStart, index, imagIndex, x0r, x0i, mask);
            SolveFixedRow1<GROUPS, WIDTH>(aa, bb, laneStart, index, imagIndex, x0r, x0i, x1r, x1i, mask);
            SolveFixedRow2<GROUPS, WIDTH>(aa, bb, laneStart, index, imagIndex, x0r, x0i, x1r, x1i, x2r, x2i, mask);
            SolveFixedRow3<GROUPS, WIDTH>(
                aa, bb, laneStart, index, imagIndex, x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i, mask);
            SolveFixedRow4<GROUPS, WIDTH>(
                aa, bb, laneStart, index, imagIndex, x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i, x4r, x4i, mask);
            SolveFixedRow5<GROUPS, WIDTH>(
                aa, bb, laneStart, index, imagIndex, x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i, x4r, x4i, x5r, x5i, mask);
            SolveFixedRow6<GROUPS, WIDTH>(
                aa, bb, laneStart, index, imagIndex, x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i, x4r, x4i, x5r, x5i, x6r,
                x6i, mask);
            SolveFixedRow7<GROUPS, WIDTH>(
                aa, bb, laneStart, index, imagIndex, x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i, x4r, x4i, x5r, x5i, x6r,
                x6i, x7r, x7i, mask);
        }
    }
}
template <
    uint32_t MAX_K, uint32_t GROUPS, uint32_t WIDTH, bool STORE, uint32_t FK, uint32_t FR,
    uint32_t TRANSFER_THREADS = 256>
__aicore__ inline void CallTransfer(
    GM_ADDR a, GM_ADDR b, CtrsmBatchedTilingData t, uint64_t base, __ubuf__ float* aa, __ubuf__ float* bb,
    __ubuf__ uint32_t* indices, __ubuf__ uint32_t* bad)
{
    if constexpr (FK > 0 || (GROUPS == 1 && MAX_K <= 64)) {
        if (t.side == ACLBLAS_SIDE_LEFT && t.trans == ACLBLAS_OP_N && t.diag == ACLBLAS_NON_UNIT) {
            asc_vf_call<Transfer<MAX_K, GROUPS, WIDTH, STORE, false, FK, FR, 1, TRANSFER_THREADS>>(
                dim3{TRANSFER_THREADS, 1, 1}, a, b, t, base, aa, bb, indices, bad, MAX_K);
            return;
        }
        if (t.side == ACLBLAS_SIDE_LEFT && t.trans == ACLBLAS_OP_C && t.diag == ACLBLAS_UNIT) {
            asc_vf_call<Transfer<MAX_K, GROUPS, WIDTH, STORE, false, FK, FR, 2, TRANSFER_THREADS>>(
                dim3{TRANSFER_THREADS, 1, 1}, a, b, t, base, aa, bb, indices, bad, MAX_K);
            return;
        }
        if (t.side == ACLBLAS_SIDE_RIGHT && t.trans == ACLBLAS_OP_N && t.diag == ACLBLAS_UNIT) {
            asc_vf_call<Transfer<MAX_K, GROUPS, WIDTH, STORE, false, FK, FR, 3, TRANSFER_THREADS>>(
                dim3{TRANSFER_THREADS, 1, 1}, a, b, t, base, aa, bb, indices, bad, MAX_K);
            return;
        }
        if (t.side == ACLBLAS_SIDE_RIGHT && t.trans == ACLBLAS_OP_C && t.diag == ACLBLAS_UNIT) {
            asc_vf_call<Transfer<MAX_K, GROUPS, WIDTH, STORE, false, FK, FR, 4, TRANSFER_THREADS>>(
                dim3{TRANSFER_THREADS, 1, 1}, a, b, t, base, aa, bb, indices, bad, MAX_K);
            return;
        }
    }
    asc_vf_call<Transfer<MAX_K, GROUPS, WIDTH, STORE, false, FK, FR, 0, TRANSFER_THREADS>>(
        dim3{TRANSFER_THREADS, 1, 1}, a, b, t, base, aa, bb, indices, bad, MAX_K);
}

// The complete short UNIT solve already follows the reference order, including
// nonfinite values. Pack it directly into input staging for the final DMA.
template <uint32_t MAX_K, uint32_t WIDTH, uint32_t FIXED_K, uint32_t T, bool LEFT>
__simt_vf__ __aicore__ LAUNCH_BOUND(T) inline void PackUnitOutput(
    __ubuf__ float* bb, __ubuf__ uint32_t* bad, uint32_t rows, uint32_t count, bool forward)
{
    constexpr uint32_t STRIDE = 2 * WIDTH + 8;
    const uint32_t k = FIXED_K ? FIXED_K : rows;
    auto raw = reinterpret_cast<__ubuf__ aclblasComplex*>(bad + (T + 8) / 8 * 8);
    if (threadIdx.x == 0)
        bad[T] = 0;
    if constexpr (LEFT) {
        for (uint32_t q = threadIdx.x; q < k * count; q += T) {
            const uint32_t step = q % k, rhs = q / k;
            const uint32_t row = forward ? step : k - 1 - step;
            raw[rhs * MAX_K + row].real = bb[step * STRIDE + rhs];
            raw[rhs * MAX_K + row].imag = bb[step * STRIDE + WIDTH + rhs];
        }
    } else {
        for (uint32_t q = threadIdx.x; q < k * WIDTH; q += T) {
            const uint32_t step = q / WIDTH, rhs = q % WIDTH;
            if (rhs >= count)
                continue;
            const uint32_t row = forward ? step : k - 1 - step;
            raw[row * WIDTH + rhs].real = bb[step * STRIDE + rhs];
            raw[row * WIDTH + rhs].imag = bb[step * STRIDE + WIDTH + rhs];
        }
    }
}

template <
    uint32_t MAX_K, uint32_t GROUPS, uint32_t WIDTH = 128, uint32_t FIXED_K = 0, uint32_t FIXED_RHS = 0,
    uint32_t TRANSFER_THREADS = 256>
__aicore__ inline void PrefetchSmall(
    GM_ADDR a, GM_ADDR b, CtrsmBatchedTilingData t, uint64_t base, uint64_t jobs, uint32_t originalK, uint32_t k,
    uint32_t rhs, __ubuf__ uint32_t* bad)
{
    using namespace AscendC;
    constexpr bool PREFETCH = Staging<MAX_K, GROUPS, WIDTH, TRANSFER_THREADS>::B;
    constexpr bool PREFETCH_A = Staging<MAX_K, GROUPS, WIDTH, TRANSFER_THREADS>::A;
    if constexpr (PREFETCH) {
        constexpr uint32_t RHS = WIDTH / GROUPS;
        const bool left = t.side == ACLBLAS_SIDE_LEFT;
        const bool upper = (t.uplo == ACLBLAS_UPPER) != (t.trans != ACLBLAS_OP_N);
        const bool forward = left ? !upper : upper;
        const uint32_t tiles = (rhs + RHS - 1) / RHS;
        auto raw = reinterpret_cast<__ubuf__ float*>(bad + (TRANSFER_THREADS + 8) / 8 * 8);
        for (uint32_t group = 0; group < GROUPS; ++group) {
            const uint64_t job = base + group;
            if (job >= jobs)
                continue;
            const uint32_t first = job % tiles * RHS;
            const uint32_t count = rhs - first < RHS ? rhs - first : RHS;
            const uint32_t row = forward ? 0 : originalK - k;
            if constexpr (PREFETCH_A) {
                auto ap = reinterpret_cast<__gm__ float*>(reinterpret_cast<__gm__ uint64_t*>(a)[job / tiles]);
                asc_copy_gm2ub_align(
                    raw + 2 * MAX_K * WIDTH + group * 2 * MAX_K * MAX_K, ap, k, k * 2 * sizeof(float), 0, 0, false, 0,
                    static_cast<uint64_t>(t.lda) * 2 * sizeof(float), MAX_K * 2 * sizeof(float));
            }
            auto bp = reinterpret_cast<__gm__ float*>(reinterpret_cast<__gm__ uint64_t*>(b)[job / tiles]);
            const uint64_t off =
                left ? row + static_cast<uint64_t>(first) * t.ldb : first + static_cast<uint64_t>(row) * t.ldb;
            asc_copy_gm2ub_align(
                raw + group * MAX_K * RHS * 2, bp + 2 * off, left ? count : k, (left ? k : count) * 2 * sizeof(float),
                0, 0, false, 0, static_cast<uint64_t>(t.ldb) * 2 * sizeof(float),
                (left ? MAX_K : RHS) * 2 * sizeof(float));
        }
        PipeBarrier<PIPE_ALL>();
    }
}

template <
    uint32_t MAX_K, uint32_t GROUPS, uint32_t WIDTH = 128, uint32_t FIXED_K = 0, uint32_t FIXED_RHS = 0,
    uint32_t TRANSFER_THREADS = 256>
__aicore__ inline void ComputeSmall(
    __ubuf__ float* aa, __ubuf__ float* bb, __ubuf__ uint32_t* indices, CtrsmBatchedTilingData t, uint64_t base,
    uint16_t k, uint32_t rhs, uint16_t activeRhs)
{
    if constexpr (MAX_K >= 16 && MAX_K <= 40 && GROUPS == 1 && WIDTH == 256) {
        // This shape is instantiated only by the UNIT translation unit.
        const uint32_t firstRhs = (base % ((rhs + WIDTH - 1) / WIDTH)) * WIDTH;
        CtrsmBatchedBlockedVector::Run<WIDTH>(aa, bb, k, t, firstRhs, activeRhs);
    } else if constexpr (MAX_K == 8) {
        if (k == 8)
            ComputeFixed8<MAX_K, GROUPS, WIDTH, FIXED_K ? 1 : 0>(aa, bb, indices);
        else
            Compute<MAX_K, GROUPS, WIDTH, FIXED_K ? 1 : 0>(aa, bb, indices, k, 0, activeRhs);
    } else {
        Compute<MAX_K, GROUPS, WIDTH, FIXED_K ? 1 : 0>(aa, bb, indices, k, 0, activeRhs);
    }
}

template <
    uint32_t MAX_K, uint32_t GROUPS, uint32_t WIDTH = 128, uint32_t FIXED_K = 0, uint32_t FIXED_RHS = 0,
    uint32_t TRANSFER_THREADS = 256>
__aicore__ inline void StoreSmall(
    GM_ADDR a, GM_ADDR b, CtrsmBatchedTilingData t, uint64_t base, __ubuf__ float* aa, __ubuf__ float* bb,
    __ubuf__ uint32_t* indices, __ubuf__ uint32_t* bad, uint16_t k, uint16_t activeRhs, bool left, bool forward)
{
    constexpr bool PREFETCH = Staging<MAX_K, GROUPS, WIDTH, TRANSFER_THREADS>::B;
    if constexpr (MAX_K >= 16 && MAX_K <= 40 && GROUPS == 1 && WIDTH == 256) {
        static_assert(PREFETCH);
        if (t.diag == ACLBLAS_UNIT) {
            if (left)
                asc_vf_call<PackUnitOutput<MAX_K, WIDTH, FIXED_K, TRANSFER_THREADS, true>>(
                    dim3{TRANSFER_THREADS, 1, 1}, bb, bad, k, activeRhs, forward);
            else
                asc_vf_call<PackUnitOutput<MAX_K, WIDTH, FIXED_K, TRANSFER_THREADS, false>>(
                    dim3{TRANSFER_THREADS, 1, 1}, bb, bad, k, activeRhs, forward);
        } else
            CallTransfer<MAX_K, GROUPS, WIDTH, true, FIXED_K, FIXED_RHS, TRANSFER_THREADS>(
                a, b, t, base, aa, bb, indices, bad);
    } else {
        CallTransfer<MAX_K, GROUPS, WIDTH, true, FIXED_K, FIXED_RHS, TRANSFER_THREADS>(
            a, b, t, base, aa, bb, indices, bad);
    }
}

template <
    uint32_t MAX_K, uint32_t GROUPS, uint32_t WIDTH = 128, uint32_t FIXED_K = 0, uint32_t FIXED_RHS = 0,
    uint32_t TRANSFER_THREADS = 256>
__aicore__ inline void StoreSmallDma(
    GM_ADDR a, GM_ADDR b, CtrsmBatchedTilingData t, uint64_t base, uint64_t jobs, uint32_t originalK, uint32_t k,
    uint32_t rhs, __ubuf__ uint32_t* bad, bool left)
{
    using namespace AscendC;
    constexpr bool PREFETCH = Staging<MAX_K, GROUPS, WIDTH, TRANSFER_THREADS>::B;
    if constexpr (PREFETCH) {
        if (reinterpret_cast<volatile __ubuf__ uint32_t*>(bad)[TRANSFER_THREADS] != 2) {
            constexpr uint32_t RHS = WIDTH / GROUPS;
            const uint32_t tiles = (rhs + RHS - 1) / RHS;
            auto raw = reinterpret_cast<__ubuf__ float*>(bad + (TRANSFER_THREADS + 8) / 8 * 8);
            for (uint32_t group = 0; group < GROUPS; ++group) {
                const uint64_t job = base + group;
                if (job >= jobs)
                    continue;
                const uint32_t first = job % tiles * RHS, count = rhs - first < RHS ? rhs - first : RHS;
                auto bp = reinterpret_cast<__gm__ float*>(reinterpret_cast<__gm__ uint64_t*>(b)[job / tiles]);
                const uint64_t off = left ? static_cast<uint64_t>(first) * t.ldb : first;
                asc_copy_ub2gm_align(
                    bp + 2 * off, raw + group * 2 * MAX_K * RHS, left ? count : k,
                    (left ? k : count) * 2 * sizeof(float), 0, static_cast<uint64_t>(t.ldb) * 2 * sizeof(float),
                    (left ? MAX_K : RHS) * 2 * sizeof(float));
            }
            PipeBarrier<PIPE_ALL>();
        }
    }
}

template <
    uint32_t MAX_K, uint32_t GROUPS, uint32_t WIDTH = 128, uint32_t FIXED_K = 0, uint32_t FIXED_RHS = 0,
    uint32_t TRANSFER_THREADS = 256>
__aicore__ inline void Run(GM_ADDR a, GM_ADDR b, CtrsmBatchedTilingData t, __ubuf__ float* storage)
{
    using namespace AscendC;
    constexpr uint32_t B_STRIDE = 2 * WIDTH + 8;
    constexpr uint32_t A_SIZE = MAX_K * (MAX_K + 1);
    constexpr bool PREFETCH = Staging<MAX_K, GROUPS, WIDTH, TRANSFER_THREADS>::B;
    constexpr bool PREFETCH_A = Staging<MAX_K, GROUPS, WIDTH, TRANSFER_THREADS>::A;
    static_assert(
        Staging<MAX_K, GROUPS, WIDTH, TRANSFER_THREADS>::BASE + (PREFETCH ? 2 * MAX_K * WIDTH : 0) +
            (PREFETCH_A ? 2 * GROUPS * MAX_K * MAX_K : 0) <=
        STORAGE_FLOATS);
    auto aa = storage, bb = aa + GROUPS * A_SIZE;
    auto indices = reinterpret_cast<__ubuf__ uint32_t*>(bb + MAX_K * B_STRIDE);
    auto bad = indices + WIDTH;
    const uint32_t originalK = FIXED_K ? FIXED_K : (t.side == ACLBLAS_SIDE_LEFT ? t.m : t.n);
    const uint32_t k = originalK < MAX_K ? originalK : MAX_K;
    const uint32_t rhs = FIXED_RHS ? FIXED_RHS : (t.side == ACLBLAS_SIDE_LEFT ? t.n : t.m);
    const uint64_t jobs = static_cast<uint64_t>((rhs + WIDTH / GROUPS - 1) / (WIDTH / GROUPS)) * t.batchCount;
    uint64_t firstBase = static_cast<uint64_t>(GetBlockIdx()) * GROUPS, lastBase = jobs,
             stepBase = static_cast<uint64_t>(GetBlockNum()) * GROUPS;
    if constexpr (GROUPS == 1) {
        const uint64_t quotient = jobs / GetBlockNum(), remainder = jobs % GetBlockNum();
        firstBase = GetBlockIdx() * quotient + (GetBlockIdx() < remainder ? GetBlockIdx() : remainder);
        lastBase = firstBase + quotient + (GetBlockIdx() < remainder ? 1 : 0);
        stepBase = 1;
    }
    for (uint64_t base = firstBase; base < lastBase; base += stepBase) {
        PrefetchSmall<MAX_K, GROUPS, WIDTH, FIXED_K, FIXED_RHS, TRANSFER_THREADS>(
            a, b, t, base, jobs, originalK, k, rhs, bad);
        CallTransfer<MAX_K, GROUPS, WIDTH, false, FIXED_K, FIXED_RHS, TRANSFER_THREADS>(
            a, b, t, base, aa, bb, indices, bad);
        PipeBarrier<PIPE_ALL>();
        const bool left = t.side == ACLBLAS_SIDE_LEFT;
        const bool upper = (t.uplo == ACLBLAS_UPPER) != (t.trans != ACLBLAS_OP_N);
        const bool forward = left ? !upper : upper;
        const uint32_t remainingRhs = GROUPS == 1 ? rhs - (base % ((rhs + WIDTH - 1) / WIDTH)) * WIDTH : WIDTH;
        const uint16_t activeRhs = remainingRhs < WIDTH ? remainingRhs : WIDTH;
        ComputeSmall<MAX_K, GROUPS, WIDTH, FIXED_K, FIXED_RHS, TRANSFER_THREADS>(
            aa, bb, indices, t, base, k, rhs, activeRhs);
        PipeBarrier<PIPE_ALL>();
        StoreSmall<MAX_K, GROUPS, WIDTH, FIXED_K, FIXED_RHS, TRANSFER_THREADS>(
            a, b, t, base, aa, bb, indices, bad, k, activeRhs, left, forward);
        PipeBarrier<PIPE_ALL>();
        StoreSmallDma<MAX_K, GROUPS, WIDTH, FIXED_K, FIXED_RHS, TRANSFER_THREADS>(
            a, b, t, base, jobs, originalK, k, rhs, bad, left);
    }
}
} // namespace CtrsmBatchedSmallVector

#ifndef CTRSM_BATCHED_SHARED_ONLY
// Grouped inverse of small diagonal panels.
namespace CtrsmBatchedGroupedInverse {
using namespace AscendC;
constexpr uint32_t K = 32, GROUPS = 4, WIDTH = 128, STRIDE = 264, OUT_STRIDE = 72;
constexpr uint32_t OUT_SIZE = 64 * OUT_STRIDE;

__simt_vf__ __aicore__ LAUNCH_BOUND(256) inline void Load(
    GM_ADDR workspace, uint32_t kp, uint32_t k, uint32_t batches, uint64_t base, __ubuf__ float* aa, __ubuf__ float* bb,
    __ubuf__ uint32_t* indices)
{
    const uint32_t panels = (k + K - 1) / K;
    const auto source = reinterpret_cast<__gm__ float*>(workspace);
    if (threadIdx.x < WIDTH)
        indices[threadIdx.x] = threadIdx.x / K * 2;
    for (uint32_t q = threadIdx.x; q < GROUPS * K; q += 256) {
        const uint32_t group = q / K, r = q % K;
        const uint64_t job = base + group;
        const uint32_t start = job % panels * K, size = k - start < K ? k - start : K;
        aclblasComplex value{1, 0};
        if (job < static_cast<uint64_t>(batches) * panels && r < size) {
            const uint64_t off = job / panels * 4ULL * kp * kp + 2ULL * (start + r) * (2 * kp) + 2 * (start + r);
            value = {-source[off], -source[off + 2 * kp]};
            const float inverse = __fdividef(1.0f, value.real * value.real + value.imag * value.imag);
            value = {value.real * inverse, -value.imag * inverse};
        }
        const uint32_t dst = (r * (r + 1) + 2 * r) * GROUPS + 2 * group;
        aa[dst] = value.real;
        aa[dst + 1] = value.imag;
    }
    for (uint32_t q = threadIdx.x; q < GROUPS * K * K; q += 256) {
        const uint32_t group = q / (K * K), r = q / K % K, c = q % K;
        const uint64_t job = base + group;
        const uint32_t start = job % panels * K;
        const uint32_t size = k - start < K ? k - start : K;
        const bool valid = job < static_cast<uint64_t>(batches) * panels;
        if (r > c) {
            aclblasComplex value{0, 0};
            if (valid && r < size && c < size) {
                const uint64_t off = job / panels * 4ULL * kp * kp + 2ULL * (start + r) * (2 * kp) + 2 * (start + c);
                value = {-source[off], -source[off + 2 * kp]};
            }
            const uint32_t dst = (r * (r + 1) + 2 * c) * GROUPS + 2 * group;
            aa[dst] = value.real;
            aa[dst + 1] = value.imag;
        }
        bb[r * STRIDE + group * K + c] = valid && r < size && r == c ? 1.0f : 0.0f;
        bb[r * STRIDE + WIDTH + group * K + c] = 0.0f;
    }
}

__simt_vf__ __aicore__ LAUNCH_BOUND(256) inline void Pack(__ubuf__ float* bb, __ubuf__ float* out)
{
    for (uint32_t q = threadIdx.x; q < GROUPS * K * K; q += 256) {
        const uint32_t group = q / (K * K), r = q / K % K, c = q % K;
        const float real = bb[r * STRIDE + group * K + c];
        const float imag = bb[r * STRIDE + WIDTH + group * K + c];
        const uint32_t offset = group * OUT_SIZE + 2 * r * OUT_STRIDE + 2 * c;
        out[offset] = real;
        out[offset + 1] = -imag;
        out[offset + OUT_STRIDE] = imag;
        out[offset + OUT_STRIDE + 1] = real;
    }
}

__aicore__ inline void Run(GM_ADDR workspace, uint32_t kp, uint32_t k, uint32_t batches, __ubuf__ float* storage)
{
    auto aa = storage;
    auto bb = aa + GROUPS * K * (K + 1);
    auto indices = reinterpret_cast<__ubuf__ uint32_t*>(bb + K * STRIDE);
    auto out = reinterpret_cast<__ubuf__ float*>(indices + WIDTH);
    static_assert(GROUPS * K * (K + 1) + K * STRIDE + WIDTH + GROUPS * OUT_SIZE <= 53248);
    const uint32_t panels = (k + K - 1) / K;
    const uint64_t jobs = static_cast<uint64_t>(batches) * panels;
    for (uint64_t base = static_cast<uint64_t>(GetBlockIdx()) * GROUPS; base < jobs;
         base += static_cast<uint64_t>(GetBlockNum()) * GROUPS) {
        asc_vf_call<Load>(dim3{256, 1, 1}, workspace, kp, k, batches, base, aa, bb, indices);
        PipeBarrier<PIPE_ALL>();
        CtrsmBatchedSmallVector::Compute<K, GROUPS, WIDTH, 2>(aa, bb, indices, K);
        PipeBarrier<PIPE_ALL>();
        asc_vf_call<Pack>(dim3{256, 1, 1}, bb, out);
        PipeBarrier<PIPE_ALL>();
        for (uint32_t group = 0; group < GROUPS && base + group < jobs; ++group) {
            const uint64_t job = base + group, batch = job / panels;
            const uint32_t start = job % panels * K, size = k - start < K ? k - start : K;
            auto dst = reinterpret_cast<__gm__ float*>(workspace) + batch * 4ULL * kp * kp + 2ULL * start * (2 * kp) +
                       2 * start;
            asc_copy_ub2gm_align(
                dst, out + group * OUT_SIZE, 2 * size, 2 * size * sizeof(float), 0, 2ULL * kp * sizeof(float),
                OUT_STRIDE * sizeof(float));
        }
        PipeBarrier<PIPE_ALL>();
    }
}
} // namespace CtrsmBatchedGroupedInverse

// Kernel entry points and small-vector dispatch.
namespace {
constexpr uint32_t THREADS = 128;
__simt_vf__ __aicore__ LAUNCH_BOUND(THREADS) inline void CtrsmBatchedSimt(
    GM_ADDR a, GM_ADDR b, CtrsmBatchedTilingData t, uint32_t block, uint32_t blocks)
{
    const uint64_t rhsCount = t.side == ACLBLAS_SIDE_LEFT ? t.n : t.m;
    const uint64_t work = rhsCount * t.batchCount;
    const bool zero = t.alpha.real == 0.0f && t.alpha.imag == 0.0f;
    for (uint64_t job = static_cast<uint64_t>(block) * THREADS + threadIdx.x; job < work;
         job += static_cast<uint64_t>(blocks) * THREADS) {
        const uint64_t batch = job / rhsCount;
        const auto bp = reinterpret_cast<__gm__ aclblasComplex*>(reinterpret_cast<__gm__ uint64_t*>(b)[batch]);
        const __gm__ aclblasComplex* ap = nullptr;
        if (!zero) {
            ap = reinterpret_cast<const __gm__ aclblasComplex*>(reinterpret_cast<__gm__ uint64_t*>(a)[batch]);
        }
        CtrsmBatched::SolveRhs(ap, bp, static_cast<uint32_t>(job % rhsCount), t);
    }
}
} // namespace

extern "C" __global__ __aicore__ void ctrsm_batched_kernel(GM_ADDR a, GM_ADDR b, CtrsmBatchedTilingData t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    asc_vf_call<CtrsmBatchedSimt>(dim3{THREADS, 1, 1}, a, b, t, AscendC::GetBlockIdx(), AscendC::GetBlockNum());
}

void ctrsm_batched_vector_do(uint8_t* a, uint8_t* b, const CtrsmBatchedTilingData& t, uint32_t blocks, void* stream);

void ctrsm_batched_unit_do(uint8_t* a, uint8_t* b, const CtrsmBatchedTilingData& t, uint32_t blocks, void* stream);

template <uint32_t MAX_K, uint32_t GROUPS, uint32_t WIDTH = 128, uint32_t TRANSFER_THREADS = 256>
__global__ __aicore__ void ctrsm_batched_small_vector_kernel(GM_ADDR a, GM_ADDR b, CtrsmBatchedTilingData t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    __ubuf__ float storage[CtrsmBatchedSmallVector::STORAGE_FLOATS];
    CtrsmBatchedSmallVector::Run<MAX_K, GROUPS, WIDTH, 0, 0, TRANSFER_THREADS>(a, b, t, storage);
}

template <uint32_t K, uint32_t GROUPS, uint32_t WIDTH, bool SQUARE, uint32_t TRANSFER_THREADS = 256>
__global__ __aicore__ void ctrsm_batched_fixed_vector_kernel(GM_ADDR a, GM_ADDR b, CtrsmBatchedTilingData t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    __ubuf__ float storage[CtrsmBatchedSmallVector::STORAGE_FLOATS];
    CtrsmBatchedSmallVector::Run<K, GROUPS, WIDTH, K, SQUARE ? K : 0, TRANSFER_THREADS>(a, b, t, storage);
}

// Only instantiate group sizes whose tiles fit in UB. The shared launch plan
// chooses the same width/group pair used by Host to calculate block count.
template <uint32_t MAX_K, uint32_t WIDTH, uint32_t GROUPS>
void LaunchGrouped(
    uint8_t* a, uint8_t* b, const CtrsmBatchedTilingData& t, uint32_t groups, uint32_t blocks, void* stream)
{
    if constexpr (GROUPS > 1) {
        if (groups < GROUPS) {
            LaunchGrouped<MAX_K, WIDTH, GROUPS / 2>(a, b, t, groups, blocks, stream);
            return;
        }
    }
    ctrsm_batched_small_vector_kernel<MAX_K, GROUPS, WIDTH><<<blocks, nullptr, stream>>>(a, b, t);
}

void LaunchFixedSquare(
    uint8_t* a, uint8_t* b, const CtrsmBatchedTilingData& t, const CtrsmBatchedVectorPlan& plan, uint32_t blocks,
    void* stream)
{
    const uint32_t k = t.side == ACLBLAS_SIDE_LEFT ? t.m : t.n;
    if (k == 8 && plan.width == 64) {
        ctrsm_batched_fixed_vector_kernel<8, 8, 64, true><<<blocks, nullptr, stream>>>(a, b, t);
        return;
    }
    if (k == 8) {
        ctrsm_batched_fixed_vector_kernel<8, 32, 256, true><<<blocks, nullptr, stream>>>(a, b, t);
        return;
    }
    if (k == 16 && plan.width == 64) {
        ctrsm_batched_fixed_vector_kernel<16, 4, 64, true><<<blocks, nullptr, stream>>>(a, b, t);
        return;
    }
    if (k == 16) {
        ctrsm_batched_fixed_vector_kernel<16, 8, 128, true><<<blocks, nullptr, stream>>>(a, b, t);
        return;
    }
    if (k == 32 && plan.width == 64) {
        ctrsm_batched_fixed_vector_kernel<32, 2, 64, true><<<blocks, nullptr, stream>>>(a, b, t);
        return;
    }
    if (k == 32) {
        ctrsm_batched_fixed_vector_kernel<32, 4, 128, true><<<blocks, nullptr, stream>>>(a, b, t);
        return;
    }
    ctrsm_batched_fixed_vector_kernel<64, 1, 64, true><<<blocks, nullptr, stream>>>(a, b, t);
}

void LaunchSmallVector(uint8_t* a, uint8_t* b, const CtrsmBatchedTilingData& t, uint32_t blocks, void* stream)
{
    const uint32_t k = t.side == ACLBLAS_SIDE_LEFT ? t.m : t.n;
    const auto plan = GetCtrsmBatchedVectorPlan(t);
    if (plan.fixedSquare) {
        LaunchFixedSquare(a, b, t, plan, blocks, stream);
        return;
    }
    if (k <= 8 && plan.width == 64) {
        LaunchGrouped<8, 64, 8>(a, b, t, plan.groups, blocks, stream);
    } else if (k <= 8) {
        LaunchGrouped<8, 256, 32>(a, b, t, plan.groups, blocks, stream);
    } else if (k <= 16) {
        LaunchGrouped<16, 128, 16>(a, b, t, plan.groups, blocks, stream);
    } else if (k <= 32) {
        LaunchGrouped<32, 128, 16>(a, b, t, plan.groups, blocks, stream);
    } else if (k <= 64) {
        LaunchGrouped<64, 128, 4>(a, b, t, plan.groups, blocks, stream);
    } else {
        LaunchGrouped<128, 128, 1>(a, b, t, plan.groups, blocks, stream);
    }
}

void ctrsm_batched_kernel_do(uint8_t* a, uint8_t* b, const CtrsmBatchedTilingData& t, uint32_t blocks, void* stream)
{
    if (t.diag == ACLBLAS_UNIT) {
        const uint32_t k = t.side == ACLBLAS_SIDE_LEFT ? t.m : t.n;
        if ((t.alpha.real != 0 || t.alpha.imag != 0) && k > 1 && k <= 2048)
            ctrsm_batched_unit_do(a, b, t, blocks, stream);
        else
            ctrsm_batched_kernel<<<blocks, nullptr, stream>>>(a, b, t);
        return;
    }
    const uint32_t k = t.side == ACLBLAS_SIDE_LEFT ? t.m : t.n;
    if (t.alpha.real == 0 && t.alpha.imag == 0)
        ctrsm_batched_kernel<<<blocks, nullptr, stream>>>(a, b, t);
    else if (k < 128)
        LaunchSmallVector(a, b, t, blocks, stream);
    else
        ctrsm_batched_vector_do(a, b, t, blocks, stream);
}

extern "C" __global__ __aicore__ void ctrsm_cube_inverse_grouped(
    GM_ADDR workspace, CtrsmBatchedTilingData t, uint32_t kp)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    __ubuf__ float storage[CtrsmBatchedSmallVector::STORAGE_FLOATS];
    CtrsmBatchedGroupedInverse::Run(workspace, kp, t.side == ACLBLAS_SIDE_LEFT ? t.m : t.n, t.batchCount, storage);
}

void ctrsm_batched_grouped_inverse_do(
    uint8_t* workspace, const CtrsmBatchedTilingData& t, uint32_t kp, uint32_t blocks, void* stream)
{
    ctrsm_cube_inverse_grouped<<<blocks, nullptr, stream>>>(workspace, t, kp);
}

#endif
#endif
