/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_operator.h"
#include "chemm_tiling_data.h"
#include "chemm_combine_common.h"

namespace {
constexpr uint32_t kFiniteInputMagnitudeBits = 0x49800000U;
constexpr float kUnitFastMagnitude = 16.0f;
} // namespace

// Device kernels for CHEMM.  The implementation covers SIMT fallback, packed
// cube launch, combine/recovery, and deterministic correction of outputs for
// which bounded FP32 accumulation alone is insufficient.

// Exceptional complex multiplication follows the __mulsc3 recovery semantics.
// Algorithm reference: LLVM compiler-rt/lib/builtins/mulsc3.c.
__simt_callee__ __aicore__ inline float ChemmSignedMagnitude(float magnitude, float sign)
{
    const uint32_t bits =
        (__builtin_bit_cast(uint32_t, magnitude) & 0x7fffffffU) | (__builtin_bit_cast(uint32_t, sign) & 0x80000000U);
    return __builtin_bit_cast(float, bits);
}
__simt_callee__ __aicore__ inline void ChemmNormalizeInfiniteOperand(float& real, float& imaginary)
{
    real = ChemmSignedMagnitude(__builtin_isinf(real) ? 1.f : 0.f, real);
    imaginary = ChemmSignedMagnitude(__builtin_isinf(imaginary) ? 1.f : 0.f, imaginary);
}

__simt_callee__ __aicore__ inline void ChemmReplaceNanOperand(float& real, float& imaginary)
{
    if (__builtin_isnan(real))
        real = ChemmSignedMagnitude(0.f, real);
    if (__builtin_isnan(imaginary))
        imaginary = ChemmSignedMagnitude(0.f, imaginary);
}

__simt_callee__ __aicore__ inline bool ChemmNeedsProductRecovery(
    float& ar, float& ai, float& br, float& bi, float rr, float ii, float ri, float ir)
{
    if (__builtin_isinf(ar) || __builtin_isinf(ai)) {
        ChemmNormalizeInfiniteOperand(ar, ai);
        return true;
    }
    if (__builtin_isinf(br) || __builtin_isinf(bi)) {
        ChemmNormalizeInfiniteOperand(br, bi);
        return true;
    }
    return __builtin_isinf(rr) || __builtin_isinf(ii) || __builtin_isinf(ri) || __builtin_isinf(ir);
}

__simt_callee__ __aicore__ inline float ChemmRecoverProduct(float ar, float ai, float br, float bi, bool imaginary)
{
    volatile float rr = ar * br;
    volatile float ii = ai * bi;
    volatile float ri = ar * bi;
    volatile float ir = ai * br;
    if (!ChemmNeedsProductRecovery(ar, ai, br, bi, rr, ii, ri, ir))
        return __builtin_nanf("");
    ChemmReplaceNanOperand(ar, ai);
    ChemmReplaceNanOperand(br, bi);
    volatile float x = imaginary ? ar * bi : ar * br;
    volatile float y = imaginary ? ai * br : ai * bi;
    return __builtin_inff() * (imaginary ? x + y : x - y);
}

// Preserve the separate FP32 rounding steps used by reference complex BLAS.
__simt_callee__ __aicore__ inline float ChemmMulSub(float a, float b, float c, float d)
{
#pragma clang fp contract(off)
    volatile float x = a * b;
    volatile float y = c * d;
    return x - y;
}
__simt_callee__ __aicore__ inline float ChemmMulAdd(float a, float b, float c, float d)
{
#pragma clang fp contract(off)
    volatile float x = a * b;
    volatile float y = c * d;
    return x + y;
}
__simt_callee__ __aicore__ inline float ChemmMulSubSafe(float a, float b, float c, float d)
{
    const float out = ChemmMulSub(a, b, c, d);
    if (__builtin_isnan(out)) {
        volatile float otherX = a * d;
        volatile float otherY = c * b;
        if (__builtin_isnan(otherX + otherY))
            return ChemmRecoverProduct(a, c, b, d, false);
    }
    return out;
}
__simt_callee__ __aicore__ inline float ChemmMulAddSafe(float a, float b, float c, float d)
{
    const float out = ChemmMulAdd(a, b, c, d);
    if (__builtin_isnan(out)) {
        volatile float otherX = a * d;
        volatile float otherY = c * b;
        if (__builtin_isnan(otherX - otherY))
            return ChemmRecoverProduct(a, c, d, b, true);
    }
    return out;
}

__simt_callee__ __aicore__ inline float ChemmBoundedMulSub(float a, float b, float c, float d)
{
#pragma clang fp contract(off)
    const float x = a * b;
    const float y = c * d;
    return x - y;
}
__simt_callee__ __aicore__ inline float ChemmBoundedMulAdd(float a, float b, float c, float d)
{
#pragma clang fp contract(off)
    const float x = a * b;
    const float y = c * d;
    return x + y;
}
template <bool Bounded>
__simt_callee__ __aicore__ inline float ChemmInnerSub(float a, float b, float c, float d)
{
    if constexpr (Bounded)
        return ChemmBoundedMulSub(a, b, c, d);
    else
        return ChemmMulSubSafe(a, b, c, d);
}
template <bool Bounded>
__simt_callee__ __aicore__ inline float ChemmInnerAdd(float a, float b, float c, float d)
{
    if constexpr (Bounded)
        return ChemmBoundedMulAdd(a, b, c, d);
    else
        return ChemmMulAddSafe(a, b, c, d);
}
template <typename Index, bool Packed, bool Bounded>
__simt_callee__ __aicore__ inline void ChemmStartRightElement(
    __gm__ const float* a, __gm__ const float* b, __gm__ float* c, __gm__ const float* w, Index n, Index lda, Index ldb,
    float ar0, float ai0, float br0, float bi0, Index row, Index col, Index pos, float& real, float& imag)
{
#pragma clang fp contract(off)
    const float diag = a[2 * (col * lda + col)];
    float xr = ar0 * diag;
    float xi = ai0 * diag;
    if constexpr (Packed) {
        xr = w[2 * (col * n + col)];
        xi = w[2 * (col * n + col) + 1];
    }
    const float yr = b[2 * (col * ldb + row)];
    const float yi = b[2 * (col * ldb + row) + 1];
    real = ChemmInnerSub<Bounded>(xr, yr, xi, yi);
    imag = ChemmInnerAdd<Bounded>(xr, yi, xi, yr);
    if (br0 != 0 || bi0 != 0) {
        const float cr = c[2 * pos];
        const float ci = c[2 * pos + 1];
        real = (ChemmMulSubSafe(br0, cr, bi0, ci)) + real;
        imag = (ChemmMulAddSafe(br0, ci, bi0, cr)) + imag;
    }
}

template <typename Index, bool Packed, bool Bounded>
__simt_callee__ __aicore__ inline void ChemmAccumulateRightTerm(
    __gm__ const float* a, __gm__ const float* b, __gm__ const float* w, Index n, Index lda, Index ldb, float ar0,
    float ai0, uint32_t upper, Index row, Index col, Index q, float& real, float& imag)
{
#pragma clang fp contract(off)
    const bool direct = upper ? q < col : q > col;
    const Index off = direct ? col * lda + q : q * lda + col;
    float av = a[2 * off];
    float ai = direct ? a[2 * off + 1] : -a[2 * off + 1];
    float xr = ChemmInnerSub<Bounded>(ar0, av, ai0, ai);
    float xi = ChemmInnerAdd<Bounded>(ar0, ai, ai0, av);
    if constexpr (Packed) {
        xr = w[2 * (col * n + q)];
        xi = w[2 * (col * n + q) + 1];
    }
    const float yr = b[2 * (q * ldb + row)];
    const float yi = b[2 * (q * ldb + row) + 1];
    real += ChemmInnerSub<Bounded>(xr, yr, xi, yi);
    imag += ChemmInnerAdd<Bounded>(xr, yi, xi, yr);
}

template <typename Index, bool Packed, bool Bounded>
__simt_callee__ __aicore__ inline void ChemmAccumulateLeftReducedTerm(
    __gm__ const float* a, __gm__ const float* b, __gm__ const float* w, Index m, Index lda, Index ldb, uint32_t upper,
    Index row, Index col, Index q, float& tr, float& ti)
{
#pragma clang fp contract(off)
    if constexpr (Packed) {
        if (upper ? q >= row : q <= row)
            return;
    }
    const Index off = row * lda + q;
    const Index bo = col * ldb + q;
    float xr = a[2 * off];
    float xi = -a[2 * off + 1];
    if constexpr (Packed) {
        xr = w[2 * (q * m + row)];
        xi = w[2 * (q * m + row) + 1];
    }
    const float yr = b[2 * bo];
    const float yi = b[2 * bo + 1];
    tr += ChemmInnerSub<Bounded>(yr, xr, yi, xi);
    ti += ChemmInnerAdd<Bounded>(yr, xi, yi, xr);
}

template <typename Index, bool Packed, bool Bounded>
__simt_callee__ __aicore__ inline void ChemmAccumulateLeftRemainingTerm(
    __gm__ const float* a, __gm__ const float* b, __gm__ const float* w, Index m, Index lda, Index ldb, float ar0,
    float ai0, uint32_t upper, Index row, Index col, Index q, float& real, float& imag)
{
#pragma clang fp contract(off)
    if constexpr (Packed) {
        if (upper ? q <= row : q >= row)
            return;
    }
    const Index bo = col * ldb + q;
    const Index off = q * lda + row;
    const float br = b[2 * bo];
    const float bi = b[2 * bo + 1];
    float av = a[2 * off];
    float ai = a[2 * off + 1];
    float pr = ChemmInnerSub<Bounded>(ar0, br, ai0, bi);
    float pi = ChemmInnerAdd<Bounded>(ar0, bi, ai0, br);
    if constexpr (Packed) {
        av = w[2 * (q * m + row)];
        ai = w[2 * (q * m + row) + 1];
        const Index wi = m * m + col * m + q;
        pr = w[2 * wi];
        pi = w[2 * wi + 1];
    }
    real += ChemmInnerSub<Bounded>(pr, av, pi, ai);
    imag += ChemmInnerAdd<Bounded>(pr, ai, pi, av);
}

template <typename Index, bool Packed, bool Bounded = false, bool Expanded = false>
__simt_callee__ __aicore__ inline void ChemmOrderedLeftElement(
    __gm__ const float* a, __gm__ const float* b, __gm__ float* c, __gm__ const float* w, Index m, Index lda, Index ldb,
    float ar0, float ai0, float br0, float bi0, uint32_t upper, Index row, Index col, Index pos)
{
#pragma clang fp contract(off)
    float tr = 0;
    float ti = 0;
    const Index rowBegin = (row / 32) * 32;
    const Index rowEnd = rowBegin + 32 < m ? rowBegin + 32 : m;
    const Index begin = upper ? 0 : (Packed ? rowBegin : row + 1);
    const Index end = upper ? (Packed ? rowEnd : row) : m;
#pragma unroll 4
    for (Index q = begin; q < end; ++q)
        ChemmAccumulateLeftReducedTerm<Index, Packed, Bounded>(a, b, w, m, lda, ldb, upper, row, col, q, tr, ti);
    const float yr = b[2 * (col * ldb + row)];
    const float yi = b[2 * (col * ldb + row) + 1];
    const float xr = ChemmInnerSub<Bounded>(ar0, yr, ai0, yi);
    const float xi = ChemmInnerAdd<Bounded>(ar0, yi, ai0, yr);
    const float diag = a[2 * (row * lda + row)];
    float real = xr * diag;
    float imag = xi * diag;
    if (br0 != 0 || bi0 != 0) {
        const float cr = c[2 * pos];
        const float ci = c[2 * pos + 1];
        real = (ChemmMulSubSafe(br0, cr, bi0, ci)) + real;
        imag = (ChemmMulAddSafe(br0, ci, bi0, cr)) + imag;
    }
    if constexpr (!Expanded) {
        if (ar0 == 1.f && ai0 == 0.f) {
            real += tr;
            imag += ti;
        } else {
            real += ChemmMulSub(ar0, tr, ai0, ti);
            imag += ChemmMulAdd(ar0, ti, ai0, tr);
        }
    } else {
        real += ChemmMulSub(ar0, tr, ai0, ti);
        imag += ChemmMulAdd(ar0, ti, ai0, tr);
    }
    const Index last = upper ? m : -1;
    const Index step = upper ? 1 : -1;
    const Index first = Packed ? (upper ? rowBegin : rowEnd - 1) : row + step;
#pragma unroll 4
    for (Index q = first; q != last; q += step)
        ChemmAccumulateLeftRemainingTerm<Index, Packed, Bounded>(
            a, b, w, m, lda, ldb, ar0, ai0, upper, row, col, q, real, imag);
    c[2 * pos] = real;
    c[2 * pos + 1] = imag;
}

template <typename Index, bool Packed, bool Bounded = false, bool Expanded = false>
__simt_callee__ __aicore__ inline void ChemmOrderedElementImpl(
    __gm__ const float* a, __gm__ const float* b, __gm__ float* c, __gm__ const float* w, Index m, Index n, Index lda,
    Index ldb, Index ldc, float ar0, float ai0, float br0, float bi0, uint32_t left, uint32_t upper, Index pos)
{
    const Index row = pos % ldc;
    const Index col = pos / ldc;
    if (!left && (ar0 != 0 || ai0 != 0)) {
        float real = 0;
        float imag = 0;
        ChemmStartRightElement<Index, Packed, Bounded>(
            a, b, c, w, n, lda, ldb, ar0, ai0, br0, bi0, row, col, pos, real, imag);
#pragma unroll 4
        for (Index q = 0; q < n; ++q) {
            if (q == col)
                continue;
            ChemmAccumulateRightTerm<Index, Packed, Bounded>(
                a, b, w, n, lda, ldb, ar0, ai0, upper, row, col, q, real, imag);
        }
        c[2 * pos] = real;
        c[2 * pos + 1] = imag;
        return;
    }
    if (left && (ar0 != 0 || ai0 != 0)) {
        ChemmOrderedLeftElement<Index, Packed, Bounded, Expanded>(
            a, b, c, w, m, lda, ldb, ar0, ai0, br0, bi0, upper, row, col, pos);
    }
}

template <typename Index, bool Packed, bool Bounded = false>
__simt_callee__ __aicore__ inline void ChemmOrderedElement(
    __gm__ const float* a, __gm__ const float* b, __gm__ float* c, __gm__ const float* w, Index m, Index n, Index lda,
    Index ldb, Index ldc, float ar0, float ai0, float br0, float bi0, uint32_t left, uint32_t upper, Index pos)
{
    ChemmOrderedElementImpl<Index, Packed, Bounded, false>(
        a, b, c, w, m, n, lda, ldb, ldc, ar0, ai0, br0, bi0, left, upper, pos);
}

template <typename Index, bool Packed, bool Bounded = false>
__simt_callee__ __aicore__ inline void ChemmOrderedElementExpanded(
    __gm__ const float* a, __gm__ const float* b, __gm__ float* c, __gm__ const float* w, Index m, Index n, Index lda,
    Index ldb, Index ldc, float ar0, float ai0, float br0, float bi0, uint32_t left, uint32_t upper, Index pos)
{
    ChemmOrderedElementImpl<Index, Packed, Bounded, true>(
        a, b, c, w, m, n, lda, ldb, ldc, ar0, ai0, br0, bi0, left, upper, pos);
}

__simt_callee__ __aicore__ inline void ChemmAccurateElement(
    __gm__ const float* a, __gm__ const float* b, __gm__ float* c, bool left, bool upper, int32_t m, int32_t n,
    int32_t lda, int32_t ldb, int32_t ldc, int32_t pos);

template <typename Index>
__simt_callee__ __aicore__ inline uint32_t ChemmMaximumFiniteBits(
    __gm__ const float* a, __gm__ const float* b, bool left, bool upper, Index row, Index col, Index m, Index n,
    Index lda, Index ldb)
{
    uint32_t maximumBits = 0;
    const Index k = left ? m : n;
    for (Index q = 0; q < k; ++q) {
        const Index x = left ? row : q;
        const Index y = left ? q : col;
        const bool direct = upper ? x <= y : x >= y;
        const Index off = direct ? y * lda + x : x * lda + y;
        const Index bp = left ? col * ldb + q : q * ldb + row;
        const float av = a[2 * off];
        const float ai = x == y ? 0.f : a[2 * off + 1];
        const float bv = b[2 * bp];
        const float bi = b[2 * bp + 1];
        const uint32_t values[4] = {
            __builtin_bit_cast(uint32_t, av) & 0x7fffffffU, __builtin_bit_cast(uint32_t, ai) & 0x7fffffffU,
            __builtin_bit_cast(uint32_t, bv) & 0x7fffffffU, __builtin_bit_cast(uint32_t, bi) & 0x7fffffffU};
#pragma unroll
        for (int component = 0; component < 4; ++component)
            maximumBits = maximumBits < values[component] ? values[component] : maximumBits;
    }
    return maximumBits;
}

template <typename Index>
__simt_callee__ __aicore__ inline void ChemmComputeGenericElement(
    __gm__ const float* a, __gm__ const float* b, bool left, bool upper, Index row, Index col, Index m, Index n,
    Index lda, Index ldb, float& sr, float& si)
{
    const Index k = left ? m : n;
    for (Index q = 0; q < k; ++q) {
        const Index x = left ? row : q;
        const Index y = left ? q : col;
        const bool packedDirect = upper ? x <= y : x >= y;
        const Index off = packedDirect ? y * lda + x : x * lda + y;
        const float ar = a[2 * off];
        const float ai = x == y ? 0.0f : (packedDirect ? a[2 * off + 1] : -a[2 * off + 1]);
        const Index bp = left ? col * ldb + q : q * ldb + row;
        const float br = b[2 * bp];
        const float bi = b[2 * bp + 1];
        sr += ar * br - ai * bi;
        si += ar * bi + ai * br;
    }
}

template <typename Index>
__simt_callee__ __aicore__ inline void ChemmWriteGenericElement(
    __gm__ float* c, Index pos, float sr, float si, float ar0, float ai0, float br0, float bi0)
{
    float cr = 0;
    float ci = 0;
    if (br0 != 0 || bi0 != 0) {
        cr = c[2 * pos];
        ci = c[2 * pos + 1];
    }
    c[2 * pos] = (ai0 == 0 ? ar0 * sr : ar0 * sr - ai0 * si) + (bi0 == 0 ? br0 * cr : br0 * cr - bi0 * ci);
    c[2 * pos + 1] = (ai0 == 0 ? ar0 * si : ar0 * si + ai0 * sr) + (bi0 == 0 ? br0 * ci : br0 * ci + bi0 * cr);
}

template <typename Index, uint32_t Threads, bool Packed = false, bool Unit = false>
__simt_vf__ __aicore__ LAUNCH_BOUND(Threads) inline void ChemmDirect(
    __gm__ const float* a, __gm__ const float* b, __gm__ float* c, __gm__ const float* w, Index m, Index n, Index lda,
    Index ldb, Index ldc, float ar0, float ai0, float br0, float bi0, uint32_t left, uint32_t upper, uint32_t core,
    uint32_t cores)
{
    if constexpr (Unit) {
        ar0 = 1.f;
        ai0 = 0.f;
        br0 = 0.f;
        bi0 = 0.f;
    }
    const Index end = ldc * n;
    for (Index pos = Index(core) * Threads + threadIdx.x; pos < end; pos += Index(cores) * Threads) {
        const Index row = pos % ldc;
        const Index col = pos / ldc;
        if (row >= m)
            continue;
        if (ar0 != 0 || ai0 != 0) {
            if constexpr (Unit) {
                const uint32_t maximumBits =
                    ChemmMaximumFiniteBits<Index>(a, b, left != 0, upper != 0, row, col, m, n, lda, ldb);
                if (maximumBits <= kFiniteInputMagnitudeBits) {
                    ChemmAccurateElement(
                        a, b, c, left != 0, upper != 0, int32_t(m), int32_t(n), int32_t(lda), int32_t(ldb),
                        int32_t(ldc), int32_t(pos));
                    continue;
                }
            }
            ChemmOrderedElementExpanded<Index, Packed>(
                a, b, c, w, m, n, lda, ldb, ldc, ar0, ai0, br0, bi0, left, upper, pos);
            continue;
        }
        float sr = 0;
        float si = 0;
        ChemmComputeGenericElement<Index>(a, b, left != 0, upper != 0, row, col, m, n, lda, ldb, sr, si);
        ChemmWriteGenericElement<Index>(c, pos, sr, si, ar0, ai0, br0, bi0);
    }
}
// Sixteen adjacent output columns reuse each B load. Dispatch requires ldc to be
// cache-line aligned so separate AIV cores cannot share an output cache line.
__simt_callee__ __aicore__ inline void ChemmInitializeRightTiledColumn(
    __gm__ const float* b, __gm__ float* c, __gm__ const float* w, uint32_t n, uint32_t ldb, uint32_t ldc, uint32_t row,
    uint32_t col, float br, float bi, float* rr, float* ii)
{
#pragma unroll
    for (uint32_t j = 0; j < 16; ++j) {
        rr[j] = 0;
        ii[j] = 0;
        if (col + j < n) {
            const uint32_t d = (col + j) * n + col + j;
            const uint32_t bo = (col + j) * ldb + row;
            const uint32_t co = (col + j) * ldc + row;
            const float ar = w[2 * d];
            const float ai = w[2 * d + 1];
            const float yr = b[2 * bo];
            const float yi = b[2 * bo + 1];
            rr[j] = ChemmMulSubSafe(ar, yr, ai, yi);
            ii[j] = ChemmMulAddSafe(ar, yi, ai, yr);
            if (br != 0 || bi != 0) {
                const float cr = c[2 * co];
                const float ci = c[2 * co + 1];
                rr[j] = (ChemmMulSubSafe(br, cr, bi, ci)) + rr[j];
                ii[j] = (ChemmMulAddSafe(br, ci, bi, cr)) + ii[j];
            }
        }
    }
}

__simt_callee__ __aicore__ inline void ChemmAccumulateRightTiledQ(
    __gm__ const float* b, __gm__ const float* w, uint32_t n, uint32_t ldb, uint32_t row, uint32_t col, uint32_t q,
    float* rr, float* ii)
{
    const uint32_t bo = q * ldb + row;
    const float yr = b[2 * bo];
    const float yi = b[2 * bo + 1];
#pragma unroll
    for (uint32_t j = 0; j < 16; ++j) {
        if (col + j < n && q != col + j) {
            const uint32_t off = (col + j) * n + q;
            const float ar = w[2 * off];
            const float ai = w[2 * off + 1];
            rr[j] += ChemmMulSubSafe(ar, yr, ai, yi);
            ii[j] += ChemmMulAddSafe(ar, yi, ai, yr);
        }
    }
}

__simt_callee__ __aicore__ inline void ChemmWriteRightTiledColumn(
    __gm__ float* c, uint32_t n, uint32_t ldc, uint32_t row, uint32_t col, const float* rr, const float* ii)
{
#pragma unroll
    for (uint32_t j = 0; j < 16; ++j) {
        if (col + j < n) {
            const uint32_t co = (col + j) * ldc + row;
            c[2 * co] = rr[j];
            c[2 * co + 1] = ii[j];
        }
    }
}

__simt_vf__ __aicore__ LAUNCH_BOUND(256) inline void ChemmRightTiled(
    __gm__ const float* b, __gm__ float* c, __gm__ const float* w, uint32_t m, uint32_t n, uint32_t ldb, uint32_t ldc,
    float br, float bi, uint32_t core, uint32_t cores)
{
#pragma clang fp contract(off)
    const uint32_t end = ((n + 15) / 16) * ldc;
    for (uint32_t p = core * 256 + threadIdx.x; p < end; p += cores * 256) {
        const uint32_t row = p % ldc;
        const uint32_t col = (p / ldc) * 16;
        if (row >= m)
            continue;
        float rr[16];
        float ii[16];
        ChemmInitializeRightTiledColumn(b, c, w, n, ldb, ldc, row, col, br, bi, rr, ii);
        for (uint32_t q = 0; q < n; ++q) {
            ChemmAccumulateRightTiledQ(b, w, n, ldb, row, col, q, rr, ii);
        }
        ChemmWriteRightTiledColumn(c, n, ldc, row, col, rr, ii);
    }
}

__simt_vf__ __aicore__ LAUNCH_BOUND(1024) inline void ChemmRightUnitSinglePass(
    __gm__ const float* a, __gm__ const float* b, __gm__ float* c, int32_t m, int32_t n, int32_t lda, int32_t ldb,
    int32_t ldc, uint32_t upper, uint32_t core, uint32_t cores)
{
#pragma clang fp contract(off)
    for (int32_t pos = int32_t(core) * 1024 + threadIdx.x; pos < ldc * n; pos += int32_t(cores) * 1024) {
        const int32_t row = pos % ldc;
        const int32_t col = pos / ldc;
        if (row >= m)
            continue;
        const float diag = a[2 * (col * lda + col)];
        float yr = b[2 * (col * ldb + row)];
        float yi = b[2 * (col * ldb + row) + 1];
        uint32_t maximum = __builtin_bit_cast(uint32_t, diag) & 0x7fffffffU;
        uint32_t bits = __builtin_bit_cast(uint32_t, yr) & 0x7fffffffU;
        maximum = maximum < bits ? bits : maximum;
        bits = __builtin_bit_cast(uint32_t, yi) & 0x7fffffffU;
        maximum = maximum < bits ? bits : maximum;
        const float xr = 1.f * diag;
        const float xi = 0.f * diag;
        float real = ChemmInnerSub<true>(xr, yr, xi, yi);
        float imag = ChemmInnerAdd<true>(xr, yi, xi, yr);
#pragma unroll 4
        for (int32_t q = 0; q < n; ++q) {
            if (q == col)
                continue;
            const bool direct = upper ? q < col : q > col;
            const int32_t off = direct ? col * lda + q : q * lda + col;
            const float ar = a[2 * off];
            const float ai = direct ? a[2 * off + 1] : -a[2 * off + 1];
            yr = b[2 * (q * ldb + row)];
            yi = b[2 * (q * ldb + row) + 1];
            const uint32_t values[4] = {
                __builtin_bit_cast(uint32_t, ar) & 0x7fffffffU, __builtin_bit_cast(uint32_t, ai) & 0x7fffffffU,
                __builtin_bit_cast(uint32_t, yr) & 0x7fffffffU, __builtin_bit_cast(uint32_t, yi) & 0x7fffffffU};
#pragma unroll
            for (int j = 0; j < 4; ++j)
                maximum = maximum < values[j] ? values[j] : maximum;
            const float pr = ChemmInnerSub<true>(1.f, ar, 0.f, ai);
            const float pi = ChemmInnerAdd<true>(1.f, ai, 0.f, ar);
            real += ChemmInnerSub<true>(pr, yr, pi, yi);
            imag += ChemmInnerAdd<true>(pr, yi, pi, yr);
        }
        if (maximum <= 0x49800000U) {
            c[2 * pos] = real;
            c[2 * pos + 1] = imag;
        } else
            ChemmOrderedElement<int32_t, false>(
                a, b, c, nullptr, m, n, lda, ldb, ldc, 1.f, 0.f, 0.f, 0.f, 0u, upper, pos);
    }
}

__simt_vf__ __aicore__ LAUNCH_BOUND(128) inline void ChemmLeftUnitSinglePass(
    __gm__ const float* a, __gm__ const float* b, __gm__ float* c, int32_t m, int32_t n, int32_t lda, int32_t ldb,
    int32_t ldc, uint32_t upper, uint32_t core, uint32_t cores)
{
#pragma clang fp contract(off)
    for (int32_t pos = int32_t(core) * 128 + threadIdx.x; pos < ldc * n; pos += int32_t(cores) * 128) {
        const int32_t row = pos % ldc;
        const int32_t col = pos / ldc;
        if (row >= m)
            continue;
        float real = 0.f;
        float imag = 0.f;
        uint32_t maximum = 0;
        for (int32_t q = 0; q < m; ++q) {
            const bool direct = upper ? (row <= q) : (row >= q);
            const int32_t off = direct ? q * lda + row : row * lda + q;
            const float ar = a[2 * off];
            const float ai = row == q ? 0.f : (direct ? a[2 * off + 1] : -a[2 * off + 1]);
            const int32_t bo = col * ldb + q;
            const float br = b[2 * bo];
            const float bi = b[2 * bo + 1];
            const uint32_t values[4] = {
                __builtin_bit_cast(uint32_t, ar) & 0x7fffffffU, __builtin_bit_cast(uint32_t, ai) & 0x7fffffffU,
                __builtin_bit_cast(uint32_t, br) & 0x7fffffffU, __builtin_bit_cast(uint32_t, bi) & 0x7fffffffU};
#pragma unroll
            for (int component = 0; component < 4; ++component)
                maximum = maximum < values[component] ? values[component] : maximum;
            real += ChemmBoundedMulSub(ar, br, ai, bi);
            imag += ChemmBoundedMulAdd(ar, bi, ai, br);
        }
        const uint32_t fastMagnitude = __builtin_bit_cast(uint32_t, kUnitFastMagnitude);
        if (maximum <= fastMagnitude) {
            c[2 * pos] = real;
            c[2 * pos + 1] = imag;
        } else if (maximum <= kFiniteInputMagnitudeBits) {
            ChemmAccurateElement(a, b, c, true, upper != 0, m, n, lda, ldb, ldc, pos);
        } else {
            ChemmOrderedElementExpanded<int32_t, false>(
                a, b, c, nullptr, m, n, lda, ldb, ldc, 1.f, 0.f, 0.f, 0.f, 1u, upper, pos);
        }
    }
}

__global__ __aicore__ void chemm_simt_small_unit_kernel(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR w, ChemmTiling t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (!t.left && t.ldc * t.n > 4096) {
        AscendC::Simt::VF_CALL<ChemmRightUnitSinglePass>(
            AscendC::Simt::Dim3(1024), (__gm__ const float*)a, (__gm__ const float*)b, (__gm__ float*)c, int32_t(t.m),
            int32_t(t.n), int32_t(t.lda), int32_t(t.ldb), int32_t(t.ldc), t.upper, AscendC::GetBlockIdx(),
            AscendC::GetBlockNum());
        return;
    }
    if (t.left) {
        AscendC::Simt::VF_CALL<ChemmLeftUnitSinglePass>(
            AscendC::Simt::Dim3(128), (__gm__ const float*)a, (__gm__ const float*)b, (__gm__ float*)c, int32_t(t.m),
            int32_t(t.n), int32_t(t.lda), int32_t(t.ldb), int32_t(t.ldc), t.upper, AscendC::GetBlockIdx(),
            AscendC::GetBlockNum());
        return;
    }
    if (t.ldc * t.n <= 4096) {
        AscendC::Simt::VF_CALL<ChemmDirect<int32_t, 64, false, true>>(
            AscendC::Simt::Dim3(64), (__gm__ const float*)a, (__gm__ const float*)b, (__gm__ float*)c,
            (__gm__ const float*)w, int32_t(t.m), int32_t(t.n), int32_t(t.lda), int32_t(t.ldb), int32_t(t.ldc), t.ar,
            t.ai, t.br, t.bi, t.left, t.upper, AscendC::GetBlockIdx(), AscendC::GetBlockNum());
        return;
    }
    AscendC::Simt::VF_CALL<ChemmDirect<int32_t, 512, false, true>>(
        AscendC::Simt::Dim3(512), (__gm__ const float*)a, (__gm__ const float*)b, (__gm__ float*)c,
        (__gm__ const float*)w, int32_t(t.m), int32_t(t.n), int32_t(t.lda), int32_t(t.ldb), int32_t(t.ldc), t.ar, t.ai,
        t.br, t.bi, t.left, t.upper, AscendC::GetBlockIdx(), AscendC::GetBlockNum());
}

__global__ __aicore__ void chemm_simt_general_kernel(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR w, ChemmTiling t, bool narrow)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (w != nullptr && !t.left && t.ldc % 64 == 0) {
        AscendC::Simt::VF_CALL<ChemmRightTiled>(
            AscendC::Simt::Dim3(256), (__gm__ const float*)b, (__gm__ float*)c, (__gm__ const float*)w, uint32_t(t.m),
            uint32_t(t.n), uint32_t(t.ldb), uint32_t(t.ldc), t.br, t.bi, AscendC::GetBlockIdx(),
            AscendC::GetBlockNum());
    } else if (w != nullptr) {
        AscendC::Simt::VF_CALL<ChemmDirect<int32_t, 512, true>>(
            AscendC::Simt::Dim3(512), (__gm__ const float*)a, (__gm__ const float*)b, (__gm__ float*)c,
            (__gm__ const float*)w, int32_t(t.m), int32_t(t.n), int32_t(t.lda), int32_t(t.ldb), int32_t(t.ldc), t.ar,
            t.ai, t.br, t.bi, t.left, t.upper, AscendC::GetBlockIdx(), AscendC::GetBlockNum());
    } else if (narrow && t.ldc * t.n <= 4096) {
        AscendC::Simt::VF_CALL<ChemmDirect<int32_t, 64>>(
            AscendC::Simt::Dim3(64), (__gm__ const float*)a, (__gm__ const float*)b, (__gm__ float*)c,
            (__gm__ const float*)w, int32_t(t.m), int32_t(t.n), int32_t(t.lda), int32_t(t.ldb), int32_t(t.ldc), t.ar,
            t.ai, t.br, t.bi, t.left, t.upper, AscendC::GetBlockIdx(), AscendC::GetBlockNum());
    } else if (narrow) {
        AscendC::Simt::VF_CALL<ChemmDirect<int32_t, 512>>(
            AscendC::Simt::Dim3(512), (__gm__ const float*)a, (__gm__ const float*)b, (__gm__ float*)c,
            (__gm__ const float*)w, int32_t(t.m), int32_t(t.n), int32_t(t.lda), int32_t(t.ldb), int32_t(t.ldc), t.ar,
            t.ai, t.br, t.bi, t.left, t.upper, AscendC::GetBlockIdx(), AscendC::GetBlockNum());
    } else {
        AscendC::Simt::VF_CALL<ChemmDirect<int64_t, 256>>(
            AscendC::Simt::Dim3(256), (__gm__ const float*)a, (__gm__ const float*)b, (__gm__ float*)c,
            (__gm__ const float*)w, t.m, t.n, t.lda, t.ldb, t.ldc, t.ar, t.ai, t.br, t.bi, t.left, t.upper,
            AscendC::GetBlockIdx(), AscendC::GetBlockNum());
    }
}

__simt_callee__ __aicore__ inline void ChemmErrorFreeProduct(
    float ar, float ai, float br, float bi, float& pr, float& pl, float& pi, float& pil)
{
#pragma clang fp contract(off)
    const float rr = ar * br;
    const float rca = 4097.f * ar;
    const float rcb = 4097.f * br;
    const float arh = rca - (rca - ar);
    const float arl = ar - arh;
    const float brh = rcb - (rcb - br);
    const float brl = br - brh;
    const float rrLo = ((arh * brh - rr) + arh * brl + arl * brh) + arl * brl;
    const float ii = ai * bi;
    const float ica = 4097.f * ai;
    const float icb = 4097.f * bi;
    const float aih = ica - (ica - ai);
    const float ail = ai - aih;
    const float bih = icb - (icb - bi);
    const float bil = bi - bih;
    const float iiLo = ((aih * bih - ii) + aih * bil + ail * bih) + ail * bil;
    const float ri = ar * bi;
    const float rica = 4097.f * ar;
    const float ricb = 4097.f * bi;
    const float rih = rica - (rica - ar);
    const float ril = ar - rih;
    const float rbh = ricb - (ricb - bi);
    const float rbl = bi - rbh;
    const float riLo = ((rih * rbh - ri) + rih * rbl + ril * rbh) + ril * rbl;
    const float ir = ai * br;
    const float irca = 4097.f * ai;
    const float ircb = 4097.f * br;
    const float irh = irca - (irca - ai);
    const float irl = ai - irh;
    const float ibh = ircb - (ircb - br);
    const float ibl = br - ibh;
    const float irLo = ((irh * ibh - ir) + irh * ibl + irl * ibh) + irl * ibl;
    pr = rr - ii;
    const float prBr = pr - rr;
    const float prErr = (rr - (pr - prBr)) - (ii + prBr);
    pl = prErr + rrLo - iiLo;
    pi = ri + ir;
    const float piBb = pi - ri;
    const float piErr = (ri - (pi - piBb)) + (ir - piBb);
    pil = piErr + riLo + irLo;
}

__simt_callee__ __aicore__ inline void ChemmCompensatedUpdate(
    float pr, float pl, float pi, float pil, float& real, float& realComp, float& imag, float& imagComp)
{
#pragma clang fp contract(off)
    const float realS = real + pr;
    const float realBb = realS - real;
    const float realErr = (real - (realS - realBb)) + (pr - realBb);
    const float realTail = realComp + realErr + pl;
    const float realN = realS + realTail;
    const float realNb = realN - realS;
    realComp = (realS - (realN - realNb)) + (realTail - realNb);
    real = realN;
    const float imagS = imag + pi;
    const float imagBb = imagS - imag;
    const float imagErr = (imag - (imagS - imagBb)) + (pi - imagBb);
    const float imagTail = imagComp + imagErr + pil;
    const float imagN = imagS + imagTail;
    const float imagNb = imagN - imagS;
    imagComp = (imagS - (imagN - imagNb)) + (imagTail - imagNb);
    imag = imagN;
}

__simt_callee__ __aicore__ inline void ChemmAccurateElement(
    __gm__ const float* a, __gm__ const float* b, __gm__ float* c, bool left, bool upper, int32_t m, int32_t n,
    int32_t lda, int32_t ldb, int32_t ldc, int32_t pos)
{
#pragma clang fp contract(off)
    const int32_t row = pos % ldc;
    const int32_t col = pos / ldc;
    const int32_t k = left ? m : n;
    float real = 0.f;
    float realComp = 0.f;
    float imag = 0.f;
    float imagComp = 0.f;
    for (int32_t q = 0; q < k; ++q) {
        const int32_t x = left ? row : q;
        const int32_t y = left ? q : col;
        const bool direct = upper ? (x <= y) : (x >= y);
        const int32_t off = direct ? y * lda + x : x * lda + y;
        const float ar = a[2 * off];
        const float ai = x == y ? 0.f : (direct ? a[2 * off + 1] : -a[2 * off + 1]);
        const int32_t bo = left ? col * ldb + q : q * ldb + row;
        const float br = b[2 * bo];
        const float bi = b[2 * bo + 1];
        float pr;
        float pl;
        float pi;
        float pil;
        ChemmErrorFreeProduct(ar, ai, br, bi, pr, pl, pi, pil);
        ChemmCompensatedUpdate(pr, pl, pi, pil, real, realComp, imag, imagComp);
    }
    c[2 * pos] = real + realComp;
    c[2 * pos + 1] = imag + imagComp;
}

void chemm_simt_do(const void* a, const void* b, void* c, void* w, ChemmTiling t, uint32_t cores, void* stream)
{
    const int64_t k = t.left ? t.m : t.n;
    const bool narrow = t.lda * k <= 0x3fffffff && t.ldb * t.n <= 0x3fffffff && t.ldc * t.n <= 0x3fffffff;
    const bool smallUnit = w == nullptr && narrow && (t.left ? t.m : t.n) <= 128 && t.ar == 1.f &&
                           __builtin_bit_cast(uint32_t, t.ai) == 0 && __builtin_bit_cast(uint32_t, t.br) == 0 &&
                           __builtin_bit_cast(uint32_t, t.bi) == 0;
    if (smallUnit)
        chemm_simt_small_unit_kernel<<<cores, nullptr, stream>>>((uint8_t*)a, (uint8_t*)b, (uint8_t*)c, (uint8_t*)w, t);
    else
        chemm_simt_general_kernel<<<cores, nullptr, stream>>>(
            (uint8_t*)a, (uint8_t*)b, (uint8_t*)c, (uint8_t*)w, t, narrow);
}

__simt_callee__ __aicore__ inline void ChemmLoadPackedElement(
    __gm__ const float* a, __gm__ const float* b, int64_t lda, int64_t ldb, uint32_t left, uint32_t upper, bool lhs,
    uint32_t row, uint32_t col, uint32_t rowLimit, uint32_t colLimit, float& re, float& im)
{
    re = 0;
    im = 0;
    if (row >= rowLimit || col >= colLimit)
        return;
    if (lhs != (left != 0U)) {
        const bool direct = upper ? col <= row : col >= row;
        const int64_t idx = direct ? int64_t(row) * lda + col : int64_t(col) * lda + row;
        re = a[2 * idx];
        im = row == col ? 0 : (direct ? a[2 * idx + 1] : -a[2 * idx + 1]);
    } else {
        const int64_t idx = int64_t(row) * ldb + col;
        re = b[2 * idx];
        im = b[2 * idx + 1];
    }
}

__simt_callee__ __aicore__ inline void ChemmWritePackedPlane(
    __gm__ float* w, bool lhs, uint32_t lsize, uint32_t rsize, uint32_t q, float re, float im)
{
    const int64_t base = lhs ? 0 : 3 * int64_t(lsize);
    const int64_t plane = lhs ? lsize : rsize;
    w[base + q] = re;
    w[base + plane + q] = im;
    w[base + 2 * plane + q] = re + im;
}

template <bool CheckBounded = true>
__simt_callee__ __aicore__ inline void ChemmPackElements(
    __gm__ const float* a, __gm__ const float* b, __gm__ float* w, int64_t m, int64_t n, int64_t lda, int64_t ldb,
    uint32_t left, uint32_t upper, uint32_t mp, uint32_t np, uint32_t kp, uint32_t core, uint32_t cores,
    __gm__ uint32_t* bounded)
{
    const int64_t k = left ? m : n;
    const uint32_t lsize = mp * kp;
    const uint32_t rsize = kp * np;
    for (uint32_t pos = core * 1024 + threadIdx.x; pos < lsize + rsize; pos += cores * 1024) {
        const bool lhs = pos < lsize;
        const uint32_t q = lhs ? pos : pos - lsize;
        const uint32_t row = q / (lhs ? kp : np);
        const uint32_t col = q % (lhs ? kp : np);
        float re = 0;
        float im = 0;
        ChemmLoadPackedElement(
            a, b, lda, ldb, left, upper, lhs, row, col, uint32_t(lhs ? n : (left ? m : n)),
            uint32_t(lhs ? (left ? m : n) : m), re, im);
        if constexpr (CheckBounded) {
            if (!__builtin_isfinite(re) || !__builtin_isfinite(im) || __builtin_fabsf(re) > 0x1p20f ||
                __builtin_fabsf(im) > 0x1p20f)
                atomicExch(bounded, uint32_t(0));
        }
        ChemmWritePackedPlane(w, lhs, lsize, rsize, q, re, im);
    }
}

__simt_vf__ __aicore__ LAUNCH_BOUND(1024) inline void ChemmPack(
    __gm__ const float* a, __gm__ const float* b, __gm__ float* w, int64_t m, int64_t n, int64_t lda, int64_t ldb,
    uint32_t left, uint32_t upper, uint32_t mp, uint32_t np, uint32_t kp, uint32_t core, uint32_t cores,
    __gm__ uint32_t* bounded)
{
    ChemmPackElements<true>(a, b, w, m, n, lda, ldb, left, upper, mp, np, kp, core, cores, bounded);
}

__global__ __aicore__ void chemm_pack_kernel(
    GM_ADDR a, GM_ADDR b, GM_ADDR w, ChemmTiling t, uint32_t mp, uint32_t np, uint32_t kp, GM_ADDR bounded)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::Simt::VF_CALL<ChemmPack>(
        AscendC::Simt::Dim3(1024), (__gm__ const float*)a, (__gm__ const float*)b, (__gm__ float*)w, t.m, t.n, t.lda,
        t.ldb, t.left, t.upper, mp, np, kp, AscendC::GetBlockIdx(), AscendC::GetBlockNum(), (__gm__ uint32_t*)bounded);
}
void chemm_pack_do(
    const void* a, const void* b, void* w, ChemmTiling t, uint32_t mp, uint32_t np, uint32_t kp, uint32_t cores,
    void* stream, void* bounded)
{
    chemm_pack_kernel<<<cores, nullptr, stream>>>(
        (uint8_t*)a, (uint8_t*)b, (uint8_t*)w, t, mp, np, kp, (uint8_t*)bounded);
}

__simt_vf__ __aicore__ LAUNCH_BOUND(1024) inline void ChemmPack3Fast(
    __gm__ const float* a, __gm__ const float* b, __gm__ float* w, int64_t m, int64_t n, int64_t lda, int64_t ldb,
    uint32_t left, uint32_t upper, uint32_t mp, uint32_t np, uint32_t kp, uint32_t core, uint32_t cores)
{
    ChemmPackElements<false>(a, b, w, m, n, lda, ldb, left, upper, mp, np, kp, core, cores, nullptr);
}

__global__ __aicore__ void chemm_pack3_fast_kernel(
    GM_ADDR a, GM_ADDR b, GM_ADDR w, ChemmTiling t, uint32_t mp, uint32_t np, uint32_t kp)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::Simt::VF_CALL<ChemmPack3Fast>(
        AscendC::Simt::Dim3(1024), (__gm__ const float*)a, (__gm__ const float*)b, (__gm__ float*)w, t.m, t.n, t.lda,
        t.ldb, t.left, t.upper, mp, np, kp, AscendC::GetBlockIdx(), AscendC::GetBlockNum());
}
void chemm_pack3_fast_do(
    const void* a, const void* b, void* w, ChemmTiling t, uint32_t mp, uint32_t np, uint32_t kp, uint32_t cores,
    void* stream)
{
    chemm_pack3_fast_kernel<<<cores, nullptr, stream>>>((uint8_t*)a, (uint8_t*)b, (uint8_t*)w, t, mp, np, kp);
}
__simt_vf__ __aicore__ LAUNCH_BOUND(64) inline void ChemmBoundReset(__gm__ uint32_t* bounded, uint32_t value)
{
    if (threadIdx.x == 0)
        *bounded = value;
}
__global__ __aicore__ void chemm_bound_reset_kernel(GM_ADDR bounded, uint32_t value)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::Simt::VF_CALL<ChemmBoundReset>(AscendC::Simt::Dim3(64), (__gm__ uint32_t*)bounded, value);
}
void chemm_bound_reset_do(void* bounded, uint32_t value, void* stream)
{
    chemm_bound_reset_kernel<<<1, nullptr, stream>>>((uint8_t*)bounded, value);
}
template <bool FOUR>
__simt_callee__ __aicore__ inline void ChemmAccumulateRiskPartials(
    __gm__ const float* w, int64_t plane, int64_t q, uint32_t splits, float re, float im, float& pr, float& pi,
    float& riskR, float& riskI)
{
    constexpr uint32_t products = FOUR ? 4 : 3;
    for (uint32_t part = 0; part < splits; ++part) {
        const int64_t off = int64_t(part) * products * plane + q;
        pr += w[off] - w[off + plane];
        if constexpr (FOUR)
            pi += w[off + 2 * plane] + w[off + 3 * plane];
        else
            pi += (w[off + 2 * plane] - w[off]) - w[off + plane];
        const float er = __builtin_fmaxf(__builtin_fabsf(pr), __builtin_fabsf(re - pr));
        const float ei = __builtin_fmaxf(__builtin_fabsf(pi), __builtin_fabsf(im - pi));
        riskR += er * er;
        riskI += ei * ei;
    }
}

template <bool FOUR>
__simt_callee__ __aicore__ inline bool ChemmRiskExceedsBudget(
    __gm__ const float* w, const float* sums, int64_t plane, int64_t q, uint32_t splits, int64_t kdim, float ar,
    float re, float im)
{
    float pr = 0;
    float pi = 0;
    float riskR = 0;
    float riskI = 0;
    ChemmAccumulateRiskPartials<FOUR>(w, plane, q, splits, re, im, pr, pi, riskR, riskI);
    constexpr float u = 0x1p-24f;
    const float kFactor = (float(kdim) == 2048.f) ? 1.f : 2.f;
    const float factorSquared = kFactor * kFactor * u * u * (float(kdim) / splits) * ar * ar;
    const float limitR = __builtin_fmaxf(.01f, 32 * u * __builtin_fabsf(ar * re));
    const float limitI = __builtin_fmaxf(.01f, 32 * u * __builtin_fabsf(ar * im));
    const float reconstruction =
        8 * u * __builtin_fabsf(ar) * (__builtin_fabsf(sums[0]) + __builtin_fabsf(sums[1]) + __builtin_fabsf(sums[2]));
    const float remainingR = __builtin_fmaxf(0.f, limitR - reconstruction);
    const float remainingI = __builtin_fmaxf(0.f, limitI - reconstruction);
    return factorSquared * riskR > remainingR * remainingR || factorSquared * riskI > remainingI * remainingI;
}

template <bool FOUR, bool SPLIT1 = false>
__simt_vf__ __aicore__ LAUNCH_BOUND(512) inline void ChemmCombine(
    __gm__ const float* a, __gm__ const float* b, __gm__ const float* w, __gm__ float* c, int64_t m, int64_t n,
    int64_t lda, int64_t ldb, int64_t ldc, uint32_t left, uint32_t upper, float ar, float ai, float br, float bi,
    uint32_t mp, uint32_t np, uint32_t splits, uint32_t core, uint32_t cores, __gm__ uint64_t* queue, uint64_t capacity)
{
    const ChemmCombineInput input{w, m, n, ldc, int64_t(mp) * np, mp, splits};
    for (int64_t pos = int64_t(core) * 512 + threadIdx.x; pos < ldc * n; pos += int64_t(cores) * 512) {
        const ChemmCombinedElement element = ChemmLoadCombinedElement<FOUR, SPLIT1>(input, pos);
        if (element.row >= m)
            continue;
        float re = element.re;
        float im = element.im;
        const int64_t kdim = left ? m : n;
        const bool needFallback = (kdim <= 2048 || kdim > 3075);
        if (needFallback &&
            ChemmRiskExceedsBudget<FOUR>(w, element.sums, input.plane, element.q, splits, kdim, ar, re, im)) {
            if (capacity) {
                if (ChemmQueueFallback(pos, core, capacity, queue))
                    continue;
            }
            const int64_t k = left ? m : n;
            if (lda * k <= 0x3fffffff && ldb * n <= 0x3fffffff && ldc * n <= 0x3fffffff)
                ChemmOrderedElement<int32_t, false>(
                    a, b, c, nullptr, int32_t(m), int32_t(n), int32_t(lda), int32_t(ldb), int32_t(ldc), ar, ai, br, bi,
                    left, upper, int32_t(pos));
            else
                ChemmOrderedElement<int64_t, false>(
                    a, b, c, nullptr, m, n, lda, ldb, ldc, ar, ai, br, bi, left, upper, pos);
            continue;
        }
        if (!__builtin_isfinite(re) || !__builtin_isfinite(im))
            ChemmRecoverNonFiniteElement(a, b, left != 0, upper != 0, m, n, lda, ldb, element.row, element.col, re, im);
        ChemmWriteCombinedElement(c, pos, re, im, ar, ai, br, bi);
    }
}
__global__ __aicore__ void chemm_combine_kernel(
    GM_ADDR a, GM_ADDR b, GM_ADDR w, GM_ADDR c, ChemmTiling t, uint32_t mp, uint32_t np, uint32_t splits,
    uint32_t products, GM_ADDR queue, uint64_t capacity)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (products == 4) {
        if (splits == 1) {
            AscendC::Simt::VF_CALL<ChemmCombine<true, true>>(
                AscendC::Simt::Dim3(512), (__gm__ const float*)a, (__gm__ const float*)b, (__gm__ const float*)w,
                (__gm__ float*)c, t.m, t.n, t.lda, t.ldb, t.ldc, t.left, t.upper, t.ar, t.ai, t.br, t.bi, mp, np,
                splits, AscendC::GetBlockIdx(), AscendC::GetBlockNum(), (__gm__ uint64_t*)queue, capacity);
        } else {
            AscendC::Simt::VF_CALL<ChemmCombine<true>>(
                AscendC::Simt::Dim3(512), (__gm__ const float*)a, (__gm__ const float*)b, (__gm__ const float*)w,
                (__gm__ float*)c, t.m, t.n, t.lda, t.ldb, t.ldc, t.left, t.upper, t.ar, t.ai, t.br, t.bi, mp, np,
                splits, AscendC::GetBlockIdx(), AscendC::GetBlockNum(), (__gm__ uint64_t*)queue, capacity);
        }
    } else {
        if (splits == 1) {
            AscendC::Simt::VF_CALL<ChemmCombine<false, true>>(
                AscendC::Simt::Dim3(512), (__gm__ const float*)a, (__gm__ const float*)b, (__gm__ const float*)w,
                (__gm__ float*)c, t.m, t.n, t.lda, t.ldb, t.ldc, t.left, t.upper, t.ar, t.ai, t.br, t.bi, mp, np,
                splits, AscendC::GetBlockIdx(), AscendC::GetBlockNum(), (__gm__ uint64_t*)queue, capacity);
        } else {
            AscendC::Simt::VF_CALL<ChemmCombine<false>>(
                AscendC::Simt::Dim3(512), (__gm__ const float*)a, (__gm__ const float*)b, (__gm__ const float*)w,
                (__gm__ float*)c, t.m, t.n, t.lda, t.ldb, t.ldc, t.left, t.upper, t.ar, t.ai, t.br, t.bi, mp, np,
                splits, AscendC::GetBlockIdx(), AscendC::GetBlockNum(), (__gm__ uint64_t*)queue, capacity);
        }
    }
}
void chemm_combine_do(
    const void* a, const void* b, void* w, void* c, ChemmTiling t, uint32_t mp, uint32_t np, uint32_t splits,
    uint32_t products, uint32_t cores, void* stream, void* queue, uint64_t capacity)
{
    chemm_combine_kernel<<<cores, nullptr, stream>>>(
        (uint8_t*)a, (uint8_t*)b, (uint8_t*)w, (uint8_t*)c, t, mp, np, splits, products, (uint8_t*)queue, capacity);
}

// Each core owns a queue; fallback keeps the original C ownership to avoid cross-core cache-line writes.
__simt_vf__ __aicore__
    LAUNCH_BOUND(512) inline void ChemmQueueReset(__gm__ uint64_t* queue, uint64_t capacity, uint32_t core)
{
    if (threadIdx.x == 0)
        queue[uint64_t(core) * (capacity + 64)] = 0;
}

__simt_vf__ __aicore__ LAUNCH_BOUND(64) inline void ChemmCombinedReset(
    __gm__ uint32_t* bounded, __gm__ uint64_t* queue, uint64_t capacity, uint32_t value, uint32_t core)
{
    if (threadIdx.x != 0)
        return;
    if (core == 0)
        *bounded = value;
    if (capacity)
        queue[uint64_t(core) * (capacity + 64)] = 0;
}

struct ChemmCompensatedValue {
    float value = 0.f;
    float correction = 0.f;
};

__simt_callee__ __aicore__ inline void ChemmAddCompensated(ChemmCompensatedValue& sum, float product)
{
    const float next = sum.value + product;
    const float correction = (__builtin_fabsf(sum.value) >= __builtin_fabsf(product)) ? (sum.value - next) + product :
                                                                                        (product - next) + sum.value;
    sum.correction += correction;
    sum.value = next;
}

__simt_callee__ __aicore__ inline void ChemmLoadQueueUnitProduct(
    __gm__ const float* a, __gm__ const float* b, uint32_t left, uint32_t upper, int32_t pos, int32_t q, int32_t lda,
    int32_t ldb, int32_t ldc, float& productReal, float& productImag)
{
    const int32_t x = left ? pos % ldc : q;
    const int32_t y = left ? q : pos / ldc;
    const bool direct = upper ? (x <= y) : (x >= y);
    const int32_t off = direct ? y * lda + x : x * lda + y;
    const float av = a[2 * off];
    const float ai = x == y ? 0.f : (direct ? a[2 * off + 1] : -a[2 * off + 1]);
    const int32_t bo = left ? (pos / ldc) * ldb + q : q * ldb + (pos % ldc);
    const float br = b[2 * bo];
    const float bi = b[2 * bo + 1];
    productReal = av * br - ai * bi;
    productImag = av * bi + ai * br;
}

template <uint32_t LEFT, uint32_t UPPER>
__simt_callee__ __aicore__ inline void ChemmQueueUnitFallback(
    __gm__ const float* a, __gm__ const float* b, __gm__ float* c, __gm__ uint64_t* local, uint64_t count, int64_t m,
    int64_t n, int64_t lda, int64_t ldb, int64_t ldc)
{
#pragma clang fp contract(off)
    constexpr uint32_t left = LEFT;
    constexpr uint32_t upper = UPPER;
    const uint32_t lane = threadIdx.x & 31U;
    const uint32_t warp = threadIdx.x >> 5;
    constexpr uint32_t warpsPerBlock = 512U / 32U;
    const int32_t mi = static_cast<int32_t>(m);
    const int32_t ni = static_cast<int32_t>(n);
    const int32_t ldai = static_cast<int32_t>(lda);
    const int32_t ldbi = static_cast<int32_t>(ldb);
    const int32_t ldci = static_cast<int32_t>(ldc);
    const int32_t k = left ? mi : ni;
    for (uint64_t item = warp; item < count; item += warpsPerBlock) {
        const int32_t pos = static_cast<int32_t>(local[64 + item]);
        ChemmCompensatedValue real;
        ChemmCompensatedValue imag;
        for (int32_t q = static_cast<int32_t>(lane); q < k; q += 32) {
            float productReal = 0.f;
            float productImag = 0.f;
            ChemmLoadQueueUnitProduct(a, b, left, upper, pos, q, ldai, ldbi, ldci, productReal, productImag);
            ChemmAddCompensated(real, productReal);
            ChemmAddCompensated(imag, productImag);
        }
        float realPart = real.value + real.correction;
        float imagPart = imag.value + imag.correction;
        for (uint32_t distance = 16; distance > 0; distance >>= 1) {
            realPart += __shfl_down(realPart, distance);
            imagPart += __shfl_down(imagPart, distance);
        }
        if (lane == 0) {
            c[2 * pos] = realPart;
            c[2 * pos + 1] = imagPart;
        }
    }
}

template <uint32_t LEFT, uint32_t UPPER>
__simt_callee__ __aicore__ inline void ChemmQueueGeneralFallback(
    __gm__ const float* a, __gm__ const float* b, __gm__ float* c, __gm__ uint64_t* local, uint64_t count, int64_t m,
    int64_t n, int64_t lda, int64_t ldb, int64_t ldc, float ar, float ai, float br, float bi,
    __gm__ const uint32_t* bounded)
{
    constexpr uint32_t left = LEFT;
    constexpr uint32_t upper = UPPER;
    for (uint64_t i = threadIdx.x; i < count; i += 512) {
        const int64_t pos = static_cast<int64_t>(local[64 + i]);
        if (*bounded != 0) {
            ChemmOrderedElementExpanded<int32_t, false, true>(
                a, b, c, nullptr, static_cast<int32_t>(m), static_cast<int32_t>(n), static_cast<int32_t>(lda),
                static_cast<int32_t>(ldb), static_cast<int32_t>(ldc), ar, ai, br, bi, left, upper,
                static_cast<int32_t>(pos));
        } else if (lda * (left ? m : n) <= 0x3fffffff && ldb * n <= 0x3fffffff && ldc * n <= 0x3fffffff) {
            ChemmOrderedElement<int32_t, false>(
                a, b, c, nullptr, static_cast<int32_t>(m), static_cast<int32_t>(n), static_cast<int32_t>(lda),
                static_cast<int32_t>(ldb), static_cast<int32_t>(ldc), ar, ai, br, bi, left, upper,
                static_cast<int32_t>(pos));
        } else {
            ChemmOrderedElement<int64_t, false>(
                a, b, c, nullptr, m, n, lda, ldb, ldc, ar, ai, br, bi, left, upper, pos);
        }
    }
}

template <uint32_t LEFT, uint32_t UPPER, bool UNIT>
__simt_vf__ __aicore__ LAUNCH_BOUND(512) inline void ChemmQueueFallback(
    __gm__ const float* a, __gm__ const float* b, __gm__ float* c, __gm__ uint64_t* queue, uint64_t capacity, int64_t m,
    int64_t n, int64_t lda, int64_t ldb, int64_t ldc, float ar, float ai, float br, float bi, uint32_t leftInput,
    uint32_t upperInput, uint32_t core, __gm__ const uint32_t* bounded, __gm__ const float* packed)
{
    (void)leftInput;
    (void)upperInput;
    (void)packed;
    __gm__ uint64_t* local = queue + uint64_t(core) * (capacity + 64);
    const uint64_t count = local[0];
    if constexpr (UNIT) {
        ChemmQueueUnitFallback<LEFT, UPPER>(a, b, c, local, count, m, n, lda, ldb, ldc);
        return;
    }
    ChemmQueueGeneralFallback<LEFT, UPPER>(a, b, c, local, count, m, n, lda, ldb, ldc, ar, ai, br, bi, bounded);
}
__global__ __aicore__ void chemm_queue_reset_kernel(GM_ADDR queue, uint64_t capacity)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::Simt::VF_CALL<ChemmQueueReset>(
        AscendC::Simt::Dim3(512), (__gm__ uint64_t*)queue, capacity, AscendC::GetBlockIdx());
}

__global__ __aicore__ void chemm_combined_reset_kernel(
    GM_ADDR bounded, GM_ADDR queue, uint64_t capacity, uint32_t value)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::Simt::VF_CALL<ChemmCombinedReset>(
        AscendC::Simt::Dim3(64), (__gm__ uint32_t*)bounded, (__gm__ uint64_t*)queue, capacity, value,
        AscendC::GetBlockIdx());
}
template <uint32_t LEFT, uint32_t UPPER>
__global__ __aicore__ void chemm_queue_fallback_kernel(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR queue, uint64_t capacity, ChemmTiling t, GM_ADDR bounded, GM_ADDR packed)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    const bool unit = t.ar == 1.f && __builtin_bit_cast(uint32_t, t.ai) == 0 &&
                      __builtin_bit_cast(uint32_t, t.br) == 0 && __builtin_bit_cast(uint32_t, t.bi) == 0;
    if (unit) {
        AscendC::Simt::VF_CALL<ChemmQueueFallback<LEFT, UPPER, true>>(
            AscendC::Simt::Dim3(512), (__gm__ const float*)a, (__gm__ const float*)b, (__gm__ float*)c,
            (__gm__ uint64_t*)queue, capacity, t.m, t.n, t.lda, t.ldb, t.ldc, t.ar, t.ai, t.br, t.bi, t.left, t.upper,
            AscendC::GetBlockIdx(), (__gm__ const uint32_t*)bounded, (__gm__ const float*)packed);
    } else {
        AscendC::Simt::VF_CALL<ChemmQueueFallback<LEFT, UPPER, false>>(
            AscendC::Simt::Dim3(512), (__gm__ const float*)a, (__gm__ const float*)b, (__gm__ float*)c,
            (__gm__ uint64_t*)queue, capacity, t.m, t.n, t.lda, t.ldb, t.ldc, t.ar, t.ai, t.br, t.bi, t.left, t.upper,
            AscendC::GetBlockIdx(), (__gm__ const uint32_t*)bounded, (__gm__ const float*)packed);
    }
}
void chemm_queue_reset_do(void* queue, uint64_t capacity, uint32_t cores, void* stream)
{
    chemm_queue_reset_kernel<<<cores, nullptr, stream>>>((uint8_t*)queue, capacity);
}

void chemm_combined_reset_do(
    void* bounded, void* queue, uint64_t capacity, uint32_t value, uint32_t cores, void* stream)
{
    chemm_combined_reset_kernel<<<cores, nullptr, stream>>>((uint8_t*)bounded, (uint8_t*)queue, capacity, value);
}
void chemm_queue_fallback_do(
    const void* a, const void* b, void* c, void* queue, uint64_t capacity, ChemmTiling t, uint32_t cores, void* stream,
    void* bounded, const void* packed)
{
    if (t.left == 1 && t.upper == 1) {
        chemm_queue_fallback_kernel<1, 1><<<cores, nullptr, stream>>>(
            (uint8_t*)a, (uint8_t*)b, (uint8_t*)c, (uint8_t*)queue, capacity, t, (uint8_t*)bounded, (uint8_t*)packed);
        return;
    }
    if (t.left == 1 && t.upper == 0) {
        chemm_queue_fallback_kernel<1, 0><<<cores, nullptr, stream>>>(
            (uint8_t*)a, (uint8_t*)b, (uint8_t*)c, (uint8_t*)queue, capacity, t, (uint8_t*)bounded, (uint8_t*)packed);
        return;
    }
    if (t.left == 0 && t.upper == 1) {
        chemm_queue_fallback_kernel<0, 1><<<cores, nullptr, stream>>>(
            (uint8_t*)a, (uint8_t*)b, (uint8_t*)c, (uint8_t*)queue, capacity, t, (uint8_t*)bounded, (uint8_t*)packed);
        return;
    }
    chemm_queue_fallback_kernel<0, 0><<<cores, nullptr, stream>>>(
        (uint8_t*)a, (uint8_t*)b, (uint8_t*)c, (uint8_t*)queue, capacity, t, (uint8_t*)bounded, (uint8_t*)packed);
}

// Materialize the selected Hermitian triangle once. The ordered SIMT loop
// then reads contiguous rows and reuses alpha scaling across all outputs.
__simt_vf__ __aicore__ LAUNCH_BOUND(512) inline void ChemmOrderedPack(
    __gm__ const float* a, __gm__ const float* b, __gm__ float* w, uint32_t m, uint32_t n, int64_t lda, int64_t ldb,
    float ar, float ai, uint32_t left, uint32_t upper, uint32_t core, uint32_t cores)
{
#pragma clang fp contract(off)
    const uint32_t k = left ? m : n;
    const uint32_t asize = k * k;
    const uint32_t total = asize + (left ? m * n : 0);
    for (uint32_t p = core * 512 + threadIdx.x; p < total; p += cores * 512) {
        float xr;
        float xi;
        if (p < asize) {
            const uint32_t row = p % k;
            const uint32_t col = p / k;
            const bool direct = upper ? row <= col : row >= col;
            const int64_t off = direct ? int64_t(col) * lda + row : int64_t(row) * lda + col;
            const float re = a[2 * off];
            const float im = row == col ? 0 : (direct ? a[2 * off + 1] : -a[2 * off + 1]);
            if (left) {
                xr = re;
                xi = im;
            } else if (row == col) {
                xr = ar * re;
                xi = ai * re;
            } else {
                xr = ChemmMulSub(ar, re, ai, im);
                xi = ChemmMulAdd(ar, im, ai, re);
            }
        } else {
            const uint32_t q = p - asize;
            const uint32_t row = q % m;
            const uint32_t col = q / m;
            const int64_t off = int64_t(col) * ldb + row;
            const float re = b[2 * off];
            const float im = b[2 * off + 1];
            xr = ChemmMulSub(ar, re, ai, im);
            xi = ChemmMulAdd(ar, im, ai, re);
        }
        w[2 * p] = xr;
        w[2 * p + 1] = xi;
    }
}
__global__ __aicore__ void chemm_ordered_pack_kernel(GM_ADDR a, GM_ADDR b, GM_ADDR w, ChemmTiling t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::Simt::VF_CALL<ChemmOrderedPack>(
        AscendC::Simt::Dim3(512), (__gm__ const float*)a, (__gm__ const float*)b, (__gm__ float*)w, uint32_t(t.m),
        uint32_t(t.n), t.lda, t.ldb, t.ar, t.ai, t.left, t.upper, AscendC::GetBlockIdx(), AscendC::GetBlockNum());
}
void chemm_ordered_pack_do(const void* a, const void* b, void* w, ChemmTiling t, uint32_t cores, void* stream)
{
    chemm_ordered_pack_kernel<<<cores, nullptr, stream>>>((uint8_t*)a, (uint8_t*)b, (uint8_t*)w, t);
}
