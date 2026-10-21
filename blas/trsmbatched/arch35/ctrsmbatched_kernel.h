/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once
#include <cstdint>
#include <algorithm>
#include "cann_ops_blas_common.h"

struct CtrsmBatchedTilingData {
    uint32_t m, n, lda, ldb, batchCount;
    int32_t side, uplo, trans, diag;
    aclblasComplex alpha;
};

// Host and kernel launchers share one description of each small-vector tile.
// WIDTH lanes are divided into GROUPS independent matrices/RHS tiles.
struct CtrsmBatchedVectorPlan {
    uint32_t groups;
    uint32_t width;
    bool fixedSquare;
};

inline CtrsmBatchedVectorPlan GetCtrsmBatchedVectorPlan(const CtrsmBatchedTilingData& t)
{
    const uint32_t k = t.side == ACLBLAS_SIDE_LEFT ? t.m : t.n;
    const uint32_t rhs = t.side == ACLBLAS_SIDE_LEFT ? t.n : t.m;
    if (t.m == t.n) {
        switch (k) {
            case 8:
                return t.batchCount <= 256 ? CtrsmBatchedVectorPlan{8, 64, true} :
                                             CtrsmBatchedVectorPlan{32, 256, true};
            case 16:
                return t.batchCount <= 64 ? CtrsmBatchedVectorPlan{4, 64, true} : CtrsmBatchedVectorPlan{8, 128, true};
            case 32:
                return t.batchCount <= 256 ? CtrsmBatchedVectorPlan{2, 64, true} : CtrsmBatchedVectorPlan{4, 128, true};
            case 64:
                return {1, 64, true};
            default:
                break;
        }
    }
    const uint32_t width = k <= 8 ? (t.batchCount <= 256 ? 64 : 256) : 128;
    const uint32_t minRhs = k <= 32 ? 8 : (k <= 64 ? 32 : 128);
    uint32_t groups = width / minRhs;
    while (groups > 1 && rhs > width / groups) {
        groups /= 2;
    }
    return {groups, width, false};
}

inline bool ValidCtrsmBatchedModes(
    aclblasSideMode_t side, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag)
{
    return (side == ACLBLAS_SIDE_LEFT || side == ACLBLAS_SIDE_RIGHT) &&
           (uplo == ACLBLAS_UPPER || uplo == ACLBLAS_LOWER) &&
           (trans == ACLBLAS_OP_N || trans == ACLBLAS_OP_T || trans == ACLBLAS_OP_C) &&
           (diag == ACLBLAS_NON_UNIT || diag == ACLBLAS_UNIT);
}

inline aclblasStatus_t ValidateCtrsmBatched(
    aclblasSideMode_t side, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, int m, int n,
    const aclblasComplex* alpha, const void* a, int lda, const void* b, int ldb, int batchCount)
{
    if (!ValidCtrsmBatchedModes(side, uplo, trans, diag) || m < 0 || n < 0 || batchCount < 1 || lda < 0 || ldb < 0 ||
        alpha == nullptr || b == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    // Empty matrices are no-ops; their unused leading dimensions may be zero.
    // Validate metadata and required host pointers, but do not access A or B.
    if (m == 0 || n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (lda < std::max(1, side == ACLBLAS_SIDE_LEFT ? m : n) || ldb < std::max(1, m) ||
        ((alpha->real != 0.0f || alpha->imag != 0.0f) && a == nullptr)) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

#if defined(CTRSM_BATCHED_DEVICE) || defined(CTRSM_BATCHED_PORTABLE_CHECK)
#include <cmath>
#ifndef CTRSM_BATCHED_PORTABLE_CHECK
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "c_api/asc_simd.h"
#endif

// Ordered scalar arithmetic and exceptional-value fallback.
#ifdef __clang__
#define CTRSM_INLINE __attribute__((always_inline))
#else
#define CTRSM_INLINE
#endif

namespace CtrsmBatched {
// The SIMT backend does not lower fcopysign. Keep sign extraction as integer ops.
__simt_callee__ CTRSM_INLINE inline float CopySign(float magnitude, float sign)
{
    volatile uint32_t signBits = __builtin_bit_cast(uint32_t, sign) & 0x80000000U;
    const uint32_t bits = (__builtin_bit_cast(uint32_t, magnitude) & 0x7fffffffU) | signBits;
    return __builtin_bit_cast(float, bits);
}

// Preserve the sign while replacing infinities with one and other values with zero.
__simt_callee__ CTRSM_INLINE inline float InfinityUnit(float value)
{
    return CopySign(__builtin_isinf(value) ? 1.0f : 0.0f, value);
}

__simt_callee__ CTRSM_INLINE inline float ClearNan(float value)
{
    return __builtin_isnan(value) ? CopySign(0.0f, value) : value;
}

// Fused complex products retain one rounded cross term and fuse the other
// product with the final add/subtract. Operand order matters for the imaginary
// component; rank updates pass the solved value first, dot updates pass A first.
__simt_callee__ CTRSM_INLINE inline aclblasComplex FusedMul(aclblasComplex a, aclblasComplex b)
{
#ifdef __clang__
#pragma clang fp contract(off)
#endif
    const float ii = a.imag * b.imag, ir = a.imag * b.real;
    return {fmaf(a.real, b.real, -ii), fmaf(a.real, b.imag, ir)};
}

__simt_callee__ CTRSM_INLINE inline aclblasComplex Mul(aclblasComplex a, aclblasComplex b) { return FusedMul(a, b); }

// Cooperlake's GEMM beta kernel scales B using separately rounded products.
// FMA belongs to the block solve/update, not this initial alpha multiplication.
__simt_callee__ CTRSM_INLINE inline aclblasComplex ScaleInput(aclblasComplex alpha, aclblasComplex value)
{
#ifdef __clang__
#pragma clang fp contract(off)
#endif
    // The SIMT O3 backend can fuse across the frontend contract(off) pragma.
    // One reused volatile slot makes each product's rounding observable.
    volatile float product = alpha.real * value.real;
    const float rr = product;
    product = alpha.imag * value.imag;
    const float ii = product;
    product = alpha.imag * value.real;
    const float ir = product;
    product = alpha.real * value.imag;
    const float ri = product;
    return {rr - ii, ir + ri};
}

// Scale before squaring to avoid overflow in |diagonal|^2 for large finite diagonals.
__simt_callee__ CTRSM_INLINE inline aclblasComplex Div(aclblasComplex a, aclblasComplex b)
{
    const float ar = b.real < 0 ? -b.real : b.real;
    const float ai = b.imag < 0 ? -b.imag : b.imag;
    const float scale = ar > ai ? ar : ai;
    const float r = b.real / scale, i = b.imag / scale;
    const float d = r * r + i * i;
    const float x = a.real / scale, y = a.imag / scale;
    aclblasComplex result{(x * r + y * i) / d, (y * r - x * i) / d};
    if (__builtin_isnan(result.real) && __builtin_isnan(result.imag)) {
        if ((__builtin_isinf(a.real) || __builtin_isinf(a.imag)) && __builtin_isfinite(scale)) {
            const float xr = InfinityUnit(a.real), xi = InfinityUnit(a.imag);
            result = {__builtin_inff() * (xr * r + xi * i), __builtin_inff() * (xi * r - xr * i)};
        } else if (
            (__builtin_isinf(b.real) || __builtin_isinf(b.imag)) && __builtin_isfinite(a.real) &&
            __builtin_isfinite(a.imag)) {
            const float br = InfinityUnit(b.real), bi = InfinityUnit(b.imag);
            result = {0.0f * (a.real * br + a.imag * bi), 0.0f * (a.imag * br - a.real * bi)};
        }
    }
    return result;
}

template <typename APtr>
__simt_callee__ CTRSM_INLINE inline aclblasComplex LoadA(APtr a, uint32_t row, uint32_t col, CtrsmBatchedTilingData t)
{
    const uint64_t offset =
        t.trans == ACLBLAS_OP_N ? row + static_cast<uint64_t>(col) * t.lda : col + static_cast<uint64_t>(row) * t.lda;
    aclblasComplex v = {a[offset].real, a[offset].imag};
    if (t.trans == ACLBLAS_OP_C)
        v.imag = -v.imag;
    return v;
}

// Two-float expansion for cancellation-sensitive scalar division. The AI Core
// does not support double arithmetic; FMA recovers the product rounding errors.
struct FloatExpansion {
    float high, low;
};

__simt_callee__ CTRSM_INLINE inline FloatExpansion ProductSum(float a, float b, float c, float d)
{
    volatile float p = a * b, q = c * d;
    const float pe = fmaf(a, b, -p), qe = fmaf(c, d, -q);
    const float sum = p + q, v = sum - p;
    return {sum, ((p - (sum - v)) + (q - v)) + (pe + qe)};
}

__simt_callee__ CTRSM_INLINE inline float ExpansionQuotient(FloatExpansion a, FloatExpansion b)
{
    const float q = a.high / b.high;
    const float remainder = fmaf(-q, b.high, a.high) + (a.low - q * b.low);
    return q + remainder / b.high;
}

__simt_callee__ CTRSM_INLINE inline int ScaleExponent(aclblasComplex v)
{
    const uint32_t rb = __builtin_bit_cast(uint32_t, v.real) & 0x7fffffffU;
    const uint32_t ib = __builtin_bit_cast(uint32_t, v.imag) & 0x7fffffffU;
    const int exponent = static_cast<int>((rb > ib ? rb : ib) >> 23);
    return exponent == 0 ? -126 : exponent - 127;
}

// Exact power-of-two scaling, including round-to-nearest-even subnormals.
// The device scalbnf implementation uses an approximate powf internally.
__simt_callee__ CTRSM_INLINE inline float ScalePowerOfTwo(float value, int power)
{
    const uint32_t bits = __builtin_bit_cast(uint32_t, value), sign = bits & 0x80000000U;
    uint32_t mantissa = bits & 0x7fffffU;
    int exponent = (bits >> 23) & 0xffU;
    if (exponent == 255 || (exponent == 0 && mantissa == 0))
        return value;
    if (exponent == 0) {
        const int shift = __builtin_clz(mantissa) - 8;
        mantissa <<= shift;
        exponent = 1 - shift;
    } else {
        mantissa |= 0x800000U;
    }
    exponent += power;
    if (exponent >= 255)
        return __builtin_bit_cast(float, sign | 0x7f800000U);
    if (exponent > 0)
        return __builtin_bit_cast(float, sign | (static_cast<uint32_t>(exponent) << 23) | (mantissa & 0x7fffffU));
    const int shift = 1 - exponent;
    if (shift > 24)
        return __builtin_bit_cast(float, sign);
    const uint32_t remainder = mantissa & ((1U << shift) - 1), half = 1U << (shift - 1);
    uint32_t rounded = mantissa >> shift;
    rounded += remainder > half || (remainder == half && (rounded & 1));
    return __builtin_bit_cast(float, sign | rounded);
}

__simt_callee__ CTRSM_INLINE inline aclblasComplex OrderedDiv(aclblasComplex a, aclblasComplex b)
{
    if (__builtin_isfinite(a.real) && __builtin_isfinite(a.imag) && __builtin_isfinite(b.real) &&
        __builtin_isfinite(b.imag) && (b.real != 0 || b.imag != 0)) {
        const int ae = ScaleExponent(a), be = ScaleExponent(b);
        const float ar = ScalePowerOfTwo(a.real, -ae), ai = ScalePowerOfTwo(a.imag, -ae);
        const float br = ScalePowerOfTwo(b.real, -be), bi = ScalePowerOfTwo(b.imag, -be);
        const auto denominator = ProductSum(br, br, bi, bi);
        const float real = ExpansionQuotient(ProductSum(ar, br, ai, bi), denominator);
        const float imag = ExpansionQuotient(ProductSum(ai, br, -ar, bi), denominator);
        return {ScalePowerOfTwo(real, ae - be), ScalePowerOfTwo(imag, ae - be)};
    }
    return Div(a, b);
}

__simt_callee__ CTRSM_INLINE inline float IeeeDivide(float numerator, float denominator)
{
    if (denominator != 0.0f)
        return numerator / denominator;
    if (numerator == 0.0f || __builtin_isnan(numerator))
        return __builtin_nanf("");
    const uint32_t sign =
        (__builtin_bit_cast(uint32_t, numerator) ^ __builtin_bit_cast(uint32_t, denominator)) & 0x80000000U;
    return __builtin_bit_cast(float, sign | 0x7f800000U);
}

// Smith division with explicit fused numerator/denominator operations.
// This matches the fused FP32 solve, including its IEEE exceptional values.
__simt_callee__ CTRSM_INLINE inline aclblasComplex FusedDiv(aclblasComplex a, aclblasComplex b)
{
    if (fabsf(b.real) < fabsf(b.imag)) {
        const float ratio = IeeeDivide(b.real, b.imag);
        const float denominator = fmaf(b.real, ratio, b.imag);
        return {
            IeeeDivide(fmaf(a.real, ratio, a.imag), denominator),
            IeeeDivide(fmaf(a.imag, ratio, -a.real), denominator)};
    }
    const float ratio = IeeeDivide(b.imag, b.real);
    const float denominator = fmaf(b.imag, ratio, b.real);
    return {
        IeeeDivide(fmaf(a.imag, ratio, a.real), denominator), IeeeDivide(fmaf(-a.real, ratio, a.imag), denominator)};
}

// OpenBLAS 0.3.25 Cooperlake uses 192-row panels, 8-row LEFT solves and
// 2-column RIGHT solves. Preserve the block updates for cancellation-sensitive
// systems; a fused scalar substitution is not numerically interchangeable.
__simt_callee__ CTRSM_INLINE inline uint32_t BlockWidth(uint32_t index, uint32_t total, uint32_t width)
{
    if (width == 0)
        return 1;
    const uint32_t tail = total % width;
    if (index < total - tail)
        return width;
    uint32_t at = total - tail;
    for (uint32_t w = width / 2; w != 0; w /= 2) {
        if (tail & w) {
            if (index < at + w)
                return w;
            at += w;
        }
    }
    return 1;
}

__simt_callee__ CTRSM_INLINE inline float RefinedDivide(float numerator, float denominator)
{
    const float quotient = IeeeDivide(numerator, denominator);
    if (__builtin_isfinite(numerator) && __builtin_isfinite(denominator) && denominator != 0 &&
        __builtin_isfinite(quotient)) {
        const float remainder = fmaf(-quotient, denominator, numerator);
        return quotient + remainder / denominator;
    }
    return quotient;
}

__simt_callee__ CTRSM_INLINE inline aclblasComplex BlockInverse(aclblasComplex z)
{
#ifdef __clang__
#pragma clang fp contract(off)
#endif
    if (fabsf(z.real) >= fabsf(z.imag)) {
        const float ratio = RefinedDivide(z.imag, z.real);
        const float inverse = RefinedDivide(1.0f, z.real * fmaf(ratio, ratio, 1.0f));
        return {inverse, -ratio * inverse};
    }
    const float ratio = RefinedDivide(z.real, z.imag);
    const float inverse = RefinedDivide(1.0f, z.imag * fmaf(ratio, ratio, 1.0f));
    return {ratio * inverse, -inverse};
}

template <typename APtr, typename BPtr>
__simt_callee__ CTRSM_INLINE inline aclblasComplex BlockDot(
    APtr a, BPtr b, uint32_t rhs, CtrsmBatchedTilingData t, uint32_t row, uint32_t begin, uint32_t end, uint32_t height)
{
#ifdef __clang__
#pragma clang fp contract(off)
#endif
    const bool left = t.side == ACLBLAS_SIDE_LEFT;
    const uint64_t base = left ? static_cast<uint64_t>(rhs) * t.ldb : rhs;
    const uint64_t stride = left ? 1 : t.ldb;
    float rr = 0, ri = 0, ir = 0, ii = 0;
    for (uint32_t j = begin; j < end; ++j) {
        const auto av = left ? LoadA(a, row, j, t) : LoadA(a, j, row, t);
        const uint64_t off = base + j * stride;
        const aclblasComplex xv{b[off].real, b[off].imag};
        const auto x = left ? av : xv, y = left ? xv : av;
        if (height == 8) {
            rr = fmaf(x.real, y.real, rr);
            ri = fmaf(x.real, y.imag, ri);
            ir = fmaf(x.imag, y.real, ir);
            ii = fmaf(x.imag, y.imag, ii);
        } else {
            rr = fmaf(x.imag, y.imag, -rr);
            rr = fmaf(x.real, y.real, -rr);
            ri = fmaf(x.real, y.imag, ri);
            ri = fmaf(x.imag, y.real, ri);
        }
    }
    return height == 8 ? aclblasComplex{rr - ii, ri + ir} : aclblasComplex{rr, ri};
}

__simt_callee__ CTRSM_INLINE inline aclblasComplex BlockSubtract(aclblasComplex b, aclblasComplex sum)
{
    // Retain the complex (-1, 0) update, including zero times infinity.
    return {fmaf(-1.0f, sum.real, -fmaf(0.0f, sum.imag, -b.real)), fmaf(-1.0f, sum.imag, fmaf(0.0f, sum.real, b.imag))};
}

} // namespace CtrsmBatched
#endif // device arithmetic
