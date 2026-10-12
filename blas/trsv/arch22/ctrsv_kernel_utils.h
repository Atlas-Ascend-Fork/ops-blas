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

// Shared by the device compatibility path and its host arithmetic checks.
#ifndef CTRSV_SCALAR_INLINE
#define CTRSV_SCALAR_INLINE __aicore__ inline
#endif

namespace ctrsv_netlib {

CTRSV_SCALAR_INLINE uint32_t Bits(float value)
{
    union {
        float value;
        uint32_t bits;
    } data{value};
    return data.bits;
}

CTRSV_SCALAR_INLINE float Value(uint32_t bits)
{
    union {
        uint32_t bits;
        float value;
    } data{bits};
    return data.value;
}

CTRSV_SCALAR_INLINE int HighBit(uint64_t value)
{
    int bit = 0;
    if (value >> 32U) {
        value >>= 32U;
        bit += 32;
    }
    if (value >> 16U) {
        value >>= 16U;
        bit += 16;
    }
    if (value >> 8U) {
        value >>= 8U;
        bit += 8;
    }
    if (value >> 4U) {
        value >>= 4U;
        bit += 4;
    }
    if (value >> 2U) {
        value >>= 2U;
        bit += 2;
    }
    if (value >> 1U) {
        bit += 1;
    }
    return bit;
}

CTRSV_SCALAR_INLINE uint64_t ShiftJam(uint64_t value, int distance)
{
    if (distance == 0)
        return value;
    if (distance >= 64)
        return value != 0U;
    return (value >> distance) | ((value << (64 - distance)) != 0U);
}

// Binary32 round-to-nearest-even FMA. The exact 48-bit product and 24-bit
// addend are aligned with sticky bits in 64 bits; no intermediate FP product
// can overflow or underflow before cancellation with the addend.
CTRSV_SCALAR_INLINE float RoundBinary32(uint64_t magnitude, int exponent, uint32_t sign)
{
    const int bit = HighBit(magnitude);
    int resultExponent = exponent - 62 + bit;
    if (resultExponent > 127)
        return Value(sign | 0x7f800000U);
    int shift = bit - 23;
    if (resultExponent < -126)
        shift = -exponent - 87;
    uint64_t rounded;
    if (shift <= 0) {
        rounded = magnitude << -shift;
    } else if (shift > 64) {
        rounded = 0U;
    } else if (shift == 64) {
        rounded = magnitude > (uint64_t{1} << 63U);
    } else {
        rounded = magnitude >> shift;
        const uint64_t remainder = magnitude & ((uint64_t{1} << shift) - 1U);
        const uint64_t half = uint64_t{1} << (shift - 1);
        rounded += remainder > half || (remainder == half && (rounded & 1U));
    }
    if (resultExponent < -126) {
        return Value(sign | static_cast<uint32_t>(rounded));
    }
    if (rounded >= 0x1000000U) {
        rounded >>= 1U;
        ++resultExponent;
    }
    if (resultExponent > 127)
        return Value(sign | 0x7f800000U);
    return Value(
        sign | (static_cast<uint32_t>(resultExponent + 127) << 23U) | (static_cast<uint32_t>(rounded) & 0x7fffffU));
}

CTRSV_SCALAR_INLINE float FmaFinite(uint32_t aa, uint32_t ab, uint32_t ac, uint32_t productSign, uint32_t addendSign)
{
    const int rawA = aa >> 23U, rawB = ab >> 23U, rawC = ac >> 23U;
    const uint64_t ma = (aa & 0x7fffffU) | (rawA ? 0x800000U : 0U);
    const uint64_t mb = (ab & 0x7fffffU) | (rawB ? 0x800000U : 0U);
    const uint64_t mc = (ac & 0x7fffffU) | (rawC ? 0x800000U : 0U);
    uint64_t product = ma * mb;
    const int productBit = HighBit(product);
    int exponent = (rawA ? rawA : 1) + (rawB ? rawB : 1) - 300 + productBit;
    product <<= 62 - productBit;
    uint64_t addend = 0U;
    if (mc != 0U) {
        const int addendBit = HighBit(mc);
        const int addendExponent = (rawC ? rawC : 1) - 150 + addendBit;
        addend = mc << (62 - addendBit);
        if (exponent >= addendExponent) {
            addend = ShiftJam(addend, exponent - addendExponent);
        } else {
            product = ShiftJam(product, addendExponent - exponent);
            exponent = addendExponent;
        }
    }
    uint32_t sign = productSign;
    uint64_t magnitude;
    if (productSign == addendSign) {
        magnitude = product + addend;
    } else if (product >= addend) {
        magnitude = product - addend;
    } else {
        magnitude = addend - product;
        sign = addendSign;
    }
    if (magnitude == 0U)
        return Value(0U);

    return RoundBinary32(magnitude, exponent, sign);
}

CTRSV_SCALAR_INLINE float Fma(float a, float b, float c)
{
    const uint32_t ua = Bits(a), ub = Bits(b), uc = Bits(c);
    const uint32_t aa = ua & 0x7fffffffU, ab = ub & 0x7fffffffU, ac = uc & 0x7fffffffU;
    const uint32_t productSign = (ua ^ ub) & 0x80000000U;
    const uint32_t addendSign = uc & 0x80000000U;
    if (aa > 0x7f800000U || ab > 0x7f800000U || ac > 0x7f800000U) {
        return Value(0x7fc00000U);
    }
    if (aa == 0x7f800000U || ab == 0x7f800000U) {
        if (aa == 0U || ab == 0U || (ac == 0x7f800000U && productSign != addendSign)) {
            return Value(0x7fc00000U);
        }
        return Value(productSign | 0x7f800000U);
    }
    if (ac == 0x7f800000U)
        return c;
    if (aa == 0U || ab == 0U) {
        return ac == 0U ? Value(productSign & addendSign) : c;
    }

    return FmaFinite(aa, ab, ac, productSign, addendSign);
}

CTRSV_SCALAR_INLINE float Mul(float a, float b) { return Fma(a, b, Value((Bits(a) ^ Bits(b)) & 0x80000000U)); }

struct Complex {
    float real;
    float imag;
};

CTRSV_SCALAR_INLINE bool IsNan(float x) { return (Bits(x) & 0x7fffffffU) > 0x7f800000U; }

CTRSV_SCALAR_INLINE bool IsInf(float x) { return (Bits(x) & 0x7fffffffU) == 0x7f800000U; }

CTRSV_SCALAR_INLINE float CopySign(float magnitude, float sign)
{
    return Value((Bits(magnitude) & 0x7fffffffU) | (Bits(sign) & 0x80000000U));
}

CTRSV_SCALAR_INLINE bool HasInf(Complex value) { return IsInf(value.real) || IsInf(value.imag); }

CTRSV_SCALAR_INLINE bool IsFinite(Complex value)
{
    return (Bits(value.real) & 0x7fffffffU) < 0x7f800000U && (Bits(value.imag) & 0x7fffffffU) < 0x7f800000U;
}

CTRSV_SCALAR_INLINE bool NeedsSoftwareComponent(float value)
{
    const uint32_t magnitude = Bits(value) & 0x7fffffffU;
    return magnitude > 0x4f800000U || (magnitude != 0U && magnitude < 0x2f800000U);
}

CTRSV_SCALAR_INLINE bool NeedsSoftwareProduct(Complex value)
{
    return NeedsSoftwareComponent(value.real) || NeedsSoftwareComponent(value.imag);
}

CTRSV_SCALAR_INLINE bool NeedsFullRangeDiagonal(Complex value)
{
    const uint32_t real = Bits(value.real) & 0x7fffffffU;
    const uint32_t imag = Bits(value.imag) & 0x7fffffffU;
    return NeedsSoftwareProduct(value) || (real < 0x2f800000U && imag < 0x2f800000U);
}

CTRSV_SCALAR_INLINE Complex NormalizeInfinity(Complex value)
{
    return {
        CopySign(IsInf(value.real) ? 1.0f : 0.0f, value.real), CopySign(IsInf(value.imag) ? 1.0f : 0.0f, value.imag)};
}

CTRSV_SCALAR_INLINE void ClearNanComponents(Complex& value)
{
    if (IsNan(value.real))
        value.real = CopySign(0.0f, value.real);
    if (IsNan(value.imag))
        value.imag = CopySign(0.0f, value.imag);
}

// Full complex arithmetic recovers infinities after a NaN intermediate.
// This is different from the limited-range complex multiply used by some
// BLAS builds. Only the exceptional path needs the four separately rounded
// products and recovery; ordinary operands retain the existing FMA order.
CTRSV_SCALAR_INLINE Complex RecoverProduct(Complex a, Complex b)
{
    const float rr = Mul(a.real, b.real), ii = Mul(a.imag, b.imag);
    const float ri = Mul(a.real, b.imag), ir = Mul(a.imag, b.real);
    volatile float real = rr - ii;
    volatile float imag = ri + ir;
    if (!IsNan(real) || !IsNan(imag))
        return {real, imag};

    bool recover = false;
    if (HasInf(a)) {
        a = NormalizeInfinity(a);
        ClearNanComponents(b);
        recover = true;
    }
    if (HasInf(b)) {
        b = NormalizeInfinity(b);
        ClearNanComponents(a);
        recover = true;
    }
    if (!recover && (IsInf(rr) || IsInf(ii) || IsInf(ri) || IsInf(ir))) {
        ClearNanComponents(a);
        ClearNanComponents(b);
        recover = true;
    }
    if (recover) {
        const float infinity = Value(0x7f800000U);
        return {
            Mul(infinity, Fma(a.real, b.real, -Mul(a.imag, b.imag))),
            Mul(infinity, Fma(a.real, b.imag, Mul(a.imag, b.real)))};
    }
    return {real, imag};
}

CTRSV_SCALAR_INLINE Complex Product(Complex a, Complex b)
{
    const float ii = Mul(a.imag, b.imag);
    const float ir = Mul(a.imag, b.real);
    const Complex result{Fma(a.real, b.real, -ii), Fma(a.real, b.imag, ir)};
    return IsNan(result.real) && IsNan(result.imag) ? RecoverProduct(a, b) : result;
}

CTRSV_SCALAR_INLINE Complex SubProduct(Complex sum, Complex a, Complex b)
{
    const Complex product = Product(a, b);
    // Netlib's complex multiplication is rounded before the outer subtraction.
    volatile float real = sum.real - product.real;
    volatile float imag = sum.imag - product.imag;
    return {real, imag};
}

// A finite binary64 value represented without device FP64 instructions.
// Its value is (-1)^sign * mantissa * 2^(exponent-52).
struct Wide {
    uint64_t mantissa;
    int exponent;
    uint32_t sign;
};

struct ExactBinaryProduct {
    uint64_t mantissa;
    int exponent;
    uint32_t sign;
};

// Exact value: (-1)^sign * mantissa * 2^exponent, before normalization.
CTRSV_SCALAR_INLINE ExactBinaryProduct ExactProduct(float a, float b)
{
    const uint32_t ua = Bits(a), ub = Bits(b);
    const int ea = (ua >> 23U) & 255U, eb = (ub >> 23U) & 255U;
    const uint64_t product =
        static_cast<uint64_t>((ua & 0x7fffffU) | (ea ? 0x800000U : 0U)) * ((ub & 0x7fffffU) | (eb ? 0x800000U : 0U));
    return {product, (ea ? ea : 1) + (eb ? eb : 1) - 300, (ua ^ ub) & 0x80000000U};
}

CTRSV_SCALAR_INLINE Wide RoundWide(uint64_t magnitude, int ep, uint32_t sign)
{
    const int bit = HighBit(magnitude);
    int exponent = ep - 62 + bit;
    const int shift = bit - 52;
    uint64_t mantissa;
    if (shift <= 0)
        mantissa = magnitude << -shift;
    else {
        mantissa = magnitude >> shift;
        const uint64_t remainder = magnitude & ((uint64_t{1} << shift) - 1U);
        const uint64_t half = uint64_t{1} << (shift - 1);
        mantissa += remainder > half || (remainder == half && (mantissa & 1U));
    }
    if (mantissa == (uint64_t{1} << 53U)) {
        mantissa >>= 1U;
        ++exponent;
    }
    return {mantissa, exponent, sign};
}

CTRSV_SCALAR_INLINE Wide SumProductsWide(float a, float b, float c, float d)
{
    const ExactBinaryProduct left = ExactProduct(a, b), right = ExactProduct(c, d);
    uint64_t p = left.mantissa, q = right.mantissa;
    uint32_t sign = left.sign;
    const uint32_t otherSign = right.sign;
    if (p == 0U && q == 0U)
        return {0U, 0, sign & otherSign};
    const int bp = HighBit(p), bq = HighBit(q);
    int ep = left.exponent + bp, eq = right.exponent + bq;
    p <<= 62 - bp;
    q <<= 62 - bq;
    if (p == 0U)
        ep = eq;
    if (q == 0U)
        eq = ep;
    if (ep >= eq)
        q = ShiftJam(q, ep - eq);
    else {
        p = ShiftJam(p, eq - ep);
        ep = eq;
    }
    uint64_t magnitude;
    if (sign == otherSign)
        magnitude = p + q;
    else if (p >= q)
        magnitude = p - q;
    else {
        magnitude = q - p;
        sign = otherSign;
    }
    if (magnitude == 0U)
        return {0U, 0, 0U};
    return RoundWide(magnitude, ep, sign);
}

CTRSV_SCALAR_INLINE float DivideWide(Wide numerator, Wide denominator)
{
    const uint32_t sign = numerator.sign ^ denominator.sign;
    if (numerator.mantissa == 0U)
        return Value(sign);
    int exponent = numerator.exponent - denominator.exponent;
    uint64_t remainder = numerator.mantissa;
    if (remainder < denominator.mantissa) {
        remainder <<= 1U;
        --exponent;
    }
    remainder -= denominator.mantissa;
    uint64_t quotient = 1U;
    // Round the quotient to binary64 first, then to binary32, matching
    // promoted complex division even at a binary32 midpoint.
    for (int i = 0; i < 52; ++i) {
        remainder <<= 1U;
        quotient <<= 1U;
        if (remainder >= denominator.mantissa) {
            remainder -= denominator.mantissa;
            quotient |= 1U;
        }
    }
    const uint64_t twice = remainder << 1U;
    quotient += twice > denominator.mantissa || (twice == denominator.mantissa && (quotient & 1U));
    if (quotient == (uint64_t{1} << 53U)) {
        quotient >>= 1U;
        ++exponent;
    }
    if (exponent > 127)
        return Value(sign | 0x7f800000U);
    const int shift = exponent >= -126 ? 29 : -exponent - 97;
    if (shift > 53)
        return Value(sign);
    uint64_t rounded = quotient >> shift;
    const uint64_t tail = quotient & ((uint64_t{1} << shift) - 1U);
    const uint64_t half = uint64_t{1} << (shift - 1);
    rounded += tail > half || (tail == half && (rounded & 1U));
    if (exponent < -126)
        return Value(sign | static_cast<uint32_t>(rounded));
    if (rounded == 0x1000000U) {
        rounded >>= 1U;
        ++exponent;
    }
    if (exponent > 127)
        return Value(sign | 0x7f800000U);
    return Value(sign | (static_cast<uint32_t>(exponent + 127) << 23U) | (static_cast<uint32_t>(rounded) & 0x7fffffU));
}

CTRSV_SCALAR_INLINE Complex NonfiniteNumerator(Complex x, Complex a)
{
    // With a nonfinite numerator, each numerator sum has a nonfinite
    // term. Represent the other (finite binary64) product by signed zero
    // so that an intermediate binary32 overflow cannot change its class.
    const float rr =
        (IsInf(x.real) || IsNan(x.real)) ? Mul(x.real, a.real) : Value((Bits(x.real) ^ Bits(a.real)) & 0x80000000U);
    const float ri =
        (IsInf(x.real) || IsNan(x.real)) ? Mul(x.real, a.imag) : Value((Bits(x.real) ^ Bits(a.imag)) & 0x80000000U);
    const float ir =
        (IsInf(x.imag) || IsNan(x.imag)) ? Mul(x.imag, a.real) : Value((Bits(x.imag) ^ Bits(a.real)) & 0x80000000U);
    const float ii =
        (IsInf(x.imag) || IsNan(x.imag)) ? Mul(x.imag, a.imag) : Value((Bits(x.imag) ^ Bits(a.imag)) & 0x80000000U);
    const Complex initial{rr + ii, ir - ri};
    return initial;
}

CTRSV_SCALAR_INLINE Complex DivideFullRange(Complex x, Complex a)
{
    const float infinity = Value(0x7f800000U), nan = Value(0x7fc00000U);
    const bool finiteX = IsFinite(x);
    const bool finiteA = IsFinite(a);
    if (a.real == 0.0f && a.imag == 0.0f && (!IsNan(x.real) || !IsNan(x.imag))) {
        return {Mul(CopySign(infinity, a.real), x.real), Mul(CopySign(infinity, a.real), x.imag)};
    }
    if (finiteX && finiteA) {
        const Wide denominator = SumProductsWide(a.real, a.real, a.imag, a.imag);
        return {
            DivideWide(SumProductsWide(x.real, a.real, x.imag, a.imag), denominator),
            DivideWide(SumProductsWide(x.imag, a.real, -x.real, a.imag), denominator)};
    }
    if (finiteA) {
        const Complex initial = NonfiniteNumerator(x, a);
        if (!IsNan(initial.real) || !IsNan(initial.imag))
            return initial;
    }
    if (HasInf(x) && finiteA) {
        x = NormalizeInfinity(x);
        return {
            Mul(infinity, Fma(a.real, x.real, Mul(a.imag, x.imag))),
            Mul(infinity, Fma(a.real, x.imag, -Mul(a.imag, x.real)))};
    }
    if (HasInf(a) && finiteX) {
        a = NormalizeInfinity(a);
        return {
            Mul(0.0f, Fma(x.real, a.real, Mul(x.imag, a.imag))), Mul(0.0f, Fma(x.imag, a.real, -Mul(x.real, a.imag)))};
    }
    return {nan, nan};
}

CTRSV_SCALAR_INLINE Complex Divide(Complex x, Complex a)
{
    const uint32_t xr = Bits(x.real) & 0x7fffffffU, xi = Bits(x.imag) & 0x7fffffffU;
    const uint32_t dr = Bits(a.real) & 0x7fffffffU, di = Bits(a.imag) & 0x7fffffffU;
    if (xr > 0x4f800000U || xi > 0x4f800000U || dr > 0x4f800000U || di > 0x4f800000U ||
        (dr < 0x2f800000U && di < 0x2f800000U)) {
        return DivideFullRange(x, a);
    }
    const float ar = a.real < 0.0f ? -a.real : a.real;
    const float ai = a.imag < 0.0f ? -a.imag : a.imag;
    if (ar >= ai) {
        if (a.real == 0.0f)
            return DivideFullRange(x, a);
        volatile float ratio = a.imag / a.real;
        const float denominator = Fma(a.imag, ratio, a.real);
        if (denominator == 0.0f)
            return DivideFullRange(x, a);
        return {Fma(x.imag, ratio, x.real) / denominator, Fma(-x.real, ratio, x.imag) / denominator};
    }
    if (a.imag == 0.0f)
        return DivideFullRange(x, a);
    volatile float ratio = a.real / a.imag;
    const float denominator = Fma(a.real, ratio, a.imag);
    if (denominator == 0.0f)
        return DivideFullRange(x, a);
    return {Fma(x.real, ratio, x.imag) / denominator, Fma(x.imag, ratio, -x.real) / denominator};
}

} // namespace ctrsv_netlib
