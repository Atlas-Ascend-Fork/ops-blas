/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Finite-value software binary64 for exceptional numerical ranges.
// Define CHBMV_SOFT_INLINE as __aicore__ inline before inclusion on a device.
// Finite float32 inputs and their HBMV products/sums fit the binary64 exponent
// range. Non-finite inputs must remain on an explicit IEEE/special-value path.
#pragma once
#include <cstdint>
#ifndef CHBMV_SOFT_INLINE
#define CHBMV_SOFT_INLINE inline
#endif

namespace chbmv_numeric_range {
CHBMV_SOFT_INLINE uint32_t Bits(float x)
{
    union {
        float f;
        uint32_t u;
    } v{x};
    return v.u;
}
CHBMV_SOFT_INLINE float Float(uint32_t x)
{
    union {
        uint32_t u;
        float f;
    } v{x};
    return v.f;
}
CHBMV_SOFT_INLINE bool Finite(float x) { return (Bits(x) & 0x7f800000U) != 0x7f800000U; }

struct Finite64 {
    uint64_t mantissa = 0; // Normalized 53-bit unsigned significand.
    int exponent = 0;      // Value = (-1)^negative * mantissa * 2^(exponent-52).
    bool negative = false;

    static CHBMV_SOFT_INLINE uint64_t RightJam(uint64_t value, int distance)
    {
        if (distance <= 0) {
            return value;
        }
        if (distance >= 64) {
            return value != 0;
        }
        const uint64_t mask = (uint64_t{1} << distance) - 1U;
        return (value >> distance) | ((value & mask) != 0);
    }

    static CHBMV_SOFT_INLINE uint64_t RoundRight(uint64_t value, int distance)
    {
        if (distance <= 0) {
            return value << (-distance);
        }
        if (distance >= 64) {
            return 0;
        }
        const uint64_t result = value >> distance;
        const uint64_t remainder = value & ((uint64_t{1} << distance) - 1U);
        const uint64_t half = uint64_t{1} << (distance - 1);
        return result + (remainder > half || (remainder == half && (result & 1U)));
    }

    static CHBMV_SOFT_INLINE Finite64 FromFloat(float x)
    {
        const uint32_t bits = Bits(x);
        const uint32_t encodedExponent = (bits >> 23U) & 255U;
        uint32_t m = bits & 0x7fffffU;
        Finite64 result;
        result.negative = (bits >> 31U) != 0;
        if (encodedExponent == 0) {
            if (m == 0) {
                return result;
            }
            result.exponent = -126;
            while (m < 0x800000U) {
                m <<= 1U;
                --result.exponent;
            }
        } else {
            // Callers must reject encodedExponent==255 before entering this class.
            m |= 0x800000U;
            result.exponent = static_cast<int>(encodedExponent) - 127;
        }
        result.mantissa = static_cast<uint64_t>(m) << 29U;
        return result;
    }

    CHBMV_SOFT_INLINE Finite64 Negate() const
    {
        Finite64 result = *this;
        result.negative = !result.negative;
        return result;
    }

    CHBMV_SOFT_INLINE Finite64 Add(const Finite64& other) const
    {
        if (mantissa == 0) {
            return other;
        }
        if (other.mantissa == 0) {
            return *this;
        }
        const bool swap = exponent < other.exponent || (exponent == other.exponent && mantissa < other.mantissa);
        const Finite64& a = swap ? other : *this;
        const Finite64& b = swap ? *this : other;
        uint64_t av = a.mantissa << 3U;
        const uint64_t bv = RightJam(b.mantissa << 3U, a.exponent - b.exponent);
        Finite64 result;
        result.exponent = a.exponent;
        result.negative = a.negative;
        uint64_t sum = a.negative == b.negative ? av + bv : av - bv;
        if (sum == 0) {
            result.negative = false;
            result.exponent = 0;
            return result;
        }
        if (sum & (uint64_t{1} << 56U)) {
            sum = RightJam(sum, 1);
            ++result.exponent;
        }
        while ((sum & (uint64_t{1} << 55U)) == 0) {
            sum <<= 1U;
            --result.exponent;
        }
        result.mantissa = RoundRight(sum, 3);
        if (result.mantissa & (uint64_t{1} << 53U)) {
            result.mantissa >>= 1U;
            ++result.exponent;
        }
        return result;
    }

    CHBMV_SOFT_INLINE Finite64 Multiply(const Finite64& other) const
    {
        Finite64 result;
        result.negative = negative != other.negative;
        if (mantissa == 0 || other.mantissa == 0) {
            return result;
        }
        const uint64_t a0 = mantissa & 0xffffffffULL;
        const uint64_t a1 = mantissa >> 32U;
        const uint64_t b0 = other.mantissa & 0xffffffffULL;
        const uint64_t b1 = other.mantissa >> 32U;
        const uint64_t w0 = a0 * b0;
        const uint64_t t = a1 * b0 + (w0 >> 32U);
        const uint64_t w1 = (t & 0xffffffffULL) + a0 * b1;
        const uint64_t high = a1 * b1 + (t >> 32U) + (w1 >> 32U);
        const uint64_t low = (w1 << 32U) | (w0 & 0xffffffffULL);
        const bool leading105 = (high & (uint64_t{1} << 41U)) != 0;
        const int shift = leading105 ? 53 : 52;
        uint64_t significand = (high << (64 - shift)) | (low >> shift);
        const uint64_t remainder = low & ((uint64_t{1} << shift) - 1U);
        const uint64_t half = uint64_t{1} << (shift - 1);
        significand += remainder > half || (remainder == half && (significand & 1U));
        result.exponent = exponent + other.exponent + leading105;
        if (significand & (uint64_t{1} << 53U)) {
            significand >>= 1U;
            ++result.exponent;
        }
        result.mantissa = significand;
        return result;
    }

    CHBMV_SOFT_INLINE float ToFloat() const
    {
        const uint32_t sign = negative ? 0x80000000U : 0;
        if (mantissa == 0) {
            return Float(sign);
        }
        if (exponent < -126) {
            // Rounding may carry from the largest subnormal to the smallest normal.
            return Float(sign | static_cast<uint32_t>(RoundRight(mantissa, -exponent - 97)));
        }
        uint64_t m = RoundRight(mantissa, 29);
        int e = exponent;
        if (m >= (uint64_t{1} << 24U)) {
            m >>= 1U;
            ++e;
        }
        if (e > 127) {
            return Float(sign | 0x7f800000U);
        }
        return Float(sign | (static_cast<uint32_t>(e + 127) << 23U) | (static_cast<uint32_t>(m) & 0x7fffffU));
    }
};

struct FiniteComplex64 {
    Finite64 real, imag;
    static CHBMV_SOFT_INLINE FiniteComplex64 FromFloat(float re, float im)
    {
        return {Finite64::FromFloat(re), Finite64::FromFloat(im)};
    }
    CHBMV_SOFT_INLINE FiniteComplex64 Add(const FiniteComplex64& x) const
    {
        return {real.Add(x.real), imag.Add(x.imag)};
    }
    CHBMV_SOFT_INLINE FiniteComplex64 Multiply(const FiniteComplex64& x) const
    {
        return {
            real.Multiply(x.real).Add(imag.Multiply(x.imag).Negate()),
            real.Multiply(x.imag).Add(imag.Multiply(x.real))};
    }
    CHBMV_SOFT_INLINE FiniteComplex64 MultiplyReal(float value) const
    {
        const auto scale = Finite64::FromFloat(value);
        return {real.Multiply(scale), imag.Multiply(scale)};
    }
};
} // namespace chbmv_numeric_range
