/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once
#include <cstdint>

#if defined(__CCE_AICORE__)
#define CGBT_FLOAT_INLINE __aicore__ inline
#else
#define CGBT_FLOAT_INLINE inline
#endif

// IEEE-754 binary32 encoding and integer accumulator geometry. The accumulator
// keeps two high bits free for addition/carry and retains low sticky bits for RNE.
constexpr int CGBT_ACCUMULATOR_BITS = 64;
constexpr int CGBT_ACCUMULATOR_TOP = CGBT_ACCUMULATOR_BITS - 1;
constexpr int CGBT_ALIGNED_PRODUCT_TOP = CGBT_ACCUMULATOR_TOP - 2;
constexpr int CGBT_FP32_FRACTION_BITS = 23;
constexpr int CGBT_FP32_EXPONENT_BIAS = 127;
constexpr int CGBT_FP32_SIGNIFICAND_BIAS = CGBT_FP32_EXPONENT_BIAS + CGBT_FP32_FRACTION_BITS;
constexpr int CGBT_FP32_MIN_UNIT_EXPONENT = 1 - CGBT_FP32_SIGNIFICAND_BIAS;
constexpr int CGBT_FP32_SPECIAL_EXPONENT = 255;
constexpr uint32_t CGBT_FP32_SIGN_MASK = 0x80000000U;
constexpr uint32_t CGBT_FP32_MAGNITUDE_MASK = 0x7fffffffU;
constexpr uint32_t CGBT_FP32_FRACTION_MASK = (1U << CGBT_FP32_FRACTION_BITS) - 1U;
constexpr uint32_t CGBT_FP32_HIDDEN_BIT = 1U << CGBT_FP32_FRACTION_BITS;
constexpr uint32_t CGBT_FP32_SIGNIFICAND_CARRY = CGBT_FP32_HIDDEN_BIT << 1U;
constexpr uint32_t CGBT_FP32_INFINITY = 0x7f800000U;
constexpr uint32_t CGBT_FP32_QUIET_NAN = 0x7fc00000U;
constexpr uint32_t CGBT_FP32_ONE = 0x3f800000U;

union CgbtFloatBits {
    float value;
    uint32_t bits;
};

CGBT_FLOAT_INLINE uint64_t CgbtShiftJam(uint64_t value, int shift)
{
    if (shift <= 0)
        return value << (-shift);
    if (shift >= CGBT_ACCUMULATOR_BITS)
        return value != 0U;
    return (value >> shift) | ((value << (CGBT_ACCUMULATOR_BITS - shift)) != 0U);
}

CGBT_FLOAT_INLINE float CgbtRoundFma(uint64_t magnitude, uint32_t sign, int scale)
{
    CgbtFloatBits result{};
    const int top = CGBT_ACCUMULATOR_TOP - __builtin_clzll(magnitude);
    const int exponent = scale + top;
    const int unit = exponent - CGBT_FP32_FRACTION_BITS > CGBT_FP32_MIN_UNIT_EXPONENT ?
                         exponent - CGBT_FP32_FRACTION_BITS :
                         CGBT_FP32_MIN_UNIT_EXPONENT;
    const int shift = unit - scale;
    uint64_t rounded = 0U;
    if (shift <= 0) {
        // unit >= scale + top - fractionBits, hence -shift <= fractionBits.
        // Cancellation can require left normalization, never a negative right shift.
        if (shift > -CGBT_ACCUMULATOR_BITS)
            rounded = magnitude << (-shift);
    } else if (shift < CGBT_ACCUMULATOR_BITS) {
        rounded = magnitude >> shift;
    }
    if (shift > 0 && shift <= CGBT_ACCUMULATOR_BITS) {
        const uint64_t halfway = 1ULL << (shift - 1);
        const uint64_t remainder = shift == CGBT_ACCUMULATOR_BITS ? magnitude : magnitude & ((1ULL << shift) - 1U);
        rounded += remainder > halfway || (remainder == halfway && (rounded & 1U));
    }
    int encodedExp = unit + CGBT_FP32_SIGNIFICAND_BIAS;
    if (rounded >= CGBT_FP32_SIGNIFICAND_CARRY) {
        rounded >>= 1U;
        ++encodedExp;
    }
    if (encodedExp >= CGBT_FP32_SPECIAL_EXPONENT)
        result.bits = sign | CGBT_FP32_INFINITY;
    else if (rounded < CGBT_FP32_HIDDEN_BIT)
        result.bits = sign | static_cast<uint32_t>(rounded);
    else
        result.bits =
            sign | (static_cast<uint32_t>(encodedExp) << CGBT_FP32_FRACTION_BITS) | (rounded & CGBT_FP32_FRACTION_MASK);
    return result.value;
}

CGBT_FLOAT_INLINE float CgbtAccumulateFma(
    uint64_t product, uint64_t mz, int productExp, int addendExp, uint32_t productSign, uint32_t addendSign)
{
    CgbtFloatBits result{};
    const int productTop = CGBT_ACCUMULATOR_TOP - __builtin_clzll(product) + productExp;
    const int addendTop = mz ? CGBT_ACCUMULATOR_TOP - __builtin_clzll(mz) + addendExp : productTop;
    const int scale = (productTop > addendTop ? productTop : addendTop) - CGBT_ALIGNED_PRODUCT_TOP;
    const uint64_t p = CgbtShiftJam(product, scale - productExp);
    const uint64_t q = mz ? CgbtShiftJam(mz, scale - addendExp) : 0U;
    uint64_t magnitude;
    uint32_t sign = productSign;
    if (productSign == addendSign)
        magnitude = p + q;
    else if (p >= q)
        magnitude = p - q;
    else {
        magnitude = q - p;
        sign = addendSign;
    }
    if (magnitude == 0U) {
        result.bits = 0U;
        return result.value;
    }
    return CgbtRoundFma(magnitude, sign, scale);
}

CGBT_FLOAT_INLINE bool CgbtFmaExceptional(
    uint32_t ax, uint32_t ay, uint32_t az, uint32_t productSign, uint32_t addendSign, float z, float& value)
{
    CgbtFloatBits result{};
    if (ax > CGBT_FP32_INFINITY || ay > CGBT_FP32_INFINITY || az > CGBT_FP32_INFINITY ||
        ((ax == 0U && ay == CGBT_FP32_INFINITY) || (ay == 0U && ax == CGBT_FP32_INFINITY))) {
        result.bits = CGBT_FP32_QUIET_NAN;
        value = result.value;
        return true;
    }
    if (ax == CGBT_FP32_INFINITY || ay == CGBT_FP32_INFINITY) {
        result.bits = az == CGBT_FP32_INFINITY && productSign != addendSign ? CGBT_FP32_QUIET_NAN :
                                                                              productSign | CGBT_FP32_INFINITY;
        value = result.value;
        return true;
    }
    if (az == CGBT_FP32_INFINITY) {
        value = z;
        return true;
    }
    return false;
}

// Exact FP32 fused multiply-add for the rare overflow-sensitive scalar path.
// A2 vector multiply-add rounds its product separately. Integer significands
// retain the exact 48-bit product and guard/round/sticky bits until final RNE.
CGBT_FLOAT_INLINE float CgbtFma(float x, float y, float z)
{
    CgbtFloatBits a{x};
    CgbtFloatBits b{y};
    CgbtFloatBits c{z};
    CgbtFloatBits result{};
    const uint32_t ax = a.bits & CGBT_FP32_MAGNITUDE_MASK;
    const uint32_t ay = b.bits & CGBT_FP32_MAGNITUDE_MASK;
    const uint32_t az = c.bits & CGBT_FP32_MAGNITUDE_MASK;
    const uint32_t productSign = (a.bits ^ b.bits) & CGBT_FP32_SIGN_MASK;
    const uint32_t addendSign = c.bits & CGBT_FP32_SIGN_MASK;
    float exceptional;
    if (CgbtFmaExceptional(ax, ay, az, productSign, addendSign, z, exceptional))
        return exceptional;
    const int ex = ax >> CGBT_FP32_FRACTION_BITS;
    const int ey = ay >> CGBT_FP32_FRACTION_BITS;
    const int ez = az >> CGBT_FP32_FRACTION_BITS;
    const uint64_t mx = (ax & CGBT_FP32_FRACTION_MASK) | (ex ? CGBT_FP32_HIDDEN_BIT : 0U);
    const uint64_t my = (ay & CGBT_FP32_FRACTION_MASK) | (ey ? CGBT_FP32_HIDDEN_BIT : 0U);
    const uint64_t mz = (az & CGBT_FP32_FRACTION_MASK) | (ez ? CGBT_FP32_HIDDEN_BIT : 0U);
    const uint64_t product = mx * my;
    const int productExp = (ex ? ex - CGBT_FP32_SIGNIFICAND_BIAS : CGBT_FP32_MIN_UNIT_EXPONENT) +
                           (ey ? ey - CGBT_FP32_SIGNIFICAND_BIAS : CGBT_FP32_MIN_UNIT_EXPONENT);
    const int addendExp = ez ? ez - CGBT_FP32_SIGNIFICAND_BIAS : CGBT_FP32_MIN_UNIT_EXPONENT;
    if (product == 0U) {
        if (mz != 0U)
            return z;
        result.bits = productSign & addendSign;
        return result.value;
    }
    return CgbtAccumulateFma(product, mz, productExp, addendExp, productSign, addendSign);
}
// Full-range complex multiplication follows the compiler runtime (__mulsc3).
// Recover an infinite result only when both components became NaN. Ordinary
// NaNs must propagate; alpha=0/k=0 keep their separate BLAS quick-return rules.
struct CgbtComplexValue {
    float real, imag;
};

CGBT_FLOAT_INLINE bool CgbtIsInf(float value)
{
    return (CgbtFloatBits{value}.bits & CGBT_FP32_MAGNITUDE_MASK) == CGBT_FP32_INFINITY;
}

CGBT_FLOAT_INLINE float CgbtSignedUnit(float value, bool one)
{
    CgbtFloatBits result{value};
    result.bits = (result.bits & CGBT_FP32_SIGN_MASK) | (one ? CGBT_FP32_ONE : 0U);
    return result.value;
}

CGBT_FLOAT_INLINE void CgbtClearNan(CgbtComplexValue& value)
{
    if (value.real != value.real)
        value.real = CgbtSignedUnit(value.real, false);
    if (value.imag != value.imag)
        value.imag = CgbtSignedUnit(value.imag, false);
}

CGBT_FLOAT_INLINE CgbtComplexValue CgbtComplexMultiply(CgbtComplexValue a, CgbtComplexValue b)
{
    volatile float rr = a.real * b.real;
    volatile float ii = a.imag * b.imag;
    volatile float ri = a.real * b.imag;
    volatile float ir = a.imag * b.real;
    CgbtComplexValue result{rr - ii, ri + ir};
    if (result.real == result.real || result.imag == result.imag)
        return result;
    bool recover = false;
    if (CgbtIsInf(a.real) || CgbtIsInf(a.imag)) {
        a = {CgbtSignedUnit(a.real, CgbtIsInf(a.real)), CgbtSignedUnit(a.imag, CgbtIsInf(a.imag))};
        CgbtClearNan(b);
        recover = true;
    }
    if (CgbtIsInf(b.real) || CgbtIsInf(b.imag)) {
        b = {CgbtSignedUnit(b.real, CgbtIsInf(b.real)), CgbtSignedUnit(b.imag, CgbtIsInf(b.imag))};
        CgbtClearNan(a);
        recover = true;
    }
    if (!recover && (CgbtIsInf(rr) || CgbtIsInf(ii) || CgbtIsInf(ri) || CgbtIsInf(ir))) {
        CgbtClearNan(a);
        CgbtClearNan(b);
        recover = true;
    }
    if (recover) {
        CgbtFloatBits infinity{};
        infinity.bits = CGBT_FP32_INFINITY;
        volatile float realReal = a.real * b.real;
        volatile float imagImag = a.imag * b.imag;
        volatile float realImag = a.real * b.imag;
        volatile float imagReal = a.imag * b.real;
        result = {infinity.value * (realReal - imagImag), infinity.value * (realImag + imagReal)};
    }
    return result;
}

#undef CGBT_FLOAT_INLINE
