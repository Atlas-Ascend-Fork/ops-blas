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

// Rare-path arithmetic: a signed 53-bit significand with an independent exponent.
// Products of FP32 significands are exact before accumulation. No device FP64
// instructions are required; final conversion rounds once to the FP32 range.
namespace scaled_scalar {
constexpr int32_t Special = 4096;
struct Value {
    int64_t significand;
    int32_t exponent;
};
union Bits {
    float value;
    uint32_t bits;
};
__aicore__ inline Value Normalize(int64_t value, int32_t exponent)
{
    if (value == 0) {
        return {0, 0};
    }
    const bool negative = value < 0;
    uint64_t magnitude = negative ? 0ULL - static_cast<uint64_t>(value) : static_cast<uint64_t>(value);
    const int32_t leading = 63 - __builtin_clzll(magnitude);
    int32_t shift = leading - 52;
    if (shift > 0) {
        const uint64_t mask = (1ULL << shift) - 1ULL;
        const uint64_t remainder = magnitude & mask;
        magnitude >>= shift;
        const uint64_t half = 1ULL << (shift - 1);
        magnitude += remainder > half || (remainder == half && (magnitude & 1ULL));
    } else {
        magnitude <<= -shift;
    }
    if (magnitude == (1ULL << 53U)) {
        magnitude >>= 1U;
        ++shift;
    }
    return {negative ? -static_cast<int64_t>(magnitude) : static_cast<int64_t>(magnitude), exponent + shift};
}
__aicore__ inline Value FromFloat(float value)
{
    Bits bits;
    bits.value = value;
    const uint32_t field = (bits.bits >> 23U) & 255U;
    const uint32_t fraction = bits.bits & 0x7fffffU;
    const int64_t sign = (bits.bits >> 31U) != 0U ? -1 : 1;
    if (field == 255U) {
        return {fraction == 0U ? sign : 0, Special};
    }
    return Normalize(
        sign * (field == 0U ? fraction : fraction | 0x800000U), field == 0U ? -149 : static_cast<int32_t>(field) - 150);
}
__aicore__ inline Value Neg(Value value) { return {-value.significand, value.exponent}; }
__aicore__ inline Value Add(Value a, Value b)
{
    if (a.exponent == Special || b.exponent == Special) {
        if (a.exponent == Special && b.exponent == Special) {
            return a.significand == b.significand ? a : Value{0, Special};
        }
        return a.exponent == Special ? a : b;
    }
    if (a.significand == 0) {
        return b;
    }
    if (b.significand == 0) {
        return a;
    }
    if (a.exponent < b.exponent) {
        const Value temporary = a;
        a = b;
        b = temporary;
    }
    const uint32_t difference = a.exponent - b.exponent;
    if (difference > 56U) {
        return a;
    }
    const uint64_t magnitude = b.significand < 0 ? -b.significand : b.significand;
    const uint64_t shifted = magnitude << 3U;
    const uint64_t mask = (1ULL << difference) - 1ULL;
    const uint64_t alignedMagnitude = (shifted >> difference) | ((shifted & mask) != 0ULL);
    const int64_t aligned =
        b.significand < 0 ? -static_cast<int64_t>(alignedMagnitude) : static_cast<int64_t>(alignedMagnitude);
    return Normalize(a.significand * 8 + aligned, a.exponent - 3);
}
__aicore__ inline Value Product(float left, float right)
{
    const Value a = FromFloat(left);
    const Value b = FromFloat(right);
    if (a.exponent == Special || b.exponent == Special) {
        return FromFloat(left * right);
    }
    // Normalized FP32 significands have 29 trailing zero bits.
    return Normalize((a.significand / (1LL << 29)) * (b.significand / (1LL << 29)), a.exponent + b.exponent + 58);
}
__aicore__ inline Value Scale(Value a, float scalar)
{
    const Value b = FromFloat(scalar);
    if (a.exponent == Special || b.exponent == Special) {
        if ((a.exponent != Special && a.significand == 0) || (b.exponent != Special && b.significand == 0)) {
            return {0, Special};
        }
        const bool negative = (a.significand < 0) != (b.significand < 0);
        const bool nan = (a.exponent == Special && a.significand == 0) || (b.exponent == Special && b.significand == 0);
        return {nan ? 0 : (negative ? -1 : 1), Special};
    }
    const int64_t high = a.significand / (1LL << 26);
    const int64_t low = a.significand - high * (1LL << 26);
    const int64_t multiplier = b.significand / (1LL << 29);
    return Add(
        Normalize(high * multiplier, a.exponent + b.exponent + 55),
        Normalize(low * multiplier, a.exponent + b.exponent + 29));
}
__aicore__ inline uint64_t RoundRight(uint64_t value, uint32_t shift)
{
    if (shift == 0U) {
        return value;
    }
    if (shift >= 64U) {
        return 0ULL;
    }
    const uint64_t half = 1ULL << (shift - 1U);
    const uint64_t remainder = value & ((1ULL << shift) - 1ULL);
    const uint64_t result = value >> shift;
    return result + (remainder > half || (remainder == half && (result & 1ULL)));
}
__aicore__ inline float ToFloat(Value value)
{
    Bits result;
    const bool negative = value.significand < 0;
    result.bits = negative ? 0x80000000U : 0U;
    if (value.exponent == Special) {
        result.bits |= value.significand == 0 ? 0x7fc00000U : 0x7f800000U;
        return result.value;
    }
    if (value.significand == 0) {
        return 0.0F;
    }
    value = Normalize(value.significand, value.exponent);
    const uint64_t magnitude = negative ? -value.significand : value.significand;
    int32_t exponent = value.exponent + 52;
    if (exponent < -126) {
        result.bits |= static_cast<uint32_t>(RoundRight(magnitude, -149 - value.exponent));
        return result.value;
    }
    uint64_t rounded = RoundRight(magnitude, 29U);
    if (rounded == (1ULL << 24U)) {
        rounded >>= 1U;
        ++exponent;
    }
    result.bits |= exponent > 127 ?
                       0x7f800000U :
                       (static_cast<uint32_t>(exponent + 127) << 23U) | (static_cast<uint32_t>(rounded) & 0x7fffffU);
    return result.value;
}
struct Complex {
    Value real{0, 0};
    Value imag{0, 0};
};
__aicore__ inline Complex AddComplex(Complex a, Complex b) { return {Add(a.real, b.real), Add(a.imag, b.imag)}; }
__aicore__ inline Complex ConjugateProduct(float ar, float ai, float br, float bi)
{
    return {Add(Product(ar, br), Product(ai, bi)), Add(Product(ai, br), Neg(Product(ar, bi)))};
}
__aicore__ inline Complex ScaleComplex(Complex a, float real, float imag)
{
    if (imag == 0.0F) {
        return {Scale(a.real, real), Scale(a.imag, real)};
    }
    if (real == 0.0F) {
        return {Neg(Scale(a.imag, imag)), Scale(a.real, imag)};
    }
    return {Add(Scale(a.real, real), Neg(Scale(a.imag, imag))), Add(Scale(a.real, imag), Scale(a.imag, real))};
}
} // namespace scaled_scalar
