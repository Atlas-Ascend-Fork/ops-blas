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

// IEEE-754 binary32 layout and scaling used by the compensated division.
constexpr int CTRSV_FLOAT_FRACTION_BITS = 23;
constexpr int CTRSV_FLOAT_SIGNIFICAND_BITS = 24;
constexpr int CTRSV_FLOAT_SPECIAL_EXPONENT = 255;
constexpr uint32_t CTRSV_FLOAT_EXPONENT_MASK = 0xffU;
constexpr uint32_t CTRSV_FLOAT_SIGN_MASK = 0x80000000U;
constexpr uint32_t CTRSV_FLOAT_FRACTION_MASK = 0x007fffffU;
constexpr uint32_t CTRSV_FLOAT_HIDDEN_BIT = 0x00800000U;
constexpr uint32_t CTRSV_FLOAT_INFINITY_BITS = 0x7f800000U;
constexpr uint32_t CTRSV_FLOAT_SIGN_FRACTION_MASK = CTRSV_FLOAT_SIGN_MASK | CTRSV_FLOAT_FRACTION_MASK;
constexpr int CTRSV_SOLUTION_SCALE_EXPONENT = 24;
constexpr float CTRSV_SOLUTION_SCALE = 0x1p24f;
constexpr float CTRSV_SUBNORMAL_NORMALIZE_SCALE = 0x1p24f;
constexpr float CTRSV_SCALED_MIN_NORMAL = 0x1p-102f;
constexpr int CTRSV_SCALED_SUBNORMAL_EXPONENT = -125;
// Below the minimum exponent of a product of two binary32 subnormals (-298).
constexpr int CTRSV_ZERO_PRODUCT_EXPONENT = -600;
constexpr float CTRSV_ROUNDING_HALF = 0.5f;
constexpr float CTRSV_EXCEPTIONAL_INPUT_SCALE = 0.25f;

struct ComplexValue {
    float real;
    float imag;
};

__simt_callee__ inline ComplexValue LoadComplex(__gm__ const float* data, uint64_t elementOffset)
{
    const uint64_t floatOffset = elementOffset * 2;
    return {data[floatOffset], data[floatOffset + 1]};
}

__simt_callee__ inline void StoreComplex(__gm__ float* data, uint64_t elementOffset, const ComplexValue& value)
{
    const uint64_t floatOffset = elementOffset * 2;
    data[floatOffset] = value.real;
    data[floatOffset + 1] = value.imag;
}

__simt_callee__ inline ComplexValue Multiply(const ComplexValue& lhs, const ComplexValue& rhs)
{
    volatile float rr = lhs.real * rhs.real;
    volatile float ii = lhs.imag * rhs.imag;
    volatile float ri = lhs.real * rhs.imag;
    volatile float ir = lhs.imag * rhs.real;
    ComplexValue result = {rr - ii, ri + ir};
    if (isnan(result.real) && isnan(result.imag)) {
        const bool leftInfinite = isinf(lhs.real) || isinf(lhs.imag);
        const bool rightInfinite = isinf(rhs.real) || isinf(rhs.imag);
        if (leftInfinite || rightInfinite) {
            // Preserve the direction of an infinite complex operand. A NaN in
            // its finite companion component must not erase a defined infinity.
            const ComplexValue left = {
                leftInfinite ? copysignf(isinf(lhs.real) ? 1.0f : 0.0f, lhs.real) :
                               (isnan(lhs.real) ? copysignf(0.0f, lhs.real) : lhs.real),
                leftInfinite ? copysignf(isinf(lhs.imag) ? 1.0f : 0.0f, lhs.imag) :
                               (isnan(lhs.imag) ? copysignf(0.0f, lhs.imag) : lhs.imag)};
            const ComplexValue right = {
                rightInfinite ? copysignf(isinf(rhs.real) ? 1.0f : 0.0f, rhs.real) :
                                (isnan(rhs.real) ? copysignf(0.0f, rhs.real) : rhs.real),
                rightInfinite ? copysignf(isinf(rhs.imag) ? 1.0f : 0.0f, rhs.imag) :
                                (isnan(rhs.imag) ? copysignf(0.0f, rhs.imag) : rhs.imag)};
            const float infinity = __builtin_huge_valf();
            result.real = infinity * (left.real * right.real - left.imag * right.imag);
            result.imag = infinity * (left.real * right.imag + left.imag * right.real);
        }
    }
    return result;
}

__simt_callee__ inline void SubtractProduct(ComplexValue& target, const ComplexValue& lhs, const ComplexValue& rhs)
{
    const ComplexValue product = Multiply(lhs, rhs);
    target.real -= product.real;
    target.imag -= product.imag;
}

struct FloatPair {
    float high;
    float low;
};

// Preserve product and division residuals until the final FLOAT32 rounding.
// The Netlib CPU complex division uses wider intermediates than a reciprocal
// rounded to FLOAT32; those lost bits matter after cancellation in later rows.
__simt_callee__ inline FloatPair SumProducts(float a, float b, float c, float d)
{
    const float first = a * b;
    const float second = c * d;
    const float sum = first + second;
    const float secondPart = sum - first;
    const float sumError = (first - (sum - secondPart)) + (second - secondPart);
    return {sum, sumError + fmaf(a, b, -first) + fmaf(c, d, -second)};
}

__simt_callee__ inline FloatPair DividePair(FloatPair numerator, FloatPair denominator)
{
    const float quotient = numerator.high / denominator.high;
    const float residual =
        fmaf(-quotient, denominator.high, numerator.high) + numerator.low - quotient * denominator.low;
    return {quotient, residual / denominator.high};
}

__simt_callee__ inline float ScalePowerOfTwo(float value, int power)
{
    uint32_t bits = __float_as_uint(value);
    int exponent = static_cast<int>((bits >> CTRSV_FLOAT_FRACTION_BITS) & CTRSV_FLOAT_EXPONENT_MASK);
    if (exponent == 0 && value != 0.0f) {
        value *= CTRSV_SUBNORMAL_NORMALIZE_SCALE;
        power -= CTRSV_FLOAT_SIGNIFICAND_BITS;
        bits = __float_as_uint(value);
        exponent = static_cast<int>((bits >> CTRSV_FLOAT_FRACTION_BITS) & CTRSV_FLOAT_EXPONENT_MASK);
    }
    const int shiftedExponent = exponent + power;
    if (exponent == 0 || exponent == CTRSV_FLOAT_SPECIAL_EXPONENT) {
        return value;
    }
    if (exponent > 0 && exponent < CTRSV_FLOAT_SPECIAL_EXPONENT && shiftedExponent > 0 &&
        shiftedExponent < CTRSV_FLOAT_SPECIAL_EXPONENT) {
        return __uint_as_float(
            (bits & CTRSV_FLOAT_SIGN_FRACTION_MASK) |
            (static_cast<uint32_t>(shiftedExponent) << CTRSV_FLOAT_FRACTION_BITS));
    }
    const uint32_t sign = bits & CTRSV_FLOAT_SIGN_MASK;
    if (shiftedExponent >= CTRSV_FLOAT_SPECIAL_EXPONENT) {
        return __uint_as_float(sign | CTRSV_FLOAT_INFINITY_BITS);
    }
    const int shift = 1 - shiftedExponent;
    if (shift > CTRSV_FLOAT_SIGNIFICAND_BITS) {
        return __uint_as_float(sign);
    }
    const uint32_t mantissa = (bits & CTRSV_FLOAT_FRACTION_MASK) | CTRSV_FLOAT_HIDDEN_BIT;
    uint32_t rounded = mantissa >> shift;
    const uint32_t remainder = mantissa & ((1U << shift) - 1U);
    const uint32_t halfway = 1U << (shift - 1);
    rounded += remainder > halfway || (remainder == halfway && (rounded & 1U));
    return __uint_as_float(sign | rounded);
}

__simt_callee__ inline float RoundDivision(FloatPair quotient, int outputExponent, float resultScale)
{
    const float scaledHigh = ScalePowerOfTwo(quotient.high, outputExponent);
    if (resultScale == CTRSV_SOLUTION_SCALE && fabsf(scaledHigh) < CTRSV_SCALED_MIN_NORMAL) {
        // FLOAT32 subnormal spacing is 2^-149, hence 2^-125 at scale 2^24.
        // Round the high/low pair directly to this grid to avoid double rounding.
        const float highUnits = ScalePowerOfTwo(quotient.high, outputExponent - CTRSV_SCALED_SUBNORMAL_EXPONENT);
        const float lowUnits = ScalePowerOfTwo(quotient.low, outputExponent - CTRSV_SCALED_SUBNORMAL_EXPONENT);
        float roundedUnits = rintf(highUnits);
        const float highRemainder = highUnits - roundedUnits;
        const float remainder = highRemainder + lowUnits;
        const float lowPart = remainder - highRemainder;
        const float remainderError = (highRemainder - (remainder - lowPart)) + (lowUnits - lowPart);
        const bool odd = (static_cast<int32_t>(roundedUnits) & 1) != 0;
        if (remainder > CTRSV_ROUNDING_HALF ||
            (remainder == CTRSV_ROUNDING_HALF && (remainderError > 0.0f || (remainderError == 0.0f && odd)))) {
            roundedUnits += 1.0f;
        } else if (
            remainder < -CTRSV_ROUNDING_HALF ||
            (remainder == -CTRSV_ROUNDING_HALF && (remainderError < 0.0f || (remainderError == 0.0f && odd)))) {
            roundedUnits -= 1.0f;
        }
        return ScalePowerOfTwo(roundedUnits, CTRSV_SCALED_SUBNORMAL_EXPONENT);
    }
    return ScalePowerOfTwo(quotient.high + quotient.low, outputExponent);
}

struct ScaledPair {
    FloatPair value;
    int exponent;
};

__simt_callee__ inline ScaledPair ScaledProduct(float a, float b)
{
    if (a == 0.0f || b == 0.0f) {
        return {{a * b, 0.0f}, CTRSV_ZERO_PRODUCT_EXPONENT};
    }
    const int ae = ilogbf(fabsf(a));
    const int be = ilogbf(fabsf(b));
    a = ScalePowerOfTwo(a, -ae);
    b = ScalePowerOfTwo(b, -be);
    const float product = a * b;
    return {{product, fmaf(a, b, -product)}, ae + be};
}

// Scale each product independently: a small component can still determine an
// entire output component even when the other output overflows.
__simt_callee__ inline ScaledPair ScaledSumProducts(float a, float b, float c, float d)
{
    const ScaledPair first = ScaledProduct(a, b);
    const ScaledPair second = ScaledProduct(c, d);
    const int exponent = first.exponent > second.exponent ? first.exponent : second.exponent;
    const float high1 = ScalePowerOfTwo(first.value.high, first.exponent - exponent);
    const float high2 = ScalePowerOfTwo(second.value.high, second.exponent - exponent);
    FloatPair sum = SumProducts(high1, 1.0f, high2, 1.0f);
    sum.low += ScalePowerOfTwo(first.value.low, first.exponent - exponent) +
               ScalePowerOfTwo(second.value.low, second.exponent - exponent);
    return {sum, exponent};
}

__simt_callee__ inline ComplexValue DivideFinite(const ComplexValue& lhs, const ComplexValue& rhs, float resultScale)
{
    const float numeratorMax = fmaxf(fabsf(lhs.real), fabsf(lhs.imag));
    if (numeratorMax == 0.0f) {
        return {0.0f, 0.0f};
    }
    const ScaledPair denominator = ScaledSumProducts(rhs.real, rhs.real, rhs.imag, rhs.imag);
    const ScaledPair real = ScaledSumProducts(lhs.real, rhs.real, lhs.imag, rhs.imag);
    const ScaledPair imag = ScaledSumProducts(lhs.imag, rhs.real, -lhs.real, rhs.imag);
    const int scaleExponent = resultScale == CTRSV_SOLUTION_SCALE ? CTRSV_SOLUTION_SCALE_EXPONENT : 0;
    return {
        RoundDivision(
            DividePair(real.value, denominator.value), real.exponent - denominator.exponent + scaleExponent,
            resultScale),
        RoundDivision(
            DividePair(imag.value, denominator.value), imag.exponent - denominator.exponent + scaleExponent,
            resultScale)};
}

__simt_callee__ inline ComplexValue DivideExceptional(
    const ComplexValue& lhs, const ComplexValue& rhs, bool numeratorFinite, bool denominatorFinite)
{
    const float infinity = __builtin_huge_valf();
    const float notANumber = __builtin_nanf("");
    ComplexValue result = {notANumber, notANumber};
    if (rhs.real == 0.0f && rhs.imag == 0.0f) {
        return {copysignf(infinity, rhs.real) * lhs.real, copysignf(infinity, rhs.real) * lhs.imag};
    }
    if (numeratorFinite && (isinf(rhs.real) || isinf(rhs.imag))) {
        const float realDirection = copysignf(isinf(rhs.real) ? 1.0f : 0.0f, rhs.real);
        const float imagDirection = copysignf(isinf(rhs.imag) ? 1.0f : 0.0f, rhs.imag);
        return {
            0.0f * ((lhs.real * CTRSV_EXCEPTIONAL_INPUT_SCALE) * realDirection +
                    (lhs.imag * CTRSV_EXCEPTIONAL_INPUT_SCALE) * imagDirection),
            0.0f * ((lhs.imag * CTRSV_EXCEPTIONAL_INPUT_SCALE) * realDirection -
                    (lhs.real * CTRSV_EXCEPTIONAL_INPUT_SCALE) * imagDirection)};
    }
    if (!numeratorFinite && denominatorFinite) {
        // For finite nonzero divisors the squared magnitude is positive.
        // Use the original divisor to retain nonzero components whose
        // reciprocal would underflow before multiplication by infinity.
        result.real = lhs.real * rhs.real + lhs.imag * rhs.imag;
        result.imag = lhs.imag * rhs.real - lhs.real * rhs.imag;
    }
    // An infinite numerator can still have a defined infinite component even
    // when ordinary complex arithmetic produces NaN in both components.
    if (isnan(result.real) && isnan(result.imag) && (isinf(lhs.real) || isinf(lhs.imag)) && denominatorFinite) {
        const float realDirection = copysignf(isinf(lhs.real) ? 1.0f : 0.0f, lhs.real);
        const float imagDirection = copysignf(isinf(lhs.imag) ? 1.0f : 0.0f, lhs.imag);
        result.real = infinity * (realDirection * rhs.real + imagDirection * rhs.imag);
        result.imag = infinity * (imagDirection * rhs.real - realDirection * rhs.imag);
    }
    return result;
}

__simt_callee__ inline ComplexValue Divide(const ComplexValue& lhs, const ComplexValue& rhs, float resultScale)
{
    const bool numeratorFinite = isfinite(lhs.real) && isfinite(lhs.imag);
    const bool denominatorFinite = isfinite(rhs.real) && isfinite(rhs.imag);
    if (numeratorFinite && denominatorFinite && (rhs.real != 0.0f || rhs.imag != 0.0f)) {
        return DivideFinite(lhs, rhs, resultScale);
    }
    return DivideExceptional(lhs, rhs, numeratorFinite, denominatorFinite);
}
