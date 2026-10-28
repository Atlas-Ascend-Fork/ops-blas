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

#ifndef CTBSV_ARITH_ATTR
#define CTBSV_ARITH_ATTR __aicore__ inline
#endif

#ifndef CTBSV_REF_ARITH_ATTR
#define CTBSV_REF_ARITH_ATTR __aicore__ inline
#endif

// IEEE 754 binary32: one sign bit, eight biased exponent bits and 23 stored
// fraction bits. Normal values have an additional implicit leading bit.
namespace CtbsvFp32 {
constexpr int32_t FRACTION_BITS = 23;
constexpr int32_t EXPONENT_BITS = 8;
constexpr int32_t SIGN_SHIFT = FRACTION_BITS + EXPONENT_BITS;
constexpr int32_t EXPONENT_BIAS = (1 << (EXPONENT_BITS - 1)) - 1;
constexpr int32_t MIN_NORMAL_EXPONENT = 1 - EXPONENT_BIAS;
constexpr int32_t MAX_NORMAL_EXPONENT = EXPONENT_BIAS;
constexpr uint32_t EXPONENT_MASK = (1U << EXPONENT_BITS) - 1U;
constexpr uint32_t FRACTION_MASK = (1U << FRACTION_BITS) - 1U;
constexpr uint32_t HIDDEN_BIT = 1U << FRACTION_BITS;
constexpr uint32_t SIGNIFICAND_CARRY = HIDDEN_BIT << 1;
constexpr uint32_t INFINITY_BITS = EXPONENT_MASK << FRACTION_BITS;
constexpr float MAX_FINITE = 3.402823466e+38f;
constexpr float OVERFLOW_RECOVERY_SCALE = 5.421010862427522e-20f; // 2^-64

// The integer FMA places the leading bit at bit 62 of a uint64_t, reserving
// bit 63 for a carry. Rounding to binary32 discards the low 62 - 23 bits.
constexpr int32_t WIDE_BITS = sizeof(uint64_t) * 8;
constexpr int32_t WIDE_CARRY_BIT = WIDE_BITS - 1;
constexpr int32_t WIDE_LEADING_BIT = WIDE_CARRY_BIT - 1;
constexpr int32_t ROUND_SHIFT = WIDE_LEADING_BIT - FRACTION_BITS;
constexpr int32_t PRODUCT_LOW_BIT = 2 * FRACTION_BITS;
constexpr int32_t PRODUCT_HIGH_BIT = PRODUCT_LOW_BIT + 1;
} // namespace CtbsvFp32

CTBSV_ARITH_ATTR float CtbsvAbsf(float x) { return (x >= 0.0f) ? x : -x; }

CTBSV_ARITH_ATTR bool CtbsvIsNan(float x) { return x != x; }

CTBSV_ARITH_ATTR bool CtbsvIsInf(float x) { return CtbsvAbsf(x) > CtbsvFp32::MAX_FINITE; }

CTBSV_ARITH_ATTR bool CtbsvIsFinite(float x) { return (!CtbsvIsNan(x)) && (!CtbsvIsInf(x)); }

CTBSV_ARITH_ATTR float CtbsvCopysign(float mag, float signSrc)
{
    float absMag = CtbsvAbsf(mag);
    return (signSrc < 0.0f) ? -absMag : absMag;
}

CTBSV_ARITH_ATTR float CtbsvInf() { return CtbsvFp32::MAX_FINITE * CtbsvFp32::MAX_FINITE; }

CTBSV_ARITH_ATTR bool CtbsvIsCzero(float re, float im) { return (re == 0.0f) && (im == 0.0f); }

CTBSV_ARITH_ATTR void CtbsvRecenterInfOperand(float& xRe, float& xIm, float& yRe, float& yIm, bool& recalc)
{
    if (!(CtbsvIsInf(xRe) || CtbsvIsInf(xIm))) {
        return;
    }
    xRe = CtbsvCopysign(CtbsvIsInf(xRe) ? 1.0f : 0.0f, xRe);
    xIm = CtbsvCopysign(CtbsvIsInf(xIm) ? 1.0f : 0.0f, xIm);
    if (CtbsvIsNan(yRe)) {
        yRe = CtbsvCopysign(0.0f, yRe);
    }
    if (CtbsvIsNan(yIm)) {
        yIm = CtbsvCopysign(0.0f, yIm);
    }
    recalc = true;
}

CTBSV_ARITH_ATTR void CtbsvCmulRecover(
    float aRe, float aIm, float bRe, float bIm, float ac, float bd, float ad, float bc, float& pRe, float& pIm)
{
    bool recalc = false;
    CtbsvRecenterInfOperand(aRe, aIm, bRe, bIm, recalc);
    CtbsvRecenterInfOperand(bRe, bIm, aRe, aIm, recalc);
    if (!recalc && (CtbsvIsInf(ac) || CtbsvIsInf(bd) || CtbsvIsInf(ad) || CtbsvIsInf(bc))) {
        const float tiny = CtbsvFp32::OVERFLOW_RECOVERY_SCALE;
        aRe *= tiny;
        aIm *= tiny;
        bRe *= tiny;
        bIm *= tiny;
        recalc = true;
    }
    if (recalc) {
        const float inf = CtbsvInf();
        pRe = inf * (aRe * bRe - aIm * bIm);
        pIm = inf * (aRe * bIm + aIm * bRe);
    }
}

// libgcc __mulsc3: recover Inf-Inf NaN so overflow matches OpenBLAS/compiler complex mul.
CTBSV_ARITH_ATTR void CtbsvCmul(float aRe, float aIm, float bRe, float bIm, float& pRe, float& pIm)
{
    float ac = aRe * bRe;
    float bd = aIm * bIm;
    float ad = aRe * bIm;
    float bc = aIm * bRe;
    pRe = ac - bd;
    pIm = ad + bc;
    if (CtbsvIsNan(pRe) && CtbsvIsNan(pIm)) {
        CtbsvCmulRecover(aRe, aIm, bRe, bIm, ac, bd, ad, bc, pRe, pIm);
    }
}

CTBSV_ARITH_ATTR float CtbsvRoundMul(float a, float b)
{
    volatile float value = a * b;
    return value;
}

union CtbsvFloatBits {
    float f;
    uint32_t u;
};

struct CtbsvDecodedFloat {
    bool sign;
    int32_t exp;
    uint32_t mant;
};

CTBSV_ARITH_ATTR uint32_t CtbsvToBits(float value)
{
    CtbsvFloatBits bits{};
    bits.f = value;
    return bits.u;
}

CTBSV_ARITH_ATTR float CtbsvFromBits(uint32_t value)
{
    CtbsvFloatBits bits{};
    bits.u = value;
    return bits.f;
}

CTBSV_ARITH_ATTR CtbsvDecodedFloat CtbsvDecodeFinite(float value)
{
    const uint32_t bits = CtbsvToBits(value);
    CtbsvDecodedFloat decoded{
        (bits >> CtbsvFp32::SIGN_SHIFT) != 0U,
        static_cast<int32_t>((bits >> CtbsvFp32::FRACTION_BITS) & CtbsvFp32::EXPONENT_MASK),
        bits & CtbsvFp32::FRACTION_MASK};
    if (decoded.exp == 0) {
        decoded.exp = CtbsvFp32::MIN_NORMAL_EXPONENT;
        while (decoded.mant != 0U && (decoded.mant & CtbsvFp32::HIDDEN_BIT) == 0U) {
            decoded.mant <<= 1;
            --decoded.exp;
        }
    } else {
        decoded.exp -= CtbsvFp32::EXPONENT_BIAS;
        decoded.mant |= CtbsvFp32::HIDDEN_BIT;
    }
    return decoded;
}

CTBSV_ARITH_ATTR uint64_t CtbsvShiftRightJam(uint64_t value, int32_t distance)
{
    if (distance <= 0) {
        return value;
    }
    if (distance < CtbsvFp32::WIDE_BITS) {
        const uint64_t extra = value << (CtbsvFp32::WIDE_BITS - distance);
        return (value >> distance) | static_cast<uint64_t>(extra != 0U);
    }
    return static_cast<uint64_t>(value != 0U);
}

// Round to nearest, ties to even. The caller guarantees 0 < shift < WIDE_BITS.
CTBSV_ARITH_ATTR uint64_t CtbsvRoundSignificand(uint64_t significand, int32_t shift)
{
    uint64_t mantissa = significand >> shift;
    const uint64_t mask = (1ULL << shift) - 1ULL;
    const uint64_t discarded = significand & mask;
    const uint64_t half = 1ULL << (shift - 1);
    if (discarded > half || (discarded == half && (mantissa & 1U) != 0U)) {
        ++mantissa;
    }
    return mantissa;
}

CTBSV_ARITH_ATTR float CtbsvPackRounded(bool sign, int32_t exp, uint64_t significand)
{
    if (significand == 0U) {
        return CtbsvFromBits(static_cast<uint32_t>(sign) << CtbsvFp32::SIGN_SHIFT);
    }
    if ((significand & (1ULL << CtbsvFp32::WIDE_CARRY_BIT)) != 0U) {
        significand = CtbsvShiftRightJam(significand, 1);
        ++exp;
    }
    while ((significand & (1ULL << CtbsvFp32::WIDE_LEADING_BIT)) == 0U) {
        significand <<= 1;
        --exp;
    }
    if (exp > CtbsvFp32::MAX_NORMAL_EXPONENT) {
        return CtbsvFromBits((static_cast<uint32_t>(sign) << CtbsvFp32::SIGN_SHIFT) | CtbsvFp32::INFINITY_BITS);
    }
    int32_t shift = CtbsvFp32::ROUND_SHIFT;
    uint32_t expField = 0U;
    if (exp >= CtbsvFp32::MIN_NORMAL_EXPONENT) {
        expField = static_cast<uint32_t>(exp + CtbsvFp32::EXPONENT_BIAS);
    } else {
        shift += CtbsvFp32::MIN_NORMAL_EXPONENT - exp;
        if (shift >= CtbsvFp32::WIDE_BITS) {
            return CtbsvFromBits(static_cast<uint32_t>(sign) << CtbsvFp32::SIGN_SHIFT);
        }
    }
    uint64_t mantissa = CtbsvRoundSignificand(significand, shift);
    if (expField == 0U) {
        if (mantissa >= CtbsvFp32::HIDDEN_BIT) {
            return CtbsvFromBits((static_cast<uint32_t>(sign) << CtbsvFp32::SIGN_SHIFT) | CtbsvFp32::HIDDEN_BIT);
        }
        return CtbsvFromBits((static_cast<uint32_t>(sign) << CtbsvFp32::SIGN_SHIFT) | static_cast<uint32_t>(mantissa));
    }
    if (mantissa >= CtbsvFp32::SIGNIFICAND_CARRY) {
        mantissa >>= 1;
        ++expField;
        if (expField >= CtbsvFp32::EXPONENT_MASK) {
            return CtbsvFromBits((static_cast<uint32_t>(sign) << CtbsvFp32::SIGN_SHIFT) | CtbsvFp32::INFINITY_BITS);
        }
    }
    const uint32_t out = (static_cast<uint32_t>(sign) << CtbsvFp32::SIGN_SHIFT) |
                         (expField << CtbsvFp32::FRACTION_BITS) |
                         (static_cast<uint32_t>(mantissa) & CtbsvFp32::FRACTION_MASK);
    return CtbsvFromBits(out);
}

CTBSV_ARITH_ATTR float CtbsvIntegerFma(float a, float b, float c)
{
    if (!CtbsvIsFinite(a) || !CtbsvIsFinite(b) || !CtbsvIsFinite(c)) {
        return a * b + c;
    }
    const CtbsvDecodedFloat decodedA = CtbsvDecodeFinite(a);
    const CtbsvDecodedFloat decodedB = CtbsvDecodeFinite(b);
    const CtbsvDecodedFloat decodedC = CtbsvDecodeFinite(c);
    if (decodedA.mant == 0U || decodedB.mant == 0U) {
        return c;
    }
    const uint64_t product = static_cast<uint64_t>(decodedA.mant) * static_cast<uint64_t>(decodedB.mant);
    const int32_t productMsb = (product & (1ULL << CtbsvFp32::PRODUCT_HIGH_BIT)) != 0U ? CtbsvFp32::PRODUCT_HIGH_BIT :
                                                                                         CtbsvFp32::PRODUCT_LOW_BIT;
    const int32_t productExp = decodedA.exp + decodedB.exp + productMsb - CtbsvFp32::PRODUCT_LOW_BIT;
    uint64_t productSig = product << (CtbsvFp32::WIDE_LEADING_BIT - productMsb);
    const bool productSign = decodedA.sign != decodedB.sign;
    if (decodedC.mant == 0U) {
        return CtbsvPackRounded(productSign, productExp, productSig);
    }
    uint64_t cSig = static_cast<uint64_t>(decodedC.mant) << CtbsvFp32::ROUND_SHIFT;
    const int32_t commonExp = productExp > decodedC.exp ? productExp : decodedC.exp;
    productSig = CtbsvShiftRightJam(productSig, commonExp - productExp);
    cSig = CtbsvShiftRightJam(cSig, commonExp - decodedC.exp);
    uint64_t sumSig = 0U;
    bool sumSign = false;
    if (productSign == decodedC.sign) {
        sumSig = productSig + cSig;
        sumSign = productSign;
    } else if (productSig >= cSig) {
        sumSig = productSig - cSig;
        sumSign = productSign;
    } else {
        sumSig = cSig - productSig;
        sumSign = decodedC.sign;
    }
    return CtbsvPackRounded(sumSign, commonExp, sumSig);
}

CTBSV_REF_ARITH_ATTR void CtbsvCmulReferenceSoft(float aRe, float aIm, float bRe, float bIm, float& pRe, float& pIm)
{
    const float bd = CtbsvRoundMul(aIm, bIm);
    const float ad = CtbsvRoundMul(aRe, bIm);
    pRe = CtbsvIntegerFma(aRe, bRe, -bd);
    pIm = CtbsvIntegerFma(aIm, bRe, ad);
}

CTBSV_ARITH_ATTR void CtbsvDivZeroDiag(float sumRe, float sumIm, float diagRe, float& outRe, float& outIm)
{
    float signedInf = CtbsvCopysign(CtbsvInf(), diagRe);
    outRe = signedInf * sumRe;
    outIm = signedInf * sumIm;
}

CTBSV_ARITH_ATTR bool CtbsvSmithDiv(
    float sumRe, float sumIm, float diagRe, float diagIm, float& outRe, float& outIm, float& denom)
{
    denom = 0.0f;
    if (CtbsvAbsf(diagRe) < CtbsvAbsf(diagIm)) {
        if (diagIm == 0.0f) {
            CtbsvDivZeroDiag(sumRe, sumIm, diagRe, outRe, outIm);
            return false;
        }
        float ratio = diagRe / diagIm;
        denom = (diagRe * ratio) + diagIm;
        if (denom == 0.0f) {
            CtbsvDivZeroDiag(sumRe, sumIm, diagRe, outRe, outIm);
            return false;
        }
        outRe = (sumRe * ratio + sumIm) / denom;
        outIm = (sumIm * ratio - sumRe) / denom;
        return true;
    }
    if (diagRe == 0.0f) {
        CtbsvDivZeroDiag(sumRe, sumIm, diagRe, outRe, outIm);
        return false;
    }
    float ratio = diagIm / diagRe;
    denom = (diagIm * ratio) + diagRe;
    if (denom == 0.0f) {
        CtbsvDivZeroDiag(sumRe, sumIm, diagRe, outRe, outIm);
        return false;
    }
    outRe = (sumRe + sumIm * ratio) / denom;
    outIm = (sumIm - sumRe * ratio) / denom;
    return true;
}

CTBSV_ARITH_ATTR void CtbsvComplexDivRecover(
    float sumRe, float sumIm, float diagRe, float diagIm, float denom, float& outRe, float& outIm)
{
    if (denom == 0.0f && (!CtbsvIsNan(sumRe) || !CtbsvIsNan(sumIm))) {
        CtbsvDivZeroDiag(sumRe, sumIm, diagRe, outRe, outIm);
        return;
    }
    if ((CtbsvIsInf(sumRe) || CtbsvIsInf(sumIm)) && CtbsvIsFinite(diagRe) && CtbsvIsFinite(diagIm)) {
        float a = CtbsvCopysign(CtbsvIsInf(sumRe) ? 1.0f : 0.0f, sumRe);
        float b = CtbsvCopysign(CtbsvIsInf(sumIm) ? 1.0f : 0.0f, sumIm);
        const float inf = CtbsvInf();
        outRe = inf * (a * diagRe + b * diagIm);
        outIm = inf * (b * diagRe - a * diagIm);
        return;
    }
    if ((CtbsvIsInf(diagRe) || CtbsvIsInf(diagIm)) && CtbsvIsFinite(sumRe) && CtbsvIsFinite(sumIm)) {
        float c = CtbsvCopysign(CtbsvIsInf(diagRe) ? 1.0f : 0.0f, diagRe);
        float d = CtbsvCopysign(CtbsvIsInf(diagIm) ? 1.0f : 0.0f, diagIm);
        outRe = 0.0f * (sumRe * c + sumIm * d);
        outIm = 0.0f * (sumIm * c - sumRe * d);
    }
}

// libgcc __divsc3 / Smith cdiv, then recover 0-diag and overflow.
CTBSV_ARITH_ATTR void CtbsvComplexDiv(float sumRe, float sumIm, float diagRe, float diagIm, float& outRe, float& outIm)
{
    float denom = 0.0f;
    if (!CtbsvSmithDiv(sumRe, sumIm, diagRe, diagIm, outRe, outIm, denom)) {
        return;
    }
    if (CtbsvIsNan(outRe) && CtbsvIsNan(outIm)) {
        CtbsvComplexDivRecover(sumRe, sumIm, diagRe, diagIm, denom, outRe, outIm);
    }
}

CTBSV_ARITH_ATTR void CtbsvComplexDivByConj(
    float sumRe, float sumIm, float diagRe, float diagIm, float& outRe, float& outIm)
{
    CtbsvComplexDiv(sumRe, sumIm, diagRe, -diagIm, outRe, outIm);
}
