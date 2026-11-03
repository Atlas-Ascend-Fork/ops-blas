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

// Preserve the exact 48-bit product and round once, nearest ties to even.
#ifndef STBSV_FMA_INLINE
#define STBSV_FMA_INLINE __aicore__ inline
#endif

// IEEE-754 binary32: a normal value is M * 2^(e - VALUE_SCALE),
// where M includes the hidden bit and therefore has FRAC_BITS + 1 bits.
#define STBSV_FMA_FRAC_BITS 23
#define STBSV_FMA_EXP_BIAS 127
#define STBSV_FMA_VALUE_SCALE (STBSV_FMA_EXP_BIAS + STBSV_FMA_FRAC_BITS)
#define STBSV_FMA_WORD_BITS 64

// Preserve the conservative fast-path cutoff: ea + eb >= 194 makes the
// product LSB exponent ea + eb - 2 * VALUE_SCALE at least -106. Even a
// one-bit cancellation result stays normal (-106 > the FP32 minimum -126).
#define STBSV_FMA_FAST_MIN_PRODUCT_EXP (-106)
#define STBSV_FMA_FAST_MIN_EXP_SUM (2 * STBSV_FMA_VALUE_SCALE + STBSV_FMA_FAST_MIN_PRODUCT_EXP)

// Reserve sign/headroom in the 64-bit accumulator. Each summand uses at
// most SUM_BITS - 1 bits, so their sum is below 2^SUM_BITS = 2^62.
// A 24-bit C significand can thus be shifted by at most 62 - 1 - 24 = 37.
#define STBSV_FMA_SUM_BITS (STBSV_FMA_WORD_BITS - 2)
#define STBSV_FMA_MAX_C_SHIFT (STBSV_FMA_SUM_BITS - 1 - (STBSV_FMA_FRAC_BITS + 1))

// rounded already contains the hidden bit. Encoding the exponent one lower
// gives ea + eb + top - (2 * VALUE_SCALE - EXP_BIAS + 1), i.e. offset 174.
#define STBSV_FMA_ENCODE_BIAS (2 * STBSV_FMA_VALUE_SCALE - STBSV_FMA_EXP_BIAS + 1)

namespace StbsvArithmetic {
union FloatBits {
    float value;
    uint32_t bits;
};

// Right shift with jamming: bit 0 records whether any discarded bit was 1.
// Keeping this sticky bit lets later rounding distinguish an exact tail from
// a nonzero tail even when the original low bits no longer fit.
STBSV_FMA_INLINE uint64_t ShiftJam(uint64_t value, uint32_t distance)
{
    if (distance == 0) {
        return value;
    }
    if (distance >= STBSV_FMA_WORD_BITS) {
        // All bits are discarded: return integer 1 for a nonzero tail, otherwise 0.
        return static_cast<uint64_t>(value != 0);
    }
    return (value >> distance) | static_cast<uint64_t>((value << (STBSV_FMA_WORD_BITS - distance)) != 0);
}

STBSV_FMA_INLINE float RoundMagnitude(uint64_t magnitude, uint32_t sign, int32_t exponent)
{
    if (magnitude == 0) {
        return 0.0f;
    }
    int32_t top = 63 - __builtin_clzll(magnitude);
    exponent += top - 61;
    int32_t shift = top - 23;
    if (exponent < -126) {
        shift += -126 - exponent;
        exponent = -126;
    }
    uint64_t rounded = 0;
    if (shift < 64) {
        if (shift > 0) {
            rounded = magnitude >> shift;
            uint64_t remainder = magnitude & ((uint64_t{1} << shift) - 1);
            uint64_t halfway = uint64_t{1} << (shift - 1);
            rounded += remainder > halfway || (remainder == halfway && (rounded & 1));
        } else {
            rounded = magnitude << -shift;
        }
    }
    if (rounded >= 0x1000000) {
        rounded >>= 1;
        ++exponent;
    }
    FloatBits result;
    if (exponent > 127) {
        result.bits = sign | 0x7f800000;
    } else if (rounded < 0x800000) {
        result.bits = sign | static_cast<uint32_t>(rounded);
    } else {
        result.bits =
            sign | (static_cast<uint32_t>(exponent + 127) << 23) | (static_cast<uint32_t>(rounded) & 0x7fffff);
    }
    return result.value;
}

STBSV_FMA_INLINE float CombineMagnitudes(
    uint64_t product, uint64_t addend, uint32_t sign, uint32_t signC, int32_t exponent)
{
    uint64_t magnitude;
    if (sign == signC) {
        magnitude = product + addend;
    } else if (product >= addend) {
        magnitude = product - addend;
    } else {
        magnitude = addend - product;
        sign = signC;
    }
    return RoundMagnitude(magnitude, sign, exponent);
}

STBSV_FMA_INLINE float FiniteFma(FloatBits ua, FloatBits ub, FloatBits uc)
{
    uint32_t ea = (ua.bits >> 23) & 255;
    uint32_t eb = (ub.bits >> 23) & 255;
    uint32_t ec = (uc.bits >> 23) & 255;
    int32_t expA = ea ? static_cast<int32_t>(ea) - 127 : -126;
    int32_t expB = eb ? static_cast<int32_t>(eb) - 127 : -126;
    int32_t expC = ec ? static_cast<int32_t>(ec) - 127 : -126;
    uint32_t ma = (ua.bits & 0x7fffff) | (ea ? 0x800000 : 0);
    uint32_t mb = (ub.bits & 0x7fffff) | (eb ? 0x800000 : 0);
    uint32_t mc = (uc.bits & 0x7fffff) | (ec ? 0x800000 : 0);
    uint64_t product = static_cast<uint64_t>(ma) * mb;
    int32_t topP = ea != 0 && eb != 0 ? 46 + ((product >> 47) != 0) : 63 - __builtin_clzll(product);
    int32_t expP = expA + expB + topP - 46;
    product <<= 61 - topP;
    uint64_t addend = 0;
    if (mc != 0) {
        int32_t topC = ec != 0 ? 23 : 31 - __builtin_clz(mc);
        expC += topC - 23;
        addend = static_cast<uint64_t>(mc) << (61 - topC);
    } else {
        expC = expP;
    }
    int32_t exponent = expP;
    if (expP > expC) {
        addend = ShiftJam(addend, expP - expC);
    } else if (expP < expC) {
        product = ShiftJam(product, expC - expP);
        exponent = expC;
    }
    return CombineMagnitudes(product, addend, (ua.bits ^ ub.bits) & 0x80000000, uc.bits & 0x80000000, exponent);
}

STBSV_FMA_INLINE float FmaFallback(float a, float b, float c)
{
    FloatBits ua{a};
    FloatBits ub{b};
    FloatBits uc{c};
    uint32_t ea = (ua.bits >> 23) & 255;
    uint32_t eb = (ub.bits >> 23) & 255;
    uint32_t ec = (uc.bits >> 23) & 255;
    if (ec == 255 && ea != 255 && eb != 255) {
        return c + c;
    }
    if (ea == 255 || eb == 255 || ec == 255 || (ea == 0 && (ua.bits & 0x7fffff) == 0) ||
        (eb == 0 && (ub.bits & 0x7fffff) == 0)) {
        return a * b + c;
    }
    return FiniteFma(ua, ub, uc);
}

STBSV_FMA_INLINE float Fma(float a, float b, float c)
{
    FloatBits ua{a};
    FloatBits ub{b};
    FloatBits uc{c};
    uint32_t ea = (ua.bits >> 23) & 255;
    uint32_t eb = (ub.bits >> 23) & 255;
    uint32_t ec = (uc.bits >> 23) & 255;
    uint32_t ma = ua.bits & 0x7fffff;
    uint32_t mb = ub.bits & 0x7fffff;
    uint32_t mc = uc.bits & 0x7fffff;
    if (ea != 0 && eb != 0 && ea != 255 && eb != 255 && ec != 0 && ec != 255 && ea + eb >= STBSV_FMA_FAST_MIN_EXP_SUM) {
        uint64_t exactProduct = static_cast<uint64_t>(ma | 0x800000) * (mb | 0x800000);
        // Align C to the product LSB: (ec - VALUE_SCALE) -
        // (ea + eb - 2 * VALUE_SCALE) = ec - ea - eb + VALUE_SCALE (150).
        int32_t cShift = static_cast<int32_t>(ec) - static_cast<int32_t>(ea + eb) + STBSV_FMA_VALUE_SCALE;
        if (cShift >= 0 && cShift <= STBSV_FMA_MAX_C_SHIFT) {
            // The signed integer sum is exact and bounded below 2^62.
            uint64_t alignedC = static_cast<uint64_t>(mc | 0x800000) << cShift;
            int64_t productSign = -static_cast<int64_t>((ua.bits ^ ub.bits) >> 31);
            int64_t cSign = -static_cast<int64_t>(uc.bits >> 31);
            int64_t signedProduct = (static_cast<int64_t>(exactProduct) ^ productSign) - productSign;
            int64_t signedC = (static_cast<int64_t>(alignedC) ^ cSign) - cSign;
            int64_t signedSum = signedProduct + signedC;
            int64_t sumSign = signedSum >> 63;
            uint32_t sign = static_cast<uint32_t>(sumSign) & 0x80000000;
            uint64_t magnitude = static_cast<uint64_t>((signedSum ^ sumSign) - sumSign);
            if (magnitude == 0) {
                return 0.0f;
            }
            int32_t top = 63 - __builtin_clzll(magnitude);
            int32_t shift = top - 23;
            uint32_t rounded;
            if (shift > 0) {
                uint64_t increment = (uint64_t{1} << (shift - 1)) - 1 + ((magnitude >> shift) & 1);
                rounded = static_cast<uint32_t>((magnitude + increment) >> shift);
            } else {
                rounded = static_cast<uint32_t>(magnitude << -shift);
            }
            FloatBits result;
            uint32_t encoded =
                static_cast<uint32_t>(static_cast<int32_t>(ea + eb) + top - STBSV_FMA_ENCODE_BIAS) * 0x800000 + rounded;
            result.bits = sign | (encoded >= 0x7f800000 ? uint32_t{0x7f800000} : encoded);
            return result.value;
        }
    }
    return FmaFallback(a, b, c);
}
} // namespace StbsvArithmetic
