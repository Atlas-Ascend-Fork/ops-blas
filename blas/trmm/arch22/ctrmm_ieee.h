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
#ifndef __CCE_AICORE__
#define CTRMM_SCALAR inline
#else
#define CTRMM_SCALAR __aicore__ inline
#endif

namespace ctrmm_ieee {
// IEEE binary32 fields and the integer accumulator used for single rounding.
constexpr uint32_t kSignMask = 0x80000000U;
constexpr uint32_t kMagnitudeMask = 0x7fffffffU;
constexpr uint32_t kInfinityBits = 0x7f800000U;
constexpr uint32_t kQuietNanBits = 0x7fc00000U;
constexpr uint32_t kFractionMask = 0x7fffffU;
constexpr uint32_t kImplicitOne = 0x800000U;
constexpr uint32_t kSignificandCarry = 0x1000000U;
constexpr uint32_t kMinNormalBits = 0x00800000U;
constexpr int kSignShift = 31;
constexpr int kFractionBits = 23;
constexpr int kExponentBias = 127;
constexpr int kMinNormalExponent = -126;
constexpr int kMinSubnormalExponent = -149;
constexpr int kSignificandExponentBias = 150;
constexpr int kMaxNormalBiasedExponent = 254;
constexpr int kAccumulatorTopBit = 61;
constexpr int kWideBits = 64;
constexpr int kWideTopBit = 63;

union Bits {
    float value;
    uint32_t bits;
};
CTRMM_SCALAR uint32_t BitsOf(float x)
{
    Bits b;
    b.value = x;
    return b.bits;
}
CTRMM_SCALAR float FromBits(uint32_t x)
{
    Bits b;
    b.bits = x;
    return b.value;
}
CTRMM_SCALAR int TopBit(uint64_t x)
{
#ifdef __CCE_AICORE__
    return kWideTopBit - static_cast<int>(AscendC::CountLeadingZero(x));
#else
    int n = 0;
    if (x >> 32) {
        x >>= 32;
        n += 32;
    }
    if (x >> 16) {
        x >>= 16;
        n += 16;
    }
    if (x >> 8) {
        x >>= 8;
        n += 8;
    }
    if (x >> 4) {
        x >>= 4;
        n += 4;
    }
    if (x >> 2) {
        x >>= 2;
        n += 2;
    }
    if (x >> 1) {
        ++n;
    }
    return n;
#endif
}
CTRMM_SCALAR uint64_t ShiftJam(uint64_t x, int shift)
{
    if (shift >= 0) {
        return x << shift;
    }
    int right = -shift;
    if (right >= kWideBits) {
        return x != 0;
    }
    return (x >> right) | ((x & ((uint64_t{1} << right) - 1)) != 0);
}
CTRMM_SCALAR float NonFiniteFma(uint32_t aa, uint32_t bb, uint32_t cc)
{
    const uint32_t ax = aa & kMagnitudeMask;
    const uint32_t bx = bb & kMagnitudeMask;
    const uint32_t cx = cc & kMagnitudeMask;
    const uint32_t signP = (aa ^ bb) >> kSignShift;
    const uint32_t signC = cc >> kSignShift;
    if (ax > kInfinityBits || bx > kInfinityBits || cx > kInfinityBits) {
        return FromBits(kQuietNanBits);
    }
    if (ax == kInfinityBits || bx == kInfinityBits) {
        if (ax == 0 || bx == 0 || (cx == kInfinityBits && signP != signC)) {
            return FromBits(kQuietNanBits);
        }
        return FromBits((signP << kSignShift) | kInfinityBits);
    }
    return FromBits(cc);
}

CTRMM_SCALAR float RoundSum(uint64_t s, uint32_t sign, int base)
{
    if (s == 0) {
        return 0.0f;
    }
    int highest = TopBit(s);
    int exponent = highest + base;
    int shift = exponent >= kMinNormalExponent ? highest - kFractionBits : kMinSubnormalExponent - base;
    uint64_t rounded;
    if (shift > kWideTopBit) {
        rounded = 0;
    } else if (shift <= 0) {
        rounded = s << (-shift);
    } else {
        rounded = s >> shift;
        uint64_t remainder = s & ((uint64_t{1} << shift) - 1);
        uint64_t half = uint64_t{1} << (shift - 1);
        if (remainder > half || (remainder == half && (rounded & 1))) {
            ++rounded;
        }
    }
    if (exponent < kMinNormalExponent) {
        return FromBits((sign << kSignShift) | static_cast<uint32_t>(rounded));
    }
    if (rounded >= kSignificandCarry) {
        rounded >>= 1;
        ++exponent;
    }
    if (exponent > kExponentBias) {
        return FromBits((sign << kSignShift) | kInfinityBits);
    }
    return FromBits(
        (sign << kSignShift) | (static_cast<uint32_t>(exponent + kExponentBias) << kFractionBits) |
        (static_cast<uint32_t>(rounded) & kFractionMask));
}

CTRMM_SCALAR float FiniteFma(uint32_t aa, uint32_t bb, uint32_t cc)
{
    const uint32_t ax = aa & kMagnitudeMask;
    const uint32_t bx = bb & kMagnitudeMask;
    const uint32_t cx = cc & kMagnitudeMask;
    const uint32_t signP = (aa ^ bb) >> kSignShift;
    const uint32_t signC = cc >> kSignShift;
    int ea = ax >> kFractionBits;
    int eb = bx >> kFractionBits;
    int ec = cx >> kFractionBits;
    uint64_t ma = (ax & kFractionMask) | (ea ? kImplicitOne : 0);
    uint64_t mb = (bx & kFractionMask) | (eb ? kImplicitOne : 0);
    uint64_t mc = (cx & kFractionMask) | (ec ? kImplicitOne : 0);
    int ep = (ea ? ea - kSignificandExponentBias : kMinSubnormalExponent) +
             (eb ? eb - kSignificandExponentBias : kMinSubnormalExponent);
    ec = ec ? ec - kSignificandExponentBias : kMinSubnormalExponent;
    uint64_t mp = ma * mb;
    if (mp == 0) {
        return cx == 0 ? FromBits((signP & signC) << kSignShift) : FromBits(cc);
    }
    int top = ep + TopBit(mp);
    if (mc != 0 && ec + TopBit(mc) > top) {
        top = ec + TopBit(mc);
    }
    uint64_t p = ShiftJam(mp, kAccumulatorTopBit + ep - top);
    uint64_t q = mc == 0 ? 0 : ShiftJam(mc, kAccumulatorTopBit + ec - top);
    uint64_t s;
    uint32_t sign;
    if (signP == signC) {
        s = p + q;
        sign = signP;
    } else if (p >= q) {
        s = p - q;
        sign = signP;
    } else {
        s = q - p;
        sign = signC;
    }
    return RoundSum(s, sign, top - kAccumulatorTopBit);
}

// Integer significands retain all product bits plus guard/round/sticky bits.
// This slow path gives IEEE binary32 fused multiply-add on scalar hardware
// without relying on vector compound-instruction rounding behavior.
CTRMM_SCALAR float Fma(float a, float b, float c)
{
    const uint32_t aa = BitsOf(a);
    const uint32_t bb = BitsOf(b);
    const uint32_t cc = BitsOf(c);
    const uint32_t ax = aa & kMagnitudeMask;
    const uint32_t bx = bb & kMagnitudeMask;
    const uint32_t cx = cc & kMagnitudeMask;
    if (ax >= kInfinityBits || bx >= kInfinityBits || cx >= kInfinityBits) {
        return NonFiniteFma(aa, bb, cc);
    }
    // Scaling a normal float by a power of two is exact when the resulting
    // exponent stays normal. In that case Mul + Add IS a single-rounding FMA.
    const int productExponent =
        static_cast<int>(ax >> kFractionBits) + static_cast<int>(bx >> kFractionBits) - kExponentBias;
    if (ax >= kMinNormalBits && bx >= kMinNormalBits && productExponent >= 1 &&
        productExponent <= kMaxNormalBiasedExponent && ((ax & kFractionMask) == 0U || (bx & kFractionMask) == 0U)) {
        return a * b + c;
    }
    return FiniteFma(aa, bb, cc);
}
} // namespace ctrmm_ieee
#undef CTRMM_SCALAR
