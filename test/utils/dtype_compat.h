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

#include <cstddef>
#include <cstdint>

#include "acl/acl.h"

// FP16 (IEEE 754 half)  <->  float conversion helpers
inline uint16_t floatToFp16(float v)
{
    uint32_t x = *reinterpret_cast<const uint32_t*>(&v);
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = static_cast<int32_t>((x >> 23) & 0xFFu) - 127 + 15;
    uint32_t mant = x & 0x7FFFFFu;

    // NaN / Inf
    if (((x >> 23) & 0xFFu) == 0xFFu)
        return static_cast<uint16_t>(sign | 0x7C00u | (mant ? 0x200u : 0));

    if (exp >= 0x1F)
        return static_cast<uint16_t>(sign | 0x7C00u);
    if (exp <= 0) {
        if (exp < -10)
            return static_cast<uint16_t>(sign);
        mant |= 0x800000u;
        int shift = 14 - exp;
        uint32_t halfMant = mant >> shift;
        uint32_t rem = mant & ((1u << shift) - 1u);
        uint32_t halfWay = 1u << (shift - 1u);
        if (rem > halfWay || (rem == halfWay && (halfMant & 1u)))
            halfMant++;
        return static_cast<uint16_t>(sign | halfMant);
    }
    uint32_t halfMant = mant >> 13;
    uint32_t rem = mant & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (halfMant & 1u))) {
        halfMant++;
        if (halfMant == 0x400u) {
            halfMant = 0;
            exp++;
        }
    }
    if (exp >= 0x1F)
        return static_cast<uint16_t>(sign | 0x7C00u);
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | halfMant);
}

inline float fp16ToFloat(uint16_t h)
{
    uint32_t hsign = (static_cast<uint32_t>(h) & 0x8000u) << 16;
    uint32_t hexp = (h >> 10) & 0x1Fu;
    uint32_t hmant = h & 0x3FFu;
    uint32_t f;
    if (hexp == 0) {
        if (hmant == 0) {
            f = hsign;
        } else {
            int e = -1;
            do {
                hmant <<= 1;
                e++;
            } while ((hmant & 0x400u) == 0);
            hmant &= 0x3FFu;
            f = hsign | (static_cast<uint32_t>(127 - 15 - e) << 23) | (hmant << 13);
        }
    } else if (hexp == 0x1Fu) {
        f = hsign | 0x7F800000u | (hmant << 13);
    } else {
        f = hsign | (static_cast<uint32_t>(hexp - 15 + 127) << 23) | (hmant << 13);
    }
    float out = *reinterpret_cast<float*>(&f);
    return out;
}

// BF16  <->  float conversion helpers  (round-to-nearest-even)
inline uint16_t floatToBf16(float v)
{
    uint32_t bits = *reinterpret_cast<const uint32_t*>(&v);
    if ((bits & 0x7FFFFFFFu) > 0x7F800000u) {
        // Preserve NaN before rounding can erase its payload or wrap the sign.
        return static_cast<uint16_t>((bits >> 16) | 0x40u);
    }
    uint32_t lsb = (bits >> 16) & 1u;
    bits += 0x7FFFu + lsb; // round-to-nearest-even
    return static_cast<uint16_t>(bits >> 16);
}

inline float bf16ToFloat(uint16_t h)
{
    uint32_t bits = static_cast<uint32_t>(h) << 16;
    float out = *reinterpret_cast<float*>(&bits);
    return out;
}

// Cast float → target dtype representation and back  (quantisation round-trip)
inline float castToDtype(float v, int32_t dtype)
{
    if (dtype == static_cast<int32_t>(ACL_FLOAT16))
        return fp16ToFloat(floatToFp16(v));
    else if (dtype == static_cast<int32_t>(ACL_BF16))
        return bf16ToFloat(floatToBf16(v));
    else
        return v; // FP32: no-op
}

// dtype byte size helper (FP16/BF16 → 2, FP32 → 4, default → 4)
inline size_t dtypeCompatSize(int32_t dtype)
{
    if (dtype == static_cast<int32_t>(ACL_FLOAT16) || dtype == static_cast<int32_t>(ACL_BF16)) {
        return 2;
    }
    return 4;
}
