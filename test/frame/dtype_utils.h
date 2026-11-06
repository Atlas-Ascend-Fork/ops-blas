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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>
#include <securec.h>

#include "acl/acl.h"
#include "dtype_cast.h"

inline size_t dtypeByteSize(aclDataType dtype)
{
    switch (dtype) {
        case ACL_FLOAT16:
            return 2;
        case ACL_BF16:
            return 2;
        case ACL_FLOAT:
            return 4;
        case ACL_INT8:
            return 1;
        case ACL_INT32:
            return 4;
        case ACL_FLOAT8_E4M3FN:
            return 1;
        case ACL_FLOAT8_E5M2:
            return 1;
        default:
            return 4;
    }
}

inline uint8_t floatToFp8E4m3(float val)
{
    if (std::isnan(val))
        return 0x7F;
    if (std::isinf(val))
        return (val < 0) ? 0xF8 : 0x7F;
    if (val == 0.0f)
        return 0;

    uint32_t sign = 0;
    float absVal = val;
    if (val < 0) {
        sign = 1;
        absVal = -val;
    }

    const float maxVal = 448.0f;
    if (absVal > maxVal)
        absVal = maxVal;

    int eExp = static_cast<int>(std::floor(std::log2(absVal)));
    float norm = absVal / std::pow(2.0f, eExp);
    int eMant = static_cast<int>((norm - 1.0f) * 8.0f + 0.5f);
    if (eMant > 7) {
        eMant = 0;
        eExp++;
    }

    eExp += 7;
    if (eExp >= 15) {
        // issue #819: E4M3FN 无 Inf 编码，指数域上限 15 仍为有限值区间：
        // 256~448（含 248 的 RNE 进位路径）按正常有限值编码，
        // 仅进位超出可表示域时饱和到 0x7E=448，不再一律写成 0x7F=NaN。
        eExp = 15;
        if (eMant > 6) {
            eMant = 6;
        }
    }
    if (eExp <= 0) {
        float subVal = absVal / std::pow(2.0f, -6.0f);
        int submant = static_cast<int>(subVal * 8.0f + 0.5f);
        if (submant > 7)
            submant = 7;
        return static_cast<uint8_t>((sign << 7) | submant);
    }
    return static_cast<uint8_t>((sign << 7) | (eExp << 3) | eMant);
}

inline float fp8E4m3ToFloat(uint8_t fp8)
{
    uint32_t sign = (fp8 >> 7) & 1;
    uint32_t exp = (fp8 >> 3) & 0xF;
    uint32_t mant = fp8 & 0x7;
    float val;
    if (exp == 0 && mant == 0) {
        val = 0.0f;
    } else if (exp == 15 && mant == 7) {
        // issue #819: E4M3FN 仅 0x7F/0xFF 为 NaN，无 Inf 编码语义；
        // exp==15 且 mant 0~6 为 256~448 的合法有限值，走下方正常解码公式。
        val = NAN;
    } else if (exp == 0) {
        val = static_cast<float>(mant) / 8.0f * std::pow(2.0f, -6.0f);
    } else {
        val = (1.0f + static_cast<float>(mant) / 8.0f) * std::pow(2.0f, static_cast<float>(exp) - 7.0f);
    }
    return sign ? -val : val;
}

inline uint8_t floatToFp8E5m2(float val)
{
    if (std::isnan(val))
        return 0x7F;
    if (std::isinf(val))
        return (val < 0) ? 0xFC : 0x7F;
    if (val == 0.0f)
        return 0;

    uint32_t sign = 0;
    float absVal = val;
    if (val < 0) {
        sign = 1;
        absVal = -val;
    }

    const float maxVal = 57344.0f;
    if (absVal > maxVal)
        absVal = maxVal;

    int eExp = static_cast<int>(std::floor(std::log2(absVal)));
    float norm = absVal / std::pow(2.0f, eExp);
    int eMant = static_cast<int>((norm - 1.0f) * 4.0f + 0.5f);
    if (eMant > 3) {
        eMant = 0;
        eExp++;
    }

    eExp += 15;
    if (eExp >= 31) {
        eExp = 31;
        eMant = 0;
    }
    if (eExp <= 0) {
        float subVal = absVal / std::pow(2.0f, -14.0f);
        int submant = static_cast<int>(subVal * 4.0f + 0.5f);
        if (submant > 3)
            submant = 3;
        return static_cast<uint8_t>((sign << 7) | submant);
    }
    return static_cast<uint8_t>((sign << 7) | (eExp << 2) | eMant);
}

inline float fp8E5m2ToFloat(uint8_t fp8)
{
    uint32_t sign = (fp8 >> 7) & 1;
    uint32_t exp = (fp8 >> 2) & 0x1F;
    uint32_t mant = fp8 & 0x3;
    float val;
    if (exp == 0 && mant == 0) {
        val = 0.0f;
    } else if (exp == 31) {
        val = (mant == 0) ? INFINITY : NAN;
    } else if (exp == 0) {
        val = static_cast<float>(mant) / 4.0f * std::pow(2.0f, -14.0f);
    } else {
        val = (1.0f + static_cast<float>(mant) / 4.0f) * std::pow(2.0f, static_cast<float>(exp) - 15.0f);
    }
    return sign ? -val : val;
}

inline void quantizeInt8ToBytes(float src, uint8_t* dst, size_t dstSize)
{
    float rounded = std::nearbyint(src);
    rounded = std::max(
        static_cast<float>(std::numeric_limits<int8_t>::min()),
        std::min(static_cast<float>(std::numeric_limits<int8_t>::max()), rounded));
    int8_t value = static_cast<int8_t>(rounded);
    memcpy_s(dst, dstSize, &value, sizeof(value));
}

inline void quantizeInt32ToBytes(float src, uint8_t* dst, size_t dstSize)
{
    double rounded = std::nearbyint(static_cast<double>(src));
    rounded = std::max(
        static_cast<double>(std::numeric_limits<int32_t>::min()),
        std::min(static_cast<double>(std::numeric_limits<int32_t>::max()), rounded));
    int32_t value = static_cast<int32_t>(rounded);
    memcpy_s(dst, dstSize, &value, sizeof(value));
}

inline std::vector<uint8_t> quantizeToBytes(const std::vector<float>& src, aclDataType dtype)
{
    if (src.empty())
        return {};
    size_t elemSize = dtypeByteSize(dtype);
    std::vector<uint8_t> dst(src.size() * elemSize);
    for (size_t i = 0; i < src.size(); i++) {
        switch (dtype) {
            case ACL_FLOAT16: {
                uint16_t v = blas_common::FloatToHalf(src[i]);
                memcpy_s(&dst[i * elemSize], dst.size() - i * elemSize, &v, sizeof(v));
                break;
            }
            case ACL_BF16: {
                uint16_t v = blas_common::FloatToBf16(src[i]);
                memcpy_s(&dst[i * elemSize], dst.size() - i * elemSize, &v, sizeof(v));
                break;
            }
            case ACL_FLOAT8_E4M3FN: {
                dst[i] = floatToFp8E4m3(src[i]);
                break;
            }
            case ACL_FLOAT8_E5M2: {
                dst[i] = floatToFp8E5m2(src[i]);
                break;
            }
            case ACL_INT8: {
                quantizeInt8ToBytes(src[i], &dst[i], dst.size() - i);
                break;
            }
            case ACL_INT32: {
                quantizeInt32ToBytes(src[i], &dst[i * elemSize], dst.size() - i * elemSize);
                break;
            }
            default: {
                memcpy_s(&dst[i * elemSize], dst.size() - i * elemSize, &src[i], elemSize);
                break;
            }
        }
    }
    return dst;
}

inline std::vector<float> dequantizeFromBytes(const std::vector<uint8_t>& src, aclDataType dtype, size_t count)
{
    if (src.empty() || count == 0)
        return {};
    size_t elemSize = dtypeByteSize(dtype);
    std::vector<float> dst(count);
    for (size_t i = 0; i < count; i++) {
        switch (dtype) {
            case ACL_FLOAT16: {
                uint16_t v = 0;
                memcpy_s(&v, sizeof(v), &src[i * elemSize], elemSize);
                dst[i] = blas_common::HalfToFloat(v);
                break;
            }
            case ACL_BF16: {
                uint16_t v = 0;
                memcpy_s(&v, sizeof(v), &src[i * elemSize], elemSize);
                dst[i] = blas_common::Bf16ToFloat(v);
                break;
            }
            case ACL_FLOAT8_E4M3FN: {
                dst[i] = fp8E4m3ToFloat(src[i]);
                break;
            }
            case ACL_FLOAT8_E5M2: {
                dst[i] = fp8E5m2ToFloat(src[i]);
                break;
            }
            case ACL_INT8: {
                int8_t v = 0;
                memcpy_s(&v, sizeof(v), &src[i], sizeof(v));
                dst[i] = static_cast<float>(v);
                break;
            }
            case ACL_INT32: {
                int32_t v = 0;
                memcpy_s(&v, sizeof(v), &src[i * elemSize], sizeof(v));
                dst[i] = static_cast<float>(v);
                break;
            }
            default: {
                memcpy_s(&dst[i], (count - i) * sizeof(float), &src[i * elemSize], elemSize);
                break;
            }
        }
    }
    return dst;
}

inline void quantizeRoundTrip(std::vector<float>& data, aclDataType dtype)
{
    auto bytes = quantizeToBytes(data, dtype);
    auto result = dequantizeFromBytes(bytes, dtype, data.size());
    data.swap(result);
}
