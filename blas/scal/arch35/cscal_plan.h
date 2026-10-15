/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
// Pure host planning functions, shared with CPU unit tests. No CANN dependency.
#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include "cscal_config.h"
#include "cscal_tiling_data.h"
namespace cscal950 {
template <uint64_t divisor>
inline uint64_t CeilDiv(uint64_t n)
{
    static_assert(divisor > 0, "CeilDiv requires a positive divisor");
    return n / divisor + (n % divisor != 0);
}
inline uint32_t FloatBits(float value)
{
    static_assert(sizeof(float) == sizeof(uint32_t), "Cscal requires 32-bit floats");
    uint32_t bits = 0;
    const auto* src = reinterpret_cast<const unsigned char*>(&value);
    auto* dst = reinterpret_cast<unsigned char*>(&bits);
    std::copy_n(src, sizeof(bits), dst);
    return bits;
}
inline bool IsFiniteAlphaComponent(float value) { return (FloatBits(value) & 0x7f800000u) != 0x7f800000u; }
inline bool StorageBytes(int n, int incx, uint64_t& bytes)
{
    bytes = 0;
    if (n <= 0 || incx <= 0)
        return true;
    const uint64_t span = uint64_t(n - 1) * uint64_t(incx) + 1;
    const uint64_t limit = std::min<uint64_t>(SIZE_MAX, PTRDIFF_MAX);
    if (span > limit / 8)
        return false;
    bytes = span * 8;
    return true;
}
inline bool MakePlan(int n, int incx, float ar, float ai, uint32_t aivCores, CscalTilingData& t)
{
    t = {};
    if (n <= 0 || incx <= 0 || aivCores == 0)
        return false;
    uint64_t bytes;
    if (!StorageBytes(n, incx, bytes))
        return false;
    t.n = static_cast<uint64_t>(n);
    t.incx = static_cast<uint64_t>(incx);
    t.alphaReal = ar;
    t.alphaImag = ai;
    t.finiteAlpha = IsFiniteAlphaComponent(ar) && IsFiniteAlphaComponent(ai);
    t.tileComplex = CSCAL_TILE_COMPLEX;
    t.simtThreads = 128;
    t.writeZero = CSCAL_ZERO_WRITES_ZERO && ar == 0.0f && ai == 0.0f;
    t.useVector = CSCAL_ENABLE_VECTOR && incx == 1 && CSCAL_KERNEL_POLICY != 1 &&
                  (CSCAL_KERNEL_POLICY == 2 || n >= CSCAL_VECTOR_THRESHOLD);
    if (t.useVector) {
        const uint64_t groups = t.n / 8;
        const uint64_t desired = CeilDiv<CSCAL_MIN_COMPLEX_PER_CORE>(t.n);
        t.coreCount =
            static_cast<uint32_t>(std::min<uint64_t>(aivCores, std::min(std::max<uint64_t>(1, groups), desired)));
    } else {
        t.coreCount = static_cast<uint32_t>(std::min<uint64_t>(aivCores, CeilDiv<256>(t.n)));
    }
    return t.coreCount != 0;
}
struct Slice {
    uint64_t start;
    uint64_t count;
};
inline Slice Partition(const CscalTilingData& t, uint32_t core)
{
    if (core >= t.coreCount || t.coreCount == 0)
        return {0, 0};
    if (!t.useVector) {
        uint64_t a = t.n * core / t.coreCount;
        uint64_t b = t.n * (core + 1) / t.coreCount;
        return {a, b - a};
    }
    const uint64_t groups = t.n / 8;
    const uint64_t q = groups / t.coreCount, r = groups % t.coreCount;
    const uint64_t start = (uint64_t(core) * q + std::min<uint64_t>(core, r)) * 8;
    uint64_t count = (q + (core < r)) * 8;
    if (core + 1 == t.coreCount)
        count += t.n % 8;
    return {start, count};
}
} // namespace cscal950
