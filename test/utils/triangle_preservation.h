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
#include <limits>
#include <cstdint>
#include <cstring>
#include "securec.h"
#include <cstdio>
#include <string>
#include <stdexcept>
#include <gtest/gtest.h>
#include "cann_ops_blas.h"
inline void PoisonUntouched(aclblasComplex* a, int n, int ld, bool upper)
{
    if (!a || n <= 0 || ld < n) {
        return;
    }
    const uint32_t bits[] = {0x7fc12345U, 0x80000000U, 0x00000000U, 0x7fc54321U, 0x7f800000U,
                             0xff800000U, 0x00000001U, 0x80000001U, 0x3f912345U, 0xbe123456U};
    for (int c = 0; c < n; ++c)
        for (int r = 0; r < ld; ++r)
            if (r >= n || (upper ? r > c : r < c)) {
                uint32_t v[2] = {bits[(size_t(c) * ld + r) % 10], bits[(size_t(c) * ld + r + 3) % 10]};
                ASSERT_EQ(memcpy_s(a + size_t(c) * ld + r, sizeof(aclblasComplex), v, sizeof(v)), EOK);
            }
}
inline double TriangleFp32Spacing(float value)
{
    const float magnitude = std::min(std::abs(value), std::numeric_limits<float>::max());
    const float next = std::nextafter(magnitude, std::numeric_limits<float>::infinity());
    if (std::isfinite(next)) {
        return static_cast<double>(next) - magnitude;
    }
    return static_cast<double>(magnitude) - std::nextafter(magnitude, 0.0F);
}
struct TrianglePrecisionStats {
    size_t count = 0;
    size_t bad = 0;
    size_t hard = 0;
    double maxerr = 0;
    double maxerrLimit = .01;
    size_t specialBad = 0;
    void Add(float a, double g, double atol = 0x1p-13, double rtol = 0x1p-13)
    {
        ++count;
        // A finite FP64 reference may overflow the FP32 output range.
        // Only special-value classification uses the rounded output value.
        const float rounded = static_cast<float>(g);
        if (!std::isfinite(a) || !std::isfinite(g)) {
            bool ok = (std::isnan(a) && std::isnan(rounded)) || a == rounded;
            bad += !ok;
            specialBad += !ok;
            hard = specialBad + (maxerr > maxerrLimit);
            return;
        }
        double e = std::abs(double(a) - double(g));
        bad += e > atol + rtol * std::abs(g);
        // opbase mixed_tolerance_standard 2.1.2 anchors the hard limit at
        // the maximum absolute error point, rather than checking every point.
        if (e > maxerr) {
            maxerr = e;
            maxerrLimit = std::max(.01, 32.0 * TriangleFp32Spacing(rounded));
        }
        hard = specialBad + (maxerr > maxerrLimit);
    }
    bool Passed() const { return hard == 0 && (count == 0 || double(bad) / count <= .01); }
};
inline void VerifyTrianglePreservation(
    const char* name, const aclblasComplex* a, const aclblasComplex* g, int n, int ld, bool upper, bool quick = false)
{
    TrianglePrecisionStats st[2];
    size_t inactive = 0;
    size_t quickBad = 0;
    for (int c = 0; c < n; ++c)
        for (int r = 0; r < ld; ++r) {
            size_t i = size_t(c) * ld + r;
            bool untouched = r >= n || (upper ? r > c : r < c);
            if (quick) {
                quickBad += memcmp(a + i, g + i, 8) != 0;
            }
            if (untouched) {
                inactive += memcmp(a + i, g + i, 8) != 0;
                continue;
            }
            st[0].Add(a[i].real, g[i].real);
            st[1].Add(a[i].imag, g[i].imag);
        }
    printf(
        "TRIANGLE_CHECK case=%s n=%d ld=%d re_count=%zu re_bad=%zu re_hard=%zu re_max=%.12g im_bad=%zu im_hard=%zu "
        "im_max=%.12g inactive=%zu quickBad=%zu pass=%d\n",
        name, n, ld, st[0].count, st[0].bad, st[0].hard, st[0].maxerr, st[1].bad, st[1].hard, st[1].maxerr, inactive,
        quickBad, int(st[0].Passed() && st[1].Passed() && inactive == 0 && quickBad == 0));
    EXPECT_TRUE(st[0].Passed()) << name << " real";
    EXPECT_TRUE(st[1].Passed()) << name << " imag";
    EXPECT_EQ(inactive, 0U) << name;
    EXPECT_EQ(quickBad, 0U) << name;
}
