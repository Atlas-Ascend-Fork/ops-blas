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
#include <cstring>
#include <iomanip>
#include <iostream>
#include <vector>

#include "cann_ops_blas.h"

struct Cgemm3mComponentResult {
    size_t finite = 0;
    size_t matched = 0;
    size_t nonfinite = 0;
    size_t nonfiniteFailed = 0;
    double maxError = 0.0;
    double maxErrorLimit = 0.01;
    float maxErrorGolden = 0.0f;

    void Add(float actual, float golden)
    {
        if (!std::isfinite(actual) || !std::isfinite(golden)) {
            ++nonfinite;
            const bool bothNonfinite = !std::isfinite(actual) && !std::isfinite(golden);
            const bool equivalent = bothNonfinite && (std::isnan(actual) || std::isnan(golden) || actual == golden);
            nonfiniteFailed += equivalent ? 0 : 1;
            return;
        }
        ++finite;
        const double error = std::abs(static_cast<double>(actual) - golden);
        matched += error <= (1.0 + std::abs(static_cast<double>(golden))) / 8192.0 ? 1 : 0;
        if (error > maxError) {
            maxError = error;
            maxErrorGolden = golden;
            const int exponent = golden == 0.0f ? -126 : std::max(-126, std::ilogb(std::abs(golden)));
            maxErrorLimit = std::max(0.01, 32.0 * std::ldexp(1.0, exponent - 23));
        }
    }

    bool Passed() const
    {
        return nonfiniteFailed == 0 && maxError <= maxErrorLimit && (finite == 0 || matched * 100 >= finite * 99);
    }
};

inline bool VerifyCgemm3m(
    const std::vector<aclblasComplex>& actual, const std::vector<aclblasComplex>& golden,
    const std::vector<aclblasComplex>& initial, int m, int n, int ldc, const std::string& name)
{
    Cgemm3mComponentResult real;
    Cgemm3mComponentResult imag;
    bool paddingUnchanged = true;
    for (int col = 0; col < n; ++col) {
        for (int row = 0; row < ldc; ++row) {
            const size_t index = static_cast<size_t>(col) * ldc + row;
            if (row < m) {
                real.Add(actual[index].real, golden[index].real);
                imag.Add(actual[index].imag, golden[index].imag);
            } else {
                paddingUnchanged =
                    paddingUnchanged && std::memcmp(&actual[index], &initial[index], sizeof(aclblasComplex)) == 0;
            }
        }
    }
    for (const auto& item : {std::make_pair("real", real), std::make_pair("imag", imag)}) {
        const auto& result = item.second;
        std::cout << "CGEMM3M_METRIC," << name << ',' << item.first << ',' << result.finite << ',' << result.matched
                  << ',' << result.nonfinite << ',' << result.nonfiniteFailed << ',' << std::setprecision(10)
                  << result.maxError << ',' << result.Passed() << ',' << result.maxErrorLimit << ','
                  << result.maxErrorGolden << std::endl;
    }
    return real.Passed() && imag.Passed() && paddingUnchanged;
}
