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
#include <cmath>
#include <cstdio>
#include <limits>
#include <random>
#include <vector>
#include <cblas.h>
#include "cher2_param.h"

namespace cher2_test {
inline std::vector<aclblasComplex> MakeInput(size_t count, const std::string& fill, uint32_t seed, bool normal)
{
    if (fill == "NULLPTR") {
        return {};
    }
    std::vector<aclblasComplex> out(count);
    if (normal && fill == "RANDOM_NORM_5_5") {
        std::mt19937 gen(seed);
        std::uniform_real_distribution<float> means(-5.0f, 5.0f), sigmas(0.1f, 2.0f);
        const float meanR = means(gen), sigmaR = sigmas(gen), meanI = means(gen), sigmaI = sigmas(gen);
        std::normal_distribution<float> real(meanR, sigmaR), imag(meanI, sigmaI);
        for (auto& v : out) {
            v = {real(gen), imag(gen)};
        }
    } else {
        const auto real = makeBlasArray(static_cast<int64_t>(count), fill, seed);
        const auto imag = makeBlasArray(static_cast<int64_t>(count), fill, seed + 1U);
        for (size_t i = 0; i < count; ++i) {
            out[i] = {real[i], imag[i]};
        }
    }
    return out;
}

struct Accuracy {
    size_t finite = 0, matched = 0, nonfinite = 0, specialMatched = 0, relativeCount = 0;
    double maxAbs = 0.0, worstGolden = 0.0, relativeSum = 0.0, maxRelative = 0.0;
    void Add(float actual, float golden)
    {
        if (!std::isfinite(actual) || !std::isfinite(golden)) {
            ++nonfinite;
            const bool equal =
                !std::isfinite(actual) && !std::isfinite(golden) &&
                (std::isnan(actual) || std::isnan(golden) || std::signbit(actual) == std::signbit(golden));
            specialMatched += equal;
            return;
        }
        ++finite;
        const double error = std::abs(static_cast<double>(actual) - golden);
        matched += error <= std::ldexp(1.0, -13) * (1.0 + std::abs(static_cast<double>(golden)));
        if (error > maxAbs) {
            maxAbs = error;
            worstGolden = golden;
        }
        // Near-zero values use the absolute tolerance; relative errors there
        // are not a meaningful MERE/MARE statistic.
        if (std::abs(golden) > std::ldexp(1.0, -13)) {
            const double relative = error / (std::abs(static_cast<double>(golden)) + std::ldexp(1.0, -14));
            ++relativeCount;
            relativeSum += relative;
            maxRelative = std::max(maxRelative, relative);
        }
    }
    double Ratio() const { return finite ? static_cast<double>(matched) / finite : 1.0; }
    double Mere() const { return relativeCount ? relativeSum / relativeCount : 0.0; }
    double Limit() const
    {
        const float g = std::abs(static_cast<float>(worstGolden));
        const double ulp = g == 0.0f ? std::ldexp(1.0, -149) : std::ldexp(1.0, std::max(std::ilogb(g), -126) - 23);
        return std::max(0.01, 32.0 * ulp);
    }
    bool Pass() const { return Ratio() >= 0.99 && maxAbs <= Limit() && nonfinite == specialMatched; }
    void Print(const std::string& name, const char* part) const
    {
        std::printf(
            "[ACCURACY] case=%s part=%s finite=%zu matched=%zu ratio=%.12g max_abs=%.12g cap=%.12g "
            "nonfinite=%zu special_matched=%zu mere=%.12g mare=%.12g\n",
            name.c_str(), part, finite, matched, Ratio(), maxAbs, Limit(), nonfinite, specialMatched, Mere(),
            maxRelative);
    }
};
} // namespace cher2_test

// Shared Netlib entry point for the preserved upstream regression suite.
inline void Cher2Golden(const Cher2Param& p, const aclblasComplex* x, const aclblasComplex* y, aclblasComplex* a)
{
    cblas_cher2(
        CblasColMajor, p.uplo == ACLBLAS_UPPER ? CblasUpper : CblasLower, p.n, &p.alpha, x, p.incx, y, p.incy, a,
        p.lda);
}
