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

#include <cblas.h>
#include <cmath>
#include <vector>
#include "chbmv_param.h"

namespace ChbmvReference {
struct Complex64 {
    double real, imag;
};
static_assert(sizeof(Complex64) == 2 * sizeof(double), "CBLAS complex storage");

inline std::vector<Complex64> Promote(const std::vector<aclblasComplex>& src)
{
    std::vector<Complex64> dst(src.size());
    for (size_t i = 0; i < src.size(); ++i) {
        dst[i] = {src[i].real, src[i].imag};
    }
    return dst;
}

inline std::vector<aclblasComplex> Golden(
    const ChbmvParam& p, const std::vector<aclblasComplex>& a, const std::vector<aclblasComplex>& x,
    const std::vector<aclblasComplex>& y)
{
    auto ad = Promote(a);
    auto xd = Promote(x);
    auto yd = Promote(y);
    const Complex64 alpha{p.alpha.real, p.alpha.imag};
    const Complex64 beta{p.beta.real, p.beta.imag};
    cblas_zhbmv(
        CblasColMajor, p.uplo == ACLBLAS_UPPER ? CblasUpper : CblasLower, p.n, p.k, &alpha, ad.data(), p.lda, xd.data(),
        p.incx, &beta, yd.data(), p.incy);
    std::vector<aclblasComplex> result(y.size());
    for (size_t i = 0; i < result.size(); ++i) {
        result[i] = {static_cast<float>(yd[i].real), static_cast<float>(yd[i].imag)};
    }
    return result;
}

struct Stats {
    size_t finite = 0, matched = 0, special = 0, specialMatched = 0;
    double maxError = 0, anchor = 0;

    void Add(float actual, float expected)
    {
        if (!std::isfinite(actual) || !std::isfinite(expected)) {
            ++special;
            if ((std::isnan(actual) && !std::isfinite(expected)) || (std::isnan(expected) && !std::isfinite(actual)) ||
                (std::isinf(actual) && actual == expected)) {
                ++specialMatched;
            }
            return;
        }
        ++finite;
        const double error = std::fabs(static_cast<double>(actual) - expected);
        if (error <= std::ldexp(1.0, -13) * (1.0 + std::fabs(static_cast<double>(expected)))) {
            ++matched;
        }
        if (error > maxError) {
            maxError = error;
            anchor = std::fabs(static_cast<double>(expected));
        }
    }

    bool Pass() const
    {
        int exponent = 0;
        std::frexp(anchor, &exponent);
        const double ulp = anchor == 0.0 ? std::ldexp(1.0, -149) : std::ldexp(1.0, std::max(exponent - 1, -126) - 23);
        return (finite == 0 || static_cast<double>(matched) / finite >= 0.99) &&
               maxError <= std::max(1e-2, 32.0 * ulp) && special == specialMatched;
    }
};
} // namespace ChbmvReference
