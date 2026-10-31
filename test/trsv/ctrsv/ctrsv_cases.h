/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once
#include <iostream>
#include <vector>
#include "ctrsv_param.h"
#include "ctrsv_test_helpers.h"
#include "ctrsv_golden.h"

namespace ctrsv_test {
constexpr aclblasComplex SENTINEL{-997.0f, 991.0f};
inline size_t XOffset(const CtrsvParam& p, int i)
{
    return p.incx > 0 ? int64_t(i) * p.incx : int64_t(p.n - 1 - i) * (-int64_t(p.incx));
}

inline std::vector<aclblasComplex> MakeA(const CtrsvParam& p)
{
    if (p.n <= 0 || p.lda <= 0 || p.A.method == BlasFillMode::M_NULLPTR) {
        return {};
    }
    const float nan = std::numeric_limits<float>::quiet_NaN();
    std::vector<aclblasComplex> a(size_t(p.lda) * p.n, {nan, nan});
    ComplexGenerator generator(p.A, p.randomSeed, p.normal);
    size_t index = 0;
    for (int j = 0; j < p.n; ++j) {
        for (int i = 0; i < std::min(p.lda, p.n); ++i) {
            if ((p.uplo == ACLBLAS_UPPER && i > j) || (p.uplo == ACLBLAS_LOWER && i < j)) {
                continue;
            }
            auto value = generator.Next(index++);
            if (i == j) {
                if (p.diag == ACLBLAS_UNIT) {
                    value = {nan, nan};
                } else if (std::isfinite(value.real) && std::isfinite(value.imag)) {
                    value.real += std::copysign(float(std::max(int64_t(5), p.n)), value.real);
                }
            }
            a[size_t(j) * p.lda + i] = value;
        }
    }
    return a;
}

inline std::vector<aclblasComplex> MakeX(const CtrsvParam& p)
{
    if (p.n <= 0 || p.x.method == BlasFillMode::M_NULLPTR) {
        return {};
    }
    const int64_t stride = std::abs(int64_t(p.incx));
    std::vector<aclblasComplex> x(1 + int64_t(p.n - 1) * stride, SENTINEL);
    ComplexGenerator generator(p.x, p.randomSeed + 7919, p.normal);
    for (int i = 0; i < p.n; ++i) {
        x[XOffset(p, i)] = generator.Next(i);
    }
    return x;
}

inline bool CheckPrecision(
    const CtrsvParam& p, const std::vector<aclblasComplex>& x, const std::vector<aclblasComplex>& golden)
{
    PrecisionStats re, im;
    int reported = 0;
    for (int i = 0; i < p.n; ++i) {
        const auto offset = XOffset(p, i);
        auto before = re.matched + im.matched;
        AccumulatePrecision(x[offset].real, golden[offset].real, re);
        AccumulatePrecision(x[offset].imag, golden[offset].imag, im);
        if (re.matched + im.matched != before + 2 && reported++ < 8) {
            std::cout << "mismatch i=" << i << " actual=(" << x[offset].real << "," << x[offset].imag << ") golden=("
                      << golden[offset].real << "," << golden[offset].imag << ")" << std::endl;
        }
    }
    double rr = re.count ? double(re.matched) / re.count : 1, ir = im.count ? double(im.matched) / im.count : 1;
    std::cout << "[" << p.caseName << "] real matched_ratio=" << rr << " max_abs_error=" << re.maxAbsError
              << " max_limit_ratio=" << re.maxLimitRatio << "; imag matched_ratio=" << ir
              << " max_abs_error=" << im.maxAbsError << " max_limit_ratio=" << im.maxLimitRatio << std::endl;
    return rr >= 0.99 && ir >= 0.99 && re.absoluteLimitPassed && im.absoluteLimitPassed;
}

inline void CtrsvGolden(const CtrsvParam& p, const std::vector<aclblasComplex>& a, std::vector<aclblasComplex>& x)
{
    aclblasCtrsv_cpu(p.uplo, p.trans, p.diag, p.n, a.data(), p.lda, x.data(), p.incx);
}
} // namespace ctrsv_test
