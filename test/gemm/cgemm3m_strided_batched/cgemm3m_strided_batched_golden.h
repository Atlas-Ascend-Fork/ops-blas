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
#include <cstring>
#include "cgemm3m_strided_batched_npu_wrapper.h"

namespace cgemm3m_strided_batched_test {
inline CBLAS_TRANSPOSE CpuTrans(int trans)
{
    return trans == ACLBLAS_OP_N ? CblasNoTrans : trans == ACLBLAS_OP_T ? CblasTrans : CblasConjTrans;
}
inline void Golden(const Invocation& q, const Matrix& a, const Matrix& b, const Matrix& c, std::vector<Complex>& out)
{
    out = c.host;
    for (int batch = 0; batch < q.batch; ++batch) {
        cblas_cgemm(
            CblasColMajor, CpuTrans(q.ta), CpuTrans(q.tb), q.m, q.n, q.k, &q.alpha,
            a.host.data() + a.base + batch * q.sa, q.lda, b.host.data() + b.base + batch * q.sb, q.ldb, &q.beta,
            out.data() + c.base + batch * q.sc, q.ldc);
    }
}
struct ComponentResult {
    size_t finite = 0, matched = 0, nonfinite = 0, nonfiniteMatched = 0, excessive = 0;
    double maxError = 0;
    void Add(float actual, float expected)
    {
        if (!std::isfinite(actual) || !std::isfinite(expected)) {
            ++nonfinite;
            nonfiniteMatched +=
                (!std::isfinite(actual) && !std::isfinite(expected) &&
                 !(std::isinf(actual) && std::isinf(expected) && std::signbit(actual) != std::signbit(expected)));
            return;
        }
        ++finite;
        const double error = std::abs(double(actual) - expected);
        maxError = std::max(maxError, error);
        matched += error <= std::ldexp(1., -13) * (1. + std::abs(double(expected)));
        const double ulp = std::abs(double(std::nextafter(expected, INFINITY)) - expected);
        excessive += error > std::max(0.01, 32 * ulp);
    }
    double Ratio() const { return finite == 0 ? 1. : double(matched) / finite; }
    bool Pass() const { return Ratio() >= 0.99 && excessive == 0 && nonfiniteMatched == nonfinite; }
};
} // namespace cgemm3m_strided_batched_test
