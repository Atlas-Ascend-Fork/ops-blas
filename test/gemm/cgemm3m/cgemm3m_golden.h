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

#include <vector>

#include <gtest/gtest.h>

#include "cblas_compat.h"
#include "cgemm3m_param.h"

inline void GoldenCgemm3m(
    const Cgemm3mParam& p, const std::vector<aclblasComplex>& a, const std::vector<aclblasComplex>& b,
    std::vector<aclblasComplex>& c)
{
    // ValidateCgemmParams rejects null scalars; invalid-input cases have no golden result.
    ASSERT_FALSE(p.alphaNull || p.betaNull) << "Null scalars must be checked as INVALID_VALUE cases";
    const aclblasComplex alpha{p.alphaReal, p.alphaImag};
    const aclblasComplex beta{p.betaReal, p.betaImag};
    if (p.k == 0 || (alpha.real == 0.0f && alpha.imag == 0.0f)) {
        if (beta.real == 1.0f && beta.imag == 0.0f) {
            return;
        }
        for (int col = 0; col < p.n; ++col) {
            for (int row = 0; row < p.m; ++row) {
                auto& value = c[static_cast<size_t>(col) * p.ldc + row];
                if (beta.real == 0.0f && beta.imag == 0.0f) {
                    value = {0.0f, 0.0f};
                } else {
                    const float real = beta.real * value.real - beta.imag * value.imag;
                    value.imag = beta.real * value.imag + beta.imag * value.real;
                    value.real = real;
                }
            }
        }
        return;
    }
    cblas_cgemm(
        CblasColMajor, ToCblasOp(p.transA), ToCblasOp(p.transB), p.m, p.n, p.k, &alpha, a.data(), p.lda, b.data(),
        p.ldb, &beta, c.data(), p.ldc);
}
