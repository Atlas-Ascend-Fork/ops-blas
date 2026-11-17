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

#include <complex>
#include <vector>

#include "cblas_compat.h"
#include "triangle_preservation.h"

inline std::vector<aclblasDoubleComplex> TriangleWiden(const aclblasComplex* input, size_t count)
{
    std::vector<aclblasDoubleComplex> output(count);
    for (size_t i = 0; i < count; ++i) {
        output[i] = {static_cast<double>(input[i].real), static_cast<double>(input[i].imag)};
    }
    return output;
}

inline void VerifyTriangleFp64(
    const char* name, const aclblasComplex* actual, const aclblasComplex* original, const aclblasDoubleComplex* golden,
    int n, int ld, bool upper, bool quick = false)
{
    // Task-book COMPLEX64 output limits, evaluated against the FP64 golden.
    constexpr double atol = 0x1p-13;
    constexpr double rtol = 0x1p-13;
    TrianglePrecisionStats stats[2];
    size_t inactive = 0U;
    size_t quickBad = 0U;
    for (int col = 0; col < n; ++col) {
        for (int row = 0; row < ld; ++row) {
            const size_t i = static_cast<size_t>(col) * ld + row;
            const bool untouched = row >= n || (upper ? row > col : row < col);
            if (quick) {
                quickBad += memcmp(actual + i, original + i, sizeof(aclblasComplex)) != 0;
            }
            if (untouched) {
                inactive += memcmp(actual + i, original + i, sizeof(aclblasComplex)) != 0;
                continue;
            }
            stats[0].Add(actual[i].real, golden[i].real, atol, rtol);
            stats[1].Add(actual[i].imag, golden[i].imag, atol, rtol);
        }
    }
    printf(
        "TRIANGLE_CHECK reference=fp64 atol=2^-13 rtol=2^-13 case=%s re_max=%.17g im_max=%.17g re_bad=%zu im_bad=%zu "
        "re_hard=%zu im_hard=%zu inactive=%zu quickBad=%zu\n",
        name, stats[0].maxerr, stats[1].maxerr, stats[0].bad, stats[1].bad, stats[0].hard, stats[1].hard, inactive,
        quickBad);
    EXPECT_TRUE(stats[0].Passed()) << name << " real (FP64 reference)";
    EXPECT_TRUE(stats[1].Passed()) << name << " imag (FP64 reference)";
    EXPECT_EQ(inactive, 0U) << name;
    EXPECT_EQ(quickBad, 0U) << name;
}

inline std::vector<aclblasDoubleComplex> CherkGoldenFp64(
    aclblasFillMode_t uplo, aclblasOperation_t trans, int n, int k, float alpha, const aclblasComplex* a, int lda,
    float beta, const aclblasComplex* original, int ldc)
{
    auto output = TriangleWiden(original, static_cast<size_t>(ldc) * n);
    if ((alpha == 0.0F || k == 0) && beta == 1.0F) {
        return output;
    }
    const size_t count = alpha == 0.0F || k == 0 ? 0U : static_cast<size_t>(lda) * (trans == ACLBLAS_OP_N ? k : n);
    auto input = TriangleWiden(a, count);
    const aclblasDoubleComplex dummy{0.0, 0.0};
    cblas_zherk(
        CblasColMajor, ToCblasUplo(uplo), trans == ACLBLAS_OP_N ? CblasNoTrans : CblasConjTrans, n, k,
        static_cast<double>(alpha), input.empty() ? &dummy : input.data(), lda, static_cast<double>(beta),
        output.data(), ldc);
    return output;
}

inline int64_t TriangleVectorIndex(int i, int n, int increment)
{
    const int64_t stride = increment;
    return stride > 0 ? static_cast<int64_t>(i) * stride : (n - 1LL - i) * (-stride);
}

inline std::vector<aclblasDoubleComplex> Csyr2GoldenFp64(
    aclblasFillMode_t uplo, int n, aclblasComplex alpha, const aclblasComplex* x, int incx, const aclblasComplex* y,
    int incy, const aclblasComplex* original, int lda)
{
    auto output = TriangleWiden(original, static_cast<size_t>(lda) * n);
    if (alpha.real == 0.0F && alpha.imag == 0.0F) {
        return output;
    }
    auto wide = [](aclblasComplex value) { return std::complex<double>(value.real, value.imag); };
    const auto scalar = wide(alpha);
    for (int col = 0; col < n; ++col) {
        const auto xc = scalar * wide(x[TriangleVectorIndex(col, n, incx)]);
        const auto yc = scalar * wide(y[TriangleVectorIndex(col, n, incy)]);
        const int begin = uplo == ACLBLAS_UPPER ? 0 : col;
        const int end = uplo == ACLBLAS_UPPER ? col + 1 : n;
        for (int row = begin; row < end; ++row) {
            const auto update =
                xc * wide(y[TriangleVectorIndex(row, n, incy)]) + yc * wide(x[TriangleVectorIndex(row, n, incx)]);
            auto& value = output[static_cast<size_t>(col) * lda + row];
            value.real += update.real();
            value.imag += update.imag();
        }
    }
    return output;
}
