/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <cblas.h>
#include "blas_test.h"
#include "csv_loader.h"
#include "verify.h"
#include "../chpr2_param.h"
#include "chpr2_npu_wrapper.h"
#include "../chpr2_golden.h"

namespace {

size_t Chpr2VectorSize(int n, int inc)
{
    if (n <= 0 || inc == 0 || inc == INT_MIN) {
        return 1U;
    }
    return 1U + static_cast<size_t>(n - 1) * static_cast<size_t>(std::abs(static_cast<int64_t>(inc)));
}

std::vector<Complex> Chpr2Fill(size_t size, const std::string& pattern, uint32_t seed)
{
    std::vector<Complex> out(size);
    if (pattern == "RANDOM_NORM_5_5" || pattern == "NULLPTR") {
        auto uniform = makeBlasArray(static_cast<int64_t>(size * 2U), "RANDOM_NORM_5_5", seed);
        auto normal = makeBlasArray(static_cast<int64_t>(size * 2U), "GAUSS_NORM_5_2", seed + 1U);
        for (size_t i = 0; i < size; ++i) {
            const auto& values = (i % 2U != 0U) ? normal : uniform;
            out[i] = {values[2U * i], values[2U * i + 1U]};
        }
    } else if (pattern == "RANDOM_ALTER") {
        for (size_t i = 0; i < size; ++i) {
            out[i] = {(i % 2U != 0U) ? -1.0f : 1.0f, (i % 2U != 0U) ? 1.0f : -1.0f};
        }
    } else if (pattern == "RANDOM_EXTREME") {
        for (size_t i = 0; i < size; ++i) {
            out[i] = {(i % 2U != 0U) ? 1e10f : -1e10f, (i % 2U != 0U) ? 1e-10f : -1e-10f};
        }
    } else {
        auto values = makeBlasArray(static_cast<int64_t>(size * 2U), pattern, seed);
        for (size_t i = 0; i < size; ++i) {
            out[i] = {values[2U * i], values[2U * i + 1U]};
        }
    }
    return out;
}

size_t Chpr2Diagonal(int j, int n, aclblasFillMode_t uplo)
{
    return (uplo == ACLBLAS_UPPER) ? (static_cast<size_t>(j) * (static_cast<size_t>(j) + 1U) / 2U + j)
                                   : (static_cast<size_t>(j) * (2ULL * n - j + 1ULL) / 2ULL);
}

bool Chpr2VerifyComponent(
    const std::vector<Complex>& actual, const std::vector<Complex>& golden, bool imaginary, const Case& p)
{
    auto visit = [&](auto consume) {
        for (size_t i = 0; i < actual.size(); ++i) {
            const float a = imaginary ? actual[i].imag : actual[i].real;
            const float g = imaginary ? golden[i].imag : golden[i].real;
            if ((std::isnan(a) && std::isnan(g)) || (std::isinf(a) && std::isinf(g) && a == g)) {
                continue;
            }
            consume(a, g);
        }
    };
    size_t compared = 0;
    visit([&](float, float) { ++compared; });
    VerifyConfig mixed;
    mixed.mode = PrecisionMode::MIXED_TOLERANCE;
    mixed.mixedAtol = 1.0 / 8192.0;
    mixed.mixedRtol = 1.0 / 8192.0;
    mixed.mixedRequiredMatchedRatio = 0.99;
    mixed.mixedMaxAbsErrorLimit = 1e-2;
    return Verifier::verifyEach(visit, compared, mixed, p.name + (imaginary ? "_imag" : "_real"));
}

} // namespace

class Chpr2Test : public BlasTest<Case> {};

TEST_F(Chpr2Test, NullHandlePrecedesAll)
{
    EXPECT_EQ(
        aclblasChpr2(nullptr, static_cast<aclblasFillMode_t>(999), 0, nullptr, nullptr, INT_MIN, nullptr, 0, nullptr),
        ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
}

INSTANTIATE_TEST_SUITE_P(
    Chpr2, Chpr2Test, ::testing::ValuesIn(GetCasesFromCsv<Case>(ReplaceFileExtension2Csv(__FILE__))),
    [](const ::testing::TestParamInfo<Case>& info) { return info.param.name; });

TEST_P(Chpr2Test, CsvDriven)
{
    const auto& p = GetParam();
    const size_t xSize = Chpr2VectorSize(p.n, p.incx);
    const size_t ySize = Chpr2VectorSize(p.n, p.incy);
    const size_t apSize = (p.n > 0) ? (static_cast<size_t>(p.n) * (static_cast<size_t>(p.n) + 1U) / 2U) : 1U;
    std::vector<Complex> x = Chpr2Fill(xSize, p.xFill, p.seed);
    std::vector<Complex> y = Chpr2Fill(ySize, p.yFill, p.seed + 1U);
    std::vector<Complex> ap = Chpr2Fill(apSize, p.apFill, p.seed + 2U);
    for (int j = 0; j < p.n; ++j) {
        ap[Chpr2Diagonal(j, p.n, p.uplo)].imag = (j % 2 == 0) ? std::numeric_limits<float>::quiet_NaN() : 12.5f;
    }

    chpr2_arch22_test::Chpr2NpuWrapper dx(x);
    chpr2_arch22_test::Chpr2NpuWrapper dy(y);
    chpr2_arch22_test::Chpr2NpuWrapper dap(ap);
    const Complex* alphaPtr = p.alphaNull ? nullptr : &p.alpha;
    Complex* xPtr = (p.xFill == "NULLPTR") ? nullptr : dx.Data();
    Complex* yPtr = (p.yFill == "NULLPTR") ? nullptr : dy.Data();
    Complex* apPtr = (p.apFill == "NULLPTR") ? nullptr : dap.Data();

    const aclblasStatus_t st = aclblasChpr2(
        Chpr2Test::handle_, p.uplo, p.n, alphaPtr, xPtr, p.incx, yPtr, p.incy, apPtr);
    EXPECT_EQ(static_cast<int>(st), static_cast<int>(p.expected)) << p.name;
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return;
    }
    ASSERT_EQ(aclrtSynchronizeStream(Chpr2Test::stream_), ACL_SUCCESS);
    (void)dx.Read(true);
    (void)dy.Read(true);
    const bool noop = (p.n == 0) || (p.alpha.real == 0.0f && p.alpha.imag == 0.0f);
    const std::vector<Complex> actual = dap.Read(noop);
    if (noop) {
        return;
    }
    std::vector<Complex> golden = ap;
    Chpr2Golden(p.uplo, p.n, p.alpha, x, p.incx, y, p.incy, golden);
    EXPECT_TRUE(Chpr2VerifyComponent(actual, golden, false, p)) << p.name;
    EXPECT_TRUE(Chpr2VerifyComponent(actual, golden, true, p)) << p.name;
    for (int j = 0; j < p.n; ++j) {
        EXPECT_FLOAT_EQ(actual[Chpr2Diagonal(j, p.n, p.uplo)].imag, 0.0f) << p.name;
    }
}
