/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include <cstring>
#include <vector>
#include <cblas.h>
#include "blas_test.h"
#include "../chpmv_param.h"
#include "../chpmv_test_utils.h"
#include "chpmv_npu_wrapper.h"

using chpmv_test::Complex;
using chpmv_test::Param;

namespace {

std::vector<Param> LoadArch22ChpmvCases()
{
    return GetCasesFromCsv<Param>(ReplaceFileExtension2Csv(__FILE__));
}

void VerifyChpmvOutput(const Param& p, const std::vector<Complex>& actual, const std::vector<Complex>& golden)
{
    for (bool imaginary : {false, true}) {
        const auto result = chpmv_test::VerifyComponent(
            actual, golden, p.n, p.incy, imaginary, p.caseName + (imaginary ? "_imag" : "_real"));
        EXPECT_TRUE(result.passed) << p.caseName;
    }
}

void VerifyChpmvStrideHoles(const Param& p, const std::vector<Complex>& actual, const std::vector<Complex>& orig)
{
    const uint64_t stride = (p.incy < 0) ? static_cast<uint64_t>(-int64_t(p.incy)) : static_cast<uint64_t>(p.incy);
    for (size_t i = 0; i < orig.size(); ++i) {
        if (i % stride != 0U) {
            EXPECT_EQ(std::memcmp(&actual[i], &orig[i], sizeof(Complex)), 0) << p.caseName << " gap " << i;
        }
    }
}

} // namespace

class ChpmvTest : public BlasTest<Param> {};

TEST_F(ChpmvTest, NullHandle)
{
    Complex alpha{1.0f, 0.0f};
    Complex beta{0.0f, 0.0f};
    Complex dummy{0.0f, 0.0f};
    EXPECT_EQ(
        static_cast<int>(aclblasChpmv(nullptr, ACLBLAS_UPPER, 1, &alpha, &dummy, &dummy, 1, &beta, &dummy, 1)),
        static_cast<int>(ACLBLAS_STATUS_HANDLE_IS_NULLPTR));
}

INSTANTIATE_TEST_SUITE_P(
    Chpmv, ChpmvTest, ::testing::ValuesIn(LoadArch22ChpmvCases()), PrintCaseInfoString<Param>);

TEST_P(ChpmvTest, CsvDriven)
{
    using namespace chpmv_test;
    const auto& p = GetParam();
    auto ap = Generate(PackedCount(p.n), p.apFill, p.randomSeed);
    auto x = Generate(Count(p.n, p.incx), p.xFill, p.randomSeed + 1U);
    auto y = Generate(Count(p.n, p.incy), p.yFill, p.randomSeed + 2U);
    if (!ap.empty() && (p.uplo == ACLBLAS_UPPER || p.uplo == ACLBLAS_LOWER)) {
        for (int i = 0; i < p.n; ++i) {
            ap[Diagonal(i, p.n, p.uplo)].imag = (i % 2 != 0) ? NAN : INFINITY;
        }
    }
    chpmv_arch22_test::ChpmvNpuWrapper dAP(ap, p.prefix);
    chpmv_arch22_test::ChpmvNpuWrapper dX(x, p.prefix);
    chpmv_arch22_test::ChpmvNpuWrapper dY(y, p.prefix);
    aclblasStatus_t status = ACLBLAS_STATUS_INTERNAL_ERROR;
    ASSERT_NO_FATAL_FAILURE(chpmv_arch22_test::RunChpmvNpu(p, handle_, stream_, dAP, dX, dY, status));
    ASSERT_EQ(status, p.expectResult) << p.caseName;
    ASSERT_TRUE(dAP.GuardsIntact());
    ASSERT_TRUE(dX.GuardsIntact());
    ASSERT_TRUE(dY.GuardsIntact());
    ASSERT_TRUE(dAP.Equals(ap));
    ASSERT_TRUE(dX.Equals(x));
    if (status != ACLBLAS_STATUS_SUCCESS) {
        ASSERT_TRUE(dY.Equals(y));
        return;
    }
    if (p.n == 0) {
        return;
    }
    if (p.alpha.real == 0.0f && p.alpha.imag == 0.0f && p.beta.real == 1.0f && p.beta.imag == 0.0f) {
        ASSERT_TRUE(dY.Equals(y));
        return;
    }
    std::vector<Complex> actual(y.size());
    dY.Read(actual);
    std::vector<Complex> golden = y;
    std::vector<Complex> apClean = ap;
    for (int i = 0; i < p.n; ++i) {
        apClean[Diagonal(i, p.n, p.uplo)].imag = 0.0f;
    }
    cblas_chpmv(
        CblasColMajor, p.uplo == ACLBLAS_UPPER ? CblasUpper : CblasLower, p.n, &p.alpha, apClean.data(), x.data(),
        p.incx, &p.beta, golden.data(), p.incy);
    VerifyChpmvStrideHoles(p, actual, y);
    VerifyChpmvOutput(p, actual, golden);
}
