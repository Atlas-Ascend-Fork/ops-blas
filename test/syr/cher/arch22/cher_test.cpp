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
#include <vector>

#include "verify.h"
#include "blas_test.h"
#include "csv_loader.h"
#include "fill.h"
#include "cher_param.h"
#include "cher_golden.h"
#include "cher_npu_wrapper.h"

namespace {

bool IsUnsafeIncx(int inc) { return inc == 0 || inc == INT_MIN; }

std::vector<aclblasComplex> MakeComplexStrided(int count, int inc, const BlasFillMode& fill, uint32_t seed)
{
    if (fill.method == BlasFillMode::M_NULLPTR || count < 0 || IsUnsafeIncx(inc)) {
        return {};
    }
    std::vector<float> re = makeBlasStrided(count, inc, fill, seed);
    std::vector<float> im = makeBlasStrided(count, inc, fill, seed + 1000U);
    std::vector<aclblasComplex> out(re.size());
    for (size_t i = 0; i < re.size(); ++i) {
        out[i] = aclblasComplex{re[i], im[i]};
    }
    return out;
}

std::vector<aclblasComplex> MakeComplexMatrixHermitianPrep(
    int n, int lda, const BlasFillMode& fill, uint32_t seed)
{
    if (fill.method == BlasFillMode::M_NULLPTR || n <= 0 || lda <= 0) {
        return {};
    }
    std::vector<aclblasComplex> a = makeBlasComplexMatrix(n, n, lda, fill, seed);
    for (size_t idx = 0; idx < a.size(); ++idx) {
        const int i = static_cast<int>(idx % static_cast<size_t>(lda));
        const int j = static_cast<int>(idx / static_cast<size_t>(lda));
        if (i < n && j < n && i == j) {
            a[idx].imag = 0.0f;
        }
    }
    return a;
}

bool CherSameFloat(float a, float b)
{
    if (std::isnan(a) && std::isnan(b)) {
        return true;
    }
    if (std::isinf(a) && std::isinf(b)) {
        return (a > 0) == (b > 0);
    }
    return a == b;
}

void VerifyCherAccuracy(
    const CherParam& p, bool upper, const std::vector<aclblasComplex>& aHost, const std::vector<aclblasComplex>& aOrig,
    const std::vector<aclblasComplex>& golden)
{
    std::vector<float> outRe;
    std::vector<float> outIm;
    std::vector<float> goldRe;
    std::vector<float> goldIm;
    auto isNonFinite = [](float v) { return std::isnan(v) || std::isinf(v); };
    for (int j = 0; j < p.n; ++j) {
        const int iStart = upper ? 0 : j;
        const int iEnd = upper ? j : (p.n - 1);
        for (int i = iStart; i <= iEnd; ++i) {
            const size_t idx = static_cast<size_t>(j) * static_cast<size_t>(p.lda) + static_cast<size_t>(i);
            float orr = aHost[idx].real;
            float ori = aHost[idx].imag;
            float gr = golden[idx].real;
            float gi = golden[idx].imag;
            if (isNonFinite(orr) || isNonFinite(ori) || isNonFinite(gr) || isNonFinite(gi)) {
                orr = ori = gr = gi = 0.0f;
            }
            outRe.push_back(orr);
            outIm.push_back(ori);
            goldRe.push_back(gr);
            goldIm.push_back(gi);
        }
    }
    VerifyConfig cfgRe;
    applyMixedTolerance(cfgRe, ACL_FLOAT, goldRe.data(), goldRe.size());
    EXPECT_TRUE(Verifier::verifyVector(outRe.data(), goldRe.data(), outRe.size(), 1, cfgRe, p.caseName + "_re"));
    VerifyConfig cfgIm;
    applyMixedTolerance(cfgIm, ACL_FLOAT, goldIm.data(), goldIm.size());
    EXPECT_TRUE(Verifier::verifyVector(outIm.data(), goldIm.data(), outIm.size(), 1, cfgIm, p.caseName + "_im"));

    for (int i = 0; i < p.n; ++i) {
        const size_t idx = static_cast<size_t>(i) * static_cast<size_t>(p.lda) + static_cast<size_t>(i);
        EXPECT_LE(std::fabs(aHost[idx].imag), 1.52587890625e-5f) << p.caseName;
    }
    for (int j = 0; j < p.n; ++j) {
        for (int i = 0; i < p.lda; ++i) {
            const size_t idx = static_cast<size_t>(j) * static_cast<size_t>(p.lda) + static_cast<size_t>(i);
            const bool inUplo = (i < p.n) && (upper ? (i <= j) : (i >= j));
            if (!inUplo) {
                EXPECT_TRUE(CherSameFloat(aHost[idx].real, aOrig[idx].real)) << p.caseName;
                EXPECT_TRUE(CherSameFloat(aHost[idx].imag, aOrig[idx].imag)) << p.caseName;
            }
        }
    }
}

} // namespace

class CherTest : public BlasTest<CherParam> {};

TEST_F(CherTest, NullHandle)
{
    float alpha = 1.0f;
    aclblasComplex dummy{0.0f, 0.0f};
    aclblasStatus_t ret = aclblasCher(nullptr, ACLBLAS_UPPER, 4, &alpha, &dummy, 1, &dummy, 4);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_HANDLE_IS_NULLPTR));
}

TEST_F(CherTest, ZeroDimensionAndDeviceAlpha)
{
    EXPECT_EQ(
        static_cast<int>(aclblasCher(
            CherTest::handle_, static_cast<aclblasFillMode_t>(999), 0, nullptr, nullptr, 0, nullptr, 0)),
        static_cast<int>(ACLBLAS_STATUS_SUCCESS));
    const float alpha = 2.0f;
    aclblasComplex x[2] = {{1.0f, 1.0f}, {2.0f, -1.0f}};
    aclblasComplex matrix[4] = {};
    ASSERT_EQ(
        static_cast<int>(
            aclblasCher_npu(CherTest::handle_, ACLBLAS_LOWER, 2, &alpha, x, 1, matrix, 2, {true})),
        static_cast<int>(ACLBLAS_STATUS_SUCCESS));
    EXPECT_FLOAT_EQ(matrix[0].real, 4.0f);
    EXPECT_FLOAT_EQ(matrix[0].imag, 0.0f);
    EXPECT_FLOAT_EQ(matrix[1].real, 2.0f);
    EXPECT_FLOAT_EQ(matrix[1].imag, -6.0f);
    EXPECT_FLOAT_EQ(matrix[3].real, 10.0f);
    EXPECT_FLOAT_EQ(matrix[3].imag, 0.0f);
}

INSTANTIATE_TEST_SUITE_P(
    Cher, CherTest, ::testing::ValuesIn(GetCasesFromCsv<CherParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<CherParam>);

TEST_P(CherTest, CsvDriven)
{
    const auto& p = GetParam();
    const bool upper = (p.uplo == ACLBLAS_UPPER);
    std::vector<aclblasComplex> xHost = MakeComplexStrided(p.n, p.incx, p.xFill, p.randomSeed);
    std::vector<aclblasComplex> aHost =
        MakeComplexMatrixHermitianPrep(p.n, std::max(1, p.lda), p.aFill, p.randomSeed + 1U);
    std::vector<aclblasComplex> aOrig = aHost;

    const float* alphaPtr = p.alphaNull ? nullptr : &p.alpha;
    const aclblasComplex* xPtr = xHost.empty() ? nullptr : xHost.data();
    aclblasComplex* aPtr = aHost.empty() ? nullptr : aHost.data();

    const aclblasStatus_t ret =
        aclblasCher_npu(CherTest::handle_, p.uplo, p.n, alphaPtr, xPtr, p.incx, aPtr, p.lda);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
    if (p.expectResult != ACLBLAS_STATUS_SUCCESS) {
        return;
    }
    if (p.n == 0 || aPtr == nullptr || (alphaPtr != nullptr && *alphaPtr == 0.0f)) {
        return;
    }
    std::vector<aclblasComplex> golden = aOrig;
    ASSERT_EQ(
        static_cast<int>(aclblasCher_cpu(
            CherTest::handle_, p.uplo, p.n, alphaPtr, xHost.data(), p.incx, golden.data(), p.lda)),
        static_cast<int>(ACLBLAS_STATUS_SUCCESS));
    VerifyCherAccuracy(p, upper, aHost, aOrig, golden);
}
