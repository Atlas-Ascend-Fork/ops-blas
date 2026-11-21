/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <complex>
#include <cstring>
#include <limits>
#include <unordered_set>
#include <utility>
#include <vector>

#include "blas_test.h"
#include "cgemm3m_param.h"
#include "cgemm3m_golden.h"
#include "cgemm3m_verify.h"
#include "cgemm3m_profile.h"
#include "../cgemm/arch35/cgemm_npu_wrapper.h"
#include "cblas_compat.h"
#include "fill.h"

namespace {

aclblasStatus_t RunCgemm3m(
    aclblasHandle_t handle, const Cgemm3mParam& p, const std::vector<aclblasComplex>& a,
    const std::vector<aclblasComplex>& b, std::vector<aclblasComplex>& c)
{
    CgemmDeviceBuffers buffers;
    const aclblasComplex alpha{p.alphaReal, p.alphaImag};
    const aclblasComplex beta{p.betaReal, p.betaImag};
    aclblasStatus_t status = AllocateCgemmDeviceBuffers(
        buffers, p.k, a.empty() ? nullptr : a.data(), b.empty() ? nullptr : b.data(), c.empty() ? nullptr : c.data(),
        a.size() * sizeof(aclblasComplex), b.size() * sizeof(aclblasComplex), c.size() * sizeof(aclblasComplex));
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }
    status = aclblasCgemm3m(
        handle, p.transA, p.transB, p.m, p.n, p.k, p.alphaNull ? nullptr : &alpha,
        p.aNull ? nullptr : static_cast<const aclblasComplex*>(buffers.a), p.lda,
        p.bNull ? nullptr : static_cast<const aclblasComplex*>(buffers.b), p.ldb, p.betaNull ? nullptr : &beta,
        p.cNull ? nullptr : static_cast<aclblasComplex*>(buffers.c), p.ldc);
    return CgemmSyncAndCopyBack(buffers, c.data(), c.size() * sizeof(aclblasComplex), status);
}

void DiagnoseCgemm3m(
    const Cgemm3mParam& p, const std::vector<aclblasComplex>& a, const std::vector<aclblasComplex>& b,
    const std::vector<aclblasComplex>& actual, const std::vector<aclblasComplex>& golden)
{
    double maximum = -1.0;
    int worstRow = 0;
    int worstCol = 0;
    for (int col = 0; col < p.n; ++col) {
        for (int row = 0; row < p.m; ++row) {
            const size_t index = static_cast<size_t>(col) * p.ldc + row;
            const double error = std::max(
                std::abs(static_cast<double>(actual[index].real) - golden[index].real),
                std::abs(static_cast<double>(actual[index].imag) - golden[index].imag));
            if (error > maximum) {
                maximum = error;
                worstRow = row;
                worstCol = col;
            }
        }
    }
    if (p.k <= 0 || p.alphaReal != 1.0f || p.alphaImag != 0.0f || p.betaReal != 0.0f || p.betaImag != 0.0f) {
        return;
    }
    std::complex<double> sum(0.0, 0.0);
    for (int inner = 0; inner < p.k; ++inner) {
        const size_t ai = p.transA == ACLBLAS_OP_N ? static_cast<size_t>(inner) * p.lda + worstRow :
                                                     static_cast<size_t>(worstRow) * p.lda + inner;
        const size_t bi = p.transB == ACLBLAS_OP_N ? static_cast<size_t>(worstCol) * p.ldb + inner :
                                                     static_cast<size_t>(inner) * p.ldb + worstCol;
        std::complex<double> av(a[ai].real, a[ai].imag);
        std::complex<double> bv(b[bi].real, b[bi].imag);
        sum += (p.transA == ACLBLAS_OP_C ? std::conj(av) : av) * (p.transB == ACLBLAS_OP_C ? std::conj(bv) : bv);
    }
    const size_t index = static_cast<size_t>(worstCol) * p.ldc + worstRow;
    std::cout << "CGEMM3M_DIAGNOSTIC," << p.caseName << ',' << worstRow << ',' << worstCol << ','
              << std::setprecision(12) << actual[index].real << ',' << actual[index].imag << ',' << golden[index].real
              << ',' << golden[index].imag << ',' << sum.real() << ',' << sum.imag() << std::endl;
}

} // namespace

class Cgemm3mTest : public BlasTest<Cgemm3mParam> {};

TEST(Cgemm3mVerifier, MatchedRatioAndMaxError)
{
    Cgemm3mComponentResult result;
    for (int index = 0; index < 99; ++index) {
        result.Add(0.0f, 0.0f);
    }
    EXPECT_TRUE(result.Passed());
    result.Add(0.001f, 0.0f);
    EXPECT_TRUE(result.Passed()); // One mismatch among 100 finite values meets the task's 99% ratio.
    result.Add(0.001f, 0.0f);
    EXPECT_FALSE(result.Passed());
    for (int index = 0; index < 99; ++index) {
        result.Add(0.0f, 0.0f);
    }
    EXPECT_TRUE(result.Passed());
    Cgemm3mComponentResult maximum;
    for (int index = 0; index < 99; ++index) {
        maximum.Add(0.0f, 0.0f);
    }
    maximum.Add(0.02f, 0.0f);
    EXPECT_FALSE(maximum.Passed()); // Ratio passes, but the task's maximum-error bound fails.
}

TEST(Cgemm3mVerifier, MagnitudeScaledTolerance)
{
    Cgemm3mComponentResult result;
    result.Add(0.001f, 0.0f);
    EXPECT_FALSE(result.Passed());
    result.Add(8192.02f, 8192.0f);
    EXPECT_EQ(result.maxErrorGolden, 8192.0f);
    EXPECT_FALSE(result.Passed());
    Cgemm3mComponentResult boundary;
    boundary.Add(1.0f / 8192.0f, 0.0f);
    EXPECT_TRUE(boundary.Passed());
    boundary.Add(2.0f / 8192.0f, 0.0f);
    EXPECT_FALSE(boundary.Passed());
    Cgemm3mComponentResult large;
    large.Add(8192.5f, 8192.0f);
    EXPECT_EQ(large.matched, 1U);
    EXPECT_FALSE(large.Passed()); // Maximum-error gate is independent of the matched ratio.
    Cgemm3mComponentResult ulpBoundary;
    ulpBoundary.Add(8192.03125f, 8192.0f);
    EXPECT_EQ(ulpBoundary.matched, ulpBoundary.finite);
    EXPECT_TRUE(ulpBoundary.Passed());
    ulpBoundary.Add(8192.0625f, 8192.0f);
    EXPECT_EQ(ulpBoundary.matched, 2U);
    EXPECT_FALSE(ulpBoundary.Passed());
}

TEST(Cgemm3mVerifier, NonfiniteRules)
{
    const float inf = std::numeric_limits<float>::infinity();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    Cgemm3mComponentResult result;
    result.Add(nan, nan);
    result.Add(inf, inf);
    result.Add(-inf, -inf);
    EXPECT_TRUE(result.Passed());
    for (const auto pair : {std::pair<float, float>{nan, inf}, {inf, nan}, {-inf, nan}, {nan, -inf}}) {
        Cgemm3mComponentResult equivalent;
        equivalent.Add(pair.first, pair.second);
        EXPECT_TRUE(equivalent.Passed());
    }
    for (const auto pair : {std::pair<float, float>{inf, -inf}, {-inf, inf}}) {
        Cgemm3mComponentResult mismatch;
        mismatch.Add(pair.first, pair.second);
        EXPECT_FALSE(mismatch.Passed());
    }
    Cgemm3mComponentResult finiteMismatch;
    finiteMismatch.Add(0.0f, inf);
    EXPECT_FALSE(finiteMismatch.Passed());
}

TEST_F(Cgemm3mTest, NullHandle)
{
    const aclblasComplex alpha{1.0f, 0.0f};
    const aclblasComplex beta{0.0f, 0.0f};
    EXPECT_EQ(
        aclblasCgemm3m(nullptr, ACLBLAS_OP_N, ACLBLAS_OP_N, 0, 0, 0, &alpha, nullptr, 1, nullptr, 1, &beta, nullptr, 1),
        ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
}

TEST_F(Cgemm3mTest, EmptyOutputPreservesSentinel)
{
    Cgemm3mParam p(csv_map{{"m", "0"}, {"n", "3"}, {"k", "2"}, {"lda", "4"}, {"ldb", "4"}, {"ldc", "4"}});
    const std::vector<aclblasComplex> unused;
    const std::vector<aclblasComplex> sentinel(12, {7.0f, -3.0f});
    for (const bool emptyRows : {true, false}) {
        p.m = emptyRows ? 0 : 3;
        p.n = emptyRows ? 3 : 0;
        auto actual = sentinel;
        ASSERT_EQ(RunCgemm3m(handle_, p, unused, unused, actual), ACLBLAS_STATUS_SUCCESS);
        EXPECT_EQ(std::memcmp(actual.data(), sentinel.data(), actual.size() * sizeof(aclblasComplex)), 0);
    }
}

TEST_F(Cgemm3mTest, ZeroProductAllowsNullInputs)
{
    Cgemm3mParam p(csv_map{{"m", "3"}, {"n", "2"}, {"k", "7"}, {"lda", "3"}, {"ldb", "7"}, {"ldc", "5"}});
    p.aNull = true;
    p.bNull = true;
    const std::vector<aclblasComplex> unused;
    for (const bool zeroK : {true, false}) {
        p.k = zeroK ? 0 : 7;
        p.alphaReal = zeroK ? 1.0f : 0.0f;
        for (const aclblasComplex beta : {aclblasComplex{0.0f, 0.0f}, {1.0f, 0.0f}, {0.5f, -0.25f}}) {
            p.betaReal = beta.real;
            p.betaImag = beta.imag;
            const float input = beta.real == 0.0f ? std::numeric_limits<float>::quiet_NaN() : 7.0f;
            const std::vector<aclblasComplex> initial(10, {input, -3.0f});
            auto actual = initial;
            auto golden = initial;
            GoldenCgemm3m(p, unused, unused, golden);
            ASSERT_EQ(RunCgemm3m(handle_, p, unused, unused, actual), ACLBLAS_STATUS_SUCCESS);
            EXPECT_TRUE(VerifyCgemm3m(actual, golden, initial, p.m, p.n, p.ldc, "ZeroProduct"));
        }
    }
}

TEST_F(Cgemm3mTest, SegmentCancellation)
{
    constexpr int size = 2048;
    constexpr int termsPerSegment = 33; // The packed 2048-wide reduction uses 33 K terms per lane.
    constexpr int positiveSegments = 20;
    constexpr int negativeSegments = 20;
    Cgemm3mParam p(csv_map{{"m", "2048"}, {"n", "2048"}, {"k", "2048"}});
    std::vector<aclblasComplex> a(static_cast<size_t>(size) * size, {0.0f, 0.0f});
    std::vector<aclblasComplex> b(a.size(), {0.0f, 0.0f});
    const std::vector<aclblasComplex> initial(a.size(), {0.0f, 0.0f});
    auto expected = initial;
    auto actual = initial;
    for (int inner = 0; inner < (positiveSegments + negativeSegments) * termsPerSegment; ++inner) {
        a[static_cast<size_t>(inner) * size] = {1.0f, 0.0f};
        b[inner] = {inner < positiveSegments * termsPerSegment ? 4096.1f : -4096.1f, 0.0f};
    }
    // Equal positive/negative segments cancel. An independent final K term
    // also checks a small nonzero residual, rather than only exact zero.
    a[static_cast<size_t>(size - 1) * size] = {1.0f, 0.0f};
    for (float residual : {0.0f, 1.0f}) {
        b[size - 1] = {residual, 0.0f};
        expected[0].real = residual;
        actual = initial;
        ASSERT_EQ(RunCgemm3m(handle_, p, a, b, actual), ACLBLAS_STATUS_SUCCESS);
        EXPECT_NEAR(actual[0].real, residual, 0.01f);
        EXPECT_NEAR(actual[0].imag, 0.0f, 0.01f);
        EXPECT_TRUE(VerifyCgemm3m(actual, expected, initial, size, size, size, "SegmentCancellation"));
    }
}

TEST(Cgemm3mCsvIntegrity, CsvCaseCount)
{
    const csv_map special{{"alpha_real", "null"}, {"beta_real", "null"}, {"transa", "999"}, {"transb", "999"}};
    const auto normalized = Cgemm3mColumns(special);
    EXPECT_EQ(normalized.at("alpha_real"), "1");
    EXPECT_EQ(normalized.at("beta_real"), "0");
    EXPECT_EQ(normalized.at("transA"), "ACLBLAS_OP_N");
    EXPECT_EQ(normalized.at("transB"), "ACLBLAS_OP_N");
    const Cgemm3mParam invalid(special);
    EXPECT_TRUE(invalid.alphaNull && invalid.betaNull);
    EXPECT_EQ(static_cast<int>(invalid.transA), 999);
    EXPECT_EQ(static_cast<int>(invalid.transB), 999);
    const auto cases = GetCasesFromCsv<Cgemm3mParam>(ReplaceFileExtension2Csv(__FILE__));
    ASSERT_EQ(cases.size(), 1200U);
    std::unordered_set<std::string> names;
    size_t performance = 0;
    for (const auto& param : cases) {
        EXPECT_FALSE(param.caseName.empty());
        EXPECT_TRUE(names.insert(param.caseName).second) << "Duplicate case: " << param.caseName;
        if (param.caseName.rfind("TC_PF_", 0) == 0) {
            ++performance;
        }
    }
    EXPECT_EQ(performance, 200U);
    EXPECT_EQ(cases.size() - performance, 1000U);
}

INSTANTIATE_TEST_SUITE_P(
    Cgemm3m, Cgemm3mTest, ::testing::ValuesIn(GetCasesFromCsv<Cgemm3mParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<Cgemm3mParam>);

TEST_P(Cgemm3mTest, CsvDriven)
{
    const auto& p = GetParam();
    auto a = makeBlasComplexMatrix(
        p.transA == ACLBLAS_OP_N ? p.m : p.k, p.transA == ACLBLAS_OP_N ? p.k : p.m, p.lda, p.aFill, p.randomSeed);
    auto b = makeBlasComplexMatrix(
        p.transB == ACLBLAS_OP_N ? p.k : p.n, p.transB == ACLBLAS_OP_N ? p.n : p.k, p.ldb, p.bFill, p.randomSeed + 1U);
    auto initial = makeBlasComplexMatrix(p.m, p.n, p.ldc, p.cFill, p.randomSeed + 2U);
    if (std::getenv("CGEMM3M_PROFILE") != nullptr) {
        ASSERT_EQ(ProfileCgemm3m(handle_, stream_, p, a, b, initial), ACLBLAS_STATUS_SUCCESS);
        return;
    }
    auto actual = initial;
    ASSERT_EQ(RunCgemm3m(handle_, p, a, b, actual), p.expectResult);
    if (p.expectResult != ACLBLAS_STATUS_SUCCESS) {
        return;
    }
    if (p.m == 0 || p.n == 0) {
        ASSERT_EQ(actual.size(), initial.size());
        if (!actual.empty()) {
            EXPECT_EQ(std::memcmp(actual.data(), initial.data(), actual.size() * sizeof(aclblasComplex)), 0);
        }
        return;
    }
    auto golden = initial;
    GoldenCgemm3m(p, a, b, golden);
    const bool passed = VerifyCgemm3m(actual, golden, initial, p.m, p.n, p.ldc, p.caseName);
    if (!passed) {
        DiagnoseCgemm3m(p, a, b, actual, golden);
    }
    EXPECT_TRUE(passed);
    auto repeated = initial;
    ASSERT_EQ(RunCgemm3m(handle_, p, a, b, repeated), ACLBLAS_STATUS_SUCCESS);
    EXPECT_EQ(std::memcmp(actual.data(), repeated.data(), actual.size() * sizeof(aclblasComplex)), 0)
        << "Identical inputs must produce bitwise identical output";
}
