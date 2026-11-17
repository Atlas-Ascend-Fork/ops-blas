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
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

#include "verify.h"
#include "triangle_preservation.h"
#include "triangle_fp64_reference.h"
#include "blas_test.h"
#include "csv_loader.h"
#include "csv_case_contract.h"
#include "fill.h"
#include "cherk_param.h"
#include "../cherk_golden.h"
#include "cherk_npu_wrapper.h"

class CherkTest : public BlasTest<CherkParam> {};

static bool HasCasePrefix(const CherkParam& p, const char* prefix) { return p.caseName.rfind(prefix, 0U) == 0U; }

static std::vector<CherkParam> LoadCherkCorrectnessCases()
{
    const bool fullRegression = std::getenv("CHERK_FULL_REGRESSION") != nullptr;
    std::string csvPath = ReplaceFileExtension2Csv(__BASE_FILE__);
    // Only arch22 has a separate smoke/full CSV pair.
    if (fullRegression && csvPath.find("arch22") != std::string::npos) {
        csvPath.insert(csvPath.size() - 4U, "_full");
    }
    const auto allCases = GetCasesFromCsv<CherkParam>(csvPath);
    VerifyCsvComposition(allCases, 1224U, 200U, csvPath);
    std::vector<CherkParam> selectedCases;
    for (const CherkParam& p : allCases) {
        if (!fullRegression && HasCasePrefix(p, "TC_PF_")) {
            continue;
        }

        selectedCases.push_back(p);
    }
    VerifyCsvComposition(selectedCases, fullRegression ? 1224U : 1024U, fullRegression ? 200U : 0U, "CHERK selection");
    return selectedCases;
}

// ── TEST_F: null handle (not CSV-driven) ──
TEST_F(CherkTest, NullHandle)
{
    float alpha = 1.0f;
    float beta = 0.0f;
    aclblasStatus_t ret =
        aclblasCherk_npu(nullptr, ACLBLAS_UPPER, ACLBLAS_OP_N, 4, 4, &alpha, nullptr, 4, &beta, nullptr, 4);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_HANDLE_IS_NULLPTR));
}

INSTANTIATE_TEST_SUITE_P(
    Cherk, CherkTest, ::testing::ValuesIn(LoadCherkCorrectnessCases()), PrintCaseInfoString<CherkParam>);

// ═══════════════════════════════════════════════════════════════════════════════
// CSV-driven parameterised test (5-step flow)
//   1. Generate host data  2. Run NPU  3. Check return code
//   4. Run CPU golden       5. Verify precision
// ═══════════════════════════════════════════════════════════════════════════════

struct CherkHostData {
    std::vector<aclblasComplex> aHost;
    std::vector<aclblasComplex> cHost;
    std::vector<aclblasComplex> cGolden;
    const aclblasComplex* aPtr = nullptr;
    aclblasComplex* cPtr = nullptr;
    aclblasComplex* cGoldenPtr = nullptr;
    size_t cCount = 0;
};

static bool PrepareHostData(const CherkParam& p, CherkHostData& d, bool prepareGolden)
{
    const int aRows = std::max(1, p.lda);
    const int aCols = std::max(0, (p.trans == ACLBLAS_OP_N) ? p.k : p.n);

    const size_t cBytes = static_cast<size_t>(p.ldc) * static_cast<size_t>(std::max(0, p.n)) * sizeof(aclblasComplex);
    const size_t aBytes = static_cast<size_t>(p.lda) * static_cast<size_t>(aCols) * sizeof(aclblasComplex);
    const size_t cCopies = prepareGolden ? 2U : 1U;
    constexpr size_t kHostMemLimit = 8ULL * 1024ULL * 1024ULL * 1024ULL; // §8 遗留问题 4
    if (cCopies * cBytes + aBytes > kHostMemLimit) {
        std::cout << "[SKIP] host memory estimate (" << (cCopies * cBytes + aBytes) / (1024 * 1024)
                  << " MB) exceeds limit for n=" << p.n << ", ldc=" << p.ldc << std::endl;
        return false;
    }

    try {
        d.aHost = makeBlasComplexMatrix(aRows, aCols, p.lda, p.aFill, p.randomSeed);
        d.cHost =
            makeBlasComplexMatrix(std::max(0, p.n), std::max(0, p.n), std::max(1, p.ldc), p.cFill, p.randomSeed + 1U);
    } catch (const std::bad_alloc&) {
        std::cout << "[SKIP] host memory allocation failed for n=" << p.n << ", k=" << p.k << std::endl;
        return false;
    }

    PoisonUntouched(d.cHost.data(), p.n, p.ldc, p.uplo == ACLBLAS_UPPER);
    d.aPtr = (d.aHost.empty() || p.nullA) ? nullptr : d.aHost.data();
    d.cPtr = (d.cHost.empty() || p.nullC) ? nullptr : d.cHost.data();
    d.cCount = static_cast<size_t>(p.ldc) * static_cast<size_t>(std::max(0, p.n));

    if (prepareGolden && d.cPtr != nullptr) {
        try {
            d.cGolden = d.cHost; // golden starts from input-old C; non-uplo triangle stays untouched
        } catch (const std::bad_alloc&) {
            std::cout << "[SKIP] host memory allocation failed for golden copy (n=" << p.n << ")" << std::endl;
            return false;
        }
        d.cGoldenPtr = d.cGolden.data();
    }
    return true;
}

static void VerifyCherkFp64(const CherkParam& p, const CherkHostData& data)
{
    const auto golden =
        CherkGoldenFp64(p.uplo, p.trans, p.n, p.k, p.alpha, data.aPtr, p.lda, p.beta, data.cGoldenPtr, p.ldc);
    VerifyTriangleFp64(
        p.caseName.c_str(), data.cPtr, data.cGoldenPtr, golden.data(), p.n, p.ldc, p.uplo == ACLBLAS_UPPER,
        (p.alpha == 0.0F || p.k == 0) && p.beta == 1.0F);
}

TEST_P(CherkTest, CsvDriven)
{
    const auto& p = GetParam();
    CherkHostData d;
    if (!PrepareHostData(p, d, true)) {
        GTEST_SKIP() << "Skipped: host memory limit";
    }

    // nullAlpha: wrapper forwards nullptr for alpha so the API's INVALID_VALUE
    // path is exercised. The alpha value in p.alpha is irrelevant in that case.
    aclblasStatus_t ret = aclblasCherk_npu(
        CherkTest::handle_, p.uplo, p.trans, p.n, p.k, &p.alpha, d.aPtr, p.lda, p.nullBeta ? nullptr : &p.beta, d.cPtr,
        p.ldc, p.nullAlpha);

    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
    if (p.expectResult != ACLBLAS_STATUS_SUCCESS || p.n == 0 || d.cPtr == nullptr) {
        return;
    }

    VerifyCherkFp64(p, d);
}

TEST(CherkVerifierRegression, NonFiniteMismatch)
{
    VerifyConfig cfg;
    applyMixedTolerance(cfg, ACL_FLOAT, nullptr, 1U);
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    const float one = 1.0f;
    EXPECT_FALSE(Verifier::verifyVector(&nan, &one, 1U, 1, cfg, "nan_finite"));
    EXPECT_FALSE(Verifier::verifyVector(&one, &nan, 1U, 1, cfg, "finite_nan"));
    EXPECT_FALSE(Verifier::verifyVector(&inf, &one, 1U, 1, cfg, "inf_finite"));
    const float negativeInf = -inf;
    EXPECT_FALSE(Verifier::verifyVector(&inf, &negativeInf, 1U, 1, cfg, "opposite_inf"));
    EXPECT_TRUE(Verifier::verifyVector(&nan, &nan, 1U, 1, cfg, "matching_nan"));
    EXPECT_TRUE(Verifier::verifyVector(&inf, &inf, 1U, 1, cfg, "matching_inf"));
    EXPECT_FALSE(Verifier::verifyEach([&](auto consume) { consume(nan, one); }, 1U, cfg, "visitor_nan"));
}

static void RunArch22GoldenRegression(aclblasHandle_t handle, const CherkParam& p)
{
    CherkHostData d;
    ASSERT_TRUE(PrepareHostData(p, d, true));

    ASSERT_EQ(
        aclblasCherk_npu(handle, p.uplo, p.trans, p.n, p.k, &p.alpha, d.aPtr, p.lda, &p.beta, d.cPtr, p.ldc),
        ACLBLAS_STATUS_SUCCESS);
    VerifyCherkFp64(p, d);
}

// Cover panel boundaries and both sides of the outer-product geometry limits.
TEST_F(CherkTest, OuterK2GoldenRegression)
{
    const csv_map row{
        {"case_name", "OuterK2GoldenRegression"},
        {"uplo", "ACLBLAS_UPPER"},
        {"trans", "ACLBLAS_OP_N"},
        {"n", "120"},
        {"k", "2"},
        {"alpha", "1"},
        {"a_fill", "RANDOM_NORM_5_5"},
        {"lda", "120"},
        {"beta", "0"},
        {"c_fill", "RANDOM_NORM_5_5"},
        {"ldc", "120"},
        {"expect_result", "ACLBLAS_STATUS_SUCCESS"},
        {"random_seed", "20261136"},
    };
    for (int n : {7, 8, 9, 16, 112, 120, 128, 129, 136}) {
        CherkParam p(row);
        p.n = n;
        p.lda = n + 8;
        p.ldc = n + 8;
        p.caseName += "_n" + std::to_string(n);
        RunArch22GoldenRegression(CherkTest::handle_, p);
    }
}

TEST_F(CherkTest, Tiny1AlphaZeroInfRegression)
{
    const csv_map row{
        {"case_name", "Tiny1AlphaZeroInfRegression"},
        {"uplo", "ACLBLAS_UPPER"},
        {"trans", "ACLBLAS_OP_N"},
        {"n", "1"},
        {"k", "1"},
        {"alpha", "0"},
        {"a_fill", "VALUE_NORM_INF"},
        {"lda", "1"},
        {"beta", "0"},
        {"c_fill", "RANDOM_NORM_5_5"},
        {"ldc", "1"},
        {"expect_result", "ACLBLAS_STATUS_SUCCESS"},
        {"random_seed", "20261201"},
    };
    RunArch22GoldenRegression(CherkTest::handle_, CherkParam(row));
}

TEST_F(CherkTest, OuterK2AlphaZeroInfRegression)
{
    const csv_map row{
        {"case_name", "OuterK2AlphaZeroInfRegression"},
        {"uplo", "ACLBLAS_UPPER"},
        {"trans", "ACLBLAS_OP_N"},
        {"n", "120"},
        {"k", "2"},
        {"alpha", "0"},
        {"a_fill", "VALUE_NORM_INF"},
        {"lda", "120"},
        {"beta", "0"},
        {"c_fill", "RANDOM_NORM_5_5"},
        {"ldc", "120"},
        {"expect_result", "ACLBLAS_STATUS_SUCCESS"},
        {"random_seed", "20261202"},
    };
    RunArch22GoldenRegression(CherkTest::handle_, CherkParam(row));
}

TEST_F(CherkTest, ExtremeTinyScalars)
{
    const csv_map row{
        {"case_name", "ExtremeTiny"},
        {"a_fill", "RANDOM_EXTREME"},
        {"c_fill", "RANDOM_NORM_5_5"},
        {"random_seed", "20260166"}};
    for (const int n : {8, 16, 32}) {
        CherkParam p(row);
        p.n = n;
        p.k = n;
        p.lda = n;
        p.ldc = n;
        ASSERT_NO_FATAL_FAILURE(RunArch22GoldenRegression(handle_, p));
        p.alpha = 0.5F;
        p.beta = -0.25F;
        ASSERT_NO_FATAL_FAILURE(RunArch22GoldenRegression(handle_, p));
    }
}
