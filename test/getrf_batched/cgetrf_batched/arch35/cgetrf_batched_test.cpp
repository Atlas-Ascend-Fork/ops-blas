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
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <random>
#include <vector>

#include "verify.h"
#include "blas_test.h"
#include "csv_loader.h"
#include "cgetrf_batched_param.h"
#include "cgetrf_batched_golden.h"
#include "cgetrf_batched_precision.h"
#include "cgetrf_batched_npu_wrapper.h"

// ---------------------------------------------------------------------------
// Matrix generation helpers
// ---------------------------------------------------------------------------

static void GenerateRandomNonsingular(
    std::vector<aclblasComplex>& mat, int n, int lda, std::mt19937& rng, std::uniform_real_distribution<float>& dist)
{
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) {
            mat[j * lda + i] = {dist(rng), dist(rng)};
        }
        mat[j * lda + j].real += 12.0f * static_cast<float>(n);
    }
}

static void GenerateSingularZeroCol(
    std::vector<aclblasComplex>& mat, int n, int lda, std::mt19937& rng, std::uniform_real_distribution<float>& dist)
{
    GenerateRandomNonsingular(mat, n, lda, rng, dist);
    for (int i = 0; i < n; i++) {
        mat[i] = {0.0f, 0.0f};
    }
}

static void GenerateIdentity(std::vector<aclblasComplex>& mat, int n, int lda)
{
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) {
            mat[j * lda + i] = (i == j) ? aclblasComplex{1.0f, 0.0f} : aclblasComplex{0.0f, 0.0f};
        }
    }
}

static void GenerateDiagonallyDominant(
    std::vector<aclblasComplex>& mat, int n, int lda, std::mt19937& rng, std::uniform_real_distribution<float>& dist)
{
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) {
            mat[j * lda + i] = {dist(rng), dist(rng)};
        }
        mat[j * lda + j] = {12.0f * static_cast<float>(n), 0.0f};
    }
}

static void GenerateIllConditioned(std::vector<aclblasComplex>& mat, int n, int lda)
{
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) {
            mat[j * lda + i] = {1.0f / static_cast<float>(i + j + 1), 0.0f};
        }
    }
}

static void GenerateSingularDependentRow(
    std::vector<aclblasComplex>& mat, int n, int lda, std::mt19937& rng, std::uniform_real_distribution<float>& dist)
{
    GenerateRandomNonsingular(mat, n, lda, rng, dist);
    int depRow = std::min(n - 1, 2);
    if (depRow > 0) {
        for (int j = 0; j < n; j++) {
            const aclblasComplex source = mat[j * lda];
            mat[j * lda + depRow] = {2.0f * source.real, 2.0f * source.imag};
        }
    }
}

static void GenerateMixed(
    std::vector<aclblasComplex>& mat, int n, int lda, std::mt19937& rng, std::uniform_real_distribution<float>& dist,
    int batchIdx)
{
    GenerateRandomNonsingular(mat, n, lda, rng, dist);
    if (batchIdx % 2 == 1) {
        for (int i = 0; i < n; i++) {
            mat[i] = {0.0f, 0.0f};
        }
    }
}

/**
 * Generate a single n×n matrix in column-major storage (lda stride).
 * Returns a vector of size lda * n.
 */
static std::vector<aclblasComplex> generateMatrix(
    int n, int lda, GetrfMatrixType type, uint32_t seed, int batchIdx = 0, int batchSize = 1)
{
    const size_t matSize = static_cast<size_t>(lda) * std::max(1, n);
    std::vector<aclblasComplex> mat(matSize, aclblasComplex{0.0f, 0.0f});
    if (n <= 0)
        return mat;

    std::mt19937 rng(seed + batchIdx * 1000);
    std::uniform_real_distribution<float> dist(-5.0f, 5.0f);

    switch (type) {
        case GetrfMatrixType::RANDOM_NONSINGULAR:
            GenerateRandomNonsingular(mat, n, lda, rng, dist);
            break;
        case GetrfMatrixType::SINGULAR_ZERO_COL:
            GenerateSingularZeroCol(mat, n, lda, rng, dist);
            break;
        case GetrfMatrixType::IDENTITY:
            GenerateIdentity(mat, n, lda);
            break;
        case GetrfMatrixType::DIAGONALLY_DOMINANT:
            GenerateDiagonallyDominant(mat, n, lda, rng, dist);
            break;
        case GetrfMatrixType::ILL_CONDITIONED:
            GenerateIllConditioned(mat, n, lda);
            break;
        case GetrfMatrixType::SINGULAR_DEPENDENT_ROW:
            GenerateSingularDependentRow(mat, n, lda, rng, dist);
            break;
        case GetrfMatrixType::MIXED:
            GenerateMixed(mat, n, lda, rng, dist, batchIdx);
            break;
    }
    return mat;
}

// ---------------------------------------------------------------------------
// Test fixture
// ---------------------------------------------------------------------------

class CgetrfBatchedTest : public BlasTest<CgetrfBatchedParam> {};

INSTANTIATE_TEST_SUITE_P(
    CgetrfBatched, CgetrfBatchedTest,
    ::testing::ValuesIn(GetCasesFromCsv<CgetrfBatchedParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<CgetrfBatchedParam>);

static void TestErrorPath(const CgetrfBatchedParam& p, aclblasHandle_t handle)
{
    bool nullAarray = (p.matrixType == GetrfMatrixType::NULLPTR_AARRAY);
    bool nullInfo = (p.matrixType == GetrfMatrixType::NULLPTR_INFOARRAY);
    bool nullHandle = (p.matrixType == GetrfMatrixType::NULLPTR_HANDLE);
    bool usePivot = (p.pivotMode == GetrfPivotMode::PIVOT);

    int safeN = std::max(1, p.n);
    int safeBatch = std::max(1, p.batchSize);
    int safeLda = std::max(p.lda, safeN);

    std::vector<std::vector<aclblasComplex>> errMatrices;
    std::vector<aclblasComplex*> errPtrs;
    if (!nullAarray) {
        errMatrices.resize(safeBatch);
        errPtrs.resize(safeBatch);
        for (int b = 0; b < safeBatch; b++) {
            errMatrices[b].resize(static_cast<size_t>(safeLda) * safeN, aclblasComplex{1.0f, 0.0f});
            errPtrs[b] = errMatrices[b].data();
        }
    }

    std::vector<int> errPivot(usePivot ? static_cast<size_t>(safeN) * safeBatch : 0, 0);
    std::vector<int> errInfo(safeBatch, 0);

    aclblasHandle_t h = nullHandle ? nullptr : handle;
    aclblasComplex** aPtr = nullAarray ? nullptr : errPtrs.data();
    int* pivPtr = usePivot ? errPivot.data() : nullptr;
    int* infoPtr = nullInfo ? nullptr : errInfo.data();

    aclblasStatus_t ret = aclblasCgetrfBatched_npu(h, p.n, aPtr, p.lda, pivPtr, infoPtr, p.batchSize);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
}

static void TestNoOpPath(const CgetrfBatchedParam& p, aclblasHandle_t handle)
{
    std::vector<aclblasComplex*> dummyA(std::max(0, p.batchSize), nullptr);
    std::vector<int> dummyInfo(std::max(0, p.batchSize), -1);
    aclblasComplex** aPtr = dummyA.empty() ? nullptr : dummyA.data();
    int* infoPtr = dummyInfo.empty() ? nullptr : dummyInfo.data();

    aclblasStatus_t ret = aclblasCgetrfBatched_npu(handle, p.n, aPtr, p.lda, nullptr, infoPtr, p.batchSize);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
}

static bool VerifyInfoArray(
    const std::vector<int>& hostInfo, const std::vector<int>& goldenInfo, int batchSize, const std::string& caseName)
{
    bool matched = true;
    for (int b = 0; b < batchSize; b++) {
        matched = matched && hostInfo[b] == goldenInfo[b];
        EXPECT_EQ(hostInfo[b], goldenInfo[b]) << "[" << caseName << "] batch=" << b
                                              << " info mismatch: NPU=" << hostInfo[b] << " golden=" << goldenInfo[b];
    }
    std::cout << "CGETRF_INFO_CHECK," << caseName << ',' << batchSize << ',' << matched << std::endl;
    return matched;
}

static bool VerifyPivotArray(
    const std::vector<int>& hostPivot, const std::vector<int>& goldenPivot, size_t pivotSize,
    const std::string& caseName)
{
    bool pivotMatch = true;
    for (size_t i = 0; i < pivotSize; i++) {
        if (hostPivot[i] != goldenPivot[i]) {
            pivotMatch = false;
            std::cout << "[" << caseName << "] pivot mismatch at index " << i << ": NPU=" << hostPivot[i]
                      << " golden=" << goldenPivot[i] << std::endl;
            break;
        }
    }
    EXPECT_TRUE(pivotMatch) << "[" << caseName << "] PivotArray mismatch";
    std::cout << "CGETRF_PIVOT_CHECK," << caseName << ',' << pivotSize << ',' << pivotMatch << std::endl;
    return pivotMatch;
}

static bool VerifyLUMatrices(
    const std::vector<std::vector<aclblasComplex>>& hostMatrices,
    const std::vector<std::vector<aclblasComplex>>& goldenMatrices, const std::vector<int>& goldenInfo, int n, int lda,
    int batchSize, const std::string& caseName)
{
    size_t checked = 0;
    size_t singularSkipped = 0;
    bool passed = true;
    for (int b = 0; b < batchSize; b++) {
        if (!CgetrfShouldCheckLu(goldenInfo[b])) {
            ++singularSkipped;
            std::cout << "CGETRF_SINGULAR_LU_SKIPPED," << caseName << ',' << b << ',' << goldenInfo[b] << std::endl;
            continue;
        }
        ++checked;
        const size_t elementCount = static_cast<size_t>(n) * n;
        std::vector<float> npuReal(elementCount);
        std::vector<float> npuImag(elementCount);
        std::vector<float> goldenReal(elementCount);
        std::vector<float> goldenImag(elementCount);

        for (int j = 0; j < n; j++) {
            for (int i = 0; i < n; i++) {
                const size_t packedIdx = static_cast<size_t>(j) * n + i;
                const size_t matrixIdx = static_cast<size_t>(j) * lda + i;
                npuReal[packedIdx] = hostMatrices[b][matrixIdx].real;
                npuImag[packedIdx] = hostMatrices[b][matrixIdx].imag;
                goldenReal[packedIdx] = goldenMatrices[b][matrixIdx].real;
                goldenImag[packedIdx] = goldenMatrices[b][matrixIdx].imag;
            }
        }

        const std::string batchName = caseName + "_batch" + std::to_string(b);
        const auto real = CgetrfVerifyComponent(npuReal.data(), goldenReal.data(), elementCount, batchName + "_real");
        const auto imag = CgetrfVerifyComponent(npuImag.data(), goldenImag.data(), elementCount, batchName + "_imag");
        passed = passed && real.Passed() && imag.Passed();
        EXPECT_TRUE(real.Passed() && imag.Passed()) << "[" << caseName << "] batch=" << b << " L/U matrix mismatch";
    }
    std::cout << "CGETRF_PRECISION_COUNTS," << caseName << ',' << checked << ',' << singularSkipped << std::endl;
    return passed;
}

struct GetrfTestData {
    std::vector<std::vector<aclblasComplex>> hostMatrices;
    std::vector<std::vector<aclblasComplex>> originalMatrices;
    std::vector<aclblasComplex*> hostPtrs;
    std::vector<int> hostPivot;
    std::vector<int> hostInfo;
    size_t pivotSize;
};

static void PrepareTestData(
    GetrfTestData& data, const CgetrfBatchedParam& p, int n, int lda, int batchSize, bool usePivot)
{
    data.hostMatrices.resize(batchSize);
    data.hostPtrs.resize(batchSize);
    data.originalMatrices.resize(batchSize);

    for (int b = 0; b < batchSize; b++) {
        data.hostMatrices[b] = generateMatrix(n, lda, p.matrixType, p.randomSeed, b, batchSize);
        data.hostPtrs[b] = data.hostMatrices[b].data();
        data.originalMatrices[b] = data.hostMatrices[b];
    }

    data.pivotSize = usePivot ? static_cast<size_t>(n) * batchSize : 0;
    data.hostPivot.resize(data.pivotSize, 0);
    data.hostInfo.resize(batchSize, -1);
}

static void RunNpuAndGetGolden(
    GetrfTestData& data, const CgetrfBatchedParam& p, int n, int lda, int batchSize, bool usePivot,
    std::vector<std::vector<aclblasComplex>>& goldenMatrices, std::vector<int>& goldenPivot,
    std::vector<int>& goldenInfo, aclblasHandle_t handle)
{
    int* pivotPtr = usePivot ? data.hostPivot.data() : nullptr;
    int* infoPtr = data.hostInfo.data();

    aclblasStatus_t ret = aclblasCgetrfBatched_npu(handle, n, data.hostPtrs.data(), lda, pivotPtr, infoPtr, batchSize);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));

    goldenMatrices.resize(batchSize);
    std::vector<aclblasComplex*> goldenPtrs(batchSize);
    for (int b = 0; b < batchSize; b++) {
        goldenMatrices[b] = data.originalMatrices[b];
        goldenPtrs[b] = goldenMatrices[b].data();
    }

    goldenPivot.resize(data.pivotSize, 0);
    goldenInfo.resize(batchSize, -1);
    int* goldenPivotPtr = usePivot ? goldenPivot.data() : nullptr;

    aclblasCgetrfBatched_cpu(handle, n, goldenPtrs.data(), lda, goldenPivotPtr, goldenInfo.data(), batchSize);
}

static void VerifyTestData(GetrfTestData& data, const CgetrfBatchedParam& p, aclblasHandle_t handle)
{
    const bool usePivot = p.pivotMode == GetrfPivotMode::PIVOT;
    std::vector<std::vector<aclblasComplex>> goldenMatrices;
    std::vector<int> goldenPivot;
    std::vector<int> goldenInfo;
    RunNpuAndGetGolden(data, p, p.n, p.lda, p.batchSize, usePivot, goldenMatrices, goldenPivot, goldenInfo, handle);
    VerifyInfoArray(data.hostInfo, goldenInfo, p.batchSize, p.caseName);
    if (usePivot)
        VerifyPivotArray(data.hostPivot, goldenPivot, data.pivotSize, p.caseName);
    VerifyLUMatrices(data.hostMatrices, goldenMatrices, goldenInfo, p.n, p.lda, p.batchSize, p.caseName);
}

static void MeasurePerformance(GetrfTestData& data, const CgetrfBatchedParam& p, aclblasHandle_t handle)
{
    const int n = p.n;
    const int lda = p.lda;
    const int batchSize = p.batchSize;
    const bool usePivot = p.pivotMode == GetrfPivotMode::PIVOT;
    constexpr int warmups = 10;
    constexpr int repeats = 60;
    std::vector<double> sampleUs;
    aclblasStatus_t ret =
        BenchmarkCgetrfBatched(handle, n, data.hostPtrs.data(), lda, usePivot, batchSize, warmups, repeats, sampleUs);
    ASSERT_EQ(ret, ACLBLAS_STATUS_SUCCESS);
    ASSERT_EQ(sampleUs.size(), static_cast<size_t>(repeats));
    double totalUs = 0.0;
    for (size_t i = 0; i < sampleUs.size(); i++) {
        totalUs += sampleUs[i];
        std::cout << std::setprecision(12) << "CGETRF_SAMPLE," << p.caseName << ',' << i << ',' << sampleUs[i] << '\n';
    }
    std::cout << std::setprecision(12) << "CGETRF_PERFORMANCE," << p.caseName << ',' << sampleUs.size() << ','
              << totalUs / sampleUs.size() << ',' << static_cast<size_t>(lda) * n * batchSize * sizeof(aclblasComplex)
              << '\n';
}

static void RunPerformanceCase(
    GetrfTestData& data, const CgetrfBatchedParam& p, aclblasHandle_t handle, bool performanceOnly)
{
    const bool usePivot = p.pivotMode == GetrfPivotMode::PIVOT;
    if (!performanceOnly) {
        VerifyTestData(data, p, handle);
        ASSERT_FALSE(::testing::Test::HasFailure()) << "Accuracy gate failed; timing skipped";
        for (int b = 0; b < p.batchSize; ++b) {
            data.hostMatrices[b] = data.originalMatrices[b];
            data.hostPtrs[b] = data.hostMatrices[b].data();
        }
    } else {
        std::cout << "CGETRF_ACCURACY_SKIPPED," << p.caseName << std::endl;
    }
    int* pivotPtr = usePivot ? data.hostPivot.data() : nullptr;
    const char* benchmark = std::getenv("CGETRF_PERFORMANCE");
    if (benchmark && std::string(benchmark) == "1") {
        MeasurePerformance(data, p, handle);
    } else {
        aclblasStatus_t ret = aclblasCgetrfBatched_npu(
            handle, p.n, data.hostPtrs.data(), p.lda, pivotPtr, data.hostInfo.data(), p.batchSize);
        EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
    }
}

TEST_P(CgetrfBatchedTest, CsvDriven)
{
    const auto& p = GetParam();
    const int n = p.n;
    const int lda = p.lda;
    const int batchSize = p.batchSize;
    const bool usePivot = (p.pivotMode == GetrfPivotMode::PIVOT);
    const char* performanceOnlyEnv = std::getenv("CGETRF_PERFORMANCE_ONLY");
    const bool performanceOnly = performanceOnlyEnv && std::string(performanceOnlyEnv) == "1";
    if (performanceOnly && p.caseName.rfind("TC_PF", 0) != 0) {
        GTEST_SKIP() << "Accuracy tests disabled during performance-only work";
    }

    if (p.expectResult != ACLBLAS_STATUS_SUCCESS) {
        TestErrorPath(p, CgetrfBatchedTest::handle_);
        return;
    }

    if (n <= 0 || batchSize <= 0) {
        TestNoOpPath(p, CgetrfBatchedTest::handle_);
        return;
    }

    GetrfTestData data;
    PrepareTestData(data, p, n, lda, batchSize, usePivot);

    if (p.caseName.rfind("TC_PF", 0) == 0) {
        RunPerformanceCase(data, p, CgetrfBatchedTest::handle_, performanceOnly);
        return;
    }
    VerifyTestData(data, p, CgetrfBatchedTest::handle_);
}
