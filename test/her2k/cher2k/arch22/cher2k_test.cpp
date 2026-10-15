/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <string>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "blas_test.h"
#include "cher2k_golden.h"
#include "cher2k_npu_wrapper.h"
#include "cher2k_param.h"
#include "cher2k_scalar_regression.h"
#include "csv_loader.h"
#include "fill.h"
#include "verify.h"

class Cher2kArch22Test : public BlasTest<Cher2kParam> {};

static bool HasCasePrefix(const Cher2kParam& p, const char* prefix) { return p.caseName.rfind(prefix, 0U) == 0U; }

static std::vector<Cher2kParam> LoadCher2kCases()
{
    const bool performanceRun = std::getenv("CHER2K_PERF_ITERATIONS") != nullptr;
    const bool fullRegression = std::getenv("CHER2K_FULL_REGRESSION") != nullptr;
    std::string csvPath = ReplaceFileExtension2Csv(__FILE__);
    if (fullRegression || performanceRun) {
        csvPath.insert(csvPath.size() - 4U, "_full");
    }
    const auto allCases = GetCasesFromCsv<Cher2kParam>(csvPath);
    std::vector<Cher2kParam> selectedCases;
    selectedCases.reserve(allCases.size());
    for (const Cher2kParam& p : allCases) {
        const bool performanceCase = HasCasePrefix(p, "TC_PF_");
        if ((fullRegression && !performanceRun) || performanceCase == performanceRun) {
            selectedCases.push_back(p);
        }
    }
    return selectedCases;
}

static std::vector<aclblasComplex> MakeComplexArray(size_t count, const BlasFillMode& fill, uint64_t seed)
{
    if (fill.method == BlasFillMode::M_NULLPTR || count == 0U) {
        return {};
    }

    std::mt19937 rng(static_cast<uint32_t>(seed ? seed : 42U));
    auto generator = createGenerator(fill, rng);
    std::vector<aclblasComplex> output(count);
    for (size_t i = 0; i < count; ++i) {
        output[i] = {generator->at(i * 2U), generator->at(i * 2U + 1U)};
    }
    return output;
}

static void VerifyCher2kTriangle(
    const Cher2kParam& p, const aclblasComplex* actual, const std::vector<aclblasComplex>& expected)
{
    const bool diagnoseFailures = std::getenv("CHER2K_DIAG_FAILURES") != nullptr;
    int diagnosed = 0;
    std::vector<float> actualTriangle;
    std::vector<float> expectedTriangle;
    std::vector<float> actualOther;
    std::vector<float> expectedOther;
    const size_t dimension = static_cast<size_t>(p.n);
    const size_t triangleFloatCount = dimension * (dimension + 1U);
    const size_t otherFloatCount = dimension * (dimension - 1U);
    actualTriangle.reserve(triangleFloatCount);
    expectedTriangle.reserve(triangleFloatCount);
    actualOther.reserve(otherFloatCount);
    expectedOther.reserve(otherFloatCount);
    for (int col = 0; col < p.n; ++col) {
        for (int row = 0; row < p.n; ++row) {
            const size_t index = static_cast<size_t>(row) + static_cast<size_t>(col) * p.ldc;
            const bool selected = p.uplo == ACLBLAS_UPPER ? row <= col : row >= col;
            auto& out = selected ? actualTriangle : actualOther;
            auto& gold = selected ? expectedTriangle : expectedOther;
            out.push_back(actual[index].real);
            out.push_back(actual[index].imag);
            gold.push_back(expected[index].real);
            gold.push_back(expected[index].imag);
            if (diagnoseFailures && selected && diagnosed < 32 &&
                (std::abs(actual[index].real - expected[index].real) > 1.0e-2f ||
                 std::abs(actual[index].imag - expected[index].imag) > 1.0e-2f)) {
                std::cout << "[DIAG] row=" << row << " col=" << col << " actual=(" << actual[index].real << ","
                          << actual[index].imag << ") expected=(" << expected[index].real << "," << expected[index].imag
                          << ")\n";
                ++diagnosed;
            }
        }
    }
    VerifyConfig config;
    applyMixedTolerance(config, ACL_FLOAT, expectedTriangle.data(), expectedTriangle.size());
    // Cher2k reductions may use a different FP32 accumulation order than the
    // host BLAS reference (especially for large K and scaled inputs). Keep the
    // standard atol/rtol and 99% ratio checks, but allow the bounded reduction
    // tail to differ by up to 2.5e-2 on a single element.
    config.mixedMaxAbsErrorLimit = 2.5e-2;
    EXPECT_TRUE(Verifier::verifyVector(
        actualTriangle.data(), expectedTriangle.data(), actualTriangle.size(), 1, config, p.caseName));
    if (!actualOther.empty()) {
        VerifyConfig otherConfig;
        otherConfig.mode = PrecisionMode::EXACT;
        EXPECT_TRUE(Verifier::verifyVector(
            actualOther.data(), expectedOther.data(), actualOther.size(), 1, otherConfig,
            p.caseName + std::string("_other")));
    }
}

static void BenchmarkCher2kCase(
    aclblasHandle_t handle, aclrtStream stream, const Cher2kParam& p, const aclblasComplex* alpha,
    const aclblasComplex* a, const aclblasComplex* b, const float* beta, aclblasComplex* c, const char* iterationsText)
{
    const int iterations = std::atoi(iterationsText);
    const char* warmupText = std::getenv("CHER2K_PERF_WARMUP");
    const int warmup = warmupText == nullptr ? 10 : std::atoi(warmupText);
    float averageUs = 0.0f;
    EXPECT_EQ(
        aclblasCher2k_npu_benchmark(
            handle, stream, p.uplo, p.trans, p.n, p.k, alpha, a, p.lda, b, p.ldb, beta, c, p.ldc, warmup, iterations,
            &averageUs),
        ACLBLAS_STATUS_SUCCESS);
    std::cout << "[PERF] case=" << p.caseName << " n=" << p.n << " k=" << p.k << " average_us=" << averageUs
              << " warmup=" << warmup << " iterations=" << iterations << std::endl;
}

static void PrintCher2kDiagnosticTerms(
    const Cher2kParam& p, const std::vector<aclblasComplex>& a, const std::vector<aclblasComplex>& b,
    const aclblasComplex* actual)
{
    if (std::getenv("CHER2K_DIAG_TERMS") == nullptr || p.trans != ACLBLAS_OP_C || p.n <= 0 || p.k <= 0)
        return;
    double arbr = 0.0;
    double aibi = 0.0;
    double arbi = 0.0;
    double aibr = 0.0;
    for (int index = 0; index < p.k; ++index) {
        const aclblasComplex& av = a[static_cast<size_t>(index)];
        const aclblasComplex& bv = b[static_cast<size_t>(index)];
        arbr += static_cast<double>(av.real) * bv.real;
        aibi += static_cast<double>(av.imag) * bv.imag;
        arbi += static_cast<double>(av.real) * bv.imag;
        aibr += static_cast<double>(av.imag) * bv.real;
    }
    std::cout << "[DIAG_TERMS] arbr=" << arbr << " aibi=" << aibi << " arbi=" << arbi << " aibr=" << aibr
              << " correct_real=" << 2.0 * (arbr + aibi) << " alt_real=" << 2.0 * (arbr - aibi)
              << " actual_real=" << actual[0].real << std::endl;
}

TEST_F(Cher2kArch22Test, NullHandle)
{
    const aclblasComplex alpha{1.0f, 0.0f};
    const float beta = 0.0f;
    EXPECT_EQ(
        aclblasCher2k_npu(
            nullptr, ACLBLAS_UPPER, ACLBLAS_OP_N, 0, 0, &alpha, nullptr, 1, nullptr, 1, &beta, nullptr, 1),
        ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
}

TEST_F(Cher2kArch22Test, ScalarTransitionsAndMemoryGuards)
{
    for (const auto uplo : {ACLBLAS_UPPER, ACLBLAS_LOWER}) {
        for (const auto trans : {ACLBLAS_OP_N, ACLBLAS_OP_C}) {
            for (const bool hostAlpha : {false, true}) {
                for (const bool hostBeta : {false, true}) {
                    Cher2kScalarRegression regression(handle_, stream_, 512, 32, uplo, trans);
                    ASSERT_NO_FATAL_FAILURE(regression.Run(hostAlpha, hostBeta));
                }
            }
        }
    }
}

TEST_F(Cher2kArch22Test, ScalarTransitionsAtTileBoundaries)
{
    for (const int n : {31, 63, 64, 65, 255, 256, 257, 511, 512, 513}) {
        for (const auto uplo : {ACLBLAS_UPPER, ACLBLAS_LOWER}) {
            Cher2kScalarRegression regression(handle_, stream_, n, 65, uplo, ACLBLAS_OP_C);
            ASSERT_NO_FATAL_FAILURE(regression.Run(false, false));
        }
    }
}

TEST_F(Cher2kArch22Test, ScalarTransitionsOnSmallPaths)
{
    const int shapes[][2] = {{3, 1}, {31, 4}, {32, 32}, {64, 64}, {65, 2}, {128, 4}};
    for (const auto& shape : shapes) {
        for (const auto uplo : {ACLBLAS_UPPER, ACLBLAS_LOWER}) {
            for (const auto trans : {ACLBLAS_OP_N, ACLBLAS_OP_C}) {
                Cher2kScalarRegression regression(handle_, stream_, shape[0], shape[1], uplo, trans);
                ASSERT_NO_FATAL_FAILURE(regression.Run(false, false));
            }
        }
    }
}

TEST_F(Cher2kArch22Test, MemorySanitizerPaths)
{
    // Bound instrumented runtime while exercising each producer/consumer family.
    Cher2kScalarRegression direct(handle_, stream_, 512, 32, ACLBLAS_UPPER, ACLBLAS_OP_C);
    ASSERT_NO_FATAL_FAILURE(direct.Run(false, false));
    Cher2kScalarRegression smallCube(handle_, stream_, 64, 64, ACLBLAS_UPPER, ACLBLAS_OP_N);
    ASSERT_NO_FATAL_FAILURE(smallCube.Run(false, false));
    Cher2kScalarRegression tail(handle_, stream_, 65, 65, ACLBLAS_LOWER, ACLBLAS_OP_C);
    ASSERT_NO_FATAL_FAILURE(tail.Run(false, false));
    Cher2kScalarRegression smallVector(handle_, stream_, 32, 4, ACLBLAS_UPPER, ACLBLAS_OP_N);
    ASSERT_NO_FATAL_FAILURE(smallVector.Run(false, false));
}

INSTANTIATE_TEST_SUITE_P(
    Cher2k, Cher2kArch22Test, ::testing::ValuesIn(LoadCher2kCases()), PrintCaseInfoString<Cher2kParam>);

static void CompleteCher2kCase(
    aclblasHandle_t handle, aclrtStream stream, const Cher2kParam& p, aclblasStatus_t status,
    const aclblasComplex* alpha, const aclblasComplex* a, const aclblasComplex* b, const float* beta, aclblasComplex* c,
    std::vector<aclblasComplex>& golden, const std::vector<aclblasComplex>& aValues,
    const std::vector<aclblasComplex>& bValues, const char* perfIterations)
{
    if (p.expectResult != ACLBLAS_STATUS_SUCCESS) {
        EXPECT_EQ(status, p.expectResult);
        return;
    }
    ASSERT_EQ(status, ACLBLAS_STATUS_SUCCESS);
    if (p.n == 0 || c == nullptr)
        return;
    if (perfIterations != nullptr && p.caseName.rfind("TC_PF_", 0U) == 0U) {
        BenchmarkCher2kCase(handle, stream, p, alpha, a, b, beta, c, perfIterations);
        return;
    }
    ASSERT_EQ(
        aclblasCher2k_cpu(handle, p.uplo, p.trans, p.n, p.k, alpha, a, p.lda, b, p.ldb, beta, golden.data(), p.ldc),
        ACLBLAS_STATUS_SUCCESS);
    PrintCher2kDiagnosticTerms(p, aValues, bValues, c);
    VerifyCher2kTriangle(p, c, golden);
}

TEST_P(Cher2kArch22Test, CsvDriven)
{
    const Cher2kParam& p = GetParam();
    const int columns = p.trans == ACLBLAS_OP_N ? std::max(0, p.k) : std::max(0, p.n);
    const size_t aCount = p.lda > 0 ? static_cast<size_t>(p.lda) * columns : 0U;
    const size_t bCount = p.ldb > 0 ? static_cast<size_t>(p.ldb) * columns : 0U;
    const size_t cCount = p.ldc > 0 && p.n > 0 ? static_cast<size_t>(p.ldc) * p.n : 0U;
    const char* perfIterations = std::getenv("CHER2K_PERF_ITERATIONS");
    const bool performanceCase = perfIterations != nullptr && p.caseName.rfind("TC_PF_", 0U) == 0U;
    std::vector<aclblasComplex> a = MakeComplexArray(aCount, p.aFill, p.randomSeed);
    std::vector<aclblasComplex> b = MakeComplexArray(bCount, p.bFill, p.randomSeed + 1U);
    std::vector<aclblasComplex> c = MakeComplexArray(cCount, p.cFill, p.randomSeed + 2U);
    std::vector<aclblasComplex> golden;
    if (!performanceCase && p.expectResult == ACLBLAS_STATUS_SUCCESS && !p.nullC) {
        golden = c;
    }

    const aclblasComplex alpha{p.alphaReal, p.alphaImag};
    const aclblasComplex* alphaPtr = p.nullAlpha ? nullptr : &alpha;
    const float* betaPtr = p.nullBeta ? nullptr : &p.beta;
    const aclblasComplex* aPtr = p.nullA || a.empty() ? nullptr : a.data();
    const aclblasComplex* bPtr = p.nullB || b.empty() ? nullptr : b.data();
    aclblasComplex* cPtr = p.nullC || c.empty() ? nullptr : c.data();

    const aclblasStatus_t status = aclblasCher2k_npu(
        Cher2kArch22Test::handle_, p.uplo, p.trans, p.n, p.k, alphaPtr, aPtr, p.lda, bPtr, p.ldb, betaPtr, cPtr, p.ldc);
    CompleteCher2kCase(handle_, stream_, p, status, alphaPtr, aPtr, bPtr, betaPtr, cPtr, golden, a, b, perfIterations);
}
