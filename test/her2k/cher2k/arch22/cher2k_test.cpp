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
#include "csv_case_contract.h"
#include "fill.h"
#include "triangle_preservation.h"
#include "triangle_fp64_reference.h"

class Cher2kTest : public BlasTest<Cher2kParam> {};

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
    VerifyCsvComposition(allCases, 1238U, 200U, csvPath);
    std::vector<Cher2kParam> selectedCases;
    selectedCases.reserve(allCases.size());
    for (const Cher2kParam& p : allCases) {
        const bool performanceCase = HasCasePrefix(p, "TC_PF_");
        if ((fullRegression && !performanceRun) || performanceCase == performanceRun) {
            selectedCases.push_back(p);
        }
    }
    const size_t expectedTotal = performanceRun ? 200U : (fullRegression ? 1238U : 1038U);
    VerifyCsvComposition(
        selectedCases, expectedTotal, (performanceRun || fullRegression) ? 200U : 0U, "CHER2K selection");
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
    if (std::getenv("CHER2K_DIAG_TERMS") == nullptr || p.trans != ACLBLAS_OP_C || p.n <= 0 || p.k <= 0) {
        return;
    }
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

TEST_F(Cher2kTest, NullHandle)
{
    const aclblasComplex alpha{1.0f, 0.0f};
    const float beta = 0.0f;
    EXPECT_EQ(
        aclblasCher2k_npu(
            nullptr, ACLBLAS_UPPER, ACLBLAS_OP_N, 0, 0, &alpha, nullptr, 1, nullptr, 1, &beta, nullptr, 1),
        ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
}

TEST_F(Cher2kTest, Fp64CancellationReference)
{
    const aclblasComplex a[] = {{100000000.0F, 0.0F}, {1.0F, 0.0F},          {1.0F, 0.0F},
                                {1.0F, 0.0F},         {-100000000.0F, 0.0F}, {1.0F, 0.0F}};
    const aclblasComplex b[6] = {{0.5F, 0.0F}, {0.5F, 0.0F}, {0.5F, 0.0F}, {0.5F, 0.0F}, {0.5F, 0.0F}, {0.5F, 0.0F}};
    const aclblasComplex original[4] = {};
    const aclblasComplex alpha{1.0F, 0.0F};
    const float beta = 0.0F;
    Cher2kGoldenDbl golden;
    ASSERT_EQ(
        Cher2kRunGoldenDbl(handle_, ACLBLAS_UPPER, ACLBLAS_OP_N, 2, 3, &alpha, a, 2, b, 2, &beta, original, 2, golden),
        ACLBLAS_STATUS_SUCCESS);
    EXPECT_DOUBLE_EQ(golden.c[2].real, 2.0);
    EXPECT_DOUBLE_EQ(golden.c[2].imag, 0.0);
}

TEST_F(Cher2kTest, ScalarTransitionsAndMemoryGuards)
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

TEST_F(Cher2kTest, ScalarTransitionsAtTileBoundaries)
{
    for (const int n : {31, 63, 64, 65, 255, 256, 257, 511, 512, 513}) {
        for (const auto uplo : {ACLBLAS_UPPER, ACLBLAS_LOWER}) {
            Cher2kScalarRegression regression(handle_, stream_, n, 65, uplo, ACLBLAS_OP_C);
            ASSERT_NO_FATAL_FAILURE(regression.Run(false, false));
        }
    }
}

TEST_F(Cher2kTest, ScalarTransitionsOnSmallPaths)
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

TEST_F(Cher2kTest, MemorySanitizerPaths)
{
    // Bound instrumented runtime while exercising each producer/consumer family.
    Cher2kScalarRegression direct(handle_, stream_, 512, 32, ACLBLAS_UPPER, ACLBLAS_OP_C);
    ASSERT_NO_FATAL_FAILURE(direct.Run(false, false));
    Cher2kScalarRegression smallCube(handle_, stream_, 64, 64, ACLBLAS_UPPER, ACLBLAS_OP_N);
    ASSERT_NO_FATAL_FAILURE(smallCube.Run(false, false));
    Cher2kScalarRegression tail(handle_, stream_, 65, 65, ACLBLAS_LOWER, ACLBLAS_OP_C);
    ASSERT_NO_FATAL_FAILURE(tail.Run(false, false));
    for (int n : {255, 257}) {
        for (auto uplo : {ACLBLAS_UPPER, ACLBLAS_LOWER}) {
            Cher2kScalarRegression boundary(handle_, stream_, n, 65, uplo, ACLBLAS_OP_C);
            ASSERT_NO_FATAL_FAILURE(boundary.Run(false, false));
        }
    }
    Cher2kScalarRegression smallVector(handle_, stream_, 32, 4, ACLBLAS_UPPER, ACLBLAS_OP_N);
    ASSERT_NO_FATAL_FAILURE(smallVector.Run(false, false));
}

INSTANTIATE_TEST_SUITE_P(Cher2k, Cher2kTest, ::testing::ValuesIn(LoadCher2kCases()), PrintCaseInfoString<Cher2kParam>);

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
    if (p.n == 0 || c == nullptr) {
        return;
    }
    if (perfIterations != nullptr && p.caseName.rfind("TC_PF_", 0U) == 0U) {
        BenchmarkCher2kCase(handle, stream, p, alpha, a, b, beta, c, perfIterations);
        return;
    }
    Cher2kGoldenDbl wide;
    ASSERT_EQ(
        Cher2kRunGoldenDbl(
            handle, p.uplo, p.trans, p.n, p.k, alpha, a, p.lda, b, p.ldb, beta, golden.data(), p.ldc, wide),
        ACLBLAS_STATUS_SUCCESS);
    PrintCher2kDiagnosticTerms(p, aValues, bValues, c);
    VerifyTriangleFp64(
        p.caseName.c_str(), c, golden.data(), wide.c.data(), p.n, p.ldc, p.uplo == ACLBLAS_UPPER,
        ((p.alphaReal == 0.0F && p.alphaImag == 0.0F) || p.k == 0) && p.beta == 1.0F);
}

static void RunCher2kAccuracyCase(aclblasHandle_t handle, aclrtStream stream, const Cher2kParam& p)
{
    const int columns = p.trans == ACLBLAS_OP_N ? std::max(0, p.k) : std::max(0, p.n);
    const size_t aCount = p.lda > 0 ? static_cast<size_t>(p.lda) * columns : 0U;
    const size_t bCount = p.ldb > 0 ? static_cast<size_t>(p.ldb) * columns : 0U;
    const size_t cCount = p.ldc > 0 && p.n > 0 ? static_cast<size_t>(p.ldc) * p.n : 0U;
    const char* perfIterations = std::getenv("CHER2K_PERF_ITERATIONS");
    const bool performanceCase = perfIterations != nullptr && p.caseName.rfind("TC_PF_", 0U) == 0U;
    std::vector<aclblasComplex> a = MakeComplexArray(aCount, p.aFill, p.randomSeed);
    std::vector<aclblasComplex> b = MakeComplexArray(bCount, p.bFill, p.randomSeed + 1U);
    std::vector<aclblasComplex> c = MakeComplexArray(cCount, p.cFill, p.randomSeed + 2U);
    PoisonUntouched(c.data(), p.n, p.ldc, p.uplo == ACLBLAS_UPPER);
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

    const aclblasStatus_t status =
        aclblasCher2k_npu(handle, p.uplo, p.trans, p.n, p.k, alphaPtr, aPtr, p.lda, bPtr, p.ldb, betaPtr, cPtr, p.ldc);
    CompleteCher2kCase(handle, stream, p, status, alphaPtr, aPtr, bPtr, betaPtr, cPtr, golden, a, b, perfIterations);
}

TEST_P(Cher2kTest, CsvDriven) { RunCher2kAccuracyCase(handle_, stream_, GetParam()); }

TEST_F(Cher2kTest, ExtremeSmallPathScalarsAndPadding)
{
    const csv_map row{
        {"case_name", "ExtremeSmallPath"},
        {"a_fill", "RANDOM_EXTREME"},
        {"b_fill", "RANDOM_NORM_5_5"},
        {"random_seed", "20260166"}};
    for (const int n : {31, 32}) {
        Cher2kParam p(row);
        p.n = n;
        p.k = 32;
        p.lda = n + 4;
        p.ldb = n + 8;
        p.ldc = (n + 3) / 4 * 4;
        p.bFill = BlasFillMode(n == 31 ? "RANDOM_NORM_5_5" : "RANDOM_EXTREME");
        ASSERT_NO_FATAL_FAILURE(RunCher2kAccuracyCase(handle_, stream_, p));
        p.alphaReal = 0.5F;
        p.alphaImag = 0.25F;
        p.beta = -0.25F;
        ASSERT_NO_FATAL_FAILURE(RunCher2kAccuracyCase(handle_, stream_, p));
        p.alphaReal = 0.0F;
        p.alphaImag = 0.0F;
        p.beta = 0.0F;
        ASSERT_NO_FATAL_FAILURE(RunCher2kAccuracyCase(handle_, stream_, p));
    }
}
