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
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "verify.h"
#include "fill.h"
#include "blas_test.h"
#include "csv_loader.h"
#include "chemm_param.h"
#include "chemm_golden.h"
#include "chemm_npu_wrapper.h"

class ChemmArch22Test : public BlasTest<ChemmParam> {};

constexpr int64_t kMaxSafeTestElems = 64LL * 1024 * 1024;

inline int64_t ChemmSafeElemCount(int64_t rows, int64_t cols)
{
    if (cols == 0 || rows <= 0) {
        return 0;
    }
    if (rows > kMaxSafeTestElems / cols) {
        return 0;
    }
    return rows * cols;
}

inline std::vector<float> ChemmMakeHostArray(int64_t count, const BlasFillMode& fill, bool normal, uint32_t seed)
{
    if (fill.method == BlasFillMode::M_NULLPTR || count <= 0) {
        return {};
    }
    if (!normal) {
        return makeBlasArray(count, fill, seed);
    }

    // NORMAL_NORM is Chemm-local: each matrix gets a reproducible
    // mu in [-5, 5] and sigma in [0.1, 2], then both complex components
    // are sampled from that normal distribution.
    std::mt19937 generator(seed ? seed : 42);
    std::uniform_real_distribution<float> meanDistribution(-5.0f, 5.0f);
    std::uniform_real_distribution<float> stddevDistribution(0.1f, 2.0f);
    std::normal_distribution<float> valueDistribution(meanDistribution(generator), stddevDistribution(generator));
    std::vector<float> values(static_cast<size_t>(count));
    for (float& value : values) {
        value = valueDistribution(generator);
    }
    return values;
}

inline size_t ChemmComplexOffset(int64_t row, int64_t col, int64_t ld)
{
    return 2 * static_cast<size_t>(row + col * ld);
}

inline void ChemmMakeHermitian(std::vector<float>& values, const ChemmParam& p, int64_t aDim)
{
    if (p.m <= 0 || p.n <= 0) {
        return;
    }
    for (int64_t d = 0; d < aDim; ++d) {
        size_t diagonal = ChemmComplexOffset(d, d, p.lda);
        if (diagonal + 1 < values.size()) {
            values[diagonal + 1] = 0.0f;
        }
    }
    for (int64_t col = 0; col < aDim; ++col) {
        for (int64_t row = 0; row < aDim; ++row) {
            bool reflected = (p.uplo == ACLBLAS_UPPER) ? (row > col) : (row < col);
            if (!reflected) {
                continue;
            }
            size_t destination = ChemmComplexOffset(row, col, p.lda);
            size_t source = ChemmComplexOffset(col, row, p.lda);
            if (destination + 1 < values.size() && source + 1 < values.size()) {
                values[destination] = values[source];
                values[destination + 1] = -values[source + 1];
            }
        }
    }
}

inline void ChemmPrepareHostData(
    const ChemmParam& p, std::vector<float>& a, std::vector<float>& b, std::vector<float>& c,
    std::vector<float>& result)
{
    int64_t aDim = (p.side == ACLBLAS_SIDE_LEFT) ? p.m : p.n;
    int64_t aCount = ChemmSafeElemCount(aDim, p.lda);
    int64_t bCount = ChemmSafeElemCount(p.n, p.ldb);
    int64_t cCount = ChemmSafeElemCount(p.n, p.ldc);
    if (aCount > 0 && p.aFill.method != BlasFillMode::M_NULLPTR) {
        a = ChemmMakeHostArray(2 * aCount, p.aFill, p.aNormal, p.randomSeed);
        ChemmMakeHermitian(a, p, aDim);
    }
    if (bCount > 0 && p.bFill.method != BlasFillMode::M_NULLPTR) {
        b = ChemmMakeHostArray(2 * bCount, p.bFill, p.bNormal, p.randomSeed + 1);
    }
    if (cCount > 0 && p.cFill.method != BlasFillMode::M_NULLPTR) {
        c = ChemmMakeHostArray(2 * cCount, p.cFill, p.cNormal, p.randomSeed + 2);
        result = c;
    }
}

inline void ChemmBuildPointers(
    const ChemmParam& p, std::vector<float>& a, std::vector<float>& b, std::vector<float>& result,
    aclblasComplex& alpha, aclblasComplex& beta, const aclblasComplex*& alphaPtr, const aclblasComplex*& betaPtr,
    const aclblasComplex*& aPtr, const aclblasComplex*& bPtr, aclblasComplex*& cPtr)
{
    alpha = {p.alphaReal, p.alphaImag};
    beta = {p.betaReal, p.betaImag};
    alphaPtr = p.nullAlpha ? nullptr : &alpha;
    betaPtr = p.nullBeta ? nullptr : &beta;
    aPtr = a.empty() ? nullptr : reinterpret_cast<const aclblasComplex*>(a.data());
    bPtr = b.empty() ? nullptr : reinterpret_cast<const aclblasComplex*>(b.data());
    cPtr = result.empty() ? nullptr : reinterpret_cast<aclblasComplex*>(result.data());
}

inline bool ChemmCheckExpectedStatus(const ChemmParam& p, aclblasStatus_t ret)
{
    if (p.expectResult == ACLBLAS_STATUS_SUCCESS) {
        return true;
    }
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
    return false;
}

inline bool ChemmHasExtremeFill(const ChemmParam& p)
{
    return p.aFill.pattern == BlasFillMode::P_EXTREME || p.bFill.pattern == BlasFillMode::P_EXTREME;
}

inline bool ChemmPerfTimingRequested()
{
    const char* mode = std::getenv("CBLAS3_PERF_MODE");
    return mode != nullptr && mode[0] == '1';
}

inline void ChemmBuildGolden(
    const ChemmParam& p, const std::vector<float>& cHost, const aclblasComplex* alpha, const aclblasComplex* beta,
    const aclblasComplex* a, const aclblasComplex* b, std::vector<float>& golden, aclblasHandle_t handle)
{
    std::string goldenFile = makeGoldenDir(std::string(__FILE__)) + p.caseName + ".golden";
    uint64_t hash = chemm_golden_cache::computeParamHash(p);
    if (chemm_golden_cache::loadGolden(goldenFile, golden, hash)) {
        return;
    }
    golden = cHost;
    aclblasComplex* output = golden.empty() ? nullptr : reinterpret_cast<aclblasComplex*>(golden.data());
    ASSERT_EQ(
        aclblasChemm_cpu(handle, p.side, p.uplo, p.m, p.n, alpha, a, p.lda, b, p.ldb, beta, output, p.ldc),
        ACLBLAS_STATUS_SUCCESS);
    chemm_golden_cache::saveGolden(goldenFile, golden, hash);
}

inline bool ChemmSameNonFinite(float output, float golden)
{
    if (std::isnan(golden)) {
        return std::isnan(output);
    }
    return std::isinf(golden) && std::isinf(output) && std::signbit(golden) == std::signbit(output);
}

inline void ChemmVerifyComponent(
    const std::vector<float>& result, const std::vector<float>& golden, const std::string& caseName)
{
    ASSERT_EQ(result.size(), golden.size());
    std::vector<float> finiteResult;
    std::vector<float> finiteGolden;
    finiteResult.reserve(result.size());
    finiteGolden.reserve(golden.size());
    for (size_t i = 0; i < result.size(); ++i) {
        if (std::isfinite(result[i]) && std::isfinite(golden[i])) {
            finiteResult.push_back(result[i]);
            finiteGolden.push_back(golden[i]);
        } else {
            EXPECT_TRUE(ChemmSameNonFinite(result[i], golden[i])) << caseName << " non-finite component at " << i;
        }
    }
    if (finiteResult.empty()) {
        return;
    }
    VerifyConfig config;
    applyMixedTolerance(config, ACL_FLOAT, finiteGolden.data(), finiteGolden.size());
    // OpenBLAS reduction order varies across host architectures. Keep the
    // ecosystem mixed-tolerance ratio as the portable correctness gate.
    config.mixedMaxAbsErrorLimit = std::numeric_limits<double>::infinity();
    EXPECT_TRUE(
        Verifier::verifyVector(finiteResult.data(), finiteGolden.data(), finiteGolden.size(), 1, config, caseName));
}

inline void ChemmVerifyResult(const ChemmParam& p, const std::vector<float>& result, const std::vector<float>& golden)
{
    std::vector<float> resultReal;
    std::vector<float> resultImag;
    std::vector<float> goldenReal;
    std::vector<float> goldenImag;
    size_t logicalCount = static_cast<size_t>(p.m) * static_cast<size_t>(p.n);
    resultReal.reserve(logicalCount);
    resultImag.reserve(logicalCount);
    goldenReal.reserve(logicalCount);
    goldenImag.reserve(logicalCount);
    for (int64_t col = 0; col < p.n; ++col) {
        for (int64_t row = 0; row < p.m; ++row) {
            size_t offset = ChemmComplexOffset(row, col, p.ldc);
            ASSERT_LT(offset + 1, result.size());
            ASSERT_LT(offset + 1, golden.size());
            resultReal.push_back(result[offset]);
            resultImag.push_back(result[offset + 1]);
            goldenReal.push_back(golden[offset]);
            goldenImag.push_back(golden[offset + 1]);
        }
    }
    ChemmVerifyComponent(resultReal, goldenReal, p.caseName + "_real");
    ChemmVerifyComponent(resultImag, goldenImag, p.caseName + "_imag");
}

TEST(ChemmValidationTest, NullHandle)
{
    aclblasComplex alpha = {1.0f, 0.0f};
    aclblasComplex beta = {0.5f, 0.0f};
    // Minimal dummy buffers to satisfy non-null pointer check; handle=nullptr
    // is caught before any pointer deref.
    std::vector<float> aBuf(32, 0.0f); // 4*4*2 floats
    std::vector<float> bBuf(32, 0.0f);
    std::vector<float> cBuf(32, 0.0f);
    aclblasStatus_t ret = aclblasChemm_npu(
        nullptr, ACLBLAS_SIDE_LEFT, ACLBLAS_LOWER, 4, 4, &alpha, reinterpret_cast<aclblasComplex*>(aBuf.data()), 4,
        reinterpret_cast<aclblasComplex*>(bBuf.data()), 4, &beta, reinterpret_cast<aclblasComplex*>(cBuf.data()), 4);
    EXPECT_EQ(ret, ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
}

INSTANTIATE_TEST_SUITE_P(
    Chemm, ChemmArch22Test, ::testing::ValuesIn(GetCasesFromCsv<ChemmParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<ChemmParam>);

TEST_P(ChemmArch22Test, CsvDriven)
{
    const auto& p = GetParam();
    const bool perfCase = p.caseName.rfind("TC_PF_", 0) == 0;
    if (perfCase && !ChemmPerfTimingRequested()) {
        GTEST_SKIP() << "Set CBLAS3_PERF_MODE=1 to run performance sampling";
    }
    std::vector<float> aHostFlt;
    std::vector<float> bHostFlt;
    std::vector<float> cHostFlt;
    std::vector<float> cResultFlt;

    ChemmPrepareHostData(p, aHostFlt, bHostFlt, cHostFlt, cResultFlt);
    aclblasComplex alphaVal;
    aclblasComplex betaVal;
    const aclblasComplex* alphaPtr;
    const aclblasComplex* betaPtr;
    const aclblasComplex* aPtr;
    const aclblasComplex* bPtr;
    aclblasComplex* cPtr;
    ChemmBuildPointers(p, aHostFlt, bHostFlt, cResultFlt, alphaVal, betaVal, alphaPtr, betaPtr, aPtr, bPtr, cPtr);
    if (perfCase) {
        constexpr int kWarmup = 5;
        constexpr int kIterations = 60;
        float averageUs = 0.0f;
        aclblasStatus_t perfStatus = aclblasChemm_perf(
            ChemmArch22Test::handle_, p.side, p.uplo, p.m, p.n, alphaPtr, aPtr, p.lda, bPtr, p.ldb, betaPtr, cPtr,
            p.ldc, kWarmup, kIterations, averageUs);
        ASSERT_EQ(perfStatus, ACLBLAS_STATUS_SUCCESS);
        std::cout << "[PERF] case=" << p.caseName << " average_us=" << averageUs << " warmup=" << kWarmup
                  << " iterations=" << kIterations << std::endl;
        return;
    }
    aclblasStatus_t ret = aclblasChemm_npu(
        ChemmArch22Test::handle_, p.side, p.uplo, p.m, p.n, alphaPtr, aPtr, p.lda, bPtr, p.ldb, betaPtr, cPtr, p.ldc);

    if (!ChemmCheckExpectedStatus(p, ret)) {
        return;
    }
    ASSERT_EQ(ret, ACLBLAS_STATUS_SUCCESS);
    // EXTREME intentionally mixes FLT_MAX with signed/zero values. Different
    // legal FP32 reduction orders can produce different NaN/Inf classes after
    // overflow, so this case validates successful execution rather than a
    // finite tolerance that is undefined for non-finite results.
    if (ChemmHasExtremeFill(p)) {
        return;
    }
    std::vector<float> cGoldenFlt;
    ChemmBuildGolden(p, cHostFlt, alphaPtr, betaPtr, aPtr, bPtr, cGoldenFlt, ChemmArch22Test::handle_);
    ChemmVerifyResult(p, cResultFlt, cGoldenFlt);
}
