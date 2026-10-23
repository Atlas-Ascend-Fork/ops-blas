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
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "verify.h"
#include "fill.h"
#include "blas_test.h"
#include "csv_loader.h"
#include "strsmbatched_param.h"
#include "strsmbatched_golden.h"
#include "strsmbatched_npu_wrapper.h"

class StrsmbatchedTest : public BlasTest<StrsmbatchedParam> {};

TEST_F(StrsmbatchedTest, NullHandle)
{
    float alpha = 1.0f;
    aclblasStatus_t ret = aclblasStrsmBatched_npu(
        nullptr, ACLBLAS_SIDE_LEFT, ACLBLAS_UPPER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT, 5, 5, &alpha, nullptr, 5, nullptr, 5,
        1);
    EXPECT_EQ(ret, ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
}

TEST_F(StrsmbatchedTest, InvalidSide)
{
    float alpha = 1.0f;
    float aData = 1.0f;
    float bData = 1.0f;
    const float* aPtrs[1] = {&aData};
    float* bPtrs[1] = {&bData};
    aclblasSideMode_t invalidSide = static_cast<aclblasSideMode_t>(0xFF);
    aclblasStatus_t ret = aclblasStrsmBatched_npu(
        StrsmbatchedTest::handle_, invalidSide, ACLBLAS_UPPER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT, 5, 5, &alpha, aPtrs, 5,
        bPtrs, 5, 1);
    EXPECT_EQ(ret, ACLBLAS_STATUS_INVALID_VALUE);
}

TEST_F(StrsmbatchedTest, InvalidUplo)
{
    float alpha = 1.0f;
    float aData = 1.0f;
    float bData = 1.0f;
    const float* aPtrs[1] = {&aData};
    float* bPtrs[1] = {&bData};
    aclblasFillMode_t invalidUplo = static_cast<aclblasFillMode_t>(0xFF);
    aclblasStatus_t ret = aclblasStrsmBatched_npu(
        StrsmbatchedTest::handle_, ACLBLAS_SIDE_LEFT, invalidUplo, ACLBLAS_OP_N, ACLBLAS_NON_UNIT, 5, 5, &alpha, aPtrs,
        5, bPtrs, 5, 1);
    EXPECT_EQ(ret, ACLBLAS_STATUS_INVALID_VALUE);
}

TEST_F(StrsmbatchedTest, InvalidTrans)
{
    float alpha = 1.0f;
    float aData = 1.0f;
    float bData = 1.0f;
    const float* aPtrs[1] = {&aData};
    float* bPtrs[1] = {&bData};
    aclblasOperation_t invalidTrans = static_cast<aclblasOperation_t>(0xFF);
    aclblasStatus_t ret = aclblasStrsmBatched_npu(
        StrsmbatchedTest::handle_, ACLBLAS_SIDE_LEFT, ACLBLAS_UPPER, invalidTrans, ACLBLAS_NON_UNIT, 5, 5, &alpha,
        aPtrs, 5, bPtrs, 5, 1);
    EXPECT_EQ(ret, ACLBLAS_STATUS_INVALID_VALUE);
}

TEST_F(StrsmbatchedTest, InvalidDiag)
{
    float alpha = 1.0f;
    float aData = 1.0f;
    float bData = 1.0f;
    const float* aPtrs[1] = {&aData};
    float* bPtrs[1] = {&bData};
    aclblasDiagType_t invalidDiag = static_cast<aclblasDiagType_t>(0xFF);
    aclblasStatus_t ret = aclblasStrsmBatched_npu(
        StrsmbatchedTest::handle_, ACLBLAS_SIDE_LEFT, ACLBLAS_UPPER, ACLBLAS_OP_N, invalidDiag, 5, 5, &alpha, aPtrs, 5,
        bPtrs, 5, 1);
    EXPECT_EQ(ret, ACLBLAS_STATUS_INVALID_VALUE);
}

TEST_F(StrsmbatchedTest, NullAlpha)
{
    float bData = 1.0f;
    float* bPtrs[1] = {&bData};
    aclblasStatus_t ret = aclblasStrsmBatched_npu(
        StrsmbatchedTest::handle_, ACLBLAS_SIDE_LEFT, ACLBLAS_UPPER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT, 5, 5, nullptr,
        nullptr, 5, bPtrs, 5, 1);
    EXPECT_EQ(ret, ACLBLAS_STATUS_INVALID_VALUE);
}

TEST_F(StrsmbatchedTest, NullAWithNonzeroAlpha)
{
    float alpha = 1.0f;
    float bData = 1.0f;
    float* bPtrs[1] = {&bData};
    aclblasStatus_t ret = aclblasStrsmBatched_npu(
        StrsmbatchedTest::handle_, ACLBLAS_SIDE_LEFT, ACLBLAS_UPPER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT, 5, 5, &alpha,
        nullptr, 5, bPtrs, 5, 1);
    EXPECT_EQ(ret, ACLBLAS_STATUS_INVALID_VALUE);
}

TEST_F(StrsmbatchedTest, NullB)
{
    float alpha = 1.0f;
    aclblasStatus_t ret = aclblasStrsmBatched_npu(
        StrsmbatchedTest::handle_, ACLBLAS_SIDE_LEFT, ACLBLAS_UPPER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT, 5, 5, &alpha,
        nullptr, 5, nullptr, 5, 1);
    EXPECT_EQ(ret, ACLBLAS_STATUS_INVALID_VALUE);
}

static std::string GetStrsmbatchedCsvPath()
{
    std::string csvPath = ReplaceFileExtension2Csv(__FILE__);
    const char* fullTest = std::getenv("GSB_FULL_TEST");
    if (fullTest == nullptr || std::strcmp(fullTest, "1") != 0) {
        fullTest = std::getenv("STRSMBATCHED_FULL_TEST");
    }
    if (fullTest != nullptr && std::strcmp(fullTest, "1") == 0) {
        return csvPath;
    }
    csvPath.insert(csvPath.find_last_of('.'), "_smoke");
    return csvPath;
}

INSTANTIATE_TEST_SUITE_P(
    Strsmbatched, StrsmbatchedTest, ::testing::ValuesIn(GetCasesFromCsv<StrsmbatchedParam>(GetStrsmbatchedCsvPath())),
    PrintCaseInfoString<StrsmbatchedParam>);

static void ZeroOppositeTriangle(std::vector<float>& a, int aDim, int lda, bool isUpper)
{
    for (int j = 0; j < aDim; j++) {
        for (int i = 0; i < aDim; i++) {
            if (isUpper ? (i > j) : (i < j)) {
                a[static_cast<size_t>(i) + static_cast<size_t>(j) * static_cast<size_t>(lda)] = 0.0f;
            }
        }
    }
}

static void BoostDiagonal(std::vector<float>& a, int aDim, int lda)
{
    float boost = std::max(5.0f, static_cast<float>(aDim));
    for (int i = 0; i < aDim; i++) {
        float& diag = a[static_cast<size_t>(i) + static_cast<size_t>(i) * static_cast<size_t>(lda)];
        diag += (diag >= 0.0f ? boost : -boost);
    }
}

template <typename Func>
static void ParallelForBatches(int batchCount, int64_t totalWork, Func&& func)
{
    const int numThreads = (batchCount > 1 && totalWork >= 65536) ? std::min(batchCount, 16) : 1;
    if (numThreads <= 1) {
        func(0, batchCount);
        return;
    }
    std::vector<std::thread> workers;
    workers.reserve(numThreads);
    for (int t = 0; t < numThreads; ++t) {
        int bStart = (batchCount * t) / numThreads;
        int bEnd = (batchCount * (t + 1)) / numThreads;
        workers.emplace_back(func, bStart, bEnd);
    }
    for (auto& w : workers) {
        w.join();
    }
}

static void PrepareHostA(
    const StrsmbatchedParam& p, int aDim, int batchCount, std::vector<std::vector<float>>& aHost,
    std::vector<const float*>& aPtrs)
{
    aHost.resize(batchCount);
    aPtrs.assign(batchCount, nullptr);
    bool isUpper = (p.uplo == ACLBLAS_UPPER);
    BlasFillMode effectiveAFill = p.aFill;
    if (p.diag == ACLBLAS_UNIT && p.aFill.method == BlasFillMode::M_RANDOM && p.aFill.pattern == BlasFillMode::P_NORM) {
        effectiveAFill = BlasFillMode("RANDOM_NORM_P0.03_P0.03");
    }
    auto genBatch = [&](int bStart, int bEnd) {
        for (int b = bStart; b < bEnd; b++) {
            uint32_t aSeed = p.randomSeed + static_cast<uint32_t>(b) * 1000;
            aHost[b] = makeBlasMatrix(aDim, aDim, p.lda, effectiveAFill, aSeed);
            if (!aHost[b].empty() && aDim > 0 && p.lda >= aDim) {
                ZeroOppositeTriangle(aHost[b], aDim, p.lda, isUpper);
                if (p.diag == ACLBLAS_NON_UNIT && p.aFill.method != BlasFillMode::M_VALUE) {
                    BoostDiagonal(aHost[b], aDim, p.lda);
                }
            }
            aPtrs[b] = aHost[b].empty() ? nullptr : aHost[b].data();
        }
    };
    const int64_t totalWork = static_cast<int64_t>(batchCount) * std::max(1, aDim) * std::max(1, p.lda);
    ParallelForBatches(batchCount, totalWork, genBatch);
}

static void PrepareHostB(
    const StrsmbatchedParam& p, int batchCount, std::vector<std::vector<float>>& bHost,
    std::vector<std::vector<float>>& bOriginal, std::vector<float*>& bPtrs)
{
    bHost.resize(batchCount);
    bOriginal.resize(batchCount);
    bPtrs.assign(batchCount, nullptr);
    const int64_t bElems = static_cast<int64_t>(std::max(p.m, p.ldb)) * p.n;
    auto genBatch = [&](int bStart, int bEnd) {
        for (int b = bStart; b < bEnd; b++) {
            uint32_t bSeed = p.randomSeed + static_cast<uint32_t>(b) * 1000 + 500;
            bHost[b] = makeBlasArray(bElems, p.bFill, bSeed);
            bOriginal[b] = bHost[b];
            bPtrs[b] = bHost[b].empty() ? nullptr : bHost[b].data();
        }
    };
    const int64_t totalWork = static_cast<int64_t>(batchCount) * bElems;
    ParallelForBatches(batchCount, totalWork, genBatch);
}

static void VerifyResults(
    const StrsmbatchedParam& p, int batchCount, const std::vector<std::vector<float>>& bGolden,
    const std::vector<float*>& bPtrs)
{
    const size_t bCount = static_cast<size_t>(p.ldb) * static_cast<size_t>(p.n);
    if (bCount == 0 || p.m == 0 || p.n == 0) {
        return;
    }
    for (int b = 0; b < batchCount; b++) {
        VerifyConfig cfg;
        applyMixedTolerance(cfg, ACL_FLOAT, bGolden[b].data(), bCount);
        bool pass = Verifier::verifyVector(
            bPtrs[b], bGolden[b].data(), bCount, 1, cfg, p.caseName + "_batch" + std::to_string(b));
        EXPECT_TRUE(pass) << "[" << p.caseName << "] batch=" << b << " precision mismatch";
    }
}

static bool RunNegativeDimTest(const StrsmbatchedParam& p, aclblasHandle_t callHandle, const float* callAlpha)
{
    if (p.m >= 0 && p.n >= 0 && p.batchCount > 0) {
        return false;
    }
    float dummyA = 1.0f;
    float dummyB = 1.0f;
    const float* aDummyPtrs[1] = {&dummyA};
    float* bDummyPtrs[1] = {&dummyB};
    aclblasStatus_t ret = aclblasStrsmBatched_npu(
        callHandle, p.side, p.uplo, p.trans, p.diag, p.m, p.n, callAlpha,
        (p.aFill.method == BlasFillMode::M_NULLPTR) ? nullptr : aDummyPtrs, p.lda,
        (p.bFill.method == BlasFillMode::M_NULLPTR) ? nullptr : bDummyPtrs, p.ldb, p.batchCount);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
    return true;
}

static void CompareWithCpuGolden(
    aclblasHandle_t handle, const StrsmbatchedParam& p, int batchCount, const std::vector<const float*>& aPtrs,
    const std::vector<std::vector<float>>& bOriginal, const std::vector<float*>& bPtrs)
{
    if (p.m == 0 || p.n == 0 || batchCount == 0) {
        return;
    }
    const char* skipGolden = std::getenv("STRSMBATCHED_SKIP_CPU_GOLDEN");
    if (skipGolden != nullptr && std::string(skipGolden) == "1") {
        return;
    }

    std::vector<std::vector<float>> bGolden(batchCount);
    std::vector<float*> goldenPtrs(batchCount, nullptr);
    for (int b = 0; b < batchCount; b++) {
        bGolden[b] = bOriginal[b];
        goldenPtrs[b] = bGolden[b].empty() ? nullptr : bGolden[b].data();
    }

    aclblasStrsmBatched_cpu(
        handle, p.side, p.uplo, p.trans, p.diag, p.m, p.n, &p.alpha,
        (p.alpha == 0.0f) ? nullptr : const_cast<const float**>(aPtrs.data()), p.lda, goldenPtrs.data(), p.ldb,
        batchCount);

    VerifyResults(p, batchCount, bGolden, bPtrs);
}

TEST_P(StrsmbatchedTest, CsvDriven)
{
    const auto& p = GetParam();

    // 13-arg in-place aclblasStrsmBatched: matrix B is the output matrix (corresponding to C in 15-arg GEMM)

    aclblasHandle_t callHandle = (p.description == "handle_null") ? nullptr : handle_;
    const float* callAlpha = (p.alphaIsNull || p.description == "null_alpha") ? nullptr : &p.alpha;

    if (RunNegativeDimTest(p, callHandle, callAlpha)) {
        return;
    }

    const int aDim = (p.side == ACLBLAS_SIDE_LEFT) ? p.m : p.n;
    const int batchCount = p.batchCount;

    std::vector<std::vector<float>> aHost;
    std::vector<const float*> aPtrs;
    PrepareHostA(p, aDim, batchCount, aHost, aPtrs);
    if (p.description == "null_a_elem" && !aPtrs.empty()) {
        aPtrs[0] = nullptr;
    }

    std::vector<std::vector<float>> bHost;
    std::vector<std::vector<float>> bOriginal;
    std::vector<float*> bPtrs;
    PrepareHostB(p, batchCount, bHost, bOriginal, bPtrs);
    if ((p.description == "null_b_elem" || p.description == "null_c_elem") && !bPtrs.empty()) {
        bPtrs[0] = nullptr;
    }

    auto callAPtrs = (p.aFill.method == BlasFillMode::M_NULLPTR || p.description == "null_A") ? nullptr : aPtrs.data();
    auto callBPtrs =
        (p.bFill.method == BlasFillMode::M_NULLPTR || p.description == "null_B" || p.description == "null_carray") ?
            nullptr :
            bPtrs.data();
    int callLdb = (p.description == "invalid_ldc") ? (p.m - 1) : p.ldb;

    aclblasStatus_t ret = aclblasStrsmBatched_npu(
        callHandle, p.side, p.uplo, p.trans, p.diag, p.m, p.n, callAlpha, callAPtrs, p.lda, callBPtrs, callLdb,
        batchCount);

    if (p.expectResult != ACLBLAS_STATUS_SUCCESS) {
        EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
        return;
    }
    ASSERT_EQ(ret, ACLBLAS_STATUS_SUCCESS);

    CompareWithCpuGolden(handle_, p, batchCount, aPtrs, bOriginal, bPtrs);
}
