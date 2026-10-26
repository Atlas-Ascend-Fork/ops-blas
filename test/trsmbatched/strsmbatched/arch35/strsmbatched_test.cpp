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
#include <string>
#include <vector>

#include "verify.h"
#include "fill.h"
#include "blas_test.h"
#include "csv_loader.h"
#include "strsmbatched_param.h"
#include "strsmbatched_golden.h"
#include "strsmbatched_npu_wrapper.h"

class StrsmbatchedArch35Test : public BlasTest<StrsmbatchedParam> {};

// E01: Null handle test (TEST_F, not in CSV)
TEST_F(StrsmbatchedArch35Test, NullHandle)
{
    float alpha = 1.0f;
    aclblasStatus_t ret = aclblasStrsmBatched_npu(
        nullptr, ACLBLAS_SIDE_LEFT, ACLBLAS_UPPER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT,
        5, 5, &alpha, nullptr, 5, nullptr, 5, 1);
    EXPECT_EQ(ret, ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
}

// E02: Invalid side enum (TEST_F, not in CSV)
TEST_F(StrsmbatchedArch35Test, InvalidSide)
{
    float alpha = 1.0f;
    float aData = 1.0f;
    float bData = 1.0f;
    const float* aPtrs[1] = {&aData};
    float* bPtrs[1] = {&bData};
    aclblasSideMode_t invalidSide = static_cast<aclblasSideMode_t>(0xFF);
    aclblasStatus_t ret = aclblasStrsmBatched_npu(
        StrsmbatchedArch35Test::handle_, invalidSide, ACLBLAS_UPPER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT,
        5, 5, &alpha, aPtrs, 5, bPtrs, 5, 1);
    EXPECT_EQ(ret, ACLBLAS_STATUS_INVALID_VALUE);
}

// E03: Invalid uplo enum (TEST_F, not in CSV)
TEST_F(StrsmbatchedArch35Test, InvalidUplo)
{
    float alpha = 1.0f;
    float aData = 1.0f;
    float bData = 1.0f;
    const float* aPtrs[1] = {&aData};
    float* bPtrs[1] = {&bData};
    aclblasFillMode_t invalidUplo = static_cast<aclblasFillMode_t>(0xFF);
    aclblasStatus_t ret = aclblasStrsmBatched_npu(
        StrsmbatchedArch35Test::handle_, ACLBLAS_SIDE_LEFT, invalidUplo, ACLBLAS_OP_N, ACLBLAS_NON_UNIT,
        5, 5, &alpha, aPtrs, 5, bPtrs, 5, 1);
    EXPECT_EQ(ret, ACLBLAS_STATUS_INVALID_VALUE);
}

// E04: Invalid trans enum (TEST_F, not in CSV)
TEST_F(StrsmbatchedArch35Test, InvalidTrans)
{
    float alpha = 1.0f;
    float aData = 1.0f;
    float bData = 1.0f;
    const float* aPtrs[1] = {&aData};
    float* bPtrs[1] = {&bData};
    aclblasOperation_t invalidTrans = static_cast<aclblasOperation_t>(0xFF);
    aclblasStatus_t ret = aclblasStrsmBatched_npu(
        StrsmbatchedArch35Test::handle_, ACLBLAS_SIDE_LEFT, ACLBLAS_UPPER, invalidTrans, ACLBLAS_NON_UNIT,
        5, 5, &alpha, aPtrs, 5, bPtrs, 5, 1);
    EXPECT_EQ(ret, ACLBLAS_STATUS_INVALID_VALUE);
}

// E05: Invalid diag enum (TEST_F, not in CSV)
TEST_F(StrsmbatchedArch35Test, InvalidDiag)
{
    float alpha = 1.0f;
    float aData = 1.0f;
    float bData = 1.0f;
    const float* aPtrs[1] = {&aData};
    float* bPtrs[1] = {&bData};
    aclblasDiagType_t invalidDiag = static_cast<aclblasDiagType_t>(0xFF);
    aclblasStatus_t ret = aclblasStrsmBatched_npu(
        StrsmbatchedArch35Test::handle_, ACLBLAS_SIDE_LEFT, ACLBLAS_UPPER, ACLBLAS_OP_N, invalidDiag,
        5, 5, &alpha, aPtrs, 5, bPtrs, 5, 1);
    EXPECT_EQ(ret, ACLBLAS_STATUS_INVALID_VALUE);
}

// E08: Null alpha (TEST_F, not in CSV)
TEST_F(StrsmbatchedArch35Test, NullAlpha)
{
    float bData = 1.0f;
    float* bPtrs[1] = {&bData};
    aclblasStatus_t ret = aclblasStrsmBatched_npu(
        StrsmbatchedArch35Test::handle_, ACLBLAS_SIDE_LEFT, ACLBLAS_UPPER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT,
        5, 5, nullptr, nullptr, 5, bPtrs, 5, 1);
    EXPECT_EQ(ret, ACLBLAS_STATUS_INVALID_VALUE);
}

// E10: Null A with alpha != 0 (TEST_F, not in CSV)
TEST_F(StrsmbatchedArch35Test, NullAWithNonzeroAlpha)
{
    float alpha = 1.0f;
    float bData = 1.0f;
    float* bPtrs[1] = {&bData};
    aclblasStatus_t ret = aclblasStrsmBatched_npu(
        StrsmbatchedArch35Test::handle_, ACLBLAS_SIDE_LEFT, ACLBLAS_UPPER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT,
        5, 5, &alpha, nullptr, 5, bPtrs, 5, 1);
    EXPECT_EQ(ret, ACLBLAS_STATUS_INVALID_VALUE);
}

// E11: Null B (TEST_F, not in CSV)
TEST_F(StrsmbatchedArch35Test, NullB)
{
    float alpha = 1.0f;
    aclblasStatus_t ret = aclblasStrsmBatched_npu(
        StrsmbatchedArch35Test::handle_, ACLBLAS_SIDE_LEFT, ACLBLAS_UPPER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT,
        5, 5, &alpha, nullptr, 5, nullptr, 5, 1);
    EXPECT_EQ(ret, ACLBLAS_STATUS_INVALID_VALUE);
}

// E12: batchCount == 0 with null alpha — scalar validation (alpha non-null)
// runs before the batchCount == 0 no-op gate, so this still reports
// INVALID_VALUE (L1_41/L1_42/L1_43/L1_44/L1_45 lock the CSV side).
TEST_F(StrsmbatchedArch35Test, BatchZeroNullAlpha)
{
    aclblasStatus_t ret = aclblasStrsmBatched_npu(
        StrsmbatchedArch35Test::handle_, ACLBLAS_SIDE_LEFT, ACLBLAS_UPPER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT,
        5, 5, nullptr, nullptr, 5, nullptr, 5, 0);
    EXPECT_EQ(ret, ACLBLAS_STATUS_INVALID_VALUE);
}

INSTANTIATE_TEST_SUITE_P(
    Strsmbatched, StrsmbatchedArch35Test,
    ::testing::ValuesIn(GetCasesFromCsv<StrsmbatchedParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<StrsmbatchedParam>);

static void ZeroOppositeTriangle(std::vector<float>& a, int aDim, int lda, bool isUpper)
{
    for (int j = 0; j < aDim; j++) {
        for (int i = 0; i < aDim; i++) {
            if (isUpper ? (i > j) : (i < j)) {
                a[i + j * lda] = 0.0f;
            }
        }
    }
}

static void BoostDiagonal(std::vector<float>& a, int aDim, int lda)
{
    float boost = std::max(5.0f, static_cast<float>(aDim));
    for (int i = 0; i < aDim; i++) {
        float& diag = a[i + i * lda];
        diag += (diag >= 0.0f ? boost : -boost);
    }
}

static void PrepareHostA(const StrsmbatchedParam& p, int aDim, int batchCount,
    std::vector<std::vector<float>>& aHost, std::vector<const float*>& aPtrs)
{
    aHost.resize(batchCount);
    aPtrs.assign(batchCount, nullptr);
    bool isUpper = (p.uplo == ACLBLAS_UPPER);
    for (int b = 0; b < batchCount; b++) {
        uint32_t aSeed = p.randomSeed + static_cast<uint32_t>(b) * 1000;
        aHost[b] = makeBlasMatrix(aDim, aDim, p.lda, p.aFill, aSeed);
        if (!aHost[b].empty() && aDim > 0) {
            ZeroOppositeTriangle(aHost[b], aDim, p.lda, isUpper);
            if (p.diag == ACLBLAS_NON_UNIT) {
                BoostDiagonal(aHost[b], aDim, p.lda);
            }
        }
        aPtrs[b] = aHost[b].empty() ? nullptr : aHost[b].data();
    }
}

static void PrepareHostB(const StrsmbatchedParam& p, int batchCount,
    std::vector<std::vector<float>>& bHost, std::vector<std::vector<float>>& bOriginal,
    std::vector<float*>& bPtrs)
{
    bHost.resize(batchCount);
    bOriginal.resize(batchCount);
    bPtrs.assign(batchCount, nullptr);
    for (int b = 0; b < batchCount; b++) {
        uint32_t bSeed = p.randomSeed + static_cast<uint32_t>(b) * 1000 + 500;
        bHost[b] = makeBlasArray(static_cast<int64_t>(std::max(p.m, p.ldb)) * p.n, p.bFill, bSeed);
        bOriginal[b] = bHost[b];
        bPtrs[b] = bHost[b].empty() ? nullptr : bHost[b].data();
    }
}

static void VerifyResults(const StrsmbatchedParam& p, int batchCount,
    const std::vector<std::vector<float>>& bGolden, const std::vector<float*>& bPtrs)
{
    const size_t bCount = static_cast<size_t>(p.ldb) * static_cast<size_t>(p.n);
    if (bCount == 0) return;
    for (int b = 0; b < batchCount; b++) {
        VerifyConfig cfg;
        applyMixedTolerance(cfg, ACL_FLOAT, bGolden[b].data(), bCount);
        bool pass = Verifier::verifyVector(
            bPtrs[b], bGolden[b].data(), bCount, 1, cfg,
            p.caseName + "_batch" + std::to_string(b));
        EXPECT_TRUE(pass) << "[" << p.caseName << "] batch=" << b << " precision mismatch";
    }
}

// m/n < 0 negative path: size validation rejects the call before touching the
// A/B pointer arrays, so null arrays are fine (batchCount pinned to 1).
static void RunInvalidSizeCase(const StrsmbatchedParam& p, aclblasHandle_t handle)
{
    float alpha = p.alpha;
    aclblasStatus_t ret = aclblasStrsmBatched_npu(
        handle, p.side, p.uplo, p.trans, p.diag,
        p.m, p.n, &alpha, nullptr, p.lda, nullptr, p.ldb, 1);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
}

// batchCount <= 0 path: NPU and golden both run so their statuses can be
// compared. cuBLAS batched-family convention: batchCount == 0 is a legal no-op
// that returns success without touching the A/B pointer arrays (null arrays
// are acceptable); batchCount < 0 returns INVALID_VALUE. The golden mirrors
// the NPU validation order (all scalar checks incl. lda/ldb run before the
// no-op gate), so its status is comparable against the NPU result.
static void RunBatchCountNonPositiveCase(const StrsmbatchedParam& p, aclblasHandle_t handle)
{
    float alpha = p.alpha;
    aclblasStatus_t ret = aclblasStrsmBatched_npu(
        handle, p.side, p.uplo, p.trans, p.diag,
        p.m, p.n, &alpha, nullptr, p.lda, nullptr, p.ldb, p.batchCount);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
    aclblasStatus_t goldenRet = aclblasStrsmBatched_cpu(
        handle, p.side, p.uplo, p.trans, p.diag,
        p.m, p.n, &alpha, nullptr, p.lda, nullptr, p.ldb, p.batchCount);
    EXPECT_EQ(static_cast<int>(goldenRet), static_cast<int>(ret))
        << "[" << p.caseName << "] golden status diverges from NPU status";
}

TEST_P(StrsmbatchedArch35Test, CsvDriven)
{
    const auto& p = GetParam();

    if (p.m < 0 || p.n < 0) {
        RunInvalidSizeCase(p, StrsmbatchedArch35Test::handle_);
        return;
    }

    if (p.batchCount <= 0) {
        RunBatchCountNonPositiveCase(p, StrsmbatchedArch35Test::handle_);
        return;
    }

    const int aDim = (p.side == ACLBLAS_SIDE_LEFT) ? p.m : p.n;
    const int batchCount = std::max(1, p.batchCount);

    std::vector<std::vector<float>> aHost;
    std::vector<const float*> aPtrs;
    PrepareHostA(p, aDim, batchCount, aHost, aPtrs);

    std::vector<std::vector<float>> bHost;
    std::vector<std::vector<float>> bOriginal;
    std::vector<float*> bPtrs;
    PrepareHostB(p, batchCount, bHost, bOriginal, bPtrs);

    aclblasStatus_t ret = aclblasStrsmBatched_npu(
        StrsmbatchedArch35Test::handle_, p.side, p.uplo, p.trans, p.diag,
        p.m, p.n, &p.alpha,
        (p.alpha == 0.0f) ? nullptr : aPtrs.data(), p.lda,
        bPtrs.data(), p.ldb, batchCount);

    if (p.expectResult != ACLBLAS_STATUS_SUCCESS) {
        EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
        return;
    }
    ASSERT_EQ(ret, ACLBLAS_STATUS_SUCCESS);

    std::vector<std::vector<float>> bGolden(batchCount);
    std::vector<float*> goldenPtrs(batchCount, nullptr);
    for (int b = 0; b < batchCount; b++) {
        bGolden[b] = bOriginal[b];
        goldenPtrs[b] = bGolden[b].empty() ? nullptr : bGolden[b].data();
    }

    aclblasStrsmBatched_cpu(
        StrsmbatchedArch35Test::handle_, p.side, p.uplo, p.trans, p.diag,
        p.m, p.n, &p.alpha,
        (p.alpha == 0.0f) ? nullptr : const_cast<const float**>(aPtrs.data()), p.lda,
        goldenPtrs.data(), p.ldb, batchCount);

    VerifyResults(p, batchCount, bGolden, bPtrs);
}
