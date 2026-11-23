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
#include <vector>

#include "verify.h"
#include "blas_test.h"
#include "csv_loader.h"
#include "fill.h"
#include "sdgmm_param.h"
#include "sdgmm_golden.h"
#include "sdgmm_npu_wrapper.h"

class SdgmmTest : public BlasTest<SdgmmParam> {};

TEST_F(SdgmmTest, NullHandle)
{
    aclblasStatus_t ret = aclblasSdgmm_npu(nullptr, ACLBLAS_SIDE_LEFT, 4, 4, nullptr, 4, nullptr, 1, nullptr, 4);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_HANDLE_IS_NULLPTR));
}

static bool SdgmmInPlaceSetup(
    void** dX, void** dAC, const std::vector<float>& xHost, const std::vector<float>& aHost, size_t xBytes,
    size_t acBytes, size_t aCopyBytes)
{
    aclError ret = aclrtMalloc(dX, xBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    EXPECT_EQ(ret, ACL_SUCCESS);
    if (ret != ACL_SUCCESS) {
        return false;
    }
    ret = aclrtMalloc(dAC, acBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    EXPECT_EQ(ret, ACL_SUCCESS);
    if (ret != ACL_SUCCESS) {
        aclrtFree(*dX);
        return false;
    }
    ret = aclrtMemcpy(*dX, xBytes, xHost.data(), xBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    EXPECT_EQ(ret, ACL_SUCCESS);
    if (ret != ACL_SUCCESS) {
        aclrtFree(*dX);
        aclrtFree(*dAC);
        return false;
    }
    ret = aclrtMemcpy(*dAC, acBytes, aHost.data(), aCopyBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    EXPECT_EQ(ret, ACL_SUCCESS);
    if (ret != ACL_SUCCESS) {
        aclrtFree(*dX);
        aclrtFree(*dAC);
        return false;
    }
    return true;
}

TEST_F(SdgmmTest, InPlaceLdaNeLdc)
{
    constexpr int m = 4;
    constexpr int n = 4;
    constexpr int lda = 4;
    constexpr int ldc = 5;
    constexpr int incx = 1;
    const size_t acElems = static_cast<size_t>(std::max(lda, ldc)) * static_cast<size_t>(n);
    const size_t acBytes = acElems * sizeof(float);
    const size_t xBytes = static_cast<size_t>(m) * sizeof(float);
    const size_t aBytes = static_cast<size_t>(lda) * static_cast<size_t>(n) * sizeof(float);

    std::vector<float> xHost = makeBlasStrided(m, incx, "RANDOM_NORM_5_5", 42);
    std::vector<float> aHost = makeBlasMatrix(m, n, lda, "RANDOM_NORM_5_5", 42);
    void* dX = nullptr;
    void* dAC = nullptr;
    if (!SdgmmInPlaceSetup(&dX, &dAC, xHost, aHost, xBytes, acBytes, aBytes)) {
        return;
    }
    aclblasStatus_t dgmmRet = aclblasSdgmm(
        SdgmmTest::handle_, ACLBLAS_SIDE_LEFT, m, n, static_cast<const float*>(dAC), lda, static_cast<const float*>(dX),
        incx, static_cast<float*>(dAC), ldc);
    EXPECT_EQ(static_cast<int>(dgmmRet), static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE));
    aclrtFree(dX);
    aclrtFree(dAC);
}

TEST_F(SdgmmTest, InPlaceLdaEqLdc)
{
    constexpr int m = 4;
    constexpr int n = 4;
    constexpr int lda = 4;
    constexpr int ldc = 4;
    constexpr int incx = 1;
    const size_t acElems = static_cast<size_t>(lda) * static_cast<size_t>(n);
    const size_t acBytes = acElems * sizeof(float);
    const size_t xBytes = static_cast<size_t>(m) * sizeof(float);

    std::vector<float> xHost = makeBlasStrided(m, incx, "RANDOM_NORM_5_5", 42);
    std::vector<float> aHost = makeBlasMatrix(m, n, lda, "RANDOM_NORM_5_5", 42);
    void* dX = nullptr;
    void* dAC = nullptr;
    if (!SdgmmInPlaceSetup(&dX, &dAC, xHost, aHost, xBytes, acBytes, acBytes)) {
        return;
    }
    aclblasStatus_t dgmmRet = aclblasSdgmm(
        SdgmmTest::handle_, ACLBLAS_SIDE_LEFT, m, n, static_cast<const float*>(dAC), lda, static_cast<const float*>(dX),
        incx, static_cast<float*>(dAC), ldc);
    EXPECT_EQ(static_cast<int>(dgmmRet), static_cast<int>(ACLBLAS_STATUS_SUCCESS));
    if (dgmmRet != ACLBLAS_STATUS_SUCCESS) {
        aclrtFree(dX);
        aclrtFree(dAC);
        return;
    }
    EXPECT_EQ(aclrtSynchronizeStream(SdgmmTest::stream_), ACL_SUCCESS);
    std::vector<float> cHost(acElems, kBlasSentinel);
    EXPECT_EQ(aclrtMemcpy(cHost.data(), acBytes, dAC, acBytes, ACL_MEMCPY_DEVICE_TO_HOST), ACL_SUCCESS);
    std::vector<float> goldenC(acElems, kBlasSentinel);
    EXPECT_EQ(
        static_cast<int>(aclblasSdgmm_cpu(
            SdgmmTest::handle_, ACLBLAS_SIDE_LEFT, m, n, aHost.data(), lda, xHost.data(), incx, goldenC.data(), ldc)),
        static_cast<int>(ACLBLAS_STATUS_SUCCESS));
    VerifyConfig cfg;
    applyMixedTolerance(cfg, ACL_FLOAT, goldenC.data(), goldenC.size());
    EXPECT_TRUE(Verifier::verifyVector(cHost.data(), goldenC.data(), cHost.size(), 1, cfg, "InPlaceLdaEqLdc"));
    aclrtFree(dX);
    aclrtFree(dAC);
}

static void SdgmmIncxZeroRejected(aclblasHandle_t handle, aclblasSideMode_t mode)
{
    constexpr int m = 4;
    constexpr int n = 4;
    constexpr int lda = 4;
    constexpr int ldc = 4;
    const size_t acElems = static_cast<size_t>(lda) * static_cast<size_t>(n);
    const size_t acBytes = acElems * sizeof(float);
    const size_t xBytes = static_cast<size_t>(m) * sizeof(float);

    std::vector<float> xHost = makeBlasStrided(m, 1, "RANDOM_NORM_5_5", 42);
    std::vector<float> aHost = makeBlasMatrix(m, n, lda, "RANDOM_NORM_5_5", 42);
    void* dX = nullptr;
    void* dAC = nullptr;
    if (!SdgmmInPlaceSetup(&dX, &dAC, xHost, aHost, xBytes, acBytes, acBytes)) {
        return;
    }
    aclblasStatus_t dgmmRet = aclblasSdgmm(
        handle, mode, m, n, static_cast<const float*>(dAC), lda, static_cast<const float*>(dX), 0,
        static_cast<float*>(dAC), ldc);
    EXPECT_EQ(static_cast<int>(dgmmRet), static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE));
    aclrtFree(dX);
    aclrtFree(dAC);
}

TEST_F(SdgmmTest, IncxZeroRejectedLeft) { SdgmmIncxZeroRejected(handle_, ACLBLAS_SIDE_LEFT); }

TEST_F(SdgmmTest, IncxZeroRejectedRight) { SdgmmIncxZeroRejected(handle_, ACLBLAS_SIDE_RIGHT); }

INSTANTIATE_TEST_SUITE_P(
    Sdgmm, SdgmmTest, ::testing::ValuesIn(GetCasesFromCsv<SdgmmParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<SdgmmParam>);

TEST_P(SdgmmTest, CsvDriven)
{
    const auto& p = GetParam();
    const int xLen = (p.mode == ACLBLAS_SIDE_LEFT) ? p.m : p.n;
    std::vector<float> xHost;
    if (p.nullx == 0 && p.m > 0 && p.n > 0) {
        if (p.incx == INT_MIN) {
            xHost.assign(1, 1.0f);
        } else if (p.xFillRaw.rfind("MIXED", 0) == 0) {
            xHost = makeBlasMixed(xLen, p.incx, p.xFillRaw, p.randomSeed);
        } else {
            xHost = makeBlasStrided(xLen, p.incx, p.xFill, p.randomSeed);
        }
    }
    std::vector<float> aHost;
    if (p.nullA == 0 && p.m > 0 && p.n > 0) {
        aHost = makeBlasMatrix(p.m, p.n, p.lda, p.aFill, p.randomSeed);
    }
    std::vector<float> cHost;
    if (p.nullC == 0 && p.m > 0 && p.n > 0) {
        cHost.assign(static_cast<size_t>(p.ldc) * static_cast<size_t>(p.n), kBlasSentinel);
    }

    const float* xPtr = xHost.empty() ? nullptr : xHost.data();
    const float* aPtr = aHost.empty() ? nullptr : aHost.data();
    float* cPtr = cHost.empty() ? nullptr : cHost.data();
    aclblasStatus_t ret =
        aclblasSdgmm_npu(SdgmmTest::handle_, p.mode, p.m, p.n, aPtr, p.lda, xPtr, p.incx, cPtr, p.ldc);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
    if (p.expectResult != ACLBLAS_STATUS_SUCCESS || p.m == 0 || p.n == 0) {
        return;
    }

    std::vector<float> goldenC(cHost.size(), kBlasSentinel);
    EXPECT_EQ(
        static_cast<int>(
            aclblasSdgmm_cpu(SdgmmTest::handle_, p.mode, p.m, p.n, aPtr, p.lda, xPtr, p.incx, goldenC.data(), p.ldc)),
        static_cast<int>(ACLBLAS_STATUS_SUCCESS));
    VerifyConfig cfg;
    applyMixedTolerance(cfg, ACL_FLOAT, goldenC.data(), cHost.size());
    EXPECT_TRUE(Verifier::verifyVector(cPtr, goldenC.data(), cHost.size(), 1, cfg, p.caseName));
}
