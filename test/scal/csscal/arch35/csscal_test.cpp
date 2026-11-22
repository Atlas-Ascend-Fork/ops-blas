/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <algorithm>
#include <string>
#include <vector>
#include <cmath>
#include <cstdlib>

#include "fill.h"
#include "verify.h"
#include "blas_test.h"
#include "csv_loader.h"
#include "csscal_param.h"
#include "csscal_golden.h"
#include "csscal_npu_wrapper.h"

class CsscalTest : public BlasTest<CsscalParam> {};

TEST_F(CsscalTest, NullHandle)
{
    float alpha = 1.0f;
    aclblasComplex x[5] = {{1, 2}, {3, 4}, {5, 6}, {7, 8}, {9, 10}};
    aclblasStatus_t ret = aclblasCsscal_npu(nullptr, 5, &alpha, x, 1);
    EXPECT_EQ(ret, ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
}

TEST_F(CsscalTest, NullHandleTakesPrecedenceOverQuickReturn)
{
    EXPECT_EQ(aclblasCsscal(nullptr, 0, nullptr, nullptr, 1), ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
    EXPECT_EQ(aclblasCsscal(nullptr, 5, nullptr, nullptr, 0), ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
}

TEST_F(CsscalTest, QuickReturnSkipsDataPointerValidation)
{
    EXPECT_EQ(aclblasCsscal(CsscalTest::handle_, 0, nullptr, nullptr, 1), ACLBLAS_STATUS_SUCCESS);
    // n > 0 时 incx 口径先于指针校验（对齐 #1092 后的 host/README/CSV：incx<0 拒绝 NOT_SUPPORTED，
    // incx==0 返回 INVALID_VALUE，不再静默假成功）
    EXPECT_EQ(aclblasCsscal(CsscalTest::handle_, 5, nullptr, nullptr, 0), ACLBLAS_STATUS_INVALID_VALUE);
    EXPECT_EQ(aclblasCsscal(CsscalTest::handle_, 5, nullptr, nullptr, -1), ACLBLAS_STATUS_NOT_SUPPORTED);
}

// issue #1094: alpha 支持 host/device 双位置——device alpha 经同步 D2H 读取后
// 正常参与缩放，不再 host 侧裸解引用（段错误/垃圾值风险）
TEST_F(CsscalTest, DeviceAlphaScalesCorrectly)
{
    const int n = 5;
    const int incx = 1;
    const float alphaHost = 2.0f;
    std::vector<aclblasComplex> xHost = {{1, 2}, {3, 4}, {5, 6}, {7, 8}, {9, 10}};
    const size_t xBytes = xHost.size() * sizeof(aclblasComplex);

    void* dX = nullptr;
    void* dAlpha = nullptr;
    ASSERT_EQ(aclrtMalloc(&dX, xBytes, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    ASSERT_EQ(aclrtMalloc(&dAlpha, sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    ASSERT_EQ(aclrtMemcpy(dX, xBytes, xHost.data(), xBytes, ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);
    ASSERT_EQ(aclrtMemcpy(dAlpha, sizeof(float), &alphaHost, sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);

    aclblasStatus_t ret = aclblasCsscal(
        CsscalTest::handle_, n, static_cast<const float*>(dAlpha), static_cast<aclblasComplex*>(dX), incx);
    EXPECT_EQ(ret, ACLBLAS_STATUS_SUCCESS);
    ASSERT_EQ(aclrtSynchronizeDevice(), ACL_SUCCESS);

    std::vector<aclblasComplex> xOut(xHost.size());
    ASSERT_EQ(aclrtMemcpy(xOut.data(), xBytes, dX, xBytes, ACL_MEMCPY_DEVICE_TO_HOST), ACL_SUCCESS);
    for (size_t i = 0; i < xHost.size(); i++) {
        EXPECT_EQ(xOut[i].real, xHost[i].real * alphaHost);
        EXPECT_EQ(xOut[i].imag, xHost[i].imag * alphaHost);
    }

    aclrtFree(dX);
    aclrtFree(dAlpha);
}

// Helper: verify complex vector by checking real and imag parts separately
static bool verifyCsscalResult(
    const aclblasComplex* output, const aclblasComplex* golden, size_t count, int64_t stride, const VerifyConfig& cfg,
    const std::string& caseId)
{
    // Extract real parts (stride positions: 0, stride, 2*stride, ...)
    std::vector<float> outReal(count);
    std::vector<float> goldReal(count);
    std::vector<float> outImag(count);
    std::vector<float> goldImag(count);
    for (size_t i = 0; i < count; i++) {
        int64_t idx = static_cast<int64_t>(i) * stride;
        outReal[i] = output[idx].real;
        goldReal[i] = golden[idx].real;
        outImag[i] = output[idx].imag;
        goldImag[i] = golden[idx].imag;
    }

    bool passReal = Verifier::verifyVector(outReal.data(), goldReal.data(), count, 1, cfg, caseId + "_real");
    bool passImag = Verifier::verifyVector(outImag.data(), goldImag.data(), count, 1, cfg, caseId + "_imag");
    return passReal && passImag;
}

INSTANTIATE_TEST_SUITE_P(
    Csscal, CsscalTest, ::testing::ValuesIn(GetCasesFromCsv<CsscalParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<CsscalParam>);

TEST_P(CsscalTest, CsvDriven)
{
    const auto& p = GetParam();

    // Generate complex input data
    std::vector<aclblasComplex> xHost;
    if (p.incx == 1) {
        xHost = makeBlasComplexArray(p.n, p.x, p.randomSeed);
    } else {
        xHost = makeBlasComplexStrided(p.n, p.incx, p.x, p.randomSeed);
    }
    aclblasComplex* xPtr = xHost.empty() ? nullptr : xHost.data();
    float alpha = p.alpha;

    const char* sampleEnv = std::getenv("CSSCAL_PERF_SAMPLES");
    const int samples = p.caseName.find("TC_PF_") == 0 ? std::max(51, sampleEnv ? std::atoi(sampleEnv) : 60) : 1;
    const float* alphaPtr = p.nullAlpha ? nullptr : &alpha;
    aclblasStatus_t ret = aclblasCsscal_npu(CsscalTest::handle_, p.n, alphaPtr, xPtr, p.incx, samples);

    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
    if (p.expectResult != ACLBLAS_STATUS_SUCCESS)
        return;

    if (p.n <= 0)
        return;

    // Compute golden
    std::vector<aclblasComplex> goldenX;
    if (p.incx == 1) {
        goldenX = makeBlasComplexArray(p.n, p.x, p.randomSeed);
    } else {
        goldenX = makeBlasComplexStrided(p.n, p.incx, p.x, p.randomSeed);
    }
    aclblasCsscal_cpu(CsscalTest::handle_, p.n, alphaPtr, goldenX.data(), p.incx);

    VerifyConfig cfg;
    cfg.mode = PrecisionMode::MIXED_TOLERANCE;
    cfg.mixedAtol = p.mereThreshold;
    cfg.mixedRtol = p.mereThreshold;
    cfg.mixedMaxAbsErrorLimit = p.mereThreshold * 32.0;

    int absInc = std::abs(p.incx);
    EXPECT_TRUE(verifyCsscalResult(
        xPtr, goldenX.data(), static_cast<size_t>(p.n), static_cast<int64_t>(absInc), cfg, p.caseName));
}
