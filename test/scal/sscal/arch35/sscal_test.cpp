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

#include "fill.h"
#include "verify.h"
#include "blas_test.h"
#include "csv_loader.h"
#include "sscal_param.h"
#include "sscal_golden.h"
#include "sscal_npu_wrapper.h"

class SscalArch35Test : public BlasTest<SscalParam> {};

TEST_F(SscalArch35Test, NullHandle)
{
    float alpha = 1.0f;
    float x[5] = {1, 2, 3, 4, 5};
    aclblasStatus_t ret = aclblasSscal_npu(nullptr, 5, &alpha, x, 1);
    EXPECT_EQ(ret, ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
}

TEST_F(SscalArch35Test, NullHandleTakesPrecedenceOverQuickReturn)
{
    EXPECT_EQ(aclblasSscal(nullptr, 0, nullptr, nullptr, 1), ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
    EXPECT_EQ(aclblasSscal(nullptr, 5, nullptr, nullptr, 0), ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
}

TEST_F(SscalArch35Test, QuickReturnSkipsDataPointerValidation)
{
    EXPECT_EQ(aclblasSscal(SscalArch35Test::handle_, 0, nullptr, nullptr, 1), ACLBLAS_STATUS_SUCCESS);
    EXPECT_EQ(aclblasSscal(SscalArch35Test::handle_, 5, nullptr, nullptr, 0), ACLBLAS_STATUS_SUCCESS);
    EXPECT_EQ(aclblasSscal(SscalArch35Test::handle_, 5, nullptr, nullptr, -1), ACLBLAS_STATUS_SUCCESS);
}

// issue #1111: alpha 支持 host/device 双位置——device alpha 经同步 D2H 读取后
// 正常参与缩放，不再 host 侧裸解引用
TEST_F(SscalArch35Test, DeviceAlphaScalesCorrectly)
{
    const int n = 5;
    const int incx = 1;
    const float alphaHost = 2.0f;
    std::vector<float> xHost = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    const size_t xBytes = xHost.size() * sizeof(float);

    void* dX = nullptr;
    void* dAlpha = nullptr;
    ASSERT_EQ(aclrtMalloc(&dX, xBytes, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    ASSERT_EQ(aclrtMalloc(&dAlpha, sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    ASSERT_EQ(aclrtMemcpy(dX, xBytes, xHost.data(), xBytes, ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);
    ASSERT_EQ(aclrtMemcpy(dAlpha, sizeof(float), &alphaHost, sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);

    aclblasStatus_t ret =
        aclblasSscal(SscalArch35Test::handle_, n, static_cast<const float*>(dAlpha), static_cast<float*>(dX), incx);
    EXPECT_EQ(ret, ACLBLAS_STATUS_SUCCESS);
    ASSERT_EQ(aclrtSynchronizeDevice(), ACL_SUCCESS);

    std::vector<float> xOut(xHost.size());
    ASSERT_EQ(aclrtMemcpy(xOut.data(), xBytes, dX, xBytes, ACL_MEMCPY_DEVICE_TO_HOST), ACL_SUCCESS);
    for (size_t i = 0; i < xHost.size(); i++) {
        EXPECT_EQ(xOut[i], xHost[i] * alphaHost);
    }

    aclrtFree(dX);
    aclrtFree(dAlpha);
}

INSTANTIATE_TEST_SUITE_P(
    Sscal, SscalArch35Test, ::testing::ValuesIn(GetCasesFromCsv<SscalParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<SscalParam>);

TEST_P(SscalArch35Test, CsvDriven)
{
    const auto& p = GetParam();

    std::vector<float> xHost =
        (p.incx == 1) ? makeBlasArray(p.n, p.x, p.randomSeed) : makeBlasStrided(p.n, p.incx, p.x, p.randomSeed);
    float* xPtr = xHost.empty() ? nullptr : xHost.data();
    float alpha = p.alpha;

    aclblasStatus_t ret = aclblasSscal_npu(SscalArch35Test::handle_, p.n, &alpha, xPtr, p.incx);

    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
    if (p.expectResult != ACLBLAS_STATUS_SUCCESS)
        return;

    if (p.n <= 0)
        return;

    std::vector<float> goldenX =
        (p.incx == 1) ? makeBlasArray(p.n, p.x, p.randomSeed) : makeBlasStrided(p.n, p.incx, p.x, p.randomSeed);
    aclblasSscal_cpu(SscalArch35Test::handle_, p.n, &alpha, goldenX.data(), p.incx);

    VerifyConfig cfg;
    applyMixedTolerance(cfg, ACL_FLOAT, goldenX.data(), static_cast<size_t>(p.n));

    int absInc = std::abs(p.incx);
    EXPECT_TRUE(Verifier::verifyVector(
        xPtr, goldenX.data(), static_cast<size_t>(p.n), static_cast<int64_t>(absInc), cfg, p.caseName));
}
