/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>

#include "verify.h"
#include "blas_test.h"
#include "csv_loader.h"
#include "fill.h"
#include "srotm_param.h"
#include "srotm_golden.h"
#include "srotm_npu_wrapper.h"

class SrotmArch22Test : public BlasTest<SrotmParam> {};

TEST_F(SrotmArch22Test, DeviceParam)
{
    // issue #869: sparam 位于 device 内存时不崩溃，结果与 host-param 等价输入一致
    const int n = 64;
    auto x = makeBlasStrided(n, 1, "RANDOM_NORM_2_2", 11);
    auto y = makeBlasStrided(n, 1, "RANDOM_NORM_2_2", 12);
    std::array<float, 5> sparam = {-1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

    std::vector<float> hostX = x;
    std::vector<float> hostY = y;
    ASSERT_EQ(
        aclblasSrotm_npu(SrotmArch22Test::handle_, n, hostX.data(), 1, hostY.data(), 1, sparam.data()),
        ACLBLAS_STATUS_SUCCESS);

    void* paramDev = nullptr;
    ASSERT_EQ(aclrtMalloc(&paramDev, sizeof(sparam), ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    ASSERT_EQ(
        aclrtMemcpy(paramDev, sizeof(sparam), sparam.data(), sizeof(sparam), ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);

    std::vector<float> devX = x;
    std::vector<float> devY = y;
    aclblasStatus_t ret = aclblasSrotm_npu(
        SrotmArch22Test::handle_, n, devX.data(), 1, devY.data(), 1, static_cast<const float*>(paramDev));
    aclrtFree(paramDev);
    ASSERT_EQ(ret, ACLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < n; ++i) {
        ASSERT_EQ(devX[i], hostX[i]) << "device-param x mismatch at " << i;
        ASSERT_EQ(devY[i], hostY[i]) << "device-param y mismatch at " << i;
    }
}

INSTANTIATE_TEST_SUITE_P(
    Srotm, SrotmArch22Test, ::testing::ValuesIn(GetCasesFromCsv<SrotmParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<SrotmParam>);

TEST_P(SrotmArch22Test, CsvDriven)
{
    const auto& p = GetParam();

    auto x = makeBlasStrided(p.n, p.incx, "RANDOM_NORM_2_2", p.randomSeed);
    auto y = makeBlasStrided(p.n, p.incy, "RANDOM_NORM_2_2", p.randomSeed);
    std::vector<float> resultX = x;
    std::vector<float> resultY = y;
    std::vector<float> goldenX = x;
    std::vector<float> goldenY = y;

    std::array<float, 5> sparam = p.sparam;

    aclblasStatus_t ret =
        aclblasSrotm_npu(SrotmArch22Test::handle_, p.n, resultX.data(), p.incx, resultY.data(), p.incy, sparam.data());

    if (p.expectResult != ACLBLAS_STATUS_SUCCESS) {
        EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
        return;
    }
    ASSERT_EQ(ret, ACLBLAS_STATUS_SUCCESS);

    aclblasSrotm_cpu(SrotmArch22Test::handle_, p.n, goldenX.data(), p.incx, goldenY.data(), p.incy, sparam.data());

    VerifyConfig cfg;
    cfg.mode = PrecisionMode::MERE_MARE;
    cfg.mereThreshold = p.mereThreshold;
    cfg.mareMultiplier = p.mareMultiplier;

    EXPECT_TRUE(
        Verifier::verifyVector(resultX.data(), goldenX.data(), static_cast<size_t>(x.size()), 1, cfg, p.caseName));
    EXPECT_TRUE(
        Verifier::verifyVector(resultY.data(), goldenY.data(), static_cast<size_t>(y.size()), 1, cfg, p.caseName));
}
