/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <cstdlib>
#include <limits>
#include <vector>

#include "verify.h"
#include "blas_test.h"
#include "csv_loader.h"
#include "srotmg_param.h"
#include "srotmg_golden.h"
#include "srotmg_npu_wrapper.h"

class SrotmgArch35Test : public BlasTest<SrotmgParam> {};

// Null handle error path test (not in CSV)
TEST_F(SrotmgArch35Test, NullHandle)
{
    float d1 = 1.0f, d2 = 2.0f, x1 = 3.0f, y1 = 4.0f;
    float param[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    aclblasStatus_t ret = aclblasSrotmg_npu(nullptr, &d1, &d2, &x1, &y1, param);
    EXPECT_EQ(ret, ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
}

// issue #871: d1=±Inf 时 host/device 两条路径的 rescale 循环均有上限，不再挂死。
// host 路径取 +inf：-inf 会命中 srotmg_host 的 `sd1 < ZERO` 早退分支而不进入 rescale
// 循环，覆盖不到守卫；+inf 走正常分支且 rescale 不收敛，可真实触发上限退出。
TEST_F(SrotmgArch35Test, NonFiniteD1)
{
    const float inf = std::numeric_limits<float>::infinity();

    float d1Host = inf;
    float d2Host = 1.0f;
    float x1Host = 1.0f;
    const float y1Host = 1.0f;
    float paramHost[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    EXPECT_EQ(
        aclblasSrotmg(SrotmgArch35Test::handle_, &d1Host, &d2Host, &x1Host, &y1Host, paramHost),
        ACLBLAS_STATUS_SUCCESS);

    float d1Dev = inf;
    float d2Dev = 1.0f;
    float x1Dev = 1.0f;
    const float y1Dev = 1.0f;
    float paramDev[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    EXPECT_EQ(
        aclblasSrotmg_npu(SrotmgArch35Test::handle_, &d1Dev, &d2Dev, &x1Dev, &y1Dev, paramDev), ACLBLAS_STATUS_SUCCESS);
}

INSTANTIATE_TEST_SUITE_P(
    Srotmg, SrotmgArch35Test, ::testing::ValuesIn(GetCasesFromCsv<SrotmgParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<SrotmgParam>);

// CSV-driven: all-device path (via NPU wrapper)
TEST_P(SrotmgArch35Test, CsvDrivenDevice)
{
    const auto& p = GetParam();

    float d1Npu = p.d1, d2Npu = p.d2, x1Npu = p.x1, y1Val = p.y1;
    float paramNpu[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    // Golden
    float d1Golden = p.d1, d2Golden = p.d2, x1Golden = p.x1;
    float y1Golden = p.y1;
    float paramGolden[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    EXPECT_EQ(
        aclblasSrotmg_cpu(handle_, &d1Golden, &d2Golden, &x1Golden, &y1Golden, paramGolden), ACLBLAS_STATUS_SUCCESS);

    aclblasStatus_t ret = aclblasSrotmg_npu(handle_, &d1Npu, &d2Npu, &x1Npu, &y1Val, paramNpu);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
    if (p.expectResult != ACLBLAS_STATUS_SUCCESS)
        return;

    VerifyConfig cfg;
    applyMixedTolerance(cfg, ACL_FLOAT, d1Golden);

    EXPECT_TRUE(Verifier::verifyScalar(d1Npu, d1Golden, cfg, p.caseName + "_d1"));
    EXPECT_TRUE(Verifier::verifyScalar(d2Npu, d2Golden, cfg, p.caseName + "_d2"));
    EXPECT_TRUE(Verifier::verifyScalar(x1Npu, x1Golden, cfg, p.caseName + "_x1"));
    EXPECT_TRUE(Verifier::verifyScalar(paramNpu[0], paramGolden[0], cfg, p.caseName + "_param_sflag"));
    EXPECT_TRUE(Verifier::verifyScalar(paramNpu[1], paramGolden[1], cfg, p.caseName + "_param_h11"));
    EXPECT_TRUE(Verifier::verifyScalar(paramNpu[2], paramGolden[2], cfg, p.caseName + "_param_h21"));
    EXPECT_TRUE(Verifier::verifyScalar(paramNpu[3], paramGolden[3], cfg, p.caseName + "_param_h12"));
    EXPECT_TRUE(Verifier::verifyScalar(paramNpu[4], paramGolden[4], cfg, p.caseName + "_param_h22"));
}

// CSV-driven: all-host path (aclrtMallocHost pinned memory, direct call)
TEST_P(SrotmgArch35Test, CsvDrivenHost)
{
    const auto& p = GetParam();

    float *d1, *d2, *x1, *y1, *param;
    ASSERT_EQ(aclrtMallocHost(reinterpret_cast<void**>(&d1), sizeof(float)), ACL_SUCCESS);
    ASSERT_EQ(aclrtMallocHost(reinterpret_cast<void**>(&d2), sizeof(float)), ACL_SUCCESS);
    ASSERT_EQ(aclrtMallocHost(reinterpret_cast<void**>(&x1), sizeof(float)), ACL_SUCCESS);
    ASSERT_EQ(aclrtMallocHost(reinterpret_cast<void**>(&y1), sizeof(float)), ACL_SUCCESS);
    ASSERT_EQ(aclrtMallocHost(reinterpret_cast<void**>(&param), 5 * sizeof(float)), ACL_SUCCESS);

    *d1 = p.d1;
    *d2 = p.d2;
    *x1 = p.x1;
    *y1 = p.y1;

    // Golden
    float d1G = p.d1, d2G = p.d2, x1G = p.x1, y1G = p.y1;
    float paramG[5] = {};
    EXPECT_EQ(aclblasSrotmg_cpu(handle_, &d1G, &d2G, &x1G, &y1G, paramG), ACLBLAS_STATUS_SUCCESS);

    aclblasStatus_t ret = aclblasSrotmg(handle_, d1, d2, x1, y1, param);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));

    if (ret == ACLBLAS_STATUS_SUCCESS) {
        VerifyConfig cfg;
        if (p.mereThreshold > 0.0) {
            cfg.mode = PrecisionMode::MERE_MARE;
            cfg.mereThreshold = p.mereThreshold;
            cfg.mareMultiplier = p.mareMultiplier;
        } else {
            cfg.mode = PrecisionMode::EXACT;
        }

        EXPECT_TRUE(Verifier::verifyScalar(*d1, d1G, cfg, p.caseName + "_d1"));
        EXPECT_TRUE(Verifier::verifyScalar(*d2, d2G, cfg, p.caseName + "_d2"));
        EXPECT_TRUE(Verifier::verifyScalar(*x1, x1G, cfg, p.caseName + "_x1"));
        EXPECT_TRUE(Verifier::verifyScalar(param[0], paramG[0], cfg, p.caseName + "_param_sflag"));
        EXPECT_TRUE(Verifier::verifyScalar(param[1], paramG[1], cfg, p.caseName + "_param_h11"));
        EXPECT_TRUE(Verifier::verifyScalar(param[2], paramG[2], cfg, p.caseName + "_param_h21"));
        EXPECT_TRUE(Verifier::verifyScalar(param[3], paramG[3], cfg, p.caseName + "_param_h12"));
        EXPECT_TRUE(Verifier::verifyScalar(param[4], paramG[4], cfg, p.caseName + "_param_h22"));
    }

    if (d1)
        aclrtFreeHost(d1);
    if (d2)
        aclrtFreeHost(d2);
    if (x1)
        aclrtFreeHost(x1);
    if (y1)
        aclrtFreeHost(y1);
    if (param)
        aclrtFreeHost(param);
}
