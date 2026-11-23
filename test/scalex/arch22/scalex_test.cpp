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
#include <vector>

#include "verify.h"
#include "blas_test.h"
#include "csv_loader.h"
#include "scalex_param.h"
#include "scalex_golden.h"
#include "scalex_npu_wrapper.h"

class ScalexTest : public BlasTest<ScalexParam> {};

TEST_F(ScalexTest, NullHandle)
{
    float alphaVal = 2.5F;
    std::vector<float> xHost = makeBlasStrided(5, 1, "RANDOM", 42);
    const aclblasStatus_t ret =
        aclblasScalex_npu(nullptr, 5, &alphaVal, ACL_FLOAT, xHost.data(), ACL_FLOAT, 1, ACL_FLOAT);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_HANDLE_IS_NULLPTR));
}

TEST_F(ScalexTest, NullHandleTakesPrecedenceOverQuickReturn)
{
    EXPECT_EQ(
        aclblasScalex(nullptr, 0, nullptr, ACL_FLOAT, nullptr, ACL_FLOAT, 1, ACL_FLOAT),
        ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
    EXPECT_EQ(
        aclblasScalex(nullptr, 5, nullptr, ACL_FLOAT, nullptr, ACL_FLOAT, 0, ACL_FLOAT),
        ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
}

TEST_F(ScalexTest, QuickReturnSkipsDataPointerValidation)
{
    EXPECT_EQ(
        aclblasScalex(ScalexTest::handle_, 0, nullptr, ACL_FLOAT, nullptr, ACL_FLOAT, 1, ACL_FLOAT),
        ACLBLAS_STATUS_SUCCESS);
    EXPECT_EQ(
        aclblasScalex(ScalexTest::handle_, -3, nullptr, ACL_FLOAT, nullptr, ACL_FLOAT, 0, ACL_FLOAT),
        ACLBLAS_STATUS_SUCCESS);
}

TEST_F(ScalexTest, AlphaNull)
{
    std::vector<float> xHost = makeBlasStrided(5, 1, "RANDOM", 42);
    const aclblasStatus_t ret =
        aclblasScalex_npu(ScalexTest::handle_, 5, nullptr, ACL_FLOAT, xHost.data(), ACL_FLOAT, 1, ACL_FLOAT);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE));
}

INSTANTIATE_TEST_SUITE_P(
    Scalex, ScalexTest, ::testing::ValuesIn(GetCasesFromCsv<ScalexParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<ScalexParam>);

TEST_P(ScalexTest, CsvDriven)
{
    const auto& p = GetParam();
    std::vector<float> xHost;
    if (p.x.method != BlasFillMode::M_NULLPTR) {
        xHost = makeBlasStrided(p.n, p.incx, p.x, p.randomSeed);
    }

    float* xPtr = xHost.empty() ? nullptr : xHost.data();
    const float alphaLocal = p.alphaVal;
    const void* alphaPtr = &alphaLocal;
    std::vector<float> golden = xHost;

    const aclblasStatus_t ret = aclblasScalex_npu(
        ScalexTest::handle_, p.n, alphaPtr, static_cast<aclDataType>(p.alphaType), xPtr,
        static_cast<aclDataType>(p.xType), p.incx, static_cast<aclDataType>(p.executionType), p.alphaOnDevice);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
    if (p.expectResult != ACLBLAS_STATUS_SUCCESS) {
        return;
    }

    const aclblasStatus_t cpuRet = aclblasScalex_cpu(
        ScalexTest::handle_, p.n, alphaPtr, static_cast<aclDataType>(p.alphaType), golden.data(),
        static_cast<aclDataType>(p.xType), p.incx, static_cast<aclDataType>(p.executionType));
    EXPECT_EQ(static_cast<int>(cpuRet), static_cast<int>(ACLBLAS_STATUS_SUCCESS));

    VerifyConfig cfg;
    applyMixedTolerance(cfg, static_cast<aclDataType>(p.xType), golden.data(), static_cast<size_t>(std::max(0, p.n)));
    const int absIncx = std::abs(p.incx);
    EXPECT_TRUE(
        Verifier::verifyVector(xPtr, golden.data(), static_cast<size_t>(std::max(0, p.n)), absIncx, cfg, p.caseName));
}
