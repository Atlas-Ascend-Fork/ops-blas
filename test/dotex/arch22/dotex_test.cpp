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
#include "dotex_param.h"
#include "dotex_golden.h"
#include "dotex_npu_wrapper.h"

class DotexTest : public BlasTest<DotexParam> {};

TEST_F(DotexTest, NullHandle)
{
    float result = 0.0f;
    std::vector<float> xHost = makeBlasStrided(5, 1, "RANDOM", 42);
    std::vector<float> yHost = makeBlasStrided(5, 1, "RANDOM", 42);
    const aclblasStatus_t ret = aclblasDotEx_npu(
        nullptr, 5, xHost.data(), ACL_FLOAT, 1, yHost.data(), ACL_FLOAT, 1, &result, ACL_FLOAT, ACL_FLOAT);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_HANDLE_IS_NULLPTR));
}

INSTANTIATE_TEST_SUITE_P(
    Dotex, DotexTest, ::testing::ValuesIn(GetCasesFromCsv<DotexParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<DotexParam>);

TEST_P(DotexTest, CsvDriven)
{
    const auto& p = GetParam();
    std::vector<float> xHost = makeBlasStrided(p.n, static_cast<int>(p.incx), p.x, p.randomSeed);
    std::vector<float> yHost = makeBlasStrided(p.n, static_cast<int>(p.incy), p.y, p.randomSeed + 1U);
    const float* xPtr = xHost.empty() ? nullptr : xHost.data();
    const float* yPtr = yHost.empty() ? nullptr : yHost.data();
    float result = 0.0f;

    const aclblasStatus_t ret = aclblasDotEx_npu(
        DotexTest::handle_, p.n, xPtr, static_cast<aclDataType>(p.xType), p.incx, yPtr,
        static_cast<aclDataType>(p.yType), p.incy, &result, static_cast<aclDataType>(p.resultType),
        static_cast<aclDataType>(p.executionType));
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
    if (p.expectResult != ACLBLAS_STATUS_SUCCESS) {
        return;
    }

    float golden = 0.0f;
    const aclblasStatus_t cpuRet = aclblasDotEx_cpu(
        DotexTest::handle_, p.n, xPtr, static_cast<aclDataType>(p.xType), p.incx, yPtr,
        static_cast<aclDataType>(p.yType), p.incy, &golden, static_cast<aclDataType>(p.resultType),
        static_cast<aclDataType>(p.executionType));
    EXPECT_EQ(static_cast<int>(cpuRet), static_cast<int>(ACLBLAS_STATUS_SUCCESS));

    VerifyConfig cfg;
    applyMixedTolerance(cfg, static_cast<aclDataType>(p.resultType), golden);
    EXPECT_TRUE(Verifier::verifyScalar(result, golden, cfg, p.caseName));
}
