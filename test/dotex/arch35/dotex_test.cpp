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

// ── TEST_F: null handle (not in CSV) ─────────────────────────────────────────
TEST_F(DotexTest, NullHandle)
{
    float result = 0.0f;
    std::vector<float> xHost = makeBlasStrided(5, 1, "RANDOM", 42);
    std::vector<float> yHost = makeBlasStrided(5, 1, "RANDOM", 42);
    aclblasStatus_t ret = aclblasDotEx_npu(
        nullptr, 5, xHost.data(), ACL_FLOAT, 1, yHost.data(), ACL_FLOAT, 1, &result, ACL_FLOAT, ACL_FLOAT);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_HANDLE_IS_NULLPTR));
}

// issue #966: n==0 快速路径 result 为空指针时返回 INVALID_VALUE，不再以
// EXECUTION_FAILED 暴露 aclrtMemset 失败位置
TEST_F(DotexTest, NullResultQuickReturn)
{
    std::vector<float> xHost = makeBlasStrided(1, 1, "RANDOM", 42);
    aclblasStatus_t ret = aclblasDotEx_npu(
        DotexTest::handle_, 0, xHost.data(), ACL_FLOAT, 1, xHost.data(), ACL_FLOAT, 1, nullptr, ACL_FLOAT, ACL_FLOAT);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE));
}

// ── CSV parameterised test suite ─────────────────────────────────────────────
INSTANTIATE_TEST_SUITE_P(
    Dotex, DotexTest, ::testing::ValuesIn(GetCasesFromCsv<DotexParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<DotexParam>);

// ── TEST_P: 5-step CSV-driven flow ───────────────────────────────────────────
TEST_P(DotexTest, CsvDriven)
{
    const auto& p = GetParam();

    // Step 1: Generate host data
    std::vector<float> xHost = makeBlasStrided(p.n, static_cast<int>(p.incx), p.x, p.randomSeed);
    std::vector<float> yHost = makeBlasStrided(p.n, static_cast<int>(p.incy), p.y, p.randomSeed);

    const float* xPtr = xHost.empty() ? nullptr : xHost.data();
    const float* yPtr = yHost.empty() ? nullptr : yHost.data();
    float result = 0.0f;

    // Step 2: Execute NPU (wrapper handles packing, device memory)
    aclblasStatus_t ret = aclblasDotEx_npu(
        DotexTest::handle_, p.n, xPtr, static_cast<aclDataType>(p.xType), p.incx, yPtr,
        static_cast<aclDataType>(p.yType), p.incy, &result, static_cast<aclDataType>(p.resultType),
        static_cast<aclDataType>(p.executionType));

    // Step 3: Verify expected error code
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
    if (p.expectResult != ACLBLAS_STATUS_SUCCESS) {
        return;
    }

    // Step 4: Compute golden
    float golden = 0.0f;
    aclblasStatus_t cpuRet = aclblasDotEx_cpu(
        DotexTest::handle_, p.n, xPtr, static_cast<aclDataType>(p.xType), p.incx, yPtr,
        static_cast<aclDataType>(p.yType), p.incy, &golden, static_cast<aclDataType>(p.resultType),
        static_cast<aclDataType>(p.executionType));
    EXPECT_EQ(static_cast<int>(cpuRet), static_cast<int>(ACLBLAS_STATUS_SUCCESS));

    // Step 5: Precision verification with mixed tolerance
    VerifyConfig cfg;
    applyMixedTolerance(cfg, static_cast<aclDataType>(p.resultType), golden);
    EXPECT_TRUE(Verifier::verifyScalar(result, golden, cfg, p.caseName));
}
