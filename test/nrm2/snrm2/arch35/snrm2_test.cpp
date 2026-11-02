/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include <cstdint>
#include <vector>

#include "verify.h"
#include "blas_test.h"
#include "csv_loader.h"
#include "snrm2_param.h"
#include "snrm2_golden.h"
#include "snrm2_npu_wrapper.h"

class Snrm2Arch35Test : public BlasTest<Snrm2Param> {};

TEST_F(Snrm2Arch35Test, NullHandle)
{
    float result = 0.0f;
    aclblasStatus_t ret = aclblasSnrm2_npu(nullptr, 5, nullptr, 1, &result);
    EXPECT_EQ(ret, ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
}

INSTANTIATE_TEST_SUITE_P(
    Snrm2, Snrm2Arch35Test, ::testing::ValuesIn(GetCasesFromCsv<Snrm2Param>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<Snrm2Param>);

// ---------------------------------------------------------------------------
// Error path test (expectResult != SUCCESS)
// ---------------------------------------------------------------------------
static void TestErrorPath(const Snrm2Param& p, aclblasHandle_t handle)
{
    const int64_t xLen = (p.n > 0 && p.x.method != BlasFillMode::M_NULLPTR) ? p.n : 0;
    std::vector<float> xHost = makeBlasArray(xLen, p.x, p.randomSeed);
    const float* xPtr = xHost.empty() ? nullptr : xHost.data();

    float result = 0.0f;
    float* resultPtr = p.resultIsNull ? nullptr : &result;

    aclblasStatus_t ret = aclblasSnrm2_npu(handle, p.n, xPtr, p.incx, resultPtr);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
}

// ---------------------------------------------------------------------------
// No-op path test (n<=0 or incx<=0)
// ---------------------------------------------------------------------------
static void TestNoOpPath(const Snrm2Param& p, aclblasHandle_t handle)
{
    const int64_t xLen = (p.n > 0) ? p.n : 0;
    std::vector<float> xHost = makeBlasArray(xLen, p.x, p.randomSeed);
    const float* xPtr = xHost.empty() ? nullptr : xHost.data();

    float result = 123.0f;
    aclblasStatus_t ret = aclblasSnrm2_npu(handle, p.n, xPtr, p.incx, &result);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
    if (ret == ACLBLAS_STATUS_SUCCESS) {
        EXPECT_FLOAT_EQ(result, 0.0f) << "[" << p.caseName << "] early return should produce result=0.0f, got "
                                      << result;
    }
}

// ---------------------------------------------------------------------------
// Precision verification helper
// ---------------------------------------------------------------------------
static void VerifyNrm2Result(float result, float golden, const std::string& caseName)
{
    VerifyConfig cfg;
    applyMixedTolerance(cfg, ACL_FLOAT, golden);
    EXPECT_TRUE(Verifier::verifyScalar(result, golden, cfg, caseName));
}

// ---------------------------------------------------------------------------
// Normal path test helper
// ---------------------------------------------------------------------------
static void TestNormalPath(const Snrm2Param& p, aclblasHandle_t handle)
{
    const int64_t xLen = static_cast<int64_t>(1 + (p.n - 1) * p.incx);
    std::vector<float> xHost = makeBlasArray(xLen, p.x, p.randomSeed);

    float result = 0.0f;
    aclblasStatus_t ret = aclblasSnrm2_npu(handle, p.n, xHost.data(), p.incx, &result);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
    if (ret != ACLBLAS_STATUS_SUCCESS)
        return;

    float golden = 0.0f;
    aclblasSnrm2_cpu(handle, p.n, xHost.data(), p.incx, &golden);

    VerifyNrm2Result(result, golden, p.caseName);
}

// ---------------------------------------------------------------------------
// Main parameterized test
// ---------------------------------------------------------------------------
TEST_P(Snrm2Arch35Test, CsvDriven)
{
    const auto& p = GetParam();

    if (p.expectResult != ACLBLAS_STATUS_SUCCESS) {
        TestErrorPath(p, Snrm2Arch35Test::handle_);
    } else if (p.n <= 0 || p.incx <= 0) {
        TestNoOpPath(p, Snrm2Arch35Test::handle_);
    } else {
        TestNormalPath(p, Snrm2Arch35Test::handle_);
    }
}
