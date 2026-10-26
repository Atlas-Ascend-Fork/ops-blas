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
#include "sasum_param.h"
#include "sasum_golden.h"
#include "sasum_npu_wrapper.h"

class SasumArch35Test : public BlasTest<SasumParam> {};

TEST_F(SasumArch35Test, NullHandle)
{
    float result = 0.0f;
    aclblasStatus_t ret = aclblasSasum_npu(nullptr, 5, nullptr, 1, &result);
    EXPECT_EQ(ret, ACLBLAS_STATUS_NOT_INITIALIZED);
}

INSTANTIATE_TEST_SUITE_P(
    Sasum, SasumArch35Test, ::testing::ValuesIn(GetCasesFromCsv<SasumParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<SasumParam>);

// ---------------------------------------------------------------------------
// Error path test (expectResult != SUCCESS)
// ---------------------------------------------------------------------------
static void TestErrorPath(const SasumParam& p, aclblasHandle_t handle)
{
    const int64_t xLen = (p.n > 0 && p.x.method != BlasFillMode::M_NULLPTR) ? p.n : 0;
    std::vector<float> xHost = makeBlasArray(xLen, p.x, p.randomSeed);
    const float* xPtr = xHost.empty() ? nullptr : xHost.data();

    float result = 0.0f;
    float* resultPtr = p.resultIsNull ? nullptr : &result;

    aclblasStatus_t ret = aclblasSasum_npu(handle, p.n, xPtr, p.incx, resultPtr);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
}

// ---------------------------------------------------------------------------
// Quick-return path (n <= 0 or incx <= 0)
// ---------------------------------------------------------------------------
static void TestNoOpPath(const SasumParam& p, aclblasHandle_t handle)
{
    const int64_t xLen = (p.n > 0) ? p.n : 0;
    std::vector<float> xHost = makeBlasArray(xLen, p.x, p.randomSeed);
    const float* xPtr = xHost.empty() ? nullptr : xHost.data();

    float result = 123.0f;
    aclblasStatus_t ret = aclblasSasum_npu(handle, p.n, xPtr, p.incx, &result);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
    if (ret == ACLBLAS_STATUS_SUCCESS) {
        EXPECT_FLOAT_EQ(result, 0.0f) << "[" << p.caseName << "] early return should produce result=0.0f, got "
                                      << result;
    }
}

// ---------------------------------------------------------------------------
// Precision verification helper
// ---------------------------------------------------------------------------
static void VerifySasumResult(float result, float golden, const std::string& caseName)
{
    VerifyConfig cfg;
    applyMixedTolerance(cfg, ACL_FLOAT, golden);
    EXPECT_TRUE(Verifier::verifyScalar(result, golden, cfg, caseName));
}

// ---------------------------------------------------------------------------
// Normal path test helper
// ---------------------------------------------------------------------------
static void TestNormalPath(const SasumParam& p, aclblasHandle_t handle)
{
    const int64_t xLen = static_cast<int64_t>(1 + (p.n - 1) * p.incx);
    std::vector<float> xHost = makeBlasArray(xLen, p.x, p.randomSeed);

    float result = 0.0f;
    aclblasStatus_t ret = aclblasSasum_npu(handle, p.n, xHost.data(), p.incx, &result);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
    if (ret != ACLBLAS_STATUS_SUCCESS)
        return;

    float golden = 0.0f;
    // issue #606: 参考调用失败时 golden 为 0，会产生与被测实现无关的假失败
    ASSERT_EQ(aclblasSasum_cpu(handle, p.n, xHost.data(), p.incx, &golden), ACLBLAS_STATUS_SUCCESS);

    VerifySasumResult(result, golden, p.caseName);
}

// ---------------------------------------------------------------------------
// Main parameterized test
// ---------------------------------------------------------------------------
TEST_P(SasumArch35Test, CsvDriven)
{
    const auto& p = GetParam();

    if (p.expectResult != ACLBLAS_STATUS_SUCCESS) {
        TestErrorPath(p, SasumArch35Test::handle_);
    } else if (p.n <= 0 || p.incx <= 0) {
        TestNoOpPath(p, SasumArch35Test::handle_);
    } else {
        TestNormalPath(p, SasumArch35Test::handle_);
    }
}
