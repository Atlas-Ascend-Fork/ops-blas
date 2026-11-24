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
#include <chrono>

#include "verify.h"
#include "blas_test.h"
#include "csv_loader.h"
#include "scasum_param.h"
#include "scasum_golden.h"
#include "scasum_npu_wrapper.h"

class ScasumTest : public BlasTest<ScasumParam> {};

TEST_F(ScasumTest, NullHandle)
{
    float result = 0.0f;
    aclblasStatus_t ret = aclblasScasum_npu(nullptr, 5, nullptr, 1, &result);
    EXPECT_EQ(ret, ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
}

INSTANTIATE_TEST_SUITE_P(
    Scasum, ScasumTest, ::testing::ValuesIn(GetCasesFromCsv<ScasumParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<ScasumParam>);

// Negative cases: host buffer length follows the wrapper contract.
static void TestErrorPath(const ScasumParam& p, aclblasHandle_t handle)
{
    const bool quickReturn = p.n <= 0 || p.incx <= 0;
    const bool spanOverflow = !quickReturn &&
        static_cast<int64_t>(p.n - 1) * static_cast<int64_t>(p.incx) * 2 > static_cast<int64_t>(UINT32_MAX);
    const int64_t fullLen = quickReturn ? 0 : 1 + static_cast<int64_t>(p.n - 1) * p.incx;
    int64_t complexLen = 0;
    if (p.n > 0 && p.x.method != BlasFillMode::M_NULLPTR) {
        complexLen = spanOverflow ? 1 : fullLen;
        if (!spanOverflow && complexLen > (int64_t{1} << 26)) {
            GTEST_SKIP() << "[" << p.caseName << "] non-overflow input of " << complexLen
                         << " complex elements exceeds dummy-buffer safety cap";
            return;
        }
    }
    std::vector<float> xHostF = makeBlasArray(complexLen * 2, p.x, p.randomSeed);
    const aclblasComplex* xPtr = xHostF.empty() ? nullptr : reinterpret_cast<const aclblasComplex*>(xHostF.data());

    float result = 0.0f;
    float* resultPtr = p.resultIsNull ? nullptr : &result;

    aclblasStatus_t ret = aclblasScasum_npu(handle, p.n, xPtr, p.incx, resultPtr);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
}

static void TestNoOpPath(const ScasumParam& p, aclblasHandle_t handle)
{
    const int64_t complexLen = (p.n > 0) ? p.n : 0;
    std::vector<float> xHostF = makeBlasArray(complexLen * 2, p.x, p.randomSeed);
    const aclblasComplex* xPtr = xHostF.empty() ? nullptr : reinterpret_cast<const aclblasComplex*>(xHostF.data());

    float result = 123.0f;
    aclblasStatus_t ret = aclblasScasum_npu(handle, p.n, xPtr, p.incx, &result);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
    if (ret == ACLBLAS_STATUS_SUCCESS) {
        EXPECT_FLOAT_EQ(result, 0.0f) << "[" << p.caseName << "] early return should produce result=0.0f, got "
                                      << result;
    }
}

static void VerifyScasumResult(float result, float golden, const std::string& caseName)
{
    VerifyConfig cfg;
    applyMixedTolerance(cfg, ACL_FLOAT, golden);
    EXPECT_TRUE(Verifier::verifyScalar(result, golden, cfg, caseName));
}

static void TestNormalPath(const ScasumParam& p, aclblasHandle_t handle)
{
    const int64_t complexLen = (p.n > 0 && p.incx > 0) ? (1 + (p.n - 1) * p.incx) : 0;
    std::vector<float> xHostF = makeBlasArray(complexLen * 2, p.x, p.randomSeed);
    const aclblasComplex* xPtr = xHostF.empty() ? nullptr : reinterpret_cast<const aclblasComplex*>(xHostF.data());

    float result = 0.0f;
    aclblasStatus_t ret = aclblasScasum_npu(handle, p.n, xPtr, p.incx, &result);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
    if (ret != ACLBLAS_STATUS_SUCCESS)
        return;

    float golden = 0.0f;
    if (p.caseName.rfind("TC_PF_", 0) == 0) {
        const int absInc = std::abs(p.incx);
        const size_t xBytes = (p.n > 0 ? static_cast<size_t>((p.n - 1) * absInc + 1) : 1) * sizeof(aclblasComplex);
        aclblasComplex* dX = nullptr;
        float* dR = nullptr;
        if (aclrtMalloc(reinterpret_cast<void**>(&dX), xBytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
            printf("[ERROR] aclrtMalloc dX failed\n");
            return;
        }
        if (aclrtMalloc(reinterpret_cast<void**>(&dR), sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
            printf("[ERROR] aclrtMalloc dR failed\n");
            aclrtFree(dX);
            return;
        }
        if (aclrtMemcpy(dX, xBytes, xPtr, xBytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
            printf("[ERROR] aclrtMemcpy H2D failed\n");
            aclrtFree(dR);
            aclrtFree(dX);
            return;
        }

        for (int i = 0; i < 10; i++) {
            aclblasScasum(handle, p.n, dX, p.incx, dR);
        }
        aclrtSynchronizeDevice();

        const int samples = 100;
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < samples; i++) {
            aclblasScasum(handle, p.n, dX, p.incx, dR);
        }
        aclrtSynchronizeDevice();
        auto t1 = std::chrono::high_resolution_clock::now();
        double avgUs = std::chrono::duration<double, std::micro>(t1 - t0).count() / samples;
        printf("[PERF] case=%s avg_single_us=%.2f\n", p.caseName.c_str(), avgUs);

        aclrtFree(dX);
        aclrtFree(dR);
        return;
    }
    // issue #604
    if (aclblasScasum_cpu(handle, p.n, xPtr, p.incx, &golden) != ACLBLAS_STATUS_SUCCESS) {
        ADD_FAILURE() << "[" << p.caseName << "] ERROR: aclblasScasum_cpu failed, golden invalid";
        return;
    }

    VerifyScasumResult(result, golden, p.caseName);
}

TEST_P(ScasumTest, CsvDriven)
{
    const auto& p = GetParam();

    if (p.expectResult != ACLBLAS_STATUS_SUCCESS) {
        TestErrorPath(p, ScasumTest::handle_);
    } else if (p.n <= 0 || p.incx <= 0) {
        TestNoOpPath(p, ScasumTest::handle_);
    } else {
        TestNormalPath(p, ScasumTest::handle_);
    }
}
