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
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "blas_test.h"
#include "csv_loader.h"
#include "fill.h"
#include "saxpy_golden.h"
#include "saxpy_npu_wrapper.h"
#include "saxpy_param.h"
#include "verify.h"

namespace {
constexpr int SAXPY_PERF_WARMUP = 5;
constexpr int SAXPY_PERF_ITERATIONS = 20;

bool IsPerfCase(const std::string& caseName) { return caseName.compare(0, 6U, "TC_PF_") == 0; }

void RunPerfCase(const SaxpyParam& p, aclblasHandle_t handle, aclrtStream stream)
{
    const size_t bytes = static_cast<size_t>(p.n) * sizeof(float);
    std::vector<float> xHost = makeBlasStrided(p.n, 1, p.x, p.randomSeed);
    std::vector<float> yHost = makeBlasStrided(p.n, 1, p.y, p.randomSeed);
    ASSERT_EQ(xHost.size() * sizeof(float), bytes);
    ASSERT_EQ(yHost.size() * sizeof(float), bytes);

    void* deviceX = nullptr;
    void* deviceY = nullptr;
    ASSERT_EQ(aclrtMalloc(&deviceX, bytes, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    ASSERT_EQ(aclrtMalloc(&deviceY, bytes, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    ASSERT_EQ(aclrtMemcpy(deviceX, bytes, xHost.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);
    ASSERT_EQ(aclrtMemcpy(deviceY, bytes, yHost.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);

    for (int i = 0; i < SAXPY_PERF_WARMUP; ++i) {
        ASSERT_EQ(
            aclblasSaxpy(handle, p.n, &p.alpha, static_cast<float*>(deviceX), 1, static_cast<float*>(deviceY), 1),
            ACLBLAS_STATUS_SUCCESS);
    }
    ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);

    aclrtEvent startEvent = nullptr;
    aclrtEvent stopEvent = nullptr;
    ASSERT_EQ(aclrtCreateEvent(&startEvent), ACL_SUCCESS);
    ASSERT_EQ(aclrtCreateEvent(&stopEvent), ACL_SUCCESS);
    ASSERT_EQ(aclrtRecordEvent(startEvent, stream), ACL_SUCCESS);
    for (int i = 0; i < SAXPY_PERF_ITERATIONS; ++i) {
        ASSERT_EQ(
            aclblasSaxpy(handle, p.n, &p.alpha, static_cast<float*>(deviceX), 1, static_cast<float*>(deviceY), 1),
            ACLBLAS_STATUS_SUCCESS);
    }
    ASSERT_EQ(aclrtRecordEvent(stopEvent, stream), ACL_SUCCESS);
    ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);

    float elapsedMs = 0.0F;
    ASSERT_EQ(aclrtEventElapsedTime(&elapsedMs, startEvent, stopEvent), ACL_SUCCESS);
    const double averageUs = static_cast<double>(elapsedMs) * 1000.0 / SAXPY_PERF_ITERATIONS;
    std::cout << "[PERF] case=" << p.caseName << " n=" << p.n << " avg_us=" << averageUs << std::endl;

    aclrtDestroyEvent(startEvent);
    aclrtDestroyEvent(stopEvent);
    aclrtFree(deviceX);
    aclrtFree(deviceY);
}
} // namespace

class SaxpyTest : public BlasTest<SaxpyParam> {};

TEST_F(SaxpyTest, NullHandle)
{
    float alpha = 1.0F;
    float x[4] = {1.0F, 2.0F, 3.0F, 4.0F};
    float y[4] = {1.0F, 2.0F, 3.0F, 4.0F};
    aclblasStatus_t ret = aclblasSaxpy(nullptr, 4, &alpha, x, 1, y, 1);
    EXPECT_EQ(ret, ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
}

TEST_F(SaxpyTest, NullAlpha)
{
    aclblasStatus_t ret = AclblasSaxpyValidation(handle_, 4, nullptr, false, 1, false, 1);
    EXPECT_EQ(ret, ACLBLAS_STATUS_INVALID_VALUE);
}

TEST_F(SaxpyTest, NullX)
{
    float alpha = 1.0F;
    aclblasStatus_t ret = AclblasSaxpyValidation(handle_, 4, &alpha, true, 1, false, 1);
    EXPECT_EQ(ret, ACLBLAS_STATUS_INVALID_VALUE);
}

TEST_F(SaxpyTest, NullY)
{
    float alpha = 1.0F;
    aclblasStatus_t ret = AclblasSaxpyValidation(handle_, 4, &alpha, false, 1, true, 1);
    EXPECT_EQ(ret, ACLBLAS_STATUS_INVALID_VALUE);
}

INSTANTIATE_TEST_SUITE_P(
    Saxpy, SaxpyTest, ::testing::ValuesIn(GetCasesFromCsv<SaxpyParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<SaxpyParam>);

TEST_P(SaxpyTest, CsvDriven)
{
    const auto& p = GetParam();

    if (IsPerfCase(p.caseName)) {
        RunPerfCase(p, handle_, stream_);
        return;
    }

    const bool expectSuccess = (p.expectResult == ACLBLAS_STATUS_SUCCESS);
    const bool xNull = (p.x.method == BlasFillMode::M_NULLPTR);
    const bool yNull = (p.y.method == BlasFillMode::M_NULLPTR);

    // No allocation for quick-return or negative-span cases.
    if (!expectSuccess || p.n <= 0) {
        const aclblasStatus_t ret = AclblasSaxpyValidation(handle_, p.n, &p.alpha, xNull, p.incx, yNull, p.incy);
        EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult)) << p.caseName;
        return;
    }

    std::vector<float> xHost = makeBlasStrided(p.n, p.incx, p.x, p.randomSeed);
    std::vector<float> yHost = makeBlasStrided(p.n, p.incy, p.y, p.randomSeed);

    const aclblasStatus_t ret = AclblasSaxpyOnDevice(handle_, p.n, &p.alpha, xHost, p.incx, yHost, p.incy);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult)) << p.caseName;
    if (ret != ACLBLAS_STATUS_SUCCESS) {
        return;
    }

    std::vector<float> goldenX = makeBlasStrided(p.n, p.incx, p.x, p.randomSeed);
    std::vector<float> goldenY = makeBlasStrided(p.n, p.incy, p.y, p.randomSeed);
    aclblasSaxpy_cpu(handle_, p.n, &p.alpha, goldenX.data(), p.incx, goldenY.data(), p.incy);

    VerifyConfig cfg;
    applyMixedTolerance(cfg, ACL_FLOAT, goldenY.data(), static_cast<size_t>(p.n));
    const int64_t yStride = std::abs(p.incy);
    EXPECT_TRUE(
        Verifier::verifyVector(yHost.data(), goldenY.data(), static_cast<size_t>(p.n), yStride, cfg, p.caseName));
}
