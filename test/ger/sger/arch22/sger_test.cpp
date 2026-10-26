/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <climits>
#include <vector>

#include "verify.h"
#include "blas_test.h"
#include "csv_loader.h"
#include "fill.h"
#include "sger_param.h"       // shared at ../
#include "sger_golden.h"      // shared at ../
#include "sger_npu_wrapper.h" // arch22-local (int64_t signature with INT_MIN pass-through)

class SgerArch22Test : public BlasTest<SgerParam> {};

TEST_F(SgerArch22Test, NullHandle)
{
    aclblasStatus_t ret = aclblasSger_npu(nullptr, 4, 4, nullptr, nullptr, 1, nullptr, 1, nullptr, 4);
    EXPECT_EQ(ret, ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
}

INSTANTIATE_TEST_SUITE_P(
    Sger, SgerArch22Test, ::testing::ValuesIn(GetCasesFromCsv<SgerParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<SgerParam>);

TEST_P(SgerArch22Test, CsvDriven)
{
    const auto& p = GetParam();

    // Generate host data — strided storage for x/y, column-major for A.
    // makeBlasArray with NULLPTR fill returns an empty vector, which the test
    // converts to a nullptr pointer below; this is how error-path coverage works.
    const int64_t absIncX = (p.incx >= 0) ? p.incx : -static_cast<int64_t>(p.incx);
    const int64_t absIncY = (p.incy >= 0) ? p.incy : -static_cast<int64_t>(p.incy);
    // Clamp allocation size for invalid strides (INT_MIN / 0): the wrapper short-
    // circuits to INVALID_VALUE without reading the buffer, so a unit-stride
    // allocation is safe and avoids the ~32 GiB demand from |INT_MIN| * m.
    const int64_t allocIncX = (p.incx == INT_MIN || p.incx == 0) ? 1 : absIncX;
    const int64_t allocIncY = (p.incy == INT_MIN || p.incy == 0) ? 1 : absIncY;
    std::vector<float> xHost = makeBlasArray(p.m * allocIncX, p.x, p.randomSeed);
    std::vector<float> yHost = makeBlasArray(p.n * allocIncY, p.y, p.randomSeed);
    std::vector<float> aHost = makeBlasArray(static_cast<int64_t>(p.lda) * p.n, p.A, p.randomSeed + 1);

    float alphaHost = p.alphaValue;
    const float* alphaPtr = (p.alpha.method == BlasFillMode::M_NULLPTR) ? nullptr : &alphaHost;

    // Empty vector from NULLPTR-fill => pass nullptr for negative path testing.
    const float* xPtr = xHost.empty() ? nullptr : xHost.data();
    float* yPtr = yHost.empty() ? nullptr : yHost.data();
    float* aPtr = aHost.empty() ? nullptr : aHost.data();

    // Step 1: Snapshot the original A so the CPU golden can re-run sger on the
    // unmodified input (the NPU wrapper writes its result back into aHost in place).
    const std::vector<float> aOriginal = aHost;

    // Step 1: NPU execution. Wrapper takes int64_t; the int CSV values promote implicitly.
    aclblasStatus_t ret =
        aclblasSger_npu(SgerArch22Test::handle_, p.m, p.n, alphaPtr, xPtr, p.incx, yPtr, p.incy, aPtr, p.lda);

    // Step 2: Check the wrapper/kernel return code matches CSV expect_result.
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
    if (p.expectResult != ACLBLAS_STATUS_SUCCESS)
        return;

    // Step 3: CPU golden via the shared aclblasSger_cpu (validates and computes).
    // issue #606: 参考调用失败时 golden 停留在输入初值，会产生假失败
    std::vector<float> golden = aOriginal; // start from original A
    ASSERT_EQ(
        aclblasSger_cpu(SgerArch22Test::handle_, p.m, p.n, alphaPtr, xPtr, p.incx, yPtr, p.incy, golden.data(), p.lda),
        ACLBLAS_STATUS_SUCCESS);

    // Step 4: Verify precision. alpha=0 leaves A untouched → bit-exact comparison;
    // otherwise use MERE_MARE (defaults: MERE < 2^-13, MARE < 10 * 2^-13).
    VerifyConfig cfg;
    if (p.alphaValue == 0.0f) {
        cfg.mode = PrecisionMode::EXACT;
    } else {
        cfg.mode = PrecisionMode::MERE_MARE;
        if (p.mereThreshold > 0.0)
            cfg.mereThreshold = p.mereThreshold;
        if (p.mareMultiplier > 0.0)
            cfg.mareMultiplier = p.mareMultiplier;
    }
    EXPECT_TRUE(Verifier::verifyVector(aPtr, golden.data(), aHost.size(), 1, cfg, p.caseName));
}
