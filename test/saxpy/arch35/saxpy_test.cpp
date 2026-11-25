/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <algorithm>
#include <limits>
#include <string>
#include <vector>

#include "fill.h"
#include "verify.h"
#include "blas_test.h"
#include "csv_loader.h"
#include "saxpy_param.h"
#include "saxpy_golden.h"
#include "saxpy_npu_wrapper.h"

class SaxpyArch35Test : public BlasTest<SaxpyParam> {};

TEST_F(SaxpyArch35Test, NullHandle)
{
    float alpha = 1.0f;
    float x[5] = {1, 2, 3, 4, 5};
    float y[5] = {1, 2, 3, 4, 5};
    aclblasStatus_t ret = aclblasSaxpy_npu(nullptr, 5, &alpha, x, 1, y, 1);
    EXPECT_EQ(ret, ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
}

TEST_F(SaxpyArch35Test, NullAlpha)
{
    float x[5] = {1, 2, 3, 4, 5};
    float y[5] = {1, 2, 3, 4, 5};
    aclblasStatus_t ret = aclblasSaxpy_npu(SaxpyArch35Test::handle_, 5, nullptr, x, 1, y, 1);
    EXPECT_EQ(ret, ACLBLAS_STATUS_INVALID_VALUE);
}

// BLAS Level 1 quick return: alpha == 0 must not read x or touch y,
// so NaN/Inf in x must not pollute y (matches cblas_saxpy semantics)
TEST_F(SaxpyArch35Test, AlphaZeroQuickReturnKeepsYUntouched)
{
    const float nanValue = std::numeric_limits<float>::quiet_NaN();
    const float infValue = std::numeric_limits<float>::infinity();
    const int n = 8;
    float x[n] = {nanValue, infValue, -infValue, 1, 2, 3, 4, 5};
    float y[n] = {1, 1, 1, 1, 1, 1, 1, 1};
    float alpha = 0.0f;

    aclblasStatus_t ret = aclblasSaxpy_npu(SaxpyArch35Test::handle_, n, &alpha, x, 1, y, 1);
    EXPECT_EQ(ret, ACLBLAS_STATUS_SUCCESS);

    const float expected[n] = {1, 1, 1, 1, 1, 1, 1, 1};
    for (int i = 0; i < n; i++) {
        EXPECT_EQ(y[i], expected[i]) << "y[" << i << "] was modified by alpha==0 axpy";
    }
}

INSTANTIATE_TEST_SUITE_P(
    Saxpy, SaxpyArch35Test, ::testing::ValuesIn(GetCasesFromCsv<SaxpyParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<SaxpyParam>);

TEST_P(SaxpyArch35Test, CsvDriven)
{
    const auto& p = GetParam();

    std::vector<float> xHost = makeBlasStrided(p.n, p.incx, p.x, p.randomSeed);
    std::vector<float> yHost = makeBlasStrided(p.n, p.incy, p.y, p.randomSeed);

    float* xPtr = xHost.empty() ? nullptr : xHost.data();
    float* yPtr = yHost.empty() ? nullptr : yHost.data();

    aclblasStatus_t ret = aclblasSaxpy_npu(SaxpyArch35Test::handle_, p.n, &p.alpha, xPtr, p.incx, yPtr, p.incy);

    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
    if (p.expectResult != ACLBLAS_STATUS_SUCCESS)
        return;

    if (p.n <= 0)
        return;

    std::vector<float> goldenX = makeBlasStrided(p.n, p.incx, p.x, p.randomSeed);
    std::vector<float> goldenY = makeBlasStrided(p.n, p.incy, p.y, p.randomSeed);
    aclblasSaxpy_cpu(SaxpyArch35Test::handle_, p.n, &p.alpha, goldenX.data(), p.incx, goldenY.data(), p.incy);

    VerifyConfig cfg;
    applyMixedTolerance(cfg, ACL_FLOAT, goldenY.data(), static_cast<size_t>(p.n));
    int64_t yStride = std::abs(p.incy);
    EXPECT_TRUE(Verifier::verifyVector(yPtr, goldenY.data(), static_cast<size_t>(p.n), yStride, cfg, p.caseName));
}
