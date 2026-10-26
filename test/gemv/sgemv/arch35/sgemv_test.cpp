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
#include "sgemv_param.h"
#include "sgemv_golden.h"
#include "sgemv_npu_wrapper.h"

class SgemvArch35Test : public BlasTest<SgemvParam> {};

// E01: handle = nullptr → ACLBLAS_STATUS_HANDLE_IS_NULLPTR
TEST_F(SgemvArch35Test, NullHandle)
{
    float alpha = 1.0f, beta = 0.0f;
    std::vector<float> a(64, 1.0f), x(8, 1.0f), y(8, 0.0f);
    aclblasStatus_t ret =
        aclblasSgemv_npu(nullptr, ACLBLAS_OP_N, 8, 8, &alpha, a.data(), 8, x.data(), 1, &beta, y.data(), 1);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_HANDLE_IS_NULLPTR));
}

INSTANTIATE_TEST_SUITE_P(
    Sgemv, SgemvArch35Test, ::testing::ValuesIn(GetCasesFromCsv<SgemvParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<SgemvParam>);

TEST_P(SgemvArch35Test, CsvDriven)
{
    const auto& p = GetParam();

    const bool isTransN = (p.trans == ACLBLAS_OP_N);
    const int xCount = isTransN ? p.n : p.m;
    const int yCount = isTransN ? p.m : p.n;

    // Generate input data with distinct seeds for A, x, y
    auto aHost = makeBlasArray(static_cast<int64_t>(p.lda) * std::max(1, p.n), p.a, p.randomSeed);
    auto xHost = makeBlasStrided(xCount, p.incx, p.x, p.randomSeed + 1);
    auto yHost = makeBlasStrided(yCount, p.incy, p.y, p.randomSeed + 2);

    const float* alphaPtr = (p.alphaFill.method == BlasFillMode::M_NULLPTR) ? nullptr : &p.alpha;
    const float* betaPtr = (p.betaFill.method == BlasFillMode::M_NULLPTR) ? nullptr : &p.beta;

    // NPU path: copy y, run kernel, result lands in yNpu
    std::vector<float> yNpu = yHost;
    aclblasStatus_t ret = aclblasSgemv_npu(
        SgemvArch35Test::handle_, p.trans, p.m, p.n, alphaPtr, aHost.data(), p.lda, xHost.data(), p.incx, betaPtr,
        yNpu.data(), p.incy);

    if (p.expectResult != ACLBLAS_STATUS_SUCCESS) {
        EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
        return;
    }
    ASSERT_EQ(ret, ACLBLAS_STATUS_SUCCESS);

    // CPU golden path: same inputs, compute reference
    std::vector<float> yCpu = yHost;
    // issue #606: 参考调用失败时 golden 停留在输入初值，会产生假失败
    ASSERT_EQ(
        aclblasSgemv_cpu(
            SgemvArch35Test::handle_, p.trans, p.m, p.n, alphaPtr, aHost.data(), p.lda, xHost.data(), p.incx, betaPtr,
            yCpu.data(), p.incy),
        ACLBLAS_STATUS_SUCCESS);

    if (yCount == 0) {
        return;
    }

    // De-stride output arrays for verification (handles negative incy correctly)
    const int absIncy = std::abs(p.incy);
    std::vector<float> npuLogical(yCount), cpuLogical(yCount);
    for (int i = 0; i < yCount; i++) {
        int idx = (p.incy > 0) ? (i * p.incy) : ((yCount - 1 - i) * absIncy);
        npuLogical[i] = yNpu[idx];
        cpuLogical[i] = yCpu[idx];
    }

    VerifyConfig cfg;
    applyMixedTolerance(cfg, ACL_FLOAT, cpuLogical.data(), static_cast<size_t>(yCount));
    EXPECT_TRUE(
        Verifier::verifyVector(npuLogical.data(), cpuLogical.data(), static_cast<size_t>(yCount), 1, cfg, p.caseName));
}
