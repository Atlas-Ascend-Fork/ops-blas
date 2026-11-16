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
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "verify.h"
#include "blas_test.h"
#include "csv_loader.h"
#include "sgemv_param.h"
#include "sgemv_golden.h"
#include "sgemv_npu_wrapper.h"

namespace {

void RunSgemvBoundaryOrNegative(const SgemvParam &p, aclblasHandle_t handle)
{
    float dummyA = 1.0f;
    float dummyX = 1.0f;
    float dummyY = 2.0f;
    const float *alphaPtr = (p.alphaFill.method == BlasFillMode::M_NULLPTR) ? nullptr : &p.alpha;
    const float *betaPtr = (p.betaFill.method == BlasFillMode::M_NULLPTR) ? nullptr : &p.beta;
    const float *aPtr = (p.a.method == BlasFillMode::M_NULLPTR) ? nullptr : &dummyA;
    const float *xPtr = (p.x.method == BlasFillMode::M_NULLPTR) ? nullptr : &dummyX;
    float *yPtr = (p.y.method == BlasFillMode::M_NULLPTR) ? nullptr : &dummyY;

    aclblasStatus_t status = aclblasSgemv_npu(handle, p.trans, p.m, p.n, alphaPtr, aPtr, p.lda,
                                              xPtr, p.incx, betaPtr, yPtr, p.incy);
    EXPECT_EQ(static_cast<int>(status), static_cast<int>(p.expectResult));
    if (p.expectResult == ACLBLAS_STATUS_SUCCESS && yPtr != nullptr) {
        EXPECT_EQ(dummyY, 2.0f);
    }
}

void VerifySgemvLogicalVector(const SgemvParam &p, const std::vector<float> &yNpu,
                              const std::vector<float> &yCpu, int yLen, bool isPerfCase)
{
    const int strideY = std::abs(p.incy);
    std::vector<float> npuDense(static_cast<size_t>(yLen));
    std::vector<float> cpuDense(static_cast<size_t>(yLen));
    for (int i = 0; i < yLen; ++i) {
        const int offset = (p.incy > 0) ? (i * p.incy) : ((yLen - 1 - i) * strideY);
        npuDense[static_cast<size_t>(i)] = yNpu[static_cast<size_t>(offset)];
        cpuDense[static_cast<size_t>(i)] = yCpu[static_cast<size_t>(offset)];
    }

    VerifyConfig cfg;
    applyMixedTolerance(cfg, ACL_FLOAT, cpuDense.data(), static_cast<size_t>(yLen));
    if (isPerfCase) {
        cfg.mixedMaxAbsErrorLimit = std::max(cfg.mixedMaxAbsErrorLimit, 0.2);
        cfg.mixedRtol = std::max(cfg.mixedRtol, 2.0e-4);
    }
    EXPECT_TRUE(Verifier::verifyVector(npuDense.data(), cpuDense.data(), static_cast<size_t>(yLen),
                                       1, cfg, p.caseName));
}

void RunSgemvPositiveCase(const SgemvParam &p, aclblasHandle_t handle, aclrtStream stream)
{
    const bool isNoTrans = (p.trans == ACLBLAS_OP_N);
    const int xLen = isNoTrans ? p.n : p.m;
    const int yLen = isNoTrans ? p.m : p.n;

    std::vector<float> aHost = makeBlasArray(static_cast<int64_t>(p.lda) * static_cast<int64_t>(p.n),
                                             p.a, p.randomSeed);
    std::vector<float> xHost = makeBlasStrided(xLen, p.incx, p.x, p.randomSeed + 1U);
    std::vector<float> yNpu = makeBlasStrided(yLen, p.incy, p.y, p.randomSeed + 2U);
    std::vector<float> yCpu = yNpu;

    const bool isPerfCase = (p.caseName.rfind("TC_PF_", 0) == 0);
    double avgUs = 0.0;
    aclblasStatus_t npuRet = aclblasSgemv_npu_perf(handle, isPerfCase ? stream : nullptr, p.trans, p.m, p.n,
                                                   &p.alpha, aHost.data(), p.lda, xHost.data(), p.incx,
                                                   &p.beta, yNpu.data(), p.incy,
                                                   isPerfCase ? &avgUs : nullptr);
    ASSERT_EQ(npuRet, ACLBLAS_STATUS_SUCCESS);
    if (isPerfCase) {
        std::cout << "[PERF] case=" << p.caseName << " trans=" << static_cast<int>(p.trans)
                  << " m=" << p.m << " n=" << p.n << " avg_us=" << avgUs << std::endl;
    }

    aclblasStatus_t cpuRet = aclblasSgemv_cpu(handle, p.trans, p.m, p.n, &p.alpha, aHost.data(), p.lda,
                                              xHost.data(), p.incx, &p.beta, yCpu.data(), p.incy);
    ASSERT_EQ(cpuRet, ACLBLAS_STATUS_SUCCESS);
    VerifySgemvLogicalVector(p, yNpu, yCpu, yLen, isPerfCase);
}

} // namespace

class SgemvArch22Test : public BlasTest<SgemvParam> {};

TEST_F(SgemvArch22Test, NullHandle)
{
    float alpha = 1.0f;
    float beta = 0.0f;
    std::vector<float> aBuf(16, 1.0f);
    std::vector<float> xBuf(4, 1.0f);
    std::vector<float> yBuf(4, 0.0f);
    aclblasStatus_t status = aclblasSgemv_npu(nullptr, ACLBLAS_OP_N, 4, 4, &alpha, aBuf.data(), 4,
                                              xBuf.data(), 1, &beta, yBuf.data(), 1);
    EXPECT_EQ(static_cast<int>(status), static_cast<int>(ACLBLAS_STATUS_HANDLE_IS_NULLPTR));
}

INSTANTIATE_TEST_SUITE_P(
    Sgemv, SgemvArch22Test, ::testing::ValuesIn(GetCasesFromCsv<SgemvParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<SgemvParam>);

TEST_P(SgemvArch22Test, CsvDriven)
{
    const SgemvParam &p = GetParam();
    if (p.expectResult != ACLBLAS_STATUS_SUCCESS || p.m <= 0 || p.n <= 0) {
        RunSgemvBoundaryOrNegative(p, SgemvArch22Test::handle_);
        return;
    }
    RunSgemvPositiveCase(p, SgemvArch22Test::handle_, SgemvArch22Test::stream_);
}
