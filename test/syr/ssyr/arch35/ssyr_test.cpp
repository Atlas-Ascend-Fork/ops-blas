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
#include <vector>

#include "verify.h"
#include "blas_test.h"
#include "csv_loader.h"
#include "ssyr_param.h"
#include "ssyr_golden.h"
#include "ssyr_npu_wrapper.h"

class SsyrArch35Test : public BlasTest<SsyrParam> {};

TEST_F(SsyrArch35Test, NullHandle)
{
    float alpha = 1.0f;
    float dummy = 0.0f;
    aclblasStatus_t ret = aclblasSsyr(nullptr, ACLBLAS_UPPER, 4, &alpha, &dummy, 1, &dummy, 4);
    EXPECT_NE(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_SUCCESS));
}

INSTANTIATE_TEST_SUITE_P(
    Ssyr, SsyrArch35Test, ::testing::ValuesIn(GetCasesFromCsv<SsyrParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<SsyrParam>);

TEST_P(SsyrArch35Test, CsvDriven)
{
    const auto& p = GetParam();

    int absIncx = std::abs(p.incx);
    size_t aSize = (p.n > 0) ? static_cast<size_t>(p.lda) * p.n : 0;

    std::vector<float> xHost = makeBlasStrided(p.n, p.incx, p.x, p.randomSeed);
    std::vector<float> aHost =
        makeBlasArray(static_cast<int64_t>(aSize), p.a, p.randomSeed);
    std::vector<float> aOrig =
        makeBlasArray(static_cast<int64_t>(aSize), p.a, p.randomSeed);

    const float* xPtr = xHost.empty() ? nullptr : xHost.data();
    float* aPtr = aHost.empty() ? nullptr : aHost.data();

    aclblasStatus_t ret = aclblasSsyr_npu(SsyrArch35Test::handle_, p.uplo, p.n, p.alpha, xPtr, p.incx, aPtr, p.lda);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
    if (p.expectResult != ACLBLAS_STATUS_SUCCESS)
        return;
    if (p.n == 0)
        return;

    std::vector<float> golden = aOrig;
    aclblasSsyr_cpu(p.uplo, p.n, p.alpha, xHost.data(), p.incx, golden.data(), p.lda);

    std::vector<float> triOut, triGold;
    for (int j = 0; j < p.n; j++) {
        int iStart = (p.uplo == ACLBLAS_UPPER) ? 0 : j;
        int iEnd = (p.uplo == ACLBLAS_UPPER) ? j : (p.n - 1);
        for (int i = iStart; i <= iEnd; i++) {
            int64_t idx = static_cast<int64_t>(j) * p.lda + i;
            triOut.push_back(aHost[idx]);
            triGold.push_back(golden[idx]);
        }
    }

    VerifyConfig cfg;
    applyMixedTolerance(cfg, ACL_FLOAT, triGold.data(), triGold.size());
    EXPECT_TRUE(Verifier::verifyVector(triOut.data(), triGold.data(), triOut.size(), 1, cfg, p.caseName));

    uint32_t unchErr = 0;
    for (int j = 0; j < p.n; j++) {
        int iStart = (p.uplo == ACLBLAS_UPPER) ? j + 1 : 0;
        int iEnd = (p.uplo == ACLBLAS_UPPER) ? p.n - 1 : j - 1;
        for (int i = iStart; i <= iEnd; i++) {
            int64_t idx = static_cast<int64_t>(j) * p.lda + i;
            if (aHost[idx] != aOrig[idx])
                unchErr++;
        }
    }
    EXPECT_EQ(unchErr, 0u) << "[" << p.caseName << "] Unchanged region: " << unchErr << " errors";
}

// issue #510: alpha 为 Device 指针时，Host 须先经 handle stream D2H 读回再建 tiling，
// 不得直接解引用设备地址（修复前会崩溃/读到错误标量）。
TEST_F(SsyrArch35Test, DeviceAlpha)
{
    constexpr int n = 16;
    constexpr int lda = 16;
    constexpr float alpha = 2.5f;

    std::vector<float> xHost = makeBlasStrided(n, 1, "RANDOM_NORM_5_5", 42);
    std::vector<float> aOrig = makeBlasArray(static_cast<int64_t>(lda) * n, "RANDOM_NORM_5_5", 43);
    std::vector<float> aHost = aOrig;

    aclblasStatus_t ret = aclblasSsyr_npu(
        SsyrArch35Test::handle_, ACLBLAS_UPPER, n, alpha, xHost.data(), 1, aHost.data(), lda,
        /*alphaOnDevice=*/true);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_SUCCESS));
    if (ret != ACLBLAS_STATUS_SUCCESS) {
        return;
    }

    std::vector<float> golden = aOrig;
    aclblasSsyr_cpu(ACLBLAS_UPPER, n, alpha, xHost.data(), 1, golden.data(), lda);

    VerifyConfig cfg;
    applyMixedTolerance(cfg, ACL_FLOAT, golden.data(), golden.size());
    EXPECT_TRUE(Verifier::verifyVector(aHost.data(), golden.data(), aHost.size(), 1, cfg, "DeviceAlpha"));
}
