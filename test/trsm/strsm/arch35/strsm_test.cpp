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
#include <cstdint>
#include <string>
#include <vector>

#include "verify.h"
#include "fill.h"
#include "blas_test.h"
#include "csv_loader.h"
#include "strsm_param.h"
#include "strsm_golden.h"
#include "strsm_npu_wrapper.h"

class StrsmArch35Test : public BlasTest<StrsmParam> {};

TEST_F(StrsmArch35Test, NullHandle)
{
    float alpha = 1.0f;
    aclblasStatus_t ret = aclblasStrsm_npu(
        nullptr, ACLBLAS_SIDE_LEFT, ACLBLAS_LOWER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT, 5, 5, &alpha, nullptr, 5, nullptr,
        5);
    EXPECT_EQ(ret, ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
}

TEST_F(StrsmArch35Test, NullAlpha)
{
    float alpha = 1.0f;
    aclblasStatus_t ret = aclblasStrsm_npu(
        StrsmArch35Test::handle_, ACLBLAS_SIDE_LEFT, ACLBLAS_LOWER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT, 5, 5, nullptr,
        nullptr, 5, nullptr, 5);
    EXPECT_EQ(ret, ACLBLAS_STATUS_INVALID_VALUE);
}

TEST_F(StrsmArch35Test, NullB)
{
    float alpha = 1.0f;
    auto a = makeBlasArray(25, BlasFillMode("RANDOM_NORM"), 42);
    for (int i = 0; i < 5; i++)
        a[i + i * 5] += 5.0f;
    aclblasStatus_t ret = aclblasStrsm_npu(
        StrsmArch35Test::handle_, ACLBLAS_SIDE_LEFT, ACLBLAS_LOWER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT, 5, 5, &alpha,
        a.data(), 5, nullptr, 5);
    EXPECT_EQ(ret, ACLBLAS_STATUS_INVALID_VALUE);
}

INSTANTIATE_TEST_SUITE_P(
    Strsm, StrsmArch35Test, ::testing::ValuesIn(GetCasesFromCsv<StrsmParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<StrsmParam>);

TEST_P(StrsmArch35Test, CsvDriven)
{
    const auto& p = GetParam();
    const int aDim = (p.side == ACLBLAS_SIDE_LEFT) ? p.m : p.n;

    std::vector<float> aHost = makeBlasArray(static_cast<int64_t>(std::max(aDim, p.lda)) * aDim, p.aFill, p.randomSeed);
    if (!aHost.empty() && aDim > 0) {
        bool isUpper = (p.uplo == ACLBLAS_UPPER);
        for (int j = 0; j < aDim; j++) {
            for (int i = 0; i < aDim; i++) {
                if (isUpper ? (i > j) : (i < j))
                    aHost[i + j * p.lda] = 0.0f;
            }
        }
        if (p.aFill.pattern != BlasFillMode::P_ILLCOND) {
            float boost = std::max(5.0f, static_cast<float>(aDim));
            for (int i = 0; i < aDim; i++) {
                float& diag = aHost[i + i * p.lda];
                diag += (diag >= 0.0f ? boost : -boost);
            }
        }
    }
    std::vector<float> bHost =
        makeBlasArray(static_cast<int64_t>(std::max(p.m, p.ldb)) * p.n, p.bFill, p.randomSeed + 1);
    std::vector<float> bResult = bHost;

    aclblasStatus_t ret = aclblasStrsm_npu(
        StrsmArch35Test::handle_, p.side, p.uplo, p.trans, p.diag, p.m, p.n, &p.alpha,
        aHost.empty() ? nullptr : aHost.data(), p.lda, bResult.empty() ? nullptr : bResult.data(), p.ldb);

    if (p.expectResult != ACLBLAS_STATUS_SUCCESS) {
        EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
        return;
    }
    ASSERT_EQ(ret, ACLBLAS_STATUS_SUCCESS);

    std::vector<float> bGolden = bHost;
    // issue #820: golden 返回码必须断言检查，golden 失败时不得以输入初值做参考比对
    ASSERT_EQ(
        aclblasStrsm_cpu(
            StrsmArch35Test::handle_, p.side, p.uplo, p.trans, p.diag, p.m, p.n, &p.alpha,
            aHost.empty() ? nullptr : aHost.data(), p.lda, bGolden.empty() ? nullptr : bGolden.data(), p.ldb),
        ACLBLAS_STATUS_SUCCESS);

    const size_t bCount = static_cast<size_t>(p.ldb) * static_cast<size_t>(p.n);
    if (bCount == 0)
        return;

    VerifyConfig cfg;
    cfg.mode = PrecisionMode::MERE_MARE;
    cfg.mereThreshold = p.mereThreshold;
    cfg.mareMultiplier = p.mareMultiplier;
    EXPECT_TRUE(Verifier::verifyVector(bResult.data(), bGolden.data(), bCount, 1, cfg, p.caseName));
}
