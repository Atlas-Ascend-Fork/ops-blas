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
#include <climits>
#include <cmath>
#include <random>
#include <vector>

#include "verify.h"
#include "blas_test.h"
#include "csv_loader.h"
#include "fill.h"
#include "stbmv_param.h"
#include "stbmv_golden.h"
#include "stbmv_npu_wrapper.h"


class StbmvArch35Test : public BlasTest<StbmvParam> { };

TEST_F(StbmvArch35Test, RejectsMaxKBeforeKernelLaunch)
{
    float a = 1.0F;
    float x = 1.0F;
    const aclblasStatus_t ret = aclblasStbmv_npu(
        StbmvArch35Test::handle_, ACLBLAS_LOWER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT,
        1, INT_MAX, &a, 1, &x, 1);
    EXPECT_EQ(ret, ACLBLAS_STATUS_INVALID_VALUE);
}

TEST_F(StbmvArch35Test, RejectsInvalidEnums)
{
    float a = 1.0F;
    float x = 1.0F;
    const auto invalidFill = static_cast<aclblasFillMode_t>(0);
    const auto invalidTrans = static_cast<aclblasOperation_t>(0);
    const auto invalidDiag = static_cast<aclblasDiagType_t>(0);

    EXPECT_EQ(
        aclblasStbmv(StbmvArch35Test::handle_, invalidFill, ACLBLAS_OP_N, ACLBLAS_NON_UNIT, 1, 0, &a, 1, &x, 1),
        ACLBLAS_STATUS_INVALID_ENUM);
    EXPECT_EQ(
        aclblasStbmv(StbmvArch35Test::handle_, ACLBLAS_LOWER, invalidTrans, ACLBLAS_NON_UNIT, 1, 0, &a, 1, &x, 1),
        ACLBLAS_STATUS_INVALID_ENUM);
    EXPECT_EQ(
        aclblasStbmv(StbmvArch35Test::handle_, ACLBLAS_LOWER, ACLBLAS_OP_N, invalidDiag, 1, 0, &a, 1, &x, 1),
        ACLBLAS_STATUS_INVALID_ENUM);
}

INSTANTIATE_TEST_SUITE_P(
    Stbmv, StbmvArch35Test,
    ::testing::ValuesIn(GetCasesFromCsv<StbmvParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<StbmvParam>);

TEST_P(StbmvArch35Test, CsvDriven) {
    const auto& p = GetParam();

    int kl = (p.uplo == ACLBLAS_UPPER) ? 0 : p.k;
    int ku = (p.uplo == ACLBLAS_UPPER) ? p.k : 0;

    auto a = makeBlasBanded(p.n, p.n, kl, ku, p.lda, "RANDOM_0_1", p.randomSeed);
    auto x = makeBlasStrided(p.n, p.incx, "RANDOM_0_1", p.randomSeed + 1);
    std::vector<float> golden = x;

    aclblasStatus_t ret = aclblasStbmv_npu(
        StbmvArch35Test::handle_, p.uplo, p.trans, p.diag, p.n, p.k,
        a.data(), p.lda, x.data(), p.incx);

    if (p.expectResult != ACLBLAS_STATUS_SUCCESS) {
        EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
        return;
    }
    ASSERT_EQ(ret, ACLBLAS_STATUS_SUCCESS);

    aclblasStbmv_cpu(
        StbmvArch35Test::handle_, p.uplo, p.trans, p.diag, p.n, p.k,
        a.data(), p.lda, golden.data(), p.incx);

    const int absIncx = std::abs(p.incx);
    VerifyConfig cfg;
    const float* outPtr  = (p.incx < 0 && p.n > 0) ? x.data() + (p.n - 1) * absIncx : x.data();
    const float* goldPtr = (p.incx < 0 && p.n > 0) ? golden.data() + (p.n - 1) * absIncx : golden.data();
    applyMixedTolerance(cfg, ACL_FLOAT, goldPtr, static_cast<size_t>(p.n));

    EXPECT_TRUE(Verifier::verifyVector(outPtr, goldPtr,
        static_cast<size_t>(p.n), (p.incx < 0) ? -absIncx : absIncx, cfg, p.caseName));
}
