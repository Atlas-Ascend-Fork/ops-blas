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
#include "sspr_param.h"
#include "sspr_golden.h"
#include "sspr_npu_wrapper.h"

class SsprArch22Test : public BlasTest<SsprParam> {};

TEST_F(SsprArch22Test, NullHandle)
{
    float alphaVal = 1.0f;
    aclblasStatus_t ret = aclblasSspr(nullptr, ACLBLAS_UPPER, 4, &alphaVal, nullptr, 1, nullptr);
    EXPECT_EQ(ret, ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
}

INSTANTIATE_TEST_SUITE_P(
    Sspr, SsprArch22Test, ::testing::ValuesIn(GetCasesFromCsv<SsprParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<SsprParam>);

TEST_P(SsprArch22Test, CsvDriven)
{
    const auto& p = GetParam();

    size_t apLen = (p.n > 0) ? static_cast<size_t>(p.n) * (p.n + 1) / 2 : 0;

    // Generate x vector (strided, supports negative incx)
    std::vector<float> xHost = makeBlasStrided(p.n, p.incx, p.x, p.randomSeed);

    // Generate packed AP (triangular / random)
    // When ap uses UPPER/LOWER pattern in its BlasFillMode, makeBlasPacked preserves it.
    // Otherwise, use the uplo to select triangular layout.
    bool useUpper = (p.uplo == ACLBLAS_UPPER);
    std::vector<float> apHost;
    if (p.ap.pattern == BlasFillMode::P_UPPER || p.ap.pattern == BlasFillMode::P_LOWER) {
        apHost = makeBlasPacked(p.n, p.ap, p.randomSeed);
    } else {
        apHost = makeBlasTriangular(p.n, useUpper, p.ap, p.randomSeed);
    }

    // Copy apHost for alpha=0 exact verification
    std::vector<float> apOrig = apHost;

    const float* xPtr = xHost.empty() ? nullptr : xHost.data();
    float* apPtr = apHost.empty() ? nullptr : apHost.data();
    const float* alphaPtr = p.alphaNull ? nullptr : &p.alpha;

    aclblasStatus_t ret = aclblasSspr_npu(SsprArch22Test::handle_, p.uplo, p.n, alphaPtr, xPtr, p.incx, apPtr);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
    if (p.expectResult != ACLBLAS_STATUS_SUCCESS) {
        return;
    }
    if (p.n == 0) {
        return;
    }

    // Compute CPU golden
    std::vector<float> golden = apOrig;
    aclblasSspr_cpu(SsprArch22Test::handle_, p.uplo, p.n, &p.alpha, xHost.data(), p.incx, golden.data());

    VerifyConfig cfg;
    if (p.alpha == 0.0f && !p.alphaNull) {
        cfg.mode = PrecisionMode::EXACT;
        EXPECT_TRUE(Verifier::verifyVector(apPtr, golden.data(), apLen, 1, cfg, p.caseName));
    } else {
        applyMixedTolerance(cfg, ACL_FLOAT, golden.data(), apLen);
        EXPECT_TRUE(Verifier::verifyVector(apPtr, golden.data(), apLen, 1, cfg, p.caseName));
    }
}
