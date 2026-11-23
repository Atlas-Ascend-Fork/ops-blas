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
#include "stpttr_param.h"
#include "stpttr_golden.h"
#include "stpttr_npu_wrapper.h"

class StpttrTest : public BlasTest<StpttrParam> {};

TEST_F(StpttrTest, NullHandle)
{
    const aclblasStatus_t ret = aclblasStpttr_npu(nullptr, ACLBLAS_LOWER, 5, nullptr, nullptr, 5);
    EXPECT_EQ(ret, ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
}

INSTANTIATE_TEST_SUITE_P(
    Stpttr, StpttrTest, ::testing::ValuesIn(GetCasesFromCsv<StpttrParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<StpttrParam>);

TEST_P(StpttrTest, CsvDriven)
{
    const auto& p = GetParam();

    std::vector<float> apHost = makeBlasTriangular(p.n, p.uplo == ACLBLAS_UPPER, p.ap, p.randomSeed);
    std::vector<float> aHost = makeBlasArray(static_cast<int64_t>(p.lda) * p.n, p.a, p.randomSeed);

    const float* apPtr = apHost.empty() ? nullptr : apHost.data();
    float* aPtr = aHost.empty() ? nullptr : aHost.data();

    const aclblasStatus_t ret = aclblasStpttr_npu(StpttrTest::handle_, p.uplo, p.n, apPtr, aPtr, p.lda);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
    if (p.expectResult != ACLBLAS_STATUS_SUCCESS) {
        return;
    }

    std::vector<float> golden(aHost.size());
    aclblasStpttr_cpu(StpttrTest::handle_, p.uplo, p.n, apHost.data(), golden.data(), p.lda);

    VerifyConfig cfg;
    cfg.mode = PrecisionMode::EXACT;
    EXPECT_TRUE(Verifier::verifyVector(aPtr, golden.data(), aHost.size(), 1, cfg, p.caseName));
}
