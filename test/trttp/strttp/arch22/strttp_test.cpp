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
#include "strttp_param.h"
#include "strttp_golden.h"
#include "strttp_npu_wrapper.h"

class StrttpTest : public BlasTest<StrttpParam> {};

TEST_F(StrttpTest, NullHandle)
{
    const aclblasStatus_t ret = aclblasStrttp_npu(nullptr, ACLBLAS_LOWER, 5, nullptr, 5, nullptr);
    EXPECT_EQ(ret, ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
}

INSTANTIATE_TEST_SUITE_P(
    Strttp, StrttpTest, ::testing::ValuesIn(GetCasesFromCsv<StrttpParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<StrttpParam>);

TEST_P(StrttpTest, CsvDriven)
{
    const auto& p = GetParam();

    std::vector<float> aHost = makeBlasArray(static_cast<int64_t>(p.lda) * p.n, p.a, p.randomSeed);
    std::vector<float> apHost = makeBlasArray(static_cast<int64_t>(p.n) * (p.n + 1) / 2, p.ap);

    const float* aPtr = aHost.empty() ? nullptr : aHost.data();
    float* apPtr = apHost.empty() ? nullptr : apHost.data();

    const aclblasStatus_t ret = aclblasStrttp_npu(StrttpTest::handle_, p.uplo, p.n, aPtr, p.lda, apPtr);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
    if (p.expectResult != ACLBLAS_STATUS_SUCCESS) {
        return;
    }

    std::vector<float> golden(apHost.size());
    aclblasStrttp_cpu(StrttpTest::handle_, p.uplo, p.n, aHost.data(), p.lda, golden.data());

    VerifyConfig cfg;
    cfg.mode = PrecisionMode::EXACT;
    EXPECT_TRUE(Verifier::verifyVector(apPtr, golden.data(), apHost.size(), 1, cfg, p.caseName));
}
