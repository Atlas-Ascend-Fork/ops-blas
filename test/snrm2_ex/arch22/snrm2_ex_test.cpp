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
#include "dtype_utils.h"
#include "snrm2_ex_param.h"
#include "snrm2_ex_golden.h"
#include "snrm2_ex_npu_wrapper.h"

class Snrm2ExTest : public BlasTest<Snrm2ExParam> {};

TEST_F(Snrm2ExTest, NullHandle)
{
    std::vector<float> dummy(4, 0.0f);
    float result = 0.0f;
    const aclblasStatus_t ret = aclblasSnrm2Ex_npu(nullptr, ACL_FLOAT, dummy.data(), 4, 1, &result);
    EXPECT_EQ(ret, ACLBLAS_STATUS_NOT_INITIALIZED);
}

TEST_F(Snrm2ExTest, NullResult)
{
    std::vector<float> dummy(4, 0.0f);
    const aclblasStatus_t ret = aclblasSnrm2Ex_npu(Snrm2ExTest::handle_, ACL_FLOAT, dummy.data(), 4, 1, nullptr);
    EXPECT_EQ(ret, ACLBLAS_STATUS_INVALID_VALUE);
}

INSTANTIATE_TEST_SUITE_P(
    Snrm2Ex, Snrm2ExTest, ::testing::ValuesIn(GetCasesFromCsv<Snrm2ExParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<Snrm2ExParam>);

TEST_P(Snrm2ExTest, CsvDriven)
{
    const auto& p = GetParam();
    std::vector<float> xFloat = makeBlasStrided(static_cast<int>(p.n), static_cast<int>(p.incx), p.x, p.randomSeed);
    std::vector<uint8_t> xBytes;
    const void* xPtr = nullptr;
    if (!xFloat.empty()) {
        xBytes = quantizeToBytes(xFloat, p.xtype);
        xPtr = xBytes.data();
    }

    float result = 0.0f;
    const aclblasStatus_t ret = aclblasSnrm2Ex_npu(Snrm2ExTest::handle_, p.xtype, xPtr, p.n, p.incx, &result);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult)) << p.caseName;
    if (p.expectResult != ACLBLAS_STATUS_SUCCESS) {
        return;
    }
    if (p.n == 0) {
        EXPECT_FLOAT_EQ(result, 0.0f) << p.caseName;
        return;
    }

    const float golden = snrm2_ex_cpu(p.xtype, xPtr, p.n, p.incx);
    VerifyConfig cfg;
    const aclDataType scalarDtype = (p.xtype == ACL_FLOAT16) ? ACL_FLOAT16 : ACL_FLOAT;
    applyMixedTolerance(cfg, scalarDtype, golden);
    EXPECT_TRUE(Verifier::verifyScalar(result, golden, cfg, p.caseName));
}
