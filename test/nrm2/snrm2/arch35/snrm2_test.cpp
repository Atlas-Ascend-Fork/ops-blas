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
#include <cstdint>
#include <vector>

#include "verify.h"
#include "blas_test.h"
#include "csv_loader.h"
#include "snrm2_test_common.h"

class Snrm2Arch35Test : public BlasTest<Snrm2Param> {};

TEST_F(Snrm2Arch35Test, NullHandle)
{
    float result = 0.0f;
    aclblasStatus_t ret = aclblasSnrm2_npu(nullptr, 5, nullptr, 1, &result);
    EXPECT_EQ(ret, ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
}

INSTANTIATE_TEST_SUITE_P(
    Snrm2, Snrm2Arch35Test, ::testing::ValuesIn(GetCasesFromCsv<Snrm2Param>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<Snrm2Param>);

// ---------------------------------------------------------------------------
// Main parameterized test
// ---------------------------------------------------------------------------
TEST_P(Snrm2Arch35Test, CsvDriven)
{
    const auto& p = GetParam();

    if (p.expectResult != ACLBLAS_STATUS_SUCCESS) {
        TestSnrm2ErrorPath(p, Snrm2Arch35Test::handle_);
    } else if (p.n <= 0 || p.incx <= 0) {
        TestSnrm2NoOpPath(p, Snrm2Arch35Test::handle_);
    } else {
        TestSnrm2NormalPath(p, Snrm2Arch35Test::handle_);
    }
}
