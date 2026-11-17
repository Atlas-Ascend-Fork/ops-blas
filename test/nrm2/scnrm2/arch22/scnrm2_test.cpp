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
#include <string>
#include <vector>

#include "verify.h"
#include "blas_test.h"
#include "csv_loader.h"
#include "scnrm2_test_common.h"

class Scnrm2Test : public BlasTest<Scnrm2Param> {};

INSTANTIATE_TEST_SUITE_P(
    Scnrm2, Scnrm2Test, ::testing::ValuesIn(GetCasesFromCsv<Scnrm2Param>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<Scnrm2Param>);

TEST_P(Scnrm2Test, CsvDriven)
{
    const auto& p = GetParam();
    ASSERT_EQ(p.expectResult, ACLBLAS_STATUS_SUCCESS);
    ASSERT_GT(p.n, 0);
    TestScnrm2NormalPath(p, Scnrm2Test::handle_);
}

// Coverage migrated from the former standalone main() binary in snrm2_test.cpp (issue #920):
// the complex norm case keeps its original buffer, golden and verdict, and now lives in a
// scnrm2-named file testing the aclblasScnrm2 interface.
TEST_F(Scnrm2Test, UniformComplexVector)
{
    constexpr int64_t n = 8 * 1024;
    constexpr float realVal = 1.5f;
    constexpr float imagVal = 2.0f;
    constexpr int64_t incx = 1;

    std::vector<aclblasComplex> xHost(static_cast<size_t>(n), aclblasComplex{realVal, imagVal});
    float result = 0.0f;
    ASSERT_EQ(aclblasScnrm2_npu(handle_, n, xHost.data(), incx, &result), ACLBLAS_STATUS_SUCCESS);
    VerifyScnrm2Result(
        result, std::sqrt(static_cast<float>(n) * (realVal * realVal + imagVal * imagVal)), "UniformComplexVector");
}
