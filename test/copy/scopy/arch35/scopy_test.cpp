/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

#include "fill.h"
#include "verify.h"
#include "blas_test.h"
#include "csv_loader.h"
#include "scopy_param.h"
#include "scopy_golden.h"
#include "scopy_npu_wrapper.h"

class ScopyArch35Test : public BlasTest<ScopyParam> {};

// Null handle test — separate TEST_F (not in CSV)
TEST_F(ScopyArch35Test, NullHandle)
{
    aclblasStatus_t ret = aclblasScopy_npu(nullptr, 5, nullptr, 1, nullptr, 1);
    EXPECT_EQ(ret, ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
}

INSTANTIATE_TEST_SUITE_P(
    Scopy, ScopyArch35Test, ::testing::ValuesIn(GetCasesFromCsv<ScopyParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<ScopyParam>);

TEST_P(ScopyArch35Test, CsvDriven)
{
    const auto& p = GetParam();

    // Step 1: Compute buffer sizes for strided access
    // For a vector with stride incx, elements accessed at positions 0, incx, 2*incx, ..., (n-1)*incx
    // The span of memory covered is abs(incx) * (n - 1) + 1 elements
    const int64_t xSpan = (p.n > 0) ? static_cast<int64_t>(std::abs(p.incx)) * (p.n - 1) + 1 : 1;
    const int64_t ySpan = (p.n > 0) ? static_cast<int64_t>(std::abs(p.incy)) * (p.n - 1) + 1 : 1;
    const int64_t xAlloc = xSpan + std::abs(p.xAlignOffset);
    const int64_t yAlloc = ySpan + std::abs(p.yAlignOffset);

    // Step 2: Generate host data with full buffer coverage
    std::vector<float> xHost = makeBlasArray(xAlloc, p.x, p.randomSeed);
    std::vector<float> yHost = makeBlasArray(yAlloc, p.y, p.randomSeed);

    const float* xPtr = xHost.empty() ? nullptr : xHost.data() + p.xAlignOffset;
    float* yPtr = yHost.empty() ? nullptr : yHost.data() + p.yAlignOffset;

    // Step 3: Execute on NPU
    aclblasStatus_t ret = aclblasScopy_npu(ScopyArch35Test::handle_, p.n, xPtr, p.incx, yPtr, p.incy);

    // Step 4: Check return code
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
    if (p.expectResult != ACLBLAS_STATUS_SUCCESS)
        return;

    // Step 5: Skip data verification for n <= 0 (no data to verify)
    if (p.n <= 0)
        return;

    // Step 6: Compute CPU golden with same allocation strategy
    std::vector<float> goldenX = makeBlasArray(xAlloc, p.x, p.randomSeed);
    std::vector<float> goldenY = makeBlasArray(yAlloc, p.y, p.randomSeed);
    // issue #820: golden 返回码必须断言检查，golden 失败时不得以输入初值做参考比对
    ASSERT_EQ(
        aclblasScopy_cpu(
            ScopyArch35Test::handle_, p.n, goldenX.data() + p.xAlignOffset, p.incx, goldenY.data() + p.yAlignOffset,
            p.incy),
        ACLBLAS_STATUS_SUCCESS);

    // Step 7: Verify y output with EXACT precision (scopy is bit-exact data copy)
    // Compare the full strided span of y (including gaps between non-contiguous writes)
    VerifyConfig cfg;
    cfg.mode = PrecisionMode::EXACT;
    EXPECT_TRUE(Verifier::verifyVector(
        yHost.data() + p.yAlignOffset, goldenY.data() + p.yAlignOffset, static_cast<size_t>(ySpan), 1, cfg,
        p.caseName));
}
