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
#include <string>
#include <vector>

#include "fill.h"
#include "verify.h"
#include "blas_test.h"
#include "csv_loader.h"
#include "sswap_param.h"
#include "sswap_golden.h"
#include "sswap_npu_wrapper.h"

class SswapArch35Test : public BlasTest<SswapParam> {};

// Null handle test — separate TEST_F (not in CSV)
TEST_F(SswapArch35Test, NullHandle)
{
    aclblasStatus_t ret = aclblasSswap_npu(nullptr, 5, nullptr, 1, nullptr, 1);
    EXPECT_EQ(ret, ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
}

INSTANTIATE_TEST_SUITE_P(
    Sswap, SswapArch35Test, ::testing::ValuesIn(GetCasesFromCsv<SswapParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<SswapParam>);

// issue #1327 负例特例：incx/incy = INT32_MIN 的 host 校验短路验证。绕过 npu 包装
// （存储按 (n-1)*|inc|+1 分配，该步长下不可达），以最小哑设备缓冲直调公开接口；
// 期望值写死 INVALID_VALUE，保证该校验被移除时用例失败
static void RunSswapInt32MinCase(aclblasHandle_t handle, const SswapParam& p, bool incxIsMin)
{
    const size_t dummyBytes = 8 * sizeof(float);
    void* dX = nullptr;
    void* dY = nullptr;
    ASSERT_EQ(aclrtMalloc(&dX, dummyBytes, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    ASSERT_EQ(aclrtMalloc(&dY, dummyBytes, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    aclblasStatus_t ret = aclblasSswap(
        handle, p.n, static_cast<float*>(dX), incxIsMin ? INT32_MIN : p.incx, static_cast<float*>(dY),
        incxIsMin ? p.incy : INT32_MIN);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE));
    aclrtFree(dX);
    aclrtFree(dY);
}

TEST_P(SswapArch35Test, CsvDriven)
{
    const auto& p = GetParam();

    if (p.incx == INT32_MIN || p.incy == INT32_MIN) {
        RunSswapInt32MinCase(SswapArch35Test::handle_, p, p.incx == INT32_MIN);
        return;
    }

    // Step 1: Generate host data
    std::vector<float> xHost = makeBlasStrided(p.n, p.incx, p.x, p.randomSeed);
    std::vector<float> yHost = makeBlasStrided(p.n, p.incy, p.y, p.randomSeed);

    float* xPtr = xHost.empty() ? nullptr : xHost.data();
    float* yPtr = yHost.empty() ? nullptr : yHost.data();

    // Step 2: Execute on NPU
    aclblasStatus_t ret = aclblasSswap_npu(SswapArch35Test::handle_, p.n, xPtr, p.incx, yPtr, p.incy);

    // Step 3: Check return code
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
    if (p.expectResult != ACLBLAS_STATUS_SUCCESS)
        return;

    // Step 4: Skip data verification for n <= 0 (no data to verify)
    if (p.n <= 0)
        return;

    // Step 5: Compute CPU golden (swap modifies both x and y in-place)
    // We need copies of the original data for golden computation
    std::vector<float> goldenX = makeBlasStrided(p.n, p.incx, p.x, p.randomSeed);
    std::vector<float> goldenY = makeBlasStrided(p.n, p.incy, p.y, p.randomSeed);
    aclblasSswap_cpu(SswapArch35Test::handle_, p.n, goldenX.data(), p.incx, goldenY.data(), p.incy);

    // Step 6: Verify both outputs with EXACT precision
    VerifyConfig cfg;
    cfg.mode = PrecisionMode::EXACT;
    EXPECT_TRUE(Verifier::verifyVector(xPtr, goldenX.data(), xHost.size(), 1, cfg, p.caseName + "_x"));
    EXPECT_TRUE(Verifier::verifyVector(yPtr, goldenY.data(), yHost.size(), 1, cfg, p.caseName + "_y"));
}
