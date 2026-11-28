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

#include "SdotArch35Test.h"
#include "verify.h"
#include "blas_test.h"
#include "csv_loader.h"
#include "sdot_param.h"
#include "sdot_golden.h"
#include "sdot_npu_wrapper.h"

INSTANTIATE_TEST_SUITE_P(
    Sdot, SdotArch35Test, ::testing::ValuesIn(GetCasesFromCsv<SdotParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<SdotParam>);

TEST_P(SdotArch35Test, CsvDriven)
{
    const auto& p = GetParam();

    // issue #1325 负例特例：incx/incy = INT32_MIN 的 host 校验短路验证。绕过 npu 包装
    // （设备缓冲按 (n-1)*|inc|+1 分配且 std::abs 对 INT32_MIN 为 UB，该步长下不可达），
    // 以最小哑设备缓冲直调公开接口；期望值写死 INVALID_VALUE，保证该校验被移除时用例失败
    if (p.incx == INT32_MIN || p.incy == INT32_MIN) {
        const size_t dummyBytes = 8 * sizeof(float);
        void* dX = nullptr;
        void* dY = nullptr;
        void* dResult = nullptr;
        ASSERT_EQ(aclrtMalloc(&dX, dummyBytes, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
        ASSERT_EQ(aclrtMalloc(&dY, dummyBytes, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
        ASSERT_EQ(aclrtMalloc(&dResult, sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
        aclblasStatus_t minRet = aclblasSdot(
            handle_, 8, static_cast<const float*>(dX), p.incx, static_cast<const float*>(dY), p.incy,
            static_cast<float*>(dResult));
        EXPECT_EQ(static_cast<int>(minRet), static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE));
        aclrtFree(dX);
        aclrtFree(dY);
        aclrtFree(dResult);
        return;
    }

    std::vector<float> xHost = makeBlasStrided(p.n, p.incx, p.x, p.randomSeed);
    std::vector<float> yHost = makeBlasStrided(p.n, p.incy, p.y, p.randomSeed);

    const float* xPtr = xHost.empty() ? nullptr : xHost.data();
    const float* yPtr = yHost.empty() ? nullptr : yHost.data();
    float result = 0.0f;

    aclblasStatus_t ret = aclblasSdot_npu(SdotArch35Test::handle_, p.n, xPtr, p.incx, yPtr, p.incy, &result);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
    if (p.expectResult != ACLBLAS_STATUS_SUCCESS)
        return;

    float golden = aclblasSdot_cpu(p.n, xPtr, p.incx, yPtr, p.incy);

    VerifyConfig cfg;
    applyMixedTolerance(cfg, ACL_FLOAT, golden);
    EXPECT_TRUE(Verifier::verifyScalar(result, golden, cfg, p.caseName));
}
