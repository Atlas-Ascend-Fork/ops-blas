/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Shared gtest helpers for the arch22/arch35 snrm2 CSV-driven tests. Both
// architecture test files include this header instead of keeping per-arch
// copies, so the duplicate-code scan sees a single definition.
#pragma once

#include <string>
#include <vector>

#include "snrm2_golden.h"
#include "snrm2_npu_wrapper.h"
#include "snrm2_param.h"

// ---------------------------------------------------------------------------
// Error path test (expectResult != SUCCESS)
// ---------------------------------------------------------------------------
inline void TestSnrm2ErrorPath(const Snrm2Param& p, aclblasHandle_t handle)
{
    const int64_t xLen = (p.n > 0 && p.x.method != BlasFillMode::M_NULLPTR) ? p.n : 0;
    std::vector<float> xHost = makeBlasArray(xLen, p.x, p.randomSeed);
    const float* xPtr = xHost.empty() ? nullptr : xHost.data();

    float result = 0.0f;
    float* resultPtr = p.resultIsNull ? nullptr : &result;

    aclblasStatus_t ret = aclblasSnrm2_npu(handle, p.n, xPtr, p.incx, resultPtr);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
}

// ---------------------------------------------------------------------------
// No-op path test (n<=0 or incx<=0)
// ---------------------------------------------------------------------------
inline void TestSnrm2NoOpPath(const Snrm2Param& p, aclblasHandle_t handle)
{
    const int64_t xLen = (p.n > 0) ? p.n : 0;
    std::vector<float> xHost = makeBlasArray(xLen, p.x, p.randomSeed);
    const float* xPtr = xHost.empty() ? nullptr : xHost.data();

    float result = 123.0f;
    aclblasStatus_t ret = aclblasSnrm2_npu(handle, p.n, xPtr, p.incx, &result);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
    if (ret == ACLBLAS_STATUS_SUCCESS) {
        EXPECT_FLOAT_EQ(result, 0.0f) << "[" << p.caseName << "] early return should produce result=0.0f, got "
                                      << result;
    }
}

// ---------------------------------------------------------------------------
// Precision verification helper
// ---------------------------------------------------------------------------
inline void VerifySnrm2NormResult(float result, float golden, const std::string& caseName)
{
    VerifyConfig cfg;
    applyMixedTolerance(cfg, ACL_FLOAT, golden);
    EXPECT_TRUE(Verifier::verifyScalar(result, golden, cfg, caseName));
}

// ---------------------------------------------------------------------------
// Normal path test helper
// ---------------------------------------------------------------------------
inline void TestSnrm2NormalPath(const Snrm2Param& p, aclblasHandle_t handle)
{
    const int64_t xLen = (p.n > 0) ? 1 + (p.n - 1) * p.incx : 0;
    std::vector<float> xHost = makeBlasArray(xLen, p.x, p.randomSeed);

    float result = 0.0f;
    aclblasStatus_t ret = aclblasSnrm2_npu(handle, p.n, xHost.data(), p.incx, &result);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
    if (ret != ACLBLAS_STATUS_SUCCESS) {
        return;
    }

    float golden = 0.0f;
    // issue #820: golden 返回码必须断言检查，golden 失败时不得以初始值 0 做参考比对
    ASSERT_EQ(aclblasSnrm2_cpu(handle, p.n, xHost.data(), p.incx, &golden), ACLBLAS_STATUS_SUCCESS);

    VerifySnrm2NormResult(result, golden, p.caseName);
}
