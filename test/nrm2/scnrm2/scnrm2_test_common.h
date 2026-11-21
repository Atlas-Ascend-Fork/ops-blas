/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Shared gtest helpers for the arch22/arch35 scnrm2 CSV-driven tests. Both
// architecture test files include this header instead of keeping per-arch
// copies, so the duplicate-code scan sees a single definition.
#pragma once

#include <string>
#include <vector>

#include "scnrm2_golden.h"
#include "scnrm2_npu_wrapper.h"
#include "scnrm2_param.h"

inline int64_t Scnrm2BufferComplexCount(int64_t n, int64_t incx)
{
    if (n <= 0) {
        return 0;
    }
    int64_t absInc = (incx > 0) ? incx : -incx;
    if (absInc == 0) {
        absInc = 1;
    }
    return 1 + (n - 1) * absInc;
}

inline std::vector<aclblasComplex> MakeComplexInput(const Scnrm2Param& p)
{
    int64_t complexCount = Scnrm2BufferComplexCount(p.n, p.incx);
    if (p.x.method == BlasFillMode::M_NULLPTR || complexCount <= 0) {
        return {};
    }
    std::vector<float> raw = makeBlasArray(complexCount * 2, p.x, p.randomSeed);
    std::vector<aclblasComplex> data(static_cast<size_t>(complexCount));
    for (int64_t i = 0; i < complexCount; ++i) {
        data[static_cast<size_t>(i)] =
            aclblasComplex{raw[static_cast<size_t>(i * 2)], raw[static_cast<size_t>(i * 2 + 1)]};
    }
    return data;
}

inline void VerifyScnrm2Result(float result, float golden, const std::string& caseName)
{
    VerifyConfig cfg;
    applyMixedTolerance(cfg, ACL_FLOAT, golden);
    EXPECT_TRUE(Verifier::verifyScalar(result, golden, cfg, caseName));
}

// ---------------------------------------------------------------------------
// Error path test (expectResult != SUCCESS), mirrors TestSnrm2ErrorPath
// ---------------------------------------------------------------------------
inline void TestScnrm2ErrorPath(const Scnrm2Param& p, aclblasHandle_t handle)
{
    std::vector<aclblasComplex> xHost = MakeComplexInput(p);
    const aclblasComplex* xPtr = xHost.empty() ? nullptr : xHost.data();

    float result = 0.0f;
    float* resultPtr = p.resultIsNull ? nullptr : &result;

    aclblasStatus_t ret = aclblasScnrm2_npu(handle, p.n, xPtr, p.incx, resultPtr);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
}

// ---------------------------------------------------------------------------
// No-op path test (n <= 0 quick return writes 0.0f)
// ---------------------------------------------------------------------------
inline void TestScnrm2NoOpPath(const Scnrm2Param& p, aclblasHandle_t handle)
{
    std::vector<aclblasComplex> xHost = MakeComplexInput(p);
    const aclblasComplex* xPtr = xHost.empty() ? nullptr : xHost.data();

    float result = 123.0f;
    aclblasStatus_t ret = aclblasScnrm2_npu(handle, p.n, xPtr, p.incx, &result);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
    if (ret == ACLBLAS_STATUS_SUCCESS) {
        EXPECT_FLOAT_EQ(result, 0.0f) << "[" << p.caseName << "] n<=0 quick return should write 0.0f, got " << result;
    }
}

inline void TestScnrm2NormalPath(const Scnrm2Param& p, aclblasHandle_t handle)
{
    std::vector<aclblasComplex> xHost = MakeComplexInput(p);
    float result = 0.0f;
    aclblasStatus_t ret = aclblasScnrm2_npu(handle, p.n, xHost.data(), p.incx, &result);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
    if (ret != ACLBLAS_STATUS_SUCCESS) {
        return;
    }

    float golden = 0.0f;
    // issue #820: golden 返回码必须断言检查，golden 失败时不得以初始值 0 做参考比对
    ASSERT_EQ(aclblasScnrm2_cpu(handle, p.n, xHost.data(), p.incx, &golden), ACLBLAS_STATUS_SUCCESS);

    VerifyScnrm2Result(result, golden, p.caseName);
}
