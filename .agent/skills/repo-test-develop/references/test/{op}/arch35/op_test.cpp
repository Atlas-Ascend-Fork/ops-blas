/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// TEMPLATE: GTest 入口。文件落地为 test/<family>/{{op}}/arch35/{{op}}_test.cpp
// - null handle 用 TEST_F 单独测，不下 CSV
// - 其余用例 CSV 驱动，走 5 步流程：生成数据 -> _npu 执行 -> 失败比对错误码 -> _cpu 算 golden -> Verifier 比对
// - 不写 main()，由 test/frame/test_main.cpp 统一提供
// - VerifyConfig.mode 必须显式设置（EXACT / ABS / MERE_MARE，选择见 SKILL）

#include <vector>

#include "verify.h"
#include "blas_test.h"
#include "csv_loader.h"
#include "{{op}}_param.h"
#include "{{op}}_golden.h"
#include "{{op}}_npu_wrapper.h"

class {
    {
        Op
    }
} Arch35Test : public BlasTest <
{
    {
        Op
    }
}
Param > {};

TEST_F(
    {
        {
            Op
        }
    } Arch35Test,
    NullHandle)
{
    // TEMPLATE: 按 API 传占位实参，期望返回空 handle 错误码
    aclblasStatus_t ret = aclblas
    {
        {
            Op
        }
    }
    _npu(nullptr, ACLBLAS_LOWER, 5, nullptr, nullptr, 5);
    EXPECT_EQ(ret, ACLBLAS_STATUS_NOT_INITIALIZED);
}

INSTANTIATE_TEST_SUITE_P(
    {{Op}},
    {
        {
            Op
        }
    } Arch35Test,
    ::testing::ValuesIn(GetCasesFromCsv<{
        {
            Op
        }
    } Param>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<{
        {
            Op
        }
    } Param>);

TEST_P(
    {
        {
            Op
        }
    } Arch35Test,
    CsvDriven)
{
    const auto& p = GetParam();

    // 1. 生成 host 数据（按各数组参数的 BlasFillMode + randomSeed）
    std::vector<float> apHost = makeBlasTriangular(p.n, p.uplo == ACLBLAS_UPPER, p.ap, p.randomSeed);
    std::vector<float> aHost = makeBlasArray(static_cast<int64_t>(p.lda) * p.n, p.a, p.randomSeed);

    const float* apPtr = apHost.empty() ? nullptr : apHost.data();
    float* aPtr = aHost.empty() ? nullptr : aHost.data();

    // 2. _npu 执行；3. 失败先比对错误码
    aclblasStatus_t ret = aclblas
    {
        {
            Op
        }
    }
    _npu(
        {
            {
                Op
            }
        } Arch35Test::handle_,
        p.uplo, p.n, apPtr, aPtr, p.lda);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
    if (p.expectResult != ACLBLAS_STATUS_SUCCESS)
        return;

    // 4. _cpu 算 golden
    std::vector<float> golden(aHost.size());
    aclblas
    {
        {
            Op
        }
    }
    _cpu(
        {
            {
                Op
            }
        } Arch35Test::handle_,
        p.uplo, p.n, apHost.data(), golden.data(), p.lda);

    // 5. Verifier 比对（EXACT 为格式转换类示例；浮点计算类改 ABS / MERE_MARE）
    VerifyConfig cfg;
    cfg.mode = PrecisionMode::EXACT;
    EXPECT_TRUE(Verifier::verifyVector(aPtr, golden.data(), aHost.size(), 1, cfg, p.caseName));
}
