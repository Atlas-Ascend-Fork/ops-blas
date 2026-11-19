/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdio>
#include <vector>

#include "verify.h"
#include "blas_test.h"
#include "csv_loader.h"
#include "fill.h"
#include "chpr_param.h"
#include "chpr_golden.h"
#include "chpr_npu_wrapper.h"

class ChprArch22Test : public BlasTest<ChprParam> {};

static std::vector<aclblasComplex> PackComplex(const std::vector<float>& src)
{
    size_t n = src.size() / 2;
    std::vector<aclblasComplex> dst(n);
    for (size_t i = 0; i < n; ++i) {
        dst[i].real = src[2 * i];
        dst[i].imag = src[2 * i + 1];
    }
    return dst;
}

TEST_F(ChprArch22Test, NullHandle)
{
    float alphaVal = 1.0f;
    aclblasStatus_t ret = aclblasChpr(nullptr, ACLBLAS_UPPER, 4, &alphaVal, nullptr, 1, nullptr);
    EXPECT_EQ(ret, ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
}

// issue #965: alpha 为 Device 指针时经 D2H 读回再建 tiling，不得在 Host 直接解引用设备地址
TEST_F(ChprArch22Test, DeviceAlpha)
{
    constexpr int n = 8;
    constexpr float alpha = 2.5f;

    std::vector<aclblasComplex> xHost = PackComplex(makeBlasArray(n * 2, "RANDOM_2_2", 42));
    std::vector<aclblasComplex> apOrig = PackComplex(makeBlasArray(n * (n + 1), "RANDOM_2_2", 43));
    std::vector<aclblasComplex> apHost = apOrig;

    // alphaOnDevice = true：alpha 按设备指针传入（包装层同名形参，issue #965 覆盖路径）
    aclblasStatus_t ret =
        aclblasChpr_npu(ChprArch22Test::handle_, ACLBLAS_UPPER, n, &alpha, xHost.data(), 1, apHost.data(), true);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_SUCCESS));
    if (ret != ACLBLAS_STATUS_SUCCESS) {
        return;
    }

    std::vector<aclblasComplex> golden = apOrig;
    aclblasChpr_cpu(ChprArch22Test::handle_, ACLBLAS_UPPER, n, &alpha, xHost.data(), 1, golden.data());

    VerifyConfig cfg;
    applyMixedTolerance(
        cfg, ACL_FLOAT, reinterpret_cast<const float*>(golden.data()), static_cast<size_t>(golden.size()) * 2);
    EXPECT_TRUE(Verifier::verifyVector(
        reinterpret_cast<const float*>(apHost.data()), reinterpret_cast<const float*>(golden.data()),
        static_cast<size_t>(golden.size()) * 2, 1, cfg, "DeviceAlpha"));
}

INSTANTIATE_TEST_SUITE_P(
    Chpr, ChprArch22Test, ::testing::ValuesIn(GetCasesFromCsv<ChprParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<ChprParam>);

TEST_P(ChprArch22Test, CsvDriven)
{
    const auto& p = GetParam();

    int absIncx = std::abs(p.incx);
    size_t xLen = (p.n > 0) ? static_cast<size_t>((p.n - 1) * absIncx + 1) : 0;
    size_t apLen = (p.n > 0) ? static_cast<size_t>(p.n) * (p.n + 1) / 2 : 0;

    std::vector<aclblasComplex> xHost = PackComplex(makeBlasArray(static_cast<int64_t>(xLen * 2), p.x, p.randomSeed));
    std::vector<aclblasComplex> apHost =
        PackComplex(makeBlasArray(static_cast<int64_t>(apLen * 2), p.ap, p.randomSeed + 1));
    std::vector<aclblasComplex> apOrig = apHost;

    const aclblasComplex* xPtr = xHost.empty() ? nullptr : xHost.data();
    aclblasComplex* apPtr = apHost.empty() ? nullptr : apHost.data();
    const float* alphaPtr = p.alphaNull ? nullptr : &p.alpha;

    aclblasStatus_t ret = aclblasChpr_npu(ChprArch22Test::handle_, p.uplo, p.n, alphaPtr, xPtr, p.incx, apPtr);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
    if (p.expectResult != ACLBLAS_STATUS_SUCCESS)
        return;
    if (p.n == 0)
        return;

    std::vector<aclblasComplex> golden = apOrig;
    aclblasChpr_cpu(ChprArch22Test::handle_, p.uplo, p.n, &p.alpha, xHost.data(), p.incx, golden.data());

    VerifyConfig cfg;
    size_t floatCount = apLen * 2;
    const float* outF = reinterpret_cast<const float*>(apPtr);
    const float* goldF = reinterpret_cast<const float*>(golden.data());
    if (p.alpha == 0.0f && !p.alphaNull) {
        cfg.mode = PrecisionMode::EXACT;
        EXPECT_TRUE(Verifier::verifyVector(outF, goldF, floatCount, 1, cfg, p.caseName));
    } else {
        applyMixedTolerance(cfg, ACL_FLOAT, goldF, floatCount);
        EXPECT_TRUE(Verifier::verifyVector(outF, goldF, floatCount, 1, cfg, p.caseName));
    }
}
