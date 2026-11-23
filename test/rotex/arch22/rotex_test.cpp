/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include "verify.h"
#include "blas_test.h"
#include "csv_loader.h"
#include "rotex_param.h"
#include "rotex_golden.h"
#include "rotex_npu_wrapper.h"

static void RunVerifyInline(
    const RotExParam& p, const float* xOut, const float* yOut, const float* gx, const float* gy, aclDataType xT,
    aclDataType yT, int absX, int absY)
{
    VerifyConfig cfgX;
    applyMixedTolerance(cfgX, xT, gx, static_cast<size_t>(std::max(0, p.n)));
    VerifyConfig cfgY;
    applyMixedTolerance(cfgY, yT, gy, static_cast<size_t>(std::max(0, p.n)));
    const int xs = (p.incx != 0) ? absX : 1;
    const int ys = (p.incy != 0) ? absY : 1;
    EXPECT_TRUE(
        Verifier::verifyVector(xOut, gx, static_cast<size_t>(std::max(0, p.n)), xs, cfgX, p.caseName + "_x"));
    EXPECT_TRUE(
        Verifier::verifyVector(yOut, gy, static_cast<size_t>(std::max(0, p.n)), ys, cfgY, p.caseName + "_y"));
}

class RotexTest : public BlasTest<RotExParam> {};

TEST_F(RotexTest, NullHandle)
{
    float xData[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float yData[] = {5.0f, 4.0f, 3.0f, 2.0f, 1.0f};
    float cVal = 0.5f;
    float sVal = 0.5f;
    const aclblasStatus_t ret = aclblasRotEx_npu(
        nullptr, 5, xData, ACL_FLOAT, 1, yData, ACL_FLOAT, 1, &cVal, &sVal, ACL_FLOAT, ACL_FLOAT);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_HANDLE_IS_NULLPTR));
}

INSTANTIATE_TEST_SUITE_P(
    RotEx, RotexTest, ::testing::ValuesIn(GetCasesFromCsv<RotExParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<RotExParam>);

static std::vector<float> InitRotExHostVec(
    int n, int inc, int elemFloats, const BlasFillMode& fill, uint32_t seed)
{
    if (fill.method == BlasFillMode::M_NULLPTR || n <= 0) {
        return {};
    }
    if (elemFloats == 1) {
        return makeBlasStrided(n, inc, fill, seed);
    }
    const int absInc = std::abs(inc);
    return makeBlasArray(static_cast<int64_t>((n - 1) * absInc * elemFloats + elemFloats), fill, seed);
}

struct RotExScalarHolders {
    float cF{0.0f};
    float sF{0.0f};
    uint16_t cBf16{0U};
    uint16_t sBf16{0U};
    uint16_t cFp16{0U};
    uint16_t sFp16{0U};
    const void* cPtr{nullptr};
    const void* sPtr{nullptr};
};

static RotExScalarHolders MakeRotExScalars(const RotExParam& p, aclDataType csType)
{
    RotExScalarHolders h;
    h.cF = static_cast<float>(p.cValue);
    h.sF = static_cast<float>(p.sValue);
    h.cBf16 = rotExFloatToBf16(h.cF);
    h.sBf16 = rotExFloatToBf16(h.sF);
    h.cFp16 = rotExFloatToFp16(h.cF);
    h.sFp16 = rotExFloatToFp16(h.sF);
    h.cPtr = (p.cValue <= -99999.0) ? nullptr
        : (csType == ACL_BF16)    ? static_cast<const void*>(&h.cBf16)
        : (csType == ACL_FLOAT16) ? static_cast<const void*>(&h.cFp16)
                                  : static_cast<const void*>(&h.cF);
    h.sPtr = (p.sValue <= -99999.0) ? nullptr
        : (csType == ACL_BF16)    ? static_cast<const void*>(&h.sBf16)
        : (csType == ACL_FLOAT16) ? static_cast<const void*>(&h.sFp16)
                                  : static_cast<const void*>(&h.sF);
    return h;
}

TEST_P(RotexTest, CsvDriven)
{
    const auto& p = GetParam();
    const auto xType = static_cast<aclDataType>(p.xType);
    const auto yType = static_cast<aclDataType>(p.yType);
    const auto csType = static_cast<aclDataType>(p.csType);
    const auto execType = static_cast<aclDataType>(p.executionType);
    const int absIncX = std::abs(p.incx);
    const int absIncY = std::abs(p.incy);

    std::vector<float> xHost = InitRotExHostVec(p.n, p.incx, rotExElemFloats(p.xType), p.xFill, p.randomSeed);
    std::vector<float> yHost = InitRotExHostVec(p.n, p.incy, rotExElemFloats(p.yType), p.yFill, p.randomSeed);
    float* xPtr = xHost.empty() ? nullptr : xHost.data();
    float* yPtr = yHost.empty() ? nullptr : yHost.data();

    const RotExScalarHolders sc = MakeRotExScalars(p, csType);
    std::vector<float> goldenX = xHost;
    std::vector<float> goldenY = yHost;
    const aclblasStatus_t ret = aclblasRotEx_npu(
        RotexTest::handle_, p.n, xPtr, xType, p.incx, yPtr, yType, p.incy, sc.cPtr, sc.sPtr, csType, execType);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
    if (p.expectResult != ACLBLAS_STATUS_SUCCESS) {
        return;
    }
    float* gxPtr = goldenX.empty() ? nullptr : goldenX.data();
    float* gyPtr = goldenY.empty() ? nullptr : goldenY.data();
    const aclblasStatus_t cpuRet = aclblasRotEx_cpu(
        RotexTest::handle_, p.n, gxPtr, xType, p.incx, gyPtr, yType, p.incy, sc.cPtr, sc.sPtr, csType, execType);
    EXPECT_EQ(static_cast<int>(cpuRet), static_cast<int>(ACLBLAS_STATUS_SUCCESS));
    RunVerifyInline(p, xPtr, yPtr, goldenX.data(), goldenY.data(), xType, yType, absIncX, absIncY);
}
