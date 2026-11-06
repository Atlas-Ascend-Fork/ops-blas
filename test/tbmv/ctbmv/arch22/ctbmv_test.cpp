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
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

#include "blas_test.h"
#include "csv_loader.h"
#include "device.h"
#include "fill.h"
#include "ctbmv_golden.h"
#include "ctbmv_input.h"
#include "ctbmv_precision.h"
#include "ctbmv_npu_wrapper.h"
#include "ctbmv_param.h"

class CtbmvArch22Test : public BlasTest<CtbmvParam> {};

namespace {

std::string CtbmvCasePath()
{
    const char* csv = std::getenv("CTBMV_TEST_CSV");
    if (csv != nullptr && csv[0] != '\0') {
        return csv;
    }
    const std::string fullPath = ReplaceFileExtension2Csv(__FILE__);
    return fullPath.substr(0, fullPath.find_last_of("/\\") + 1U) + "ctbmv_smoke.csv";
}

bool VerifyCtbmv(
    const std::vector<aclblasComplex>& actual, const std::vector<aclblasComplex>& expected, int n, int incx,
    const std::string& caseName)
{
    CtbmvComponentMetrics real;
    CtbmvComponentMetrics imag;
    size_t diagnostics = 0U;
    for (int i = 0; i < n; ++i) {
        const size_t index = CtbmvVectorIndex(i, n, incx);
        const bool realMatch = real.Observe(actual[index].real, expected[index].real);
        const bool imagMatch = imag.Observe(actual[index].imag, expected[index].imag);
        if ((!realMatch || !imagMatch) && diagnostics++ < 4U) {
            std::cerr << caseName << " mismatch logical=" << i << " actual=(" << actual[index].real << ','
                      << actual[index].imag << ')' << " expected=(" << expected[index].real << ','
                      << expected[index].imag << ")\n";
        }
    }
    if (!real.Pass() || !imag.Pass()) {
        ADD_FAILURE() << caseName << ": real_ratio=" << real.Ratio() << ", imag_ratio=" << imag.Ratio()
                      << ", real_max_abs=" << real.maxAbs << ", imag_max_abs=" << imag.maxAbs
                      << ", real_max_ulp=" << real.maxUlp << ", imag_max_ulp=" << imag.maxUlp;
        return false;
    }
    return true;
}

} // namespace

TEST_F(CtbmvArch22Test, NullHandle)
{
    aclblasComplex a{1.0F, 0.0F};
    aclblasComplex x{1.0F, 0.0F};
    EXPECT_EQ(
        aclblasCtbmv(nullptr, ACLBLAS_UPPER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT, 1, 0, &a, 1, &x, 1),
        ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
}

TEST_F(CtbmvArch22Test, SupplementalRejectMaxBandwidth)
{
    const int maxInt = std::numeric_limits<int>::max();
    aclblasComplex a{1.0F, 0.0F};
    aclblasComplex x{1.0F, 0.0F};
    // Invalid dimensions must be rejected before accessing either pointer.
    for (const int lda : {1, maxInt}) {
        EXPECT_EQ(
            aclblasCtbmv(
                CtbmvArch22Test::handle_, ACLBLAS_UPPER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT, 1, maxInt, &a, lda, &x, 1),
            ACLBLAS_STATUS_INVALID_VALUE);
    }
}

TEST_F(CtbmvArch22Test, SupplementalUnitDiagBandSentinelsAndNegativeStride)
{
    constexpr int n = 3;
    constexpr int k = 1;
    constexpr int lda = 3;
    constexpr int incx = -2;
    const float nan = std::numeric_limits<float>::quiet_NaN();

    // Every unreferenced band/padding slot and every implicit unit diagonal is
    // NaN. Only the two referenced super-diagonal entries are finite.
    std::vector<aclblasComplex> a(static_cast<size_t>(lda) * n, {nan, nan});
    a[CtbmvBandIndex(ACLBLAS_UPPER, k, lda, 0, 1)] = {2.0F, 1.0F};
    a[CtbmvBandIndex(ACLBLAS_UPPER, k, lda, 1, 2)] = {-1.0F, 0.5F};

    std::vector<aclblasComplex> x(5U, {nan, nan});
    x[CtbmvVectorIndex(0, n, incx)] = {1.0F, 1.0F};
    x[CtbmvVectorIndex(1, n, incx)] = {2.0F, -1.0F};
    x[CtbmvVectorIndex(2, n, incx)] = {-1.0F, 2.0F};
    std::vector<aclblasComplex> golden = x;
    CtbmvGolden(ACLBLAS_UPPER, ACLBLAS_OP_N, ACLBLAS_UNIT, n, k, a, lda, golden, incx);

    ASSERT_EQ(
        aclblasCtbmv_npu(
            CtbmvArch22Test::handle_, CtbmvArch22Test::stream_, ACLBLAS_UPPER, ACLBLAS_OP_N, ACLBLAS_UNIT, n, k,
            a.data(), lda, x.data(), incx),
        ACLBLAS_STATUS_SUCCESS);
    EXPECT_TRUE(VerifyCtbmv(x, golden, n, incx, "supplemental_unit_diag_band_sentinels"));
    EXPECT_TRUE(std::isnan(x[1].real));
    EXPECT_TRUE(std::isnan(x[1].imag));
    EXPECT_TRUE(std::isnan(x[3].real));
    EXPECT_TRUE(std::isnan(x[3].imag));
}

INSTANTIATE_TEST_SUITE_P(
    Ctbmv, CtbmvArch22Test, ::testing::ValuesIn(GetCasesFromCsv<CtbmvParam>(CtbmvCasePath())),
    PrintCaseInfoString<CtbmvParam>);

// issue #838: 框架级用例——parseDataType 枚举表全集与未知 token 报错语义。
// 挂载于 ctbmv arch22（本二进制经 csv_loader.h 消费框架解析器）。
TEST_F(CtbmvArch22Test, ParseDataTypeCoversFullEnumSet)
{
    // 原表缺失的 dtype 全部可解析
    EXPECT_EQ(parseDataType("INT32"), static_cast<int32_t>(ACL_INT32));
    EXPECT_EQ(parseDataType("ACL_INT32"), static_cast<int32_t>(ACL_INT32));
    EXPECT_EQ(parseDataType("FP8_E4M3FN"), static_cast<int32_t>(ACL_FLOAT8_E4M3FN));
    EXPECT_EQ(parseDataType("ACL_FLOAT8_E4M3FN"), static_cast<int32_t>(ACL_FLOAT8_E4M3FN));
    EXPECT_EQ(parseDataType("FP8_E5M2"), static_cast<int32_t>(ACL_FLOAT8_E5M2));
    EXPECT_EQ(parseDataType("FP4"), static_cast<int32_t>(ACL_FLOAT4_E2M1));
    EXPECT_EQ(parseDataType("ACL_FLOAT4_E2M1"), static_cast<int32_t>(ACL_FLOAT4_E2M1));
    EXPECT_EQ(parseDataType("HIFLOAT8"), static_cast<int32_t>(ACL_HIFLOAT8));
    EXPECT_EQ(parseDataType("ACL_DOUBLE"), static_cast<int32_t>(ACL_DOUBLE));
    // !887 复审补齐的同仓在用扩展 dtype（blasLtMatrixTransform/blasLtMatmul 侧成员）
    EXPECT_EQ(parseDataType("COMPLEX64"), static_cast<int32_t>(ACL_COMPLEX64));
    EXPECT_EQ(parseDataType("ACL_COMPLEX64"), static_cast<int32_t>(ACL_COMPLEX64));
    EXPECT_EQ(parseDataType("ACL_FLOAT8_E8M0"), static_cast<int32_t>(ACL_FLOAT8_E8M0));
    EXPECT_EQ(parseDataType("E8M0"), static_cast<int32_t>(ACL_FLOAT8_E8M0));
    EXPECT_EQ(parseDataType("FP6"), static_cast<int32_t>(ACL_FLOAT6_E2M3));
    EXPECT_EQ(parseDataType("ACL_FLOAT6_E3M2"), static_cast<int32_t>(ACL_FLOAT6_E3M2));
    EXPECT_EQ(parseDataType("ACL_FLOAT4_E1M2"), static_cast<int32_t>(ACL_FLOAT4_E1M2));
    EXPECT_EQ(parseDataType("MXFP8_E5M2"), static_cast<int32_t>(ACL_FLOAT8_E5M2));
    EXPECT_EQ(parseDataType("MXFP4_E2M1"), static_cast<int32_t>(ACL_FLOAT4_E2M1));
    // 既有 token 行为不变
    EXPECT_EQ(parseDataType("FP32"), static_cast<int32_t>(ACL_FLOAT));
    EXPECT_EQ(parseDataType("ACL_FLOAT16"), static_cast<int32_t>(ACL_FLOAT16));
    EXPECT_EQ(parseDataType("BF16"), static_cast<int32_t>(ACL_BF16));
    EXPECT_EQ(parseDataType(""), static_cast<int32_t>(ACL_FLOAT));
    // 数值直通通道保留（负向注入口径）
    EXPECT_EQ(parseDataType("999"), 999);
    // 未知非数值 token 不再静默回退 ACL_FLOAT，必须抛出
    EXPECT_THROW(parseDataType("FP9_E9M9"), std::runtime_error);
}

TEST_P(CtbmvArch22Test, CsvDriven)
{
    const CtbmvParam& p = GetParam();
    const size_t aCount = p.n > 0 && p.lda > 0 ? static_cast<size_t>(p.lda) * p.n : 0U;
    const int64_t absIncx = p.incx >= 0 ? static_cast<int64_t>(p.incx) : -static_cast<int64_t>(p.incx);
    const size_t xCount = p.n > 0 && absIncx > 0 ? static_cast<size_t>(p.n - 1) * absIncx + 1U : 0U;

    std::vector<aclblasComplex> a = MakeCtbmvComplexArray(aCount, p.aFill, p.randomSeed);
    std::vector<aclblasComplex> x = MakeCtbmvComplexArray(xCount, p.xFill, p.randomSeed + 1U);
    std::vector<aclblasComplex> golden = x;

    const aclblasComplex* aPtr = p.aFill.method == BlasFillMode::M_NULLPTR ? nullptr : a.data();
    aclblasComplex* xPtr = p.xFill.method == BlasFillMode::M_NULLPTR ? nullptr : x.data();
    aclblasStatus_t status = aclblasCtbmv_npu(
        CtbmvArch22Test::handle_, CtbmvArch22Test::stream_, p.uplo, p.trans, p.diag, p.n, p.k, aPtr, p.lda, xPtr,
        p.incx);

    if (p.expectResult != ACLBLAS_STATUS_SUCCESS) {
        EXPECT_EQ(status, p.expectResult);
        return;
    }
    ASSERT_EQ(status, ACLBLAS_STATUS_SUCCESS);
    if (p.n == 0) {
        return;
    }

    CtbmvGolden(p.uplo, p.trans, p.diag, p.n, p.k, a, p.lda, golden, p.incx);
    EXPECT_TRUE(VerifyCtbmv(x, golden, p.n, p.incx, p.caseName));
}
