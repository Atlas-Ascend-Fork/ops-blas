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
#include <cstring>
#include <vector>

#include <securec.h>

#include "blas_test.h"
#include "cswap_golden.h"
#include "cswap_npu_wrapper.h"
#include "cswap_param.h"
#include "csv_loader.h"
#include "fill.h"

class CswapTest : public BlasTest<CswapParam> {};

TEST_F(CswapTest, NonPositiveLengthIgnoresOtherArguments)
{
    EXPECT_EQ(aclblasCswap_npu(nullptr, 0, nullptr, 0, nullptr, 0), ACLBLAS_STATUS_SUCCESS);
    EXPECT_EQ(aclblasCswap_npu(nullptr, -5, nullptr, 0, nullptr, 0), ACLBLAS_STATUS_SUCCESS);
}

TEST_F(CswapTest, NullHandle)
{
    EXPECT_EQ(aclblasCswap_npu(nullptr, 5, nullptr, 1, nullptr, 1), ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
}

INSTANTIATE_TEST_SUITE_P(
    Cswap, CswapTest, ::testing::ValuesIn(GetCasesFromCsv<CswapParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<CswapParam>);

TEST_P(CswapTest, CsvDriven)
{
    const auto& p = GetParam();
    ASSERT_GE(p.xAlignOffset, 0);
    ASSERT_GE(p.yAlignOffset, 0);
    int64_t xSpan = p.n > 0 ? static_cast<int64_t>(std::abs(p.incx)) * (p.n - 1) + 1 : 1;
    int64_t ySpan = p.n > 0 ? static_cast<int64_t>(std::abs(p.incy)) * (p.n - 1) + 1 : 1;
    int64_t xAlloc = xSpan + p.xAlignOffset;
    int64_t yAlloc = ySpan + p.yAlignOffset;

    std::vector<aclblasComplex> xHost =
        makeBlasComplexMatrix(static_cast<int>(xAlloc), 1, static_cast<int>(xAlloc), p.x, p.randomSeed);
    std::vector<aclblasComplex> yHost =
        makeBlasComplexMatrix(static_cast<int>(yAlloc), 1, static_cast<int>(yAlloc), p.y, p.randomSeed + 1U);
    std::vector<aclblasComplex> goldenX = xHost;
    std::vector<aclblasComplex> goldenY = yHost;

    aclblasComplex* xPtr = xHost.empty() ? nullptr : xHost.data() + p.xAlignOffset;
    aclblasComplex* yPtr = yHost.empty() ? nullptr : yHost.data() + p.yAlignOffset;
    aclblasStatus_t ret = aclblasCswap_npu(
        CswapTest::handle_, p.n, xPtr, p.incx, yPtr, p.incy, static_cast<size_t>(p.xAlignOffset),
        static_cast<size_t>(p.yAlignOffset));
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
    if (p.expectResult != ACLBLAS_STATUS_SUCCESS || p.n <= 0) {
        return;
    }

    ASSERT_FALSE(goldenX.empty());
    ASSERT_FALSE(goldenY.empty());
    ASSERT_EQ(
        aclblasCswap_cpu(
            CswapTest::handle_, p.n, goldenX.data() + p.xAlignOffset, p.incx, goldenY.data() + p.yAlignOffset, p.incy),
        ACLBLAS_STATUS_SUCCESS);
    EXPECT_EQ(std::memcmp(xHost.data(), goldenX.data(), xHost.size() * sizeof(aclblasComplex)), 0)
        << '[' << p.caseName << "] x output, alignment prefix, or stride holes are not bit-exact";
    EXPECT_EQ(std::memcmp(yHost.data(), goldenY.data(), yHost.size() * sizeof(aclblasComplex)), 0)
        << '[' << p.caseName << "] y output, alignment prefix, or stride holes are not bit-exact";
}

TEST_F(CswapTest, PreservesRawBitsMixedStridesAndHoles)
{
    constexpr int n = 3;
    constexpr int incx = -2;
    constexpr int incy = 3;
    std::vector<aclblasComplex> x(5);
    std::vector<aclblasComplex> y(7);
    const uint32_t xBits[10] = {0x7fc12345U, 0xff800000U, 0x3f800000U, 0x80000000U, 0x00000001U,
                                0x7f800000U, 0xbf800000U, 0x7fa54321U, 0x00000000U, 0x40490fdbU};
    const uint32_t yBits[14] = {0xdeadbeefU, 0x12345678U, 0x7fa00001U, 0x80000000U, 0x11111111U,
                                0x22222222U, 0x00000001U, 0xff800000U, 0x33333333U, 0x44444444U,
                                0x3f000000U, 0xbf000000U, 0x7fcabcdeU, 0x7f800000U};
    ASSERT_EQ(memcpy_s(x.data(), x.size() * sizeof(aclblasComplex), xBits, sizeof(xBits)), EOK);
    ASSERT_EQ(memcpy_s(y.data(), y.size() * sizeof(aclblasComplex), yBits, sizeof(yBits)), EOK);
    std::vector<aclblasComplex> goldenX = x;
    std::vector<aclblasComplex> goldenY = y;
    ASSERT_EQ(
        aclblasCswap_cpu(CswapTest::handle_, n, goldenX.data(), incx, goldenY.data(), incy), ACLBLAS_STATUS_SUCCESS);

    ASSERT_EQ(aclblasCswap_npu(CswapTest::handle_, n, x.data(), incx, y.data(), incy), ACLBLAS_STATUS_SUCCESS);
    EXPECT_EQ(std::memcmp(x.data(), goldenX.data(), x.size() * sizeof(aclblasComplex)), 0);
    EXPECT_EQ(std::memcmp(y.data(), goldenY.data(), y.size() * sizeof(aclblasComplex)), 0);
}

TEST_F(CswapTest, SupportsComplexElementAlignmentOffsets)
{
    constexpr int n = 17;
    constexpr int xOffset = 1;
    constexpr int yOffset = 3;
    std::vector<aclblasComplex> x(n + xOffset, {1.0F, -1.0F});
    std::vector<aclblasComplex> y(n + yOffset, {2.0F, -2.0F});
    std::vector<aclblasComplex> goldenX = x;
    std::vector<aclblasComplex> goldenY = y;
    ASSERT_EQ(
        aclblasCswap_cpu(CswapTest::handle_, n, goldenX.data() + xOffset, 1, goldenY.data() + yOffset, 1),
        ACLBLAS_STATUS_SUCCESS);

    ASSERT_EQ(
        aclblasCswap_npu(CswapTest::handle_, n, x.data() + xOffset, 1, y.data() + yOffset, 1, xOffset, yOffset),
        ACLBLAS_STATUS_SUCCESS);
    EXPECT_EQ(std::memcmp(x.data(), goldenX.data(), x.size() * sizeof(aclblasComplex)), 0);
    EXPECT_EQ(std::memcmp(y.data(), goldenY.data(), y.size() * sizeof(aclblasComplex)), 0);
}
