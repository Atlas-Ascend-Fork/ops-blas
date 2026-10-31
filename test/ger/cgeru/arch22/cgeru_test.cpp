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
#include <climits>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "blas_test.h"
#include "cgeru_golden.h"
#include "cgeru_npu_wrapper.h"
#include "cgeru_param.h"
#include "csv_loader.h"
#include "verify.h"

namespace {

std::vector<aclblasComplex> MakeComplexArray(int64_t count, const BlasFillMode& fill, uint32_t seed)
{
    if (count <= 0 || fill.method == BlasFillMode::M_NULLPTR)
        return {};
    const std::vector<float> real = makeBlasArray(count, fill, seed);
    const std::vector<float> imag = makeBlasArray(count, fill, seed + 1000U);
    std::vector<aclblasComplex> result(static_cast<size_t>(count));
    for (size_t i = 0; i < result.size(); ++i) {
        result[i] = aclblasComplex{real[i], imag[i]};
    }
    return result;
}

int64_t PhysicalVectorLength(int logicalLength, int increment)
{
    if (logicalLength <= 0 || increment == 0)
        return 0;
    const int64_t stride = (increment < 0) ? -static_cast<int64_t>(increment) : static_cast<int64_t>(increment);
    return 1 + static_cast<int64_t>(logicalLength - 1) * stride;
}

} // namespace

class CgeruTest : public BlasTest<CgeruParam> {};

TEST_F(CgeruTest, NullHandle)
{
    EXPECT_EQ(
        aclblasCgeru(nullptr, 4, 4, nullptr, nullptr, 1, nullptr, 1, nullptr, 4), ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
}

TEST_F(CgeruTest, QuickReturnDoesNotAccessDevicePointers)
{
    const aclblasComplex zeroAlpha{0.0f, -0.0f};
    EXPECT_EQ(
        aclblasCgeru(
            CgeruTest::handle_, INT_MAX, INT_MAX, &zeroAlpha, nullptr, INT_MIN, nullptr, INT_MIN, nullptr, INT_MAX),
        ACLBLAS_STATUS_SUCCESS);

    const aclblasComplex oneAlpha{1.0f, 0.0f};
    EXPECT_EQ(
        aclblasCgeru(CgeruTest::handle_, 0, INT_MAX, &oneAlpha, nullptr, INT_MIN, nullptr, INT_MIN, nullptr, 1),
        ACLBLAS_STATUS_SUCCESS);
}

TEST_F(CgeruTest, ZeroYColumnSkipsInfAndNanX)
{
    const aclblasComplex alpha{1.0f, 0.0f};
    const std::vector<aclblasComplex> x{
        {std::numeric_limits<float>::infinity(), 0.0f}, {std::numeric_limits<float>::quiet_NaN(), 1.0f}};
    const std::vector<aclblasComplex> y{{0.0f, -0.0f}, {1.0f, 0.0f}};
    std::vector<aclblasComplex> matrix{{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}, {7.0f, 8.0f}};
    const std::vector<aclblasComplex> original = matrix;

    ASSERT_EQ(
        aclblasCgeru_npu(CgeruTest::handle_, 2, 2, &alpha, x.data(), 1, y.data(), 1, matrix.data(), 2),
        ACLBLAS_STATUS_SUCCESS);
    EXPECT_EQ(matrix[0].real, original[0].real);
    EXPECT_EQ(matrix[0].imag, original[0].imag);
    EXPECT_EQ(matrix[1].real, original[1].real);
    EXPECT_EQ(matrix[1].imag, original[1].imag);

    // Column 1 (matrix[2]/matrix[3]) must be updated normally (issue #645):
    // alpha=1, y[1]={1,0}; x[0]={inf,0} contributes {inf, NaN} (inf*0 in the
    // imaginary part), x[1]={NaN,1} contributes {NaN, NaN}. A lost write-back
    // would leave the original {5,6}/{7,8} in place and fail these checks.
    EXPECT_TRUE(std::isinf(matrix[2].real) && matrix[2].real > 0.0f);
    EXPECT_TRUE(std::isnan(matrix[2].imag));
    EXPECT_TRUE(std::isnan(matrix[3].real));
    EXPECT_TRUE(std::isnan(matrix[3].imag));
}

TEST_F(CgeruTest, FastPathZeroYColumnGuardsInfNanX)
{
    // issue #694: m=8 使 m*2 分量满足 32B 对齐（快路径门槛）且 lda=m 连续，命中快路径；
    // 组合"零 y 列 + 对应 x 行含 Inf/NaN"此前仅有通用路径（m=2）用例覆盖。若快路径
    // 改造时丢失 y[j]==0 整列跳过守卫（如改为 coeff 乘加），0*Inf→NaN 会写穿该列。
    const int m = 8;
    const int n = 4;
    const aclblasComplex alpha{1.0f, 0.0f};
    std::vector<aclblasComplex> x(m);
    for (int i = 0; i < m; ++i) {
        x[i] = aclblasComplex{static_cast<float>(i + 1), -static_cast<float>(i + 1)};
    }
    x[1] = aclblasComplex{std::numeric_limits<float>::infinity(), 0.0f};
    std::vector<aclblasComplex> y(n);
    for (int j = 0; j < n; ++j) {
        y[j] = aclblasComplex{static_cast<float>(j + 1), 0.5f * static_cast<float>(j + 1)};
    }
    y[1] = aclblasComplex{0.0f, -0.0f};
    std::vector<aclblasComplex> matrix(static_cast<size_t>(m) * n);
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            const float v = static_cast<float>(i * 10 + j + 1);
            matrix[static_cast<size_t>(j) * m + i] = aclblasComplex{v, -v};
        }
    }
    const std::vector<aclblasComplex> original = matrix;

    ASSERT_EQ(
        aclblasCgeru_npu(CgeruTest::handle_, m, n, &alpha, x.data(), 1, y.data(), 1, matrix.data(), m),
        ACLBLAS_STATUS_SUCCESS);

    // y[1]==0 的整列必须逐位保持原值
    for (int i = 0; i < m; ++i) {
        const size_t idx = static_cast<size_t>(1) * m + i;
        EXPECT_EQ(matrix[idx].real, original[idx].real) << "row " << i;
        EXPECT_EQ(matrix[idx].imag, original[idx].imag) << "row " << i;
    }
    // 非零列正常更新：A(0,0) += alpha * x[0] * y[0] = (1.5, -0.5)
    EXPECT_FLOAT_EQ(matrix[0].real, original[0].real + 1.5f);
    EXPECT_FLOAT_EQ(matrix[0].imag, original[0].imag - 0.5f);
}

TEST_F(CgeruTest, NoConjugationAndIntMinStride)
{
    const aclblasComplex alpha{1.0f, 0.0f};
    const aclblasComplex x{1.0f, 2.0f};
    const aclblasComplex y{3.0f, 4.0f};
    aclblasComplex matrix{0.0f, 0.0f};

    ASSERT_EQ(
        aclblasCgeru_npu(CgeruTest::handle_, 1, 1, &alpha, &x, INT_MIN, &y, INT_MIN, &matrix, 1),
        ACLBLAS_STATUS_SUCCESS);
    EXPECT_FLOAT_EQ(matrix.real, -5.0f);
    EXPECT_FLOAT_EQ(matrix.imag, 10.0f);
}

TEST_F(CgeruTest, ContiguousTallNarrowAndBoundaryRows)
{
    const aclblasComplex alpha{0.75f, -0.5f};
    const BlasFillMode fill = parseFill("RANDOM_NORM_5_5");
    for (int m : {1, 31, 33, 63, 65, 129, 4093, 4095, 4096, 4097}) {
        // issue #695: 32768/m+1 在 m=4095/4096 时为 9，与首项重复；去重避免同一 (m,n) 跑两遍。
        // arch22 只有快路径/通用路径，用例名不含 arch35 专属的 RegBase 字样。
        const int tailN = 32768 / m + 1;
        const std::vector<int> nList =
            (tailN == 9) ? std::vector<int>{9, 17, 31, 65} : std::vector<int>{9, 17, 31, 65, tailN};
        for (int n : nList) {
            const std::string caseName = "contiguous_m" + std::to_string(m) + "_n" + std::to_string(n);
            SCOPED_TRACE(caseName);
            const std::vector<aclblasComplex> x = MakeComplexArray(m, fill, 20260918U);
            const std::vector<aclblasComplex> y = MakeComplexArray(n, fill, 20260919U);
            std::vector<aclblasComplex> matrix = MakeComplexArray(static_cast<int64_t>(m) * n, fill, 20260920U);
            std::vector<aclblasComplex> golden = matrix;
            ASSERT_EQ(
                aclblasCgeru_npu(CgeruTest::handle_, m, n, &alpha, x.data(), 1, y.data(), 1, matrix.data(), m),
                ACLBLAS_STATUS_SUCCESS);
            ASSERT_EQ(
                aclblasCgeru_cpu(CgeruTest::handle_, m, n, &alpha, x.data(), 1, y.data(), 1, golden.data(), m),
                ACLBLAS_STATUS_SUCCESS);
            VerifyConfig config;
            applyMixedTolerance(config, ACL_FLOAT, reinterpret_cast<const float*>(golden.data()), golden.size() * 2);
            EXPECT_TRUE(Verifier::verifyVector(
                reinterpret_cast<const float*>(matrix.data()), reinterpret_cast<const float*>(golden.data()),
                golden.size() * 2, 1, config, caseName));
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    Cgeru, CgeruTest, ::testing::ValuesIn(GetCasesFromCsv<CgeruParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<CgeruParam>);

TEST_P(CgeruTest, CsvDriven)
{
    const CgeruParam& p = GetParam();
    std::vector<aclblasComplex> xHost = MakeComplexArray(PhysicalVectorLength(p.m, p.incx), p.x, p.randomSeed);
    std::vector<aclblasComplex> yHost = MakeComplexArray(PhysicalVectorLength(p.n, p.incy), p.y, p.randomSeed + 1U);
    const int64_t aCount = (p.lda > 0 && p.n > 0) ? static_cast<int64_t>(p.lda) * p.n : 0;
    std::vector<aclblasComplex> aHost = MakeComplexArray(aCount, p.A, p.randomSeed + 2U);
    std::vector<aclblasComplex> golden = aHost;

    const aclblasComplex alpha{p.alphaReal, p.alphaImag};
    const aclblasComplex* alphaPtr = p.alphaIsNull ? nullptr : &alpha;
    const aclblasComplex* xPtr = xHost.empty() ? nullptr : xHost.data();
    const aclblasComplex* yPtr = yHost.empty() ? nullptr : yHost.data();
    aclblasComplex* aPtr = aHost.empty() ? nullptr : aHost.data();

    const aclblasStatus_t status =
        aclblasCgeru_npu(CgeruTest::handle_, p.m, p.n, alphaPtr, xPtr, p.incx, yPtr, p.incy, aPtr, p.lda);
    EXPECT_EQ(static_cast<int>(status), static_cast<int>(p.expectResult));
    if (p.expectResult != ACLBLAS_STATUS_SUCCESS)
        return;

    ASSERT_EQ(
        aclblasCgeru_cpu(
            CgeruTest::handle_, p.m, p.n, alphaPtr, xPtr, p.incx, yPtr, p.incy,
            golden.empty() ? nullptr : golden.data(), p.lda),
        ACLBLAS_STATUS_SUCCESS);

    VerifyConfig config;
    if (p.alphaReal == 0.0f && p.alphaImag == 0.0f) {
        config.mode = PrecisionMode::EXACT;
    } else {
        applyMixedTolerance(config, ACL_FLOAT, reinterpret_cast<const float*>(golden.data()), golden.size() * 2);
    }
    EXPECT_TRUE(Verifier::verifyVector(
        reinterpret_cast<const float*>(aHost.data()), reinterpret_cast<const float*>(golden.data()), golden.size() * 2,
        1, config, p.caseName));
}
