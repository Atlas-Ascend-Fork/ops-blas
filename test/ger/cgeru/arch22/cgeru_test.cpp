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
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "blas_test.h"
#include "cgeru_golden.h"
#include "cgeru_npu_wrapper.h"
#include "cgeru_param.h"
#include "csv_loader.h"
#include "fill.h"
#include "verify.h"

namespace {

// 由逻辑长度和步长计算物理存储长度：1 + (len - 1) * |inc|。
int64_t PhysicalVectorLength(int logicalLength, int increment)
{
    if (logicalLength <= 0 || increment == 0)
        return 0;
    const int64_t stride = (increment < 0) ? -static_cast<int64_t>(increment) : static_cast<int64_t>(increment);
    return 1 + static_cast<int64_t>(logicalLength - 1) * stride;
}

// INT_MIN 的物理跨度只有长度 <= 1 时合法；这里只按 1 个元素分配，
// 避免按 |INT_MIN| 计算物理长度时产生超大分配。
int64_t AllocationVectorLength(int logicalLength, int increment)
{
    if (increment == INT_MIN && logicalLength > 0)
        return 1;
    return PhysicalVectorLength(logicalLength, increment);
}

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

bool BitIdentical(const std::vector<aclblasComplex>& lhs, const std::vector<aclblasComplex>& rhs)
{
    return lhs.size() == rhs.size() &&
           (lhs.empty() || std::memcmp(lhs.data(), rhs.data(), lhs.size() * sizeof(aclblasComplex)) == 0);
}

// 用接口级公共参考实现（cgeru_golden.h 内的 cblas_cgeru 封装）生成 golden，
// 与 arch35 测试共用同一份 golden 口径。非法形状/空缓冲区返回 A 的副本。
std::vector<aclblasComplex> MakeCgeruGolden(
    aclblasHandle_t handle, int m, int n, const aclblasComplex& alpha, const std::vector<aclblasComplex>& x, int incx,
    const std::vector<aclblasComplex>& y, int incy, const std::vector<aclblasComplex>& a, int lda)
{
    std::vector<aclblasComplex> golden = a;
    if (m <= 0 || n <= 0 || lda < std::max(1, m) || x.empty() || y.empty() || a.empty()) {
        return golden;
    }
    aclblasCgeru_cpu(handle, m, n, &alpha, x.data(), incx, y.data(), incy, golden.data(), lda);
    return golden;
}

// 逐分量比较（复数按 FLOAT32 成对展开），判据取 test/frame/verify.h 的接口级混合容差：
// 2^-13 相对/绝对容差 + 每元素 max(1e-2, 32*ULP) 上限，匹配率 >= 0.99。

// golden 出现 Inf/NaN 时（RANDOM_EXTREME 用例）只要求结果同样非有限：BLAS 不规定
// 非有限输入的传播次序，CPU 参考实现与 NPU 内核在 0*Inf 这类步骤上可能分别得到
// inf 与 nan，这不构成精度错误；有限元素仍按上面的混合容差逐一检查。
bool VerifyAgainstGolden(const std::vector<aclblasComplex>& actual, const std::vector<aclblasComplex>& golden,
                         const std::string& caseId)
{
    if (actual.size() != golden.size()) {
        std::cerr << "[" << caseId << "] result/golden size mismatch" << std::endl;
        return false;
    }
    if (golden.empty()) {
        return true;
    }
    std::vector<float> finiteActual;
    std::vector<float> finiteGolden;
    finiteActual.reserve(actual.size() * 2U);
    finiteGolden.reserve(golden.size() * 2U);
    for (size_t i = 0; i < golden.size(); ++i) {
        const float goldenParts[2] = {golden[i].real, golden[i].imag};
        const float actualParts[2] = {actual[i].real, actual[i].imag};
        for (int part = 0; part < 2; ++part) {
            if (std::isfinite(goldenParts[part])) {
                finiteGolden.push_back(goldenParts[part]);
                finiteActual.push_back(actualParts[part]);
            } else if (std::isfinite(actualParts[part])) {
                std::cerr << "[" << caseId << "] component " << (i * 2U + static_cast<size_t>(part))
                          << ": expected a non-finite value, got " << actualParts[part] << std::endl;
                return false;
            }
        }
    }
    if (finiteGolden.empty()) {
        return true;
    }
    VerifyConfig config;
    applyMixedTolerance(config, ACL_FLOAT, finiteGolden.data(), finiteGolden.size());
    return Verifier::verifyVector(finiteActual.data(), finiteGolden.data(), finiteGolden.size(), 1, config, caseId);
}

// 跑一次 cgeru 并与接口级 golden 比较（限 alpha 有限、缓冲区非空的用例）。
void ExpectMatchesGolden(aclblasHandle_t handle, int m, int n, int lda, const aclblasComplex& alpha,
                         const std::vector<aclblasComplex>& x, int incx, const std::vector<aclblasComplex>& y,
                         int incy, const std::vector<aclblasComplex>& aBase, const std::string& caseId)
{
    std::vector<aclblasComplex> matrix = aBase;
    const std::vector<aclblasComplex> golden = MakeCgeruGolden(handle, m, n, alpha, x, incx, y, incy, aBase, lda);
    ASSERT_EQ(
        aclblasCgeru_npu(handle, m, n, &alpha, x.data(), incx, y.data(), incy, matrix.data(), lda),
        ACLBLAS_STATUS_SUCCESS);
    EXPECT_TRUE(VerifyAgainstGolden(matrix, golden, caseId));
}

// alpha 非有限时结果按 IEEE 754 传播：expectNaN 为真时全部为 NaN，否则全部非有限即可。
void ExpectNonFiniteResult(aclblasHandle_t handle, int m, int n, int lda, const aclblasComplex& alpha,
                           const std::vector<aclblasComplex>& x, const std::vector<aclblasComplex>& y,
                           const std::vector<aclblasComplex>& aBase, bool expectNaN)
{
    std::vector<aclblasComplex> matrix = aBase;
    ASSERT_EQ(
        aclblasCgeru_npu(handle, m, n, &alpha, x.data(), 1, y.data(), 1, matrix.data(), lda),
        ACLBLAS_STATUS_SUCCESS);
    for (const aclblasComplex& value : matrix) {
        if (expectNaN) {
            EXPECT_TRUE(std::isnan(value.real));
            EXPECT_TRUE(std::isnan(value.imag));
        } else {
            EXPECT_FALSE(std::isfinite(value.real));
            EXPECT_FALSE(std::isfinite(value.imag));
        }
    }
}

} // namespace

class CgeruArch22Test : public BlasTest<CgeruParam> {};

// ─────────────────────────────────────────────────────────────────────────────
// 显式边界用例
// ─────────────────────────────────────────────────────────────────────────────

TEST_F(CgeruArch22Test, NullHandle)
{
    EXPECT_EQ(
        aclblasCgeru_npu(nullptr, 4, 4, nullptr, nullptr, 1, nullptr, 1, nullptr, 4),
        ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
}

// m==0 / n==0 / m==n==0：标准 BLAS quick return，不访问 x/y/A。
TEST_F(CgeruArch22Test, ZeroDimensionReturnsSuccessWithoutBuffers)
{
    const aclblasComplex alpha{1.0f, 0.0f};
    EXPECT_EQ(
        aclblasCgeru_npu(CgeruArch22Test::handle_, 0, 5, &alpha, nullptr, 1, nullptr, 1, nullptr, 1),
        ACLBLAS_STATUS_SUCCESS);
    EXPECT_EQ(
        aclblasCgeru_npu(CgeruArch22Test::handle_, 5, 0, &alpha, nullptr, 1, nullptr, 1, nullptr, 5),
        ACLBLAS_STATUS_SUCCESS);
    EXPECT_EQ(
        aclblasCgeru_npu(CgeruArch22Test::handle_, 0, 0, &alpha, nullptr, 1, nullptr, 1, nullptr, 1),
        ACLBLAS_STATUS_SUCCESS);
}

// incx==0 / incy==0 以及 lda<max(1,m) 必须返回 INVALID_VALUE，且不能访问缓冲区。
TEST_F(CgeruArch22Test, InvalidStridesAndLeadingDimension)
{
    const aclblasComplex alpha{1.0f, 0.0f};
    std::vector<aclblasComplex> x(4, aclblasComplex{1.0f, 0.0f});
    std::vector<aclblasComplex> y(4, aclblasComplex{1.0f, 0.0f});
    std::vector<aclblasComplex> matrix(16, aclblasComplex{0.0f, 0.0f});

    EXPECT_EQ(
        aclblasCgeru_npu(
            CgeruArch22Test::handle_, 4, 4, &alpha, x.data(), 0, y.data(), 1, matrix.data(), 4),
        ACLBLAS_STATUS_INVALID_VALUE);
    EXPECT_EQ(
        aclblasCgeru_npu(
            CgeruArch22Test::handle_, 4, 4, &alpha, x.data(), 1, y.data(), 0, matrix.data(), 4),
        ACLBLAS_STATUS_INVALID_VALUE);
    EXPECT_EQ(
        aclblasCgeru_npu(
            CgeruArch22Test::handle_, 4, 4, &alpha, x.data(), 1, y.data(), 1, matrix.data(), 3),
        ACLBLAS_STATUS_INVALID_VALUE);
    EXPECT_EQ(
        aclblasCgeru_npu(
            CgeruArch22Test::handle_, 4, 4, &alpha, x.data(), 1, y.data(), 1, matrix.data(), 0),
        ACLBLAS_STATUS_INVALID_VALUE);
}

// INT_MIN 步长按实际跨度判定：m/n > 1 时跨度 2^31*(len-1) 不可寻址，返回 INVALID_VALUE；
// m==1（或 n==1）时跨度只有 1 个元素，属于合法输入，交由 kernel 按 1x1 处理。
TEST_F(CgeruArch22Test, IntMinStrideSpanIsValidated)
{
    const aclblasComplex alpha{1.0f, 0.0f};
    std::vector<aclblasComplex> x(4, aclblasComplex{1.0f, 1.0f});
    std::vector<aclblasComplex> y(4, aclblasComplex{2.0f, 2.0f});
    std::vector<aclblasComplex> matrix(16, aclblasComplex{0.0f, 0.0f});

    // 多维时跨度不可寻址：提前拒绝，不能把 INT_MIN 交给 kernel。
    EXPECT_EQ(
        aclblasCgeru_npu(
            CgeruArch22Test::handle_, 4, 4, &alpha, x.data(), INT_MIN, y.data(), 1, matrix.data(), 4),
        ACLBLAS_STATUS_INVALID_VALUE);
    EXPECT_EQ(
        aclblasCgeru_npu(
            CgeruArch22Test::handle_, 4, 4, &alpha, x.data(), 1, y.data(), INT_MIN, matrix.data(), 4),
        ACLBLAS_STATUS_INVALID_VALUE);
}

// 1x1 + INT_MIN：跨度只有 x[0]/y[0] 两个元素，结果按普通 1x1 乘加计算。
TEST_F(CgeruArch22Test, IntMinStrideOnUnitDimension)
{
    const aclblasComplex alpha{1.0f, 0.0f};
    const aclblasComplex x{2.0f, 3.0f};
    const aclblasComplex y{5.0f, -1.0f};
    aclblasComplex matrix{1.0f, 1.0f};

    ASSERT_EQ(
        aclblasCgeru_npu(CgeruArch22Test::handle_, 1, 1, &alpha, &x, INT_MIN, &y, INT_MIN, &matrix, 1),
        ACLBLAS_STATUS_SUCCESS);
    // A = 1+1i + (2+3i)*(5-1i) = 14+14i。
    EXPECT_FLOAT_EQ(matrix.real, 14.0f);
    EXPECT_FLOAT_EQ(matrix.imag, 14.0f);
}

// alpha==NULL / x==NULL / y==NULL / A==NULL 均由库返回 INVALID_VALUE。
TEST_F(CgeruArch22Test, NullPointersReturnInvalidValue)
{
    const aclblasComplex alpha{1.0f, 0.0f};
    std::vector<aclblasComplex> x(2, aclblasComplex{1.0f, 0.0f});
    std::vector<aclblasComplex> y(2, aclblasComplex{2.0f, 0.0f});
    std::vector<aclblasComplex> matrix(4, aclblasComplex{0.0f, 0.0f});

    EXPECT_EQ(
        aclblasCgeru_npu(
            CgeruArch22Test::handle_, 2, 2, nullptr, x.data(), 1, y.data(), 1, matrix.data(), 2),
        ACLBLAS_STATUS_INVALID_VALUE);
    EXPECT_EQ(
        aclblasCgeru_npu(
            CgeruArch22Test::handle_, 2, 2, &alpha, nullptr, 1, y.data(), 1, matrix.data(), 2),
        ACLBLAS_STATUS_INVALID_VALUE);
    EXPECT_EQ(
        aclblasCgeru_npu(
            CgeruArch22Test::handle_, 2, 2, &alpha, x.data(), 1, nullptr, 1, matrix.data(), 2),
        ACLBLAS_STATUS_INVALID_VALUE);
    EXPECT_EQ(
        aclblasCgeru_npu(
            CgeruArch22Test::handle_, 2, 2, &alpha, x.data(), 1, y.data(), 1, nullptr, 2),
        ACLBLAS_STATUS_INVALID_VALUE);
}

// cgeru 计算 x(i) * y(j)，不是 x(i) * conj(y(j))。1x1 时：
// (1+2i) * (3+4i) = -5 + 10i；共轭实现会得到 11 + 2i。
TEST_F(CgeruArch22Test, NoConjugationOnOneByOne)
{
    const aclblasComplex alpha{1.0f, 0.0f};
    const aclblasComplex x{1.0f, 2.0f};
    const aclblasComplex y{3.0f, 4.0f};
    aclblasComplex matrix{0.0f, 0.0f};

    ASSERT_EQ(
        aclblasCgeru_npu(CgeruArch22Test::handle_, 1, 1, &alpha, &x, 1, &y, 1, &matrix, 1),
        ACLBLAS_STATUS_SUCCESS);
    EXPECT_FLOAT_EQ(matrix.real, -5.0f);
    EXPECT_FLOAT_EQ(matrix.imag, 10.0f);
}

// alpha=(0,0) 是合法的 no-op：A 的每个元素（含 padding 与 NaN/Inf 位模式）都不变。
TEST_F(CgeruArch22Test, ZeroAlphaDoesNotWriteA)
{
    const int m = 3;
    const int n = 2;
    const int lda = 5;
    const aclblasComplex alpha{0.0f, -0.0f};
    std::vector<aclblasComplex> x = MakeComplexArray(m, parseFill("RANDOM_NORM_5_5"), 31U);
    std::vector<aclblasComplex> y = MakeComplexArray(n, parseFill("RANDOM_NORM_5_5"), 32U);
    std::vector<aclblasComplex> matrix(
        static_cast<size_t>(lda) * static_cast<size_t>(n),
        aclblasComplex{std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity()});
    const std::vector<aclblasComplex> original = matrix;

    ASSERT_EQ(
        aclblasCgeru_npu(CgeruArch22Test::handle_, m, n, &alpha, x.data(), 1, y.data(), 1, matrix.data(), lda),
        ACLBLAS_STATUS_SUCCESS);
    EXPECT_TRUE(BitIdentical(matrix, original));
}

// 负步长结果必须与相同逻辑向量的正步长结果逐位一致（kx/ky 预移位正确）。
TEST_F(CgeruArch22Test, NegativeStridesMatchPositive)
{
    const int m = 4;
    const int n = 3;
    const int lda = 5;
    const aclblasComplex alpha{0.5f, -0.25f};
    const BlasFillMode fill = parseFill("RANDOM_NORM_5_5");

    const std::vector<aclblasComplex> xPositive = MakeComplexArray(m, fill, 101U);
    const std::vector<aclblasComplex> yPositive = MakeComplexArray(n, fill, 102U);
    const std::vector<aclblasComplex> aBase = MakeComplexArray(static_cast<int64_t>(lda) * n, fill, 103U);

    std::vector<aclblasComplex> xNegative(m);
    for (int i = 0; i < m; ++i) {
        xNegative[static_cast<size_t>(m - 1 - i)] = xPositive[static_cast<size_t>(i)];
    }
    std::vector<aclblasComplex> yNegative(n);
    for (int j = 0; j < n; ++j) {
        yNegative[static_cast<size_t>(n - 1 - j)] = yPositive[static_cast<size_t>(j)];
    }

    std::vector<aclblasComplex> positive = aBase;
    std::vector<aclblasComplex> negativeX = aBase;
    std::vector<aclblasComplex> negativeY = aBase;
    ASSERT_EQ(
        aclblasCgeru_npu(
            CgeruArch22Test::handle_, m, n, &alpha, xPositive.data(), 1, yPositive.data(), 1, positive.data(), lda),
        ACLBLAS_STATUS_SUCCESS);
    ASSERT_EQ(
        aclblasCgeru_npu(
            CgeruArch22Test::handle_, m, n, &alpha, xNegative.data(), -1, yPositive.data(), 1, negativeX.data(), lda),
        ACLBLAS_STATUS_SUCCESS);
    ASSERT_EQ(
        aclblasCgeru_npu(
            CgeruArch22Test::handle_, m, n, &alpha, xPositive.data(), 1, yNegative.data(), -1, negativeY.data(), lda),
        ACLBLAS_STATUS_SUCCESS);

    EXPECT_TRUE(BitIdentical(positive, negativeX));
    EXPECT_TRUE(BitIdentical(positive, negativeY));
}

// incx=incy=2 的非连续访问，与接口级 golden 比较。
TEST_F(CgeruArch22Test, NonUnitStrideTwo)
{
    const int m = 5;
    const int n = 3;
    const int lda = 7;
    const aclblasComplex alpha{0.75f, -0.5f};
    const BlasFillMode fill = parseFill("RANDOM_NORM_5_5");
    std::vector<aclblasComplex> x = MakeComplexArray(PhysicalVectorLength(m, 2), fill, 111U);
    std::vector<aclblasComplex> y = MakeComplexArray(PhysicalVectorLength(n, 2), fill, 112U);
    std::vector<aclblasComplex> matrix = MakeComplexArray(static_cast<int64_t>(lda) * n, fill, 113U);

    ExpectMatchesGolden(CgeruArch22Test::handle_, m, n, lda, alpha, x, 2, y, 2, matrix, "NonUnitStrideTwo");
}

// lda>m 时，属于 padding 行的元素必须原样保留。
TEST_F(CgeruArch22Test, LdaPaddingUntouched)
{
    const int m = 3;
    const int n = 4;
    const int lda = 7;
    const aclblasComplex alpha{0.25f, 0.5f};
    const BlasFillMode fill = parseFill("RANDOM_NORM_5_5");
    std::vector<aclblasComplex> x = MakeComplexArray(m, fill, 121U);
    std::vector<aclblasComplex> y = MakeComplexArray(n, fill, 122U);
    std::vector<aclblasComplex> matrix = MakeComplexArray(static_cast<int64_t>(lda) * n, fill, 123U);
    const std::vector<aclblasComplex> original = matrix;

    ASSERT_EQ(
        aclblasCgeru_npu(CgeruArch22Test::handle_, m, n, &alpha, x.data(), 1, y.data(), 1, matrix.data(), lda),
        ACLBLAS_STATUS_SUCCESS);
    for (int j = 0; j < n; ++j) {
        for (int i = m; i < lda; ++i) {
            const size_t index = static_cast<size_t>(i) + static_cast<size_t>(j) * static_cast<size_t>(lda);
            EXPECT_EQ(matrix[index].real, original[index].real) << "padding index " << index;
            EXPECT_EQ(matrix[index].imag, original[index].imag) << "padding index " << index;
        }
    }
    const std::vector<aclblasComplex> golden =
        MakeCgeruGolden(CgeruArch22Test::handle_, m, n, alpha, x, 1, y, 1, original, lda);
    EXPECT_TRUE(VerifyAgainstGolden(matrix, golden, "LdaPadding"));
}

// m==1 或 n==1 的退化形状。
TEST_F(CgeruArch22Test, DegenerateOneDimension)
{
    const aclblasComplex alpha{0.75f, -0.5f};
    const BlasFillMode fill = parseFill("RANDOM_NORM_5_5");

    {
        const int m = 1;
        const int n = 8;
        const int lda = 1;
        std::vector<aclblasComplex> x = MakeComplexArray(m, fill, 131U);
        std::vector<aclblasComplex> y = MakeComplexArray(n, fill, 132U);
        std::vector<aclblasComplex> matrix = MakeComplexArray(static_cast<int64_t>(lda) * n, fill, 133U);
        ExpectMatchesGolden(CgeruArch22Test::handle_, m, n, lda, alpha, x, 1, y, 1, matrix, "DegenerateM1");
    }
    {
        const int m = 8;
        const int n = 1;
        const int lda = 8;
        std::vector<aclblasComplex> x = MakeComplexArray(m, fill, 134U);
        std::vector<aclblasComplex> y = MakeComplexArray(n, fill, 135U);
        std::vector<aclblasComplex> matrix = MakeComplexArray(static_cast<int64_t>(lda) * n, fill, 136U);
        ExpectMatchesGolden(CgeruArch22Test::handle_, m, n, lda, alpha, x, 1, y, 1, matrix, "DegenerateN1");
    }
}

// 64x64，alpha 同时含实部和虚部，验证一般的复数乘加路径。
TEST_F(CgeruArch22Test, Small64x64WithComplexAlpha)
{
    const int m = 64;
    const int n = 64;
    const int lda = 64;
    const aclblasComplex alpha{0.75f, -0.5f};
    const BlasFillMode fill = parseFill("RANDOM_NORM_5_5");
    std::vector<aclblasComplex> x = MakeComplexArray(m, fill, 141U);
    std::vector<aclblasComplex> y = MakeComplexArray(n, fill, 142U);
    std::vector<aclblasComplex> matrix = MakeComplexArray(static_cast<int64_t>(lda) * n, fill, 143U);

    ExpectMatchesGolden(CgeruArch22Test::handle_, m, n, lda, alpha, x, 1, y, 1, matrix, "Small64x64");
}

// alpha 取 (1,0) / (0,1) / (0,0) / Inf / NaN：
//   (1,0)：A += x*y，普通复数乘加；
//   (0,1)：A += i*x*y，按复数乘法展开，不做任何共轭；
//   (0,0)：标准 quick return，A 不变（见 ZeroAlphaDoesNotWriteA）；
//   Inf/NaN：BLAS 不对 alpha 做数值检查，按 IEEE 754 传播，结果非有限。
TEST_F(CgeruArch22Test, AlphaSpecialValues)
{
    const int m = 2;
    const int n = 2;
    const int lda = 2;
    const BlasFillMode fill = parseFill("RANDOM_NORM_5_5");
    const std::vector<aclblasComplex> x = MakeComplexArray(m, fill, 151U);
    const std::vector<aclblasComplex> y = MakeComplexArray(n, fill, 152U);
    const std::vector<aclblasComplex> aBase = MakeComplexArray(static_cast<int64_t>(lda) * n, fill, 153U);

    ExpectMatchesGolden(
        CgeruArch22Test::handle_, m, n, lda, aclblasComplex{1.0f, 0.0f}, x, 1, y, 1, aBase, "AlphaOne");
    ExpectMatchesGolden(
        CgeruArch22Test::handle_, m, n, lda, aclblasComplex{0.0f, 1.0f}, x, 1, y, 1, aBase, "AlphaI");
    ExpectNonFiniteResult(
        CgeruArch22Test::handle_, m, n, lda, aclblasComplex{std::numeric_limits<float>::infinity(), 0.0f}, x, y,
        aBase, false);
    ExpectNonFiniteResult(
        CgeruArch22Test::handle_, m, n, lda, aclblasComplex{std::numeric_limits<float>::quiet_NaN(), 0.0f}, x, y,
        aBase, true);
}

// y 列为 0 时整列跳过（Netlib 与 arch35 的语义）：零列不参与累加，即使 x 含 Inf/NaN
// 也不会让该列变成 NaN。
TEST_F(CgeruArch22Test, ZeroYColumnSkipsInfAndNanX)
{
    const int m = 2;
    const int n = 2;
    const int lda = 2;
    const aclblasComplex alpha{1.0f, 0.0f};
    std::vector<aclblasComplex> x(2);
    x[0] = aclblasComplex{std::numeric_limits<float>::infinity(), 0.0f};
    x[1] = aclblasComplex{std::numeric_limits<float>::quiet_NaN(), 1.0f};
    std::vector<aclblasComplex> y(2);
    y[0] = aclblasComplex{0.0f, 0.0f};
    y[1] = aclblasComplex{1.0f, 0.0f};
    std::vector<aclblasComplex> matrix{{1.0f, 0.0f}, {2.0f, 0.0f}, {3.0f, 0.0f}, {4.0f, 0.0f}};

    ASSERT_EQ(
        aclblasCgeru_npu(CgeruArch22Test::handle_, m, n, &alpha, x.data(), 1, y.data(), 1, matrix.data(), lda),
        ACLBLAS_STATUS_SUCCESS);
    // 零 y 列：A(:,0) 保持原值，不被 x 的 Inf/NaN 污染。
    EXPECT_FLOAT_EQ(matrix[0].real, 1.0f);
    EXPECT_FLOAT_EQ(matrix[0].imag, 0.0f);
    EXPECT_FLOAT_EQ(matrix[1].real, 2.0f);
    EXPECT_FLOAT_EQ(matrix[1].imag, 0.0f);
    // 非零 y 列：A(0,1) += Inf，A(1,1) += NaN。
    EXPECT_TRUE(std::isinf(matrix[2].real));
    EXPECT_TRUE(std::isnan(matrix[3].real));
}

// 非 1 步长、负步长与 lda>m 的组合，用接口级 golden 校验。
TEST_F(CgeruArch22Test, NonUnitAndNegativeStrides)
{
    struct StridePair {
        int incx;
        int incy;
    };
    const std::vector<StridePair> strides{{1, 1}, {2, 1}, {1, -2}, {-2, 3}, {-1, -3}, {3, -2}};
    const aclblasComplex alpha{0.75f, -0.5f};
    const BlasFillMode fill = parseFill("RANDOM_NORM_5_5");
    const int m = 5;
    const int n = 4;
    const int lda = 8;
    uint32_t seed = 20261101U;

    for (const StridePair& stride : strides) {
        const std::string caseName =
            "strides_incx" + std::to_string(stride.incx) + "_incy" + std::to_string(stride.incy);
        SCOPED_TRACE(caseName);
        std::vector<aclblasComplex> x = MakeComplexArray(PhysicalVectorLength(m, stride.incx), fill, seed);
        std::vector<aclblasComplex> y = MakeComplexArray(PhysicalVectorLength(n, stride.incy), fill, seed + 1U);
        std::vector<aclblasComplex> matrix = MakeComplexArray(static_cast<int64_t>(lda) * n, fill, seed + 2U);

        ExpectMatchesGolden(
            CgeruArch22Test::handle_, m, n, lda, alpha, x, stride.incx, y, stride.incy, matrix, caseName);
        seed += 10U;
    }
}

// 常见 tile 边界附近的 m/n；lda=m+2，padding 行也参与比较。
TEST_F(CgeruArch22Test, SizeBoundariesAndLdaPadding)
{
    const aclblasComplex alpha{0.75f, -0.5f};
    const BlasFillMode fill = parseFill("RANDOM_NORM_5_5");
    const std::vector<int> ms{1, 2, 31, 33, 63, 65, 129, 1025, 4095, 4097};
    const std::vector<int> ns{1, 9, 17, 33, 65};
    uint32_t seed = 20261022U;

    for (int m : ms) {
        for (int n : ns) {
            if (static_cast<int64_t>(m) * n > 300000) {
                continue;
            }
            const int lda = m + 2;
            const std::string caseName = "boundary_m" + std::to_string(m) + "_n" + std::to_string(n);
            SCOPED_TRACE(caseName);
            std::vector<aclblasComplex> x = MakeComplexArray(m, fill, seed);
            std::vector<aclblasComplex> y = MakeComplexArray(n, fill, seed + 1U);
            std::vector<aclblasComplex> matrix = MakeComplexArray(static_cast<int64_t>(lda) * n, fill, seed + 2U);

            ExpectMatchesGolden(CgeruArch22Test::handle_, m, n, lda, alpha, x, 1, y, 1, matrix, caseName);
            seed += 3U;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CSV 驱动用例
// ─────────────────────────────────────────────────────────────────────────────

INSTANTIATE_TEST_SUITE_P(
    CgeruArch22, CgeruArch22Test,
    ::testing::ValuesIn(GetCasesFromCsv<CgeruParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<CgeruParam>);

struct CsvGolden {
    std::vector<aclblasComplex> aOriginal;
    std::vector<aclblasComplex> golden;
    bool zeroAlpha = false;
    bool hasGolden = false;
};

// 根据 CSV 参数准备参考结果：alpha==0 时保留 A 原始副本做逐位比较，否则用接口级
// CPU 参考实现（cblas_cgeru）生成 golden；空指针用例、非法 lda 或空 A 用例不生成
// 参考结果（只校验状态码）。
static CsvGolden BuildCsvGolden(
    aclblasHandle_t handle, const CgeruParam& p, const std::vector<aclblasComplex>& xHost,
    const std::vector<aclblasComplex>& yHost, const std::vector<aclblasComplex>& aHost, const aclblasComplex& alpha)
{
    CsvGolden result;
    // 大尺寸用例（最大 4096x4096，A 为 128 MiB）的 host 内存控制：alpha!=0 时只保留
    // aHost + golden 两份；alpha==0 时才额外保存 aOriginal 做逐位比较（这类用例通常较小）。
    result.zeroAlpha = (p.alphaReal == 0.0f && p.alphaImag == 0.0f);
    // 空指针用例（如 x/y 为空）没有合法的 x/y 缓冲区，host 参考实现不能读取。
    const bool hasLogicalBuffers = !xHost.empty() && !yHost.empty();
    // lda < max(1,m) 是非法参数（库在分配设备内存前就返回 INVALID_VALUE）：
    // aHost 只按 lda*n 分配，此时按列主序算 golden 会写穿 aHost 的堆块，
    // 曾导致 TC_ED_117 崩溃、gtest 进程 abort、JSON 报告缺失（验收 1000 CRASH）。
    const bool validLdaForGolden = p.lda >= std::max(1, p.m);
    const bool hasLogicalProblem = p.m <= 0 || p.n <= 0 || aHost.empty() || !hasLogicalBuffers;
    if (hasLogicalProblem || !validLdaForGolden) {
        return result;
    }
    if (result.zeroAlpha) {
        result.aOriginal = aHost;
    } else {
        result.golden = MakeCgeruGolden(handle, p.m, p.n, alpha, xHost, p.incx, yHost, p.incy, aHost, p.lda);
    }
    result.hasGolden = true;
    return result;
}

TEST_P(CgeruArch22Test, CsvDriven)
{
    const CgeruParam& p = GetParam();
    const std::vector<aclblasComplex> xHost = MakeComplexArray(AllocationVectorLength(p.m, p.incx), p.x, p.randomSeed);
    const std::vector<aclblasComplex> yHost =
        MakeComplexArray(AllocationVectorLength(p.n, p.incy), p.y, p.randomSeed + 1U);
    const int64_t aCount = (p.lda > 0 && p.n > 0) ? static_cast<int64_t>(p.lda) * p.n : 0;
    std::vector<aclblasComplex> aHost = MakeComplexArray(aCount, p.A, p.randomSeed + 2U);

    const aclblasComplex alpha{p.alphaReal, p.alphaImag};
    const aclblasComplex* alphaPtr = p.alphaIsNull ? nullptr : &alpha;
    const aclblasComplex* xPtr = xHost.empty() ? nullptr : xHost.data();
    const aclblasComplex* yPtr = yHost.empty() ? nullptr : yHost.data();
    aclblasComplex* aPtr = aHost.empty() ? nullptr : aHost.data();

    // 参考结果必须在调用库之前算好：库调用会原地改写 aHost（A += alpha*x*y'）。
    const CsvGolden expect = BuildCsvGolden(CgeruArch22Test::handle_, p, xHost, yHost, aHost, alpha);

    const aclblasStatus_t status =
        aclblasCgeru_npu(CgeruArch22Test::handle_, p.m, p.n, alphaPtr, xPtr, p.incx, yPtr, p.incy, aPtr, p.lda);
    EXPECT_EQ(static_cast<int>(status), static_cast<int>(p.expectResult)) << p.caseName;
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return;
    }
    if (!expect.hasGolden) {
        return;
    }
    // alpha==0：A 必须逐位不变（A 可能含 NaN/Inf），故按字节比较。
    if (expect.zeroAlpha) {
        EXPECT_TRUE(BitIdentical(aHost, expect.aOriginal)) << p.caseName;
        return;
    }
    EXPECT_TRUE(VerifyAgainstGolden(aHost, expect.golden, p.caseName));
}
