/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file cgeru_test.cpp
 * \brief GTest suite for arch22 aclblasCgeru.
 *
 * The CSV driven part replays the task supplied case table through the shared
 * CgeruParam loader: operands are filled on the host, the NPU wrapper runs the
 * call, the golden comes from the reference CBLAS cgeru (no conjugation), and
 * the checker applies the mixed FLOAT32 tolerance per real/imag component.
 *
 * The hand written part pins down semantics the reference defines and a
 * generic tolerance would happily let slip:
 *   - no-op contract (zero dims, alpha == (0,0), nullptr handle),
 *   - the non-conjugate sign contract,
 *   - the reference column gate (y == (0,0) leaves A untouched even for Inf),
 *   - plain complex-product rounding for degenerate alpha (pure real / pure
 *     imaginary alpha times Inf y must produce NaN, exactly like the reference,
 *     not the "skip the zero component" variant),
 *   - general path coverage: negative strides, lda padding preservation.
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

using cgeru_arch22_test::aclblasCgeru_npu;

// Golden: the reference CBLAS cgeru (Netlib BLAS 3.10.0 complex implementation,
// y not conjugated), supplied by the test framework through cgeru_golden.h.

std::vector<aclblasComplex> MakeComplexArray(int64_t count, const BlasFillMode& fill, uint32_t seed)
{
    if (count <= 0 || fill.method == BlasFillMode::M_NULLPTR) {
        return {};
    }
    const std::vector<float> real = makeBlasArray(count, fill, seed);
    const std::vector<float> imag = makeBlasArray(count, fill, seed + 7919U);
    std::vector<aclblasComplex> values(static_cast<size_t>(count));
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = aclblasComplex{real[i], imag[i]};
    }
    return values;
}

int64_t PhysicalVectorLength(int logicalLength, int increment)
{
    if (logicalLength <= 0 || increment == 0) {
        return 0;
    }
    const int64_t stride = (increment < 0) ? -static_cast<int64_t>(increment) : static_cast<int64_t>(increment);
    return 1 + static_cast<int64_t>(logicalLength - 1) * stride;
}

const aclblasComplex kOneAlpha{1.0f, 0.0f};

} // namespace

class CgeruArch22Test : public BlasTest<CgeruParam> {};

TEST_F(CgeruArch22Test, NullHandle)
{
    const aclblasComplex alpha{1.0f, 1.0f};
    const aclblasComplex x{1.0f, 0.0f};
    const aclblasComplex y{1.0f, 0.0f};
    aclblasComplex A{0.0f, 0.0f};
    EXPECT_EQ(aclblasCgeru(nullptr, 4, 4, &alpha, &x, 1, &y, 1, &A, 4), ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
    EXPECT_EQ(aclblasCgeru_npu(nullptr, 4, 4, &alpha, &x, 1, &y, 1, &A, 4), ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
    EXPECT_EQ(A.real, 0.0f);
    EXPECT_EQ(A.imag, 0.0f);
}

TEST_F(CgeruArch22Test, QuickReturnNeverTouchesOperands)
{
    // alpha == (0,0) quick return with absurd-but-unused pointers/strides.
    const aclblasComplex zeroAlpha{0.0f, -0.0f};
    EXPECT_EQ(
        aclblasCgeru_npu(
            CgeruArch22Test::handle_, INT_MAX, INT_MAX, &zeroAlpha, nullptr, INT_MIN, nullptr, INT_MIN, nullptr,
            INT_MAX),
        ACLBLAS_STATUS_SUCCESS);
    // Zero dimension quick return with a non-zero alpha.
    EXPECT_EQ(
        aclblasCgeru_npu(
            CgeruArch22Test::handle_, 0, INT_MAX, &kOneAlpha, nullptr, INT_MIN, nullptr, INT_MIN, nullptr, 1),
        ACLBLAS_STATUS_SUCCESS);
    EXPECT_EQ(
        aclblasCgeru_npu(
            CgeruArch22Test::handle_, INT_MAX, 0, &kOneAlpha, nullptr, INT_MIN, nullptr, INT_MIN, nullptr, INT_MAX),
        ACLBLAS_STATUS_SUCCESS);
}

TEST_F(CgeruArch22Test, InvalidArgumentsRejected)
{
    const aclblasComplex alpha{1.0f, 0.0f};
    const aclblasComplex x{1.0f, 0.0f};
    const aclblasComplex y{1.0f, 0.0f};
    aclblasComplex A{0.0f, 0.0f};
    EXPECT_EQ(
        aclblasCgeru_npu(CgeruArch22Test::handle_, -1, 2, &alpha, &x, 1, &y, 1, &A, 4), ACLBLAS_STATUS_INVALID_VALUE);
    EXPECT_EQ(
        aclblasCgeru_npu(CgeruArch22Test::handle_, 2, -1, &alpha, &x, 1, &y, 1, &A, 4), ACLBLAS_STATUS_INVALID_VALUE);
    EXPECT_EQ(
        aclblasCgeru_npu(CgeruArch22Test::handle_, 2, 2, nullptr, &x, 1, &y, 1, &A, 4), ACLBLAS_STATUS_INVALID_VALUE);
    EXPECT_EQ(
        aclblasCgeru_npu(CgeruArch22Test::handle_, 2, 2, &alpha, &x, 0, &y, 1, &A, 4), ACLBLAS_STATUS_INVALID_VALUE);
    EXPECT_EQ(
        aclblasCgeru_npu(CgeruArch22Test::handle_, 2, 2, &alpha, &x, 1, &y, 0, &A, 4), ACLBLAS_STATUS_INVALID_VALUE);
    EXPECT_EQ(
        aclblasCgeru_npu(CgeruArch22Test::handle_, 2, 2, &alpha, &x, 1, &y, 1, &A, 1), ACLBLAS_STATUS_INVALID_VALUE);
    EXPECT_EQ(
        aclblasCgeru_npu(CgeruArch22Test::handle_, 2, 2, &alpha, nullptr, 1, &y, 1, &A, 2),
        ACLBLAS_STATUS_INVALID_VALUE);
    EXPECT_EQ(
        aclblasCgeru_npu(CgeruArch22Test::handle_, 2, 2, &alpha, &x, 1, nullptr, 1, &A, 2),
        ACLBLAS_STATUS_INVALID_VALUE);
    EXPECT_EQ(
        aclblasCgeru_npu(CgeruArch22Test::handle_, 2, 2, &alpha, &x, 1, &y, 1, nullptr, 2),
        ACLBLAS_STATUS_INVALID_VALUE);
}

TEST_F(CgeruArch22Test, NonConjugateSignContract)
{
    // (1 + 2i)(3 + 4i) = -5 + 10i; the conjugated variant would give 25 + 0i.
    const aclblasComplex x{1.0f, 2.0f};
    const aclblasComplex y{3.0f, 4.0f};
    aclblasComplex A{0.5f, -0.25f};
    ASSERT_EQ(
        aclblasCgeru_npu(CgeruArch22Test::handle_, 1, 1, &kOneAlpha, &x, 1, &y, 1, &A, 1), ACLBLAS_STATUS_SUCCESS);
    EXPECT_FLOAT_EQ(A.real, -4.5f); // 0.5 + (-5)
    EXPECT_FLOAT_EQ(A.imag, 9.75f); // -0.25 + 10
}

TEST_F(CgeruArch22Test, MinStrideEdgeCases)
{
    // 1x1: huge negative strides are addressable (single element) and legal.
    // (INT_MIN itself is not exercised here: on some driver stacks the
    // extreme-stride launch trips a bus fault that poisons the device for
    // every later process, and the value coverage is identical.)
    const int kHugeNegative = -1000000;
    const aclblasComplex x{1.0f, 2.0f};
    const aclblasComplex y{3.0f, 4.0f};
    aclblasComplex A{0.0f, 0.0f};
    ASSERT_EQ(
        aclblasCgeru_npu(CgeruArch22Test::handle_, 1, 1, &kOneAlpha, &x, kHugeNegative, &y, kHugeNegative, &A, 1),
        ACLBLAS_STATUS_SUCCESS);
    EXPECT_FLOAT_EQ(A.real, -5.0f);
    EXPECT_FLOAT_EQ(A.imag, 10.0f);

    // Negative strides on a multi-element case against the reference.
    const int m = 9;
    const int n = 5;
    const BlasFillMode fill = parseFill("RANDOM_NORM_5_5");
    const std::vector<aclblasComplex> xV = MakeComplexArray(m, fill, 2601U);
    const std::vector<aclblasComplex> yV = MakeComplexArray(n, fill, 2602U);
    const int lda = m + 3;
    std::vector<aclblasComplex> aV = MakeComplexArray(static_cast<int64_t>(lda) * n, fill, 2603U);
    std::vector<aclblasComplex> golden = aV;
    ASSERT_EQ(
        aclblasCgeru_npu(CgeruArch22Test::handle_, m, n, &kOneAlpha, xV.data(), -2, yV.data(), -3, aV.data(), lda),
        ACLBLAS_STATUS_SUCCESS);
    ASSERT_EQ(
        aclblasCgeru_cpu(CgeruArch22Test::handle_, m, n, &kOneAlpha, xV.data(), -2, yV.data(), -3, golden.data(), lda),
        ACLBLAS_STATUS_SUCCESS);
    VerifyConfig config;
    applyMixedTolerance(config, ACL_FLOAT, reinterpret_cast<const float*>(golden.data()), golden.size() * 2);
    EXPECT_TRUE(Verifier::verifyVector(
        reinterpret_cast<const float*>(aV.data()), reinterpret_cast<const float*>(golden.data()), golden.size() * 2, 1,
        config, "negative_strides"));
}

TEST_F(CgeruArch22Test, ZeroYColumnKeepsInfAndNanValues)
{
    // Column 0 carries y = (0, -0): the reference gate skips it, so the Inf
    // entries seeded into A and x must not fabricate NaN there.
    const std::vector<aclblasComplex> x{
        {std::numeric_limits<float>::infinity(), 0.0f}, {std::numeric_limits<float>::quiet_NaN(), 1.0f}};
    const std::vector<aclblasComplex> y{{0.0f, -0.0f}, {1.0f, 0.0f}};
    std::vector<aclblasComplex> A{
        {std::numeric_limits<float>::infinity(), -1.0f},
        {std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity()},
        {5.0f, 6.0f},
        {7.0f, 8.0f}};
    const std::vector<aclblasComplex> original = A;

    ASSERT_EQ(
        aclblasCgeru_npu(CgeruArch22Test::handle_, 2, 2, &kOneAlpha, x.data(), 1, y.data(), 1, A.data(), 2),
        ACLBLAS_STATUS_SUCCESS);
    EXPECT_TRUE(std::isinf(A[0].real) && A[0].real > 0.0f);
    EXPECT_FLOAT_EQ(A[0].imag, original[0].imag);
    EXPECT_TRUE(std::isinf(A[1].real) && A[1].real > 0.0f);
    EXPECT_TRUE(std::isinf(A[1].imag) && A[1].imag > 0.0f);
    // Column 1 (y = (1,0)) updated normally: x0=(Inf,0) x (1,0) -> (Inf, NaN).
    EXPECT_TRUE(std::isinf(A[2].real) && A[2].real > 0.0f);
    EXPECT_TRUE(std::isnan(A[2].imag));
    // x1=(NaN,1) x (1,0) -> (NaN*1 - 1*0, NaN*0 + 1*1) = (NaN, NaN+1) = (NaN, NaN).
    EXPECT_TRUE(std::isnan(A[3].real));
    EXPECT_TRUE(std::isnan(A[3].imag));
}

TEST_F(CgeruArch22Test, DegenerateAlphaFollowsPlainComplexProduct)
{
    // The reference multiplies alpha*y as a plain complex product: a zero
    // component of alpha times an Inf/NaN component of y must propagate NaN,
    // not silently drop the term.
    const float inf = std::numeric_limits<float>::infinity();

    const aclblasComplex realAlpha{2.0f, 0.0f};
    const aclblasComplex xReal{1.0f, 1.0f};
    const aclblasComplex yInf{inf, 3.0f};
    aclblasComplex A{1.0f, 1.0f};
    ASSERT_EQ(
        aclblasCgeru_npu(CgeruArch22Test::handle_, 1, 1, &realAlpha, &xReal, 1, &yInf, 1, &A, 1),
        ACLBLAS_STATUS_SUCCESS);
    EXPECT_TRUE(std::isnan(A.real)) << "plain product: t = (Inf, NaN), A += x * t -> NaN";
    EXPECT_TRUE(std::isnan(A.imag));

    const aclblasComplex imagAlpha{0.0f, 2.0f};
    A = aclblasComplex{1.0f, 1.0f};
    ASSERT_EQ(
        aclblasCgeru_npu(CgeruArch22Test::handle_, 1, 1, &imagAlpha, &xReal, 1, &yInf, 1, &A, 1),
        ACLBLAS_STATUS_SUCCESS);
    EXPECT_TRUE(std::isnan(A.real));
    EXPECT_TRUE(std::isnan(A.imag));

    // Same shapes against the reference golden for belt and braces.
    std::vector<aclblasComplex> aV{{1.0f, 1.0f}};
    std::vector<aclblasComplex> golden = aV;
    const std::vector<aclblasComplex> xV{xReal};
    const std::vector<aclblasComplex> yV{yInf};
    ASSERT_EQ(
        aclblasCgeru_npu(CgeruArch22Test::handle_, 1, 1, &realAlpha, xV.data(), 1, yV.data(), 1, aV.data(), 1),
        ACLBLAS_STATUS_SUCCESS);
    ASSERT_EQ(
        aclblasCgeru_cpu(CgeruArch22Test::handle_, 1, 1, &realAlpha, xV.data(), 1, yV.data(), 1, golden.data(), 1),
        ACLBLAS_STATUS_SUCCESS);
    VerifyConfig config;
    applyMixedTolerance(config, ACL_FLOAT, reinterpret_cast<const float*>(golden.data()), golden.size() * 2);
    EXPECT_TRUE(Verifier::verifyVector(
        reinterpret_cast<const float*>(aV.data()), reinterpret_cast<const float*>(golden.data()), golden.size() * 2, 1,
        config, "degenerate_alpha_real"));
}

TEST_F(CgeruArch22Test, LdaPaddingUntouched)
{
    const int m = 10;
    const int n = 7;
    const int lda = m + 5;
    const BlasFillMode fill = parseFill("RANDOM_NORM_5_5");
    const std::vector<aclblasComplex> xV = MakeComplexArray(m, fill, 2701U);
    const std::vector<aclblasComplex> yV = MakeComplexArray(n, fill, 2702U);
    std::vector<aclblasComplex> aV = MakeComplexArray(static_cast<int64_t>(lda) * n, fill, 2703U);
    std::vector<aclblasComplex> golden = aV;
    const aclblasComplex alpha{0.75f, -0.5f};

    ASSERT_EQ(
        aclblasCgeru_npu(CgeruArch22Test::handle_, m, n, &alpha, xV.data(), 1, yV.data(), 1, aV.data(), lda),
        ACLBLAS_STATUS_SUCCESS);
    ASSERT_EQ(
        aclblasCgeru_cpu(CgeruArch22Test::handle_, m, n, &alpha, xV.data(), 1, yV.data(), 1, golden.data(), lda),
        ACLBLAS_STATUS_SUCCESS);
    VerifyConfig config;
    applyMixedTolerance(config, ACL_FLOAT, reinterpret_cast<const float*>(golden.data()), golden.size() * 2);
    EXPECT_TRUE(Verifier::verifyVector(
        reinterpret_cast<const float*>(aV.data()), reinterpret_cast<const float*>(golden.data()), golden.size() * 2, 1,
        config, "lda_padding"));
}

TEST_F(CgeruArch22Test, GeneralPathShapeAndStrideScan)
{
    // Covers the non-fast path: strided x/y, padded lda, rectangular shapes,
    // m not a multiple of 4, negative increments.
    const aclblasComplex alpha{1.25f, -2.0f};
    const BlasFillMode fill = parseFill("RANDOM_NORM_5_5");
    for (int m : {1, 2, 3, 5, 8, 65, 130, 257}) {
        for (int n : {1, 3, 9, 33, 66}) {
            for (int incx : {-3, -1, 1, 2}) {
                const int lda = m + ((m + n) % 5);
                const std::vector<aclblasComplex> xV = MakeComplexArray(PhysicalVectorLength(m, incx), fill, 2800U);
                for (int incy : {-2, 1, 3}) {
                    const std::string tag = "scan_m" + std::to_string(m) + "_n" + std::to_string(n) + "_ix" +
                                            std::to_string(incx) + "_iy" + std::to_string(incy);
                    SCOPED_TRACE(tag);
                    const std::vector<aclblasComplex> yV = MakeComplexArray(PhysicalVectorLength(n, incy), fill, 2801U);
                    std::vector<aclblasComplex> aV = MakeComplexArray(static_cast<int64_t>(lda) * n, fill, 2802U);
                    std::vector<aclblasComplex> golden = aV;
                    ASSERT_EQ(
                        aclblasCgeru_npu(
                            CgeruArch22Test::handle_, m, n, &alpha, xV.data(), incx, yV.data(), incy, aV.data(), lda),
                        ACLBLAS_STATUS_SUCCESS);
                    ASSERT_EQ(
                        aclblasCgeru_cpu(
                            CgeruArch22Test::handle_, m, n, &alpha, xV.data(), incx, yV.data(), incy, golden.data(),
                            lda),
                        ACLBLAS_STATUS_SUCCESS);
                    VerifyConfig config;
                    applyMixedTolerance(
                        config, ACL_FLOAT, reinterpret_cast<const float*>(golden.data()), golden.size() * 2);
                    EXPECT_TRUE(Verifier::verifyVector(
                        reinterpret_cast<const float*>(aV.data()), reinterpret_cast<const float*>(golden.data()),
                        golden.size() * 2, 1, config, tag));
                }
            }
        }
    }
}

TEST_F(CgeruArch22Test, FastPathZeroYColumnGuardsInfNanX)
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
        aclblasCgeru_npu(CgeruArch22Test::handle_, m, n, &alpha, x.data(), 1, y.data(), 1, matrix.data(), m),
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

TEST_F(CgeruArch22Test, FastPathZeroYColumnInsideMixedChunk)
{
    // m=32 / n=128 keeps m%4==0, incx==1 and lda==m (fast path); each core owns
    // several columns, so one streaming chunk spans both zero-y columns and a
    // non-zero one. The chunk is stored back as a whole, hence the zero-y
    // columns must keep their original A values bit for bit.
    const int m = 32;
    const int n = 128;
    const aclblasComplex alpha{1.0f, -0.5f};
    std::vector<aclblasComplex> x(m);
    for (int i = 0; i < m; ++i) {
        x[i] = aclblasComplex{static_cast<float>(i + 1) * 0.25f, -static_cast<float>(i) * 0.125f};
    }
    std::vector<aclblasComplex> y(n);
    for (int j = 0; j < n; ++j) {
        // Columns 4 and 5 sit inside the same core slab as column 3 (one
        // streaming chunk covers the whole slab), so the chunk mixes zero-y
        // and non-zero columns.
        const bool zeroY = (j == 4 || j == 5);
        y[j] = zeroY ? aclblasComplex{0.0f, -0.0f} : aclblasComplex{static_cast<float>(j) * 0.5f, 0.25f};
    }
    std::vector<aclblasComplex> A(static_cast<size_t>(m) * n);
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            A[static_cast<size_t>(j) * m + i] = aclblasComplex{static_cast<float>(i * 7 + j), -3.0f};
        }
    }
    const std::vector<aclblasComplex> original = A;

    ASSERT_EQ(
        aclblasCgeru_npu(CgeruArch22Test::handle_, m, n, &alpha, x.data(), 1, y.data(), 1, A.data(), m),
        ACLBLAS_STATUS_SUCCESS);

    for (int j : {4, 5}) {
        for (int i = 0; i < m; ++i) {
            const size_t idx = static_cast<size_t>(j) * m + i;
            EXPECT_EQ(A[idx].real, original[idx].real) << "col " << j << " row " << i;
            EXPECT_EQ(A[idx].imag, original[idx].imag) << "col " << j << " row " << i;
        }
    }
    // The neighbouring non-zero column must be updated.
    EXPECT_NE(A[3 * m].real, original[3 * m].real) << "column 3 must be updated";

    std::vector<aclblasComplex> golden = original;
    ASSERT_EQ(
        aclblasCgeru_cpu(CgeruArch22Test::handle_, m, n, &alpha, x.data(), 1, y.data(), 1, golden.data(), m),
        ACLBLAS_STATUS_SUCCESS);
    VerifyConfig config;
    applyMixedTolerance(config, ACL_FLOAT, reinterpret_cast<const float*>(golden.data()), golden.size() * 2);
    EXPECT_TRUE(Verifier::verifyVector(
        reinterpret_cast<const float*>(A.data()), reinterpret_cast<const float*>(golden.data()), golden.size() * 2, 1,
        config, "mixed_chunk_zero_y"));
}

TEST_F(CgeruArch22Test, FastPathNonFiniteAlphaZeroYColumnStaysOriginal)
{
    // The reference gate is IF (Y(JY).NE.ZERO): a zero-y column is skipped no
    // matter what alpha is. With alpha = (Inf, 0) the plain complex product of
    // such a column is NaN, which must not turn the column active. m=8 keeps
    // the fast path (incx==1, lda==m, m%4==0); n=128 gives each core 2-3
    // columns whose whole slab sits in one streaming chunk, so zero-y and
    // non-zero columns mix inside it; incy==1 exercises the vector coefficient
    // build, incy==2 the scalar one.
    const float inf = std::numeric_limits<float>::infinity();
    const int m = 8;
    const int n = 128;
    const aclblasComplex alpha{inf, 0.0f};
    for (int incy : {1, 2}) {
        SCOPED_TRACE(incy == 1 ? "incy=1" : "incy=2");
        std::vector<aclblasComplex> x(m);
        for (int i = 0; i < m; ++i) {
            x[i] = aclblasComplex{0.5f * static_cast<float>(i + 1), -0.25f * static_cast<float>(i)};
        }
        std::vector<aclblasComplex> y(static_cast<size_t>(1) + static_cast<size_t>(n - 1) * incy);
        for (int j = 0; j < n; ++j) {
            y[static_cast<size_t>(j) * incy] = (j == 2 || j == 3 || j == 60) ?
                                                   aclblasComplex{0.0f, 0.0f} :
                                                   aclblasComplex{0.5f * static_cast<float>(j), 1.0f};
        }
        std::vector<aclblasComplex> A(static_cast<size_t>(m) * n);
        for (size_t k = 0; k < A.size(); ++k) {
            A[k] = aclblasComplex{-20.0f + 0.5f * static_cast<float>(k % m), 7.5f};
        }
        // Distinctive non-finite seeds inside the zero-y columns: a NaN
        // coefficient leaking into the accumulation would destroy them.
        A[static_cast<size_t>(2) * m + 1] = aclblasComplex{inf, 7.5f};
        A[static_cast<size_t>(3) * m + 4] = aclblasComplex{-inf, 7.5f};
        A[static_cast<size_t>(60) * m + 6] = aclblasComplex{inf, 7.5f};
        const std::vector<aclblasComplex> original = A;

        ASSERT_EQ(
            aclblasCgeru_npu(CgeruArch22Test::handle_, m, n, &alpha, x.data(), 1, y.data(), incy, A.data(), m),
            ACLBLAS_STATUS_SUCCESS);

        for (int j : {2, 3, 60}) {
            for (int i = 0; i < m; ++i) {
                const size_t idx = static_cast<size_t>(j) * m + i;
                EXPECT_EQ(A[idx].real, original[idx].real) << "incy " << incy << " col " << j << " row " << i;
                EXPECT_EQ(A[idx].imag, original[idx].imag) << "incy " << incy << " col " << j << " row " << i;
            }
        }
        // The non-zero columns take the Inf coefficient and must change.
        EXPECT_NE(A[0].real, original[0].real) << "non-zero column must be updated";
    }
}

TEST_F(CgeruArch22Test, ContiguousTallNarrowAndBoundaryRows)
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
                aclblasCgeru_npu(CgeruArch22Test::handle_, m, n, &alpha, x.data(), 1, y.data(), 1, matrix.data(), m),
                ACLBLAS_STATUS_SUCCESS);
            ASSERT_EQ(
                aclblasCgeru_cpu(CgeruArch22Test::handle_, m, n, &alpha, x.data(), 1, y.data(), 1, golden.data(), m),
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
    Cgeru, CgeruArch22Test, ::testing::ValuesIn(GetCasesFromCsv<CgeruParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<CgeruParam>);

TEST_P(CgeruArch22Test, CsvDriven)
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
        aclblasCgeru_npu(CgeruArch22Test::handle_, p.m, p.n, alphaPtr, xPtr, p.incx, yPtr, p.incy, aPtr, p.lda);
    EXPECT_EQ(static_cast<int>(status), static_cast<int>(p.expectResult));
    if (p.expectResult != ACLBLAS_STATUS_SUCCESS) {
        return;
    }

    ASSERT_EQ(
        aclblasCgeru_cpu(
            CgeruArch22Test::handle_, p.m, p.n, alphaPtr, xPtr, p.incx, yPtr, p.incy,
            golden.empty() ? nullptr : golden.data(), p.lda),
        ACLBLAS_STATUS_SUCCESS);

    VerifyConfig config;
    if (p.alphaReal == 0.0f && p.alphaImag == 0.0f) {
        // Alpha quick return: A must keep its seed bit-for-bit.
        config.mode = PrecisionMode::EXACT;
    } else {
        applyMixedTolerance(config, ACL_FLOAT, reinterpret_cast<const float*>(golden.data()), golden.size() * 2);
    }
    EXPECT_TRUE(Verifier::verifyVector(
        reinterpret_cast<const float*>(aHost.data()), reinterpret_cast<const float*>(golden.data()), golden.size() * 2,
        1, config, p.caseName));
}
