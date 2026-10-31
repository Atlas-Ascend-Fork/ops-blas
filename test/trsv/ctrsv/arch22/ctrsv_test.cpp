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
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>
#include "blas_test.h"
#include "csv_loader.h"
#include "fill.h"
#include "verify.h"
#include "ctrsv_param.h"
#include "ctrsv_golden.h"
#include "ctrsv_npu_wrapper.h"

namespace {
void CheckFiniteResult(const std::vector<aclblasComplex>& rhs, const std::vector<aclblasComplex>& expected)
{
    ASSERT_EQ(rhs.size(), expected.size());
    for (size_t i = 0; i < rhs.size(); ++i) {
        ASSERT_TRUE(std::isfinite(rhs[i].real) && std::isfinite(rhs[i].imag)) << i;
        ASSERT_NEAR(rhs[i].real, expected[i].real, 2.0e-5f) << i;
        ASSERT_NEAR(rhs[i].imag, expected[i].imag, 2.0e-5f) << i;
    }
}

void AccumulateKnownProduct(double& real, double& imag, aclblasComplex coefficient, aclblasComplex expected)
{
    real +=
        static_cast<double>(coefficient.real) * expected.real - static_cast<double>(coefficient.imag) * expected.imag;
    imag +=
        static_cast<double>(coefficient.real) * expected.imag + static_cast<double>(coefficient.imag) * expected.real;
}

template <typename Coefficient>
void BuildKnownSystem(
    int n, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag,
    const std::vector<aclblasComplex>& expected, std::vector<aclblasComplex>& matrix, std::vector<aclblasComplex>& rhs,
    Coefficient coefficientAt)
{
    const bool upper = (uplo == ACLBLAS_UPPER) != (trans != ACLBLAS_OP_N);
    for (int row = 0; row < n; ++row) {
        double real = 0.0, imag = 0.0;
        for (int col = 0; col < n; ++col) {
            if (upper ? col < row : col > row)
                continue;
            const auto coefficient = coefficientAt(row, col);
            if (row != col || diag != ACLBLAS_UNIT) {
                const size_t offset =
                    trans == ACLBLAS_OP_N ? static_cast<size_t>(col) * n + row : static_cast<size_t>(row) * n + col;
                matrix[offset] = {coefficient.real, trans == ACLBLAS_OP_C ? -coefficient.imag : coefficient.imag};
            }
            AccumulateKnownProduct(real, imag, coefficient, expected[col]);
        }
        rhs[row] = {static_cast<float>(real), static_cast<float>(imag)};
    }
}

aclblasComplex LifecycleCoefficient(int row, int col, aclblasDiagType_t diag, bool prepared)
{
    if (!prepared) {
        return row == col ? (diag == ACLBLAS_UNIT ? aclblasComplex{1.0f, 0.0f} : aclblasComplex{1.5f, 0.125f}) :
                            aclblasComplex{std::abs(row - col) == 1 ? -0.25f : 0.00390625f, 0.001953125f};
    }
    const float scale = diag == ACLBLAS_UNIT ? 1.0f : std::ldexp(1.0f, row % 9 - 4);
    return row == col ?
               (diag == ACLBLAS_UNIT ? aclblasComplex{1.0f, 0.0f} : aclblasComplex{scale, scale * 0.125f}) :
               aclblasComplex{
                   scale * (std::abs(row - col) == 1 ? -0.25f : 0.00006103515625f), scale * 0.000030517578125f};
}

void CheckLifecycleCase(
    aclblasHandle_t handle, int n, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag,
    bool prepared)
{
    SCOPED_TRACE(::testing::Message() << "n=" << n << " uplo=" << uplo << " trans=" << trans << " diag=" << diag);
    const float poison = std::numeric_limits<float>::quiet_NaN();
    std::vector<aclblasComplex> matrix(static_cast<size_t>(n) * n, {poison, poison});
    std::vector<aclblasComplex> expected(n), rhs(n);
    for (int i = 0; i < n; ++i)
        expected[i] = {0.75f + (i % 7) * 0.03125f, -0.125f};
    BuildKnownSystem(n, uplo, trans, diag, expected, matrix, rhs, [diag, prepared](int row, int col) {
        return LifecycleCoefficient(row, col, diag, prepared);
    });
    ASSERT_EQ(aclblasCtrsv_npu(handle, uplo, trans, diag, n, matrix.data(), n, rhs.data(), 1), ACLBLAS_STATUS_SUCCESS);
    ASSERT_NO_FATAL_FAILURE(CheckFiniteResult(rhs, expected));
}

void CheckResidentLifecycle(aclblasHandle_t handle, const std::vector<int>& sizes, bool prepared)
{
    for (int n : sizes) {
        for (auto uplo : {ACLBLAS_UPPER, ACLBLAS_LOWER}) {
            for (auto trans : {ACLBLAS_OP_N, ACLBLAS_OP_T, ACLBLAS_OP_C}) {
                for (auto diag : {ACLBLAS_UNIT, ACLBLAS_NON_UNIT}) {
                    ASSERT_NO_FATAL_FAILURE(CheckLifecycleCase(handle, n, uplo, trans, diag, prepared));
                }
            }
        }
    }
}
} // namespace

class CtrsvArch22Test : public BlasTest<CtrsvParam> {};

TEST_F(CtrsvArch22Test, NullHandle)
{
    EXPECT_EQ(
        aclblasCtrsv_npu(nullptr, ACLBLAS_LOWER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT, 1, nullptr, 1, nullptr, 1),
        ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
}

TEST_F(CtrsvArch22Test, TinyResidentLifecycle)
{
    ASSERT_NO_FATAL_FAILURE(CheckResidentLifecycle(
        handle_, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 31, 32, 33, 63, 64}, false));
}

static void CheckTinyNetlibExtremeOrderCase(
    aclblasHandle_t handle, int n, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag,
    bool extremeMatrix)
{
    SCOPED_TRACE(
        ::testing::Message() << "n=" << n << " uplo=" << uplo << " trans=" << trans << " diag=" << diag
                             << " extremeMatrix=" << extremeMatrix);
    auto matrix =
        makeBlasComplexMatrix(n, n, n, parseFill(extremeMatrix ? "RANDOM_EXTREME" : "RANDOM_NORM_5_5"), 20260080);
    auto rhs =
        makeBlasComplexMatrix(1, n, n, parseFill(extremeMatrix ? "RANDOM_NORM_5_5" : "RANDOM_EXTREME"), 20260121);
    for (int i = 0; i < n; ++i) {
        float& real = matrix[i * (n + 1)].real;
        real += real >= 0.0f ? float(n) : -float(n);
    }
    auto expected = rhs;
    aclblasCtrsv_cpu(uplo, trans, diag, n, matrix.data(), n, expected.data(), 1);
    ASSERT_EQ(aclblasCtrsv_npu(handle, uplo, trans, diag, n, matrix.data(), n, rhs.data(), 1), ACLBLAS_STATUS_SUCCESS);
    VerifyConfig cfg;
    cfg.mode = PrecisionMode::MERE_MARE;
    EXPECT_TRUE(Verifier::verifyMereMareComplexFloat(
        rhs.data(), expected.data(), static_cast<size_t>(n), cfg.mereThreshold, cfg.mareMultiplier, cfg.epsilonForRel,
        "TinyNetlibExtremeOrder n=" + std::to_string(n) + " uplo=" + std::to_string(static_cast<int>(uplo)) +
            " trans=" + std::to_string(static_cast<int>(trans)) + " diag=" + std::to_string(static_cast<int>(diag)) +
            " extremeMatrix=" + (extremeMatrix ? "true" : "false")));
}

TEST_F(CtrsvArch22Test, TinyNetlibExtremeOrder)
{
    for (const int n : {17, 31, 32}) {
        for (const auto uplo : {ACLBLAS_UPPER, ACLBLAS_LOWER}) {
            for (const auto trans : {ACLBLAS_OP_N, ACLBLAS_OP_T, ACLBLAS_OP_C}) {
                for (const auto diag : {ACLBLAS_UNIT, ACLBLAS_NON_UNIT}) {
                    for (const bool extremeMatrix : {false, true}) {
                        ASSERT_NO_FATAL_FAILURE(
                            CheckTinyNetlibExtremeOrderCase(handle_, n, uplo, trans, diag, extremeMatrix));
                    }
                }
            }
        }
    }
}

static void FillFullRangeSystem(
    int n, int incx, int lda, bool upper, aclblasOperation_t trans, aclblasDiagType_t diag,
    std::vector<aclblasComplex>& matrix, std::vector<aclblasComplex>& rhs)
{
    for (int row = 0; row < n; ++row) {
        const int x = incx > 0 ? row * incx : (n - 1 - row) * -incx;
        rhs[x] = {diag == ACLBLAS_UNIT ? 0x1p64f : 0x1p122f, -0x1p120f};
        for (int col = 0; col < n; ++col) {
            if (upper ? col < row : col > row)
                continue;
            if (row == col && diag == ACLBLAS_UNIT)
                continue;
            const aclblasComplex a = row == col ? aclblasComplex{0x1p120f, 0x1p119f} :
                                                  aclblasComplex{diag == ACLBLAS_UNIT ? 0x1p126f : 0.125f, -0.25f};
            const size_t index =
                trans == ACLBLAS_OP_N ? static_cast<size_t>(col) * lda + row : static_cast<size_t>(row) * lda + col;
            matrix[index] = {a.real, trans == ACLBLAS_OP_C ? -a.imag : a.imag};
        }
    }
}

static void CheckFullRangeResult(
    const std::vector<aclblasComplex>& rhs, const std::vector<aclblasComplex>& expected, aclblasDiagType_t diag)
{
    for (int i = 0; i < static_cast<int>(rhs.size()); ++i) {
        const float actual[] = {rhs[i].real, rhs[i].imag};
        const float golden[] = {expected[i].real, expected[i].imag};
        for (int component = 0; component < 2; ++component) {
            if (diag == ACLBLAS_NON_UNIT)
                ASSERT_TRUE(std::isfinite(actual[component]));
            if (std::isnan(golden[component]))
                EXPECT_TRUE(std::isnan(actual[component]));
            else if (std::isinf(golden[component]))
                EXPECT_EQ(actual[component], golden[component]);
            else
                EXPECT_NEAR(
                    actual[component], golden[component], 2.0e-6f * std::max(1.0f, std::abs(golden[component])));
        }
    }
}

TEST_F(CtrsvArch22Test, FullRangeArithmeticRegression)
{
    // Exercise the numerical boundary independently of CSV names/seeds,
    // including padded matrices and reversed/non-unit vector strides.
    for (const int n : {17, 31, 32}) {
        for (const int incx : {1, 2, -2}) {
            for (const auto uplo : {ACLBLAS_UPPER, ACLBLAS_LOWER}) {
                for (const auto trans : {ACLBLAS_OP_N, ACLBLAS_OP_T, ACLBLAS_OP_C}) {
                    for (const auto diag : {ACLBLAS_NON_UNIT, ACLBLAS_UNIT}) {
                        SCOPED_TRACE(
                            ::testing::Message() << "n=" << n << " incx=" << incx << " uplo=" << uplo
                                                 << " trans=" << trans << " diag=" << diag);
                        const int lda = n + (incx != 1);
                        const int length = 1 + (n - 1) * std::abs(incx);
                        const float poison = std::numeric_limits<float>::quiet_NaN();
                        std::vector<aclblasComplex> matrix(static_cast<size_t>(lda) * n, {poison, poison});
                        std::vector<aclblasComplex> rhs(length, {123.0f, -456.0f});
                        const bool upper = (uplo == ACLBLAS_UPPER) != (trans != ACLBLAS_OP_N);
                        FillFullRangeSystem(n, incx, lda, upper, trans, diag, matrix, rhs);
                        auto expected = rhs;
                        aclblasCtrsv_cpu(uplo, trans, diag, n, matrix.data(), lda, expected.data(), incx);
                        ASSERT_EQ(
                            aclblasCtrsv_npu(handle_, uplo, trans, diag, n, matrix.data(), lda, rhs.data(), incx),
                            ACLBLAS_STATUS_SUCCESS);
                        ASSERT_NO_FATAL_FAILURE(CheckFullRangeResult(rhs, expected, diag));
                    }
                }
            }
        }
    }
}

static void CheckUnitNanDependencyBoundaryCase(
    aclblasHandle_t handle, int n, aclblasFillMode_t uplo, aclblasOperation_t trans, int boundary, bool upper)
{
    SCOPED_TRACE(
        ::testing::Message() << "n=" << n << " uplo=" << uplo << " trans=" << trans << " boundary=" << boundary);
    std::vector<aclblasComplex> matrix(static_cast<size_t>(n) * n, {0.0f, 0.0f});
    std::vector<aclblasComplex> rhs(n, {0.75f, -0.125f});
    for (int col = 0; col < n; ++col) {
        for (int row = 0; row < n; ++row) {
            if (uplo == ACLBLAS_UPPER ? row > col : row < col)
                continue;
            matrix[static_cast<size_t>(col) * n + row] =
                row == col ? aclblasComplex{1.0f, 0.0f} : aclblasComplex{0x1p-10f, -0x1p-11f};
        }
    }
    const int poisonRow = upper ? n - 1 - boundary : boundary;
    const float nan = std::numeric_limits<float>::quiet_NaN();
    rhs[poisonRow] = {nan, nan};
    auto expected = rhs;
    aclblasCtrsv_cpu(uplo, trans, ACLBLAS_UNIT, n, matrix.data(), n, expected.data(), 1);
    ASSERT_EQ(
        aclblasCtrsv_npu(handle, uplo, trans, ACLBLAS_UNIT, n, matrix.data(), n, rhs.data(), 1),
        ACLBLAS_STATUS_SUCCESS);
    for (int ordinal = 0; ordinal < n; ++ordinal) {
        const int row = upper ? n - 1 - ordinal : ordinal;
        if (ordinal >= boundary) {
            ASSERT_TRUE(std::isnan(rhs[row].real) && std::isnan(rhs[row].imag)) << row;
        } else {
            ASSERT_TRUE(std::isfinite(rhs[row].real) && std::isfinite(rhs[row].imag)) << row;
            EXPECT_NEAR(rhs[row].real, expected[row].real, 2.0e-6f) << row;
            EXPECT_NEAR(rhs[row].imag, expected[row].imag, 2.0e-6f) << row;
        }
    }
}

TEST_F(CtrsvArch22Test, UnitNanDependencyBoundary)
{
    // Cover both solve directions and differently aligned leading panels.
    // NaNs in an unsolved RHS row must not disturb the independent prefix.
    for (const int n : {65, 71, 120, 127, 451, 482}) {
        for (const auto uplo : {ACLBLAS_UPPER, ACLBLAS_LOWER}) {
            for (const auto trans : {ACLBLAS_OP_N, ACLBLAS_OP_T, ACLBLAS_OP_C}) {
                const bool upper = (uplo == ACLBLAS_UPPER) != (trans != ACLBLAS_OP_N);
                for (const int boundary : {0, 1, 55, 57, 63, n - 1}) {
                    ASSERT_NO_FATAL_FAILURE(
                        CheckUnitNanDependencyBoundaryCase(handle_, n, uplo, trans, boundary, upper));
                }
            }
        }
    }
}

TEST_F(CtrsvArch22Test, MediumPreparedLifecycle)
{
    ASSERT_NO_FATAL_FAILURE(CheckResidentLifecycle(
        handle_, {65, 66, 127, 128, 149, 202, 231, 256, 331, 396, 448, 451, 477, 500, 506, 512, 3136}, true));
}

TEST_F(CtrsvArch22Test, UnitSingletonQuickReturn)
{
    const float poison = std::numeric_limits<float>::quiet_NaN();
    const aclblasComplex matrix{poison, poison};
    const aclblasComplex expected{3.25f, -1.75f};
    CtrsvDeviceBuffers buffers;
    ASSERT_EQ(CtrsvAllocAndCopy(buffers.a, &matrix, sizeof(matrix)), ACLBLAS_STATUS_SUCCESS);
    ASSERT_EQ(CtrsvAllocAndCopy(buffers.x, &expected, sizeof(expected)), ACLBLAS_STATUS_SUCCESS);
    const auto* a = static_cast<const aclblasComplex*>(buffers.a);
    auto* x = static_cast<aclblasComplex*>(buffers.x);
    for (const auto uplo : {ACLBLAS_UPPER, ACLBLAS_LOWER}) {
        for (const auto trans : {ACLBLAS_OP_N, ACLBLAS_OP_T, ACLBLAS_OP_C}) {
            for (const int incx : {-7, -1, 1, 9}) {
                ASSERT_EQ(aclblasCtrsv(handle_, uplo, trans, ACLBLAS_UNIT, 1, a, 1, x, incx), ACLBLAS_STATUS_SUCCESS);
            }
        }
    }
    EXPECT_EQ(
        aclblasCtrsv(handle_, ACLBLAS_UPPER, ACLBLAS_OP_N, ACLBLAS_UNIT, 1, nullptr, 1, x, 1),
        ACLBLAS_STATUS_INVALID_VALUE);
    EXPECT_EQ(
        aclblasCtrsv(handle_, ACLBLAS_UPPER, ACLBLAS_OP_N, ACLBLAS_UNIT, 1, a, 1, nullptr, 1),
        ACLBLAS_STATUS_INVALID_VALUE);
    EXPECT_EQ(
        aclblasCtrsv(handle_, ACLBLAS_UPPER, ACLBLAS_OP_N, ACLBLAS_UNIT, 1, a, 0, x, 1), ACLBLAS_STATUS_INVALID_VALUE);
    for (const int incx : {0, std::numeric_limits<int>::min()}) {
        EXPECT_EQ(
            aclblasCtrsv(handle_, ACLBLAS_UPPER, ACLBLAS_OP_N, ACLBLAS_UNIT, 1, a, 1, x, incx),
            ACLBLAS_STATUS_INVALID_VALUE);
    }
    EXPECT_EQ(
        aclblasCtrsv(handle_, static_cast<aclblasFillMode_t>(-1), ACLBLAS_OP_N, ACLBLAS_UNIT, 1, a, 1, x, 1),
        ACLBLAS_STATUS_INVALID_VALUE);
    EXPECT_EQ(
        aclblasCtrsv(handle_, ACLBLAS_UPPER, static_cast<aclblasOperation_t>(-1), ACLBLAS_UNIT, 1, a, 1, x, 1),
        ACLBLAS_STATUS_INVALID_VALUE);
    ASSERT_EQ(aclrtSynchronizeDevice(), ACL_SUCCESS);
    aclblasComplex actual{};
    ASSERT_EQ(aclrtMemcpy(&actual, sizeof(actual), x, sizeof(actual), ACL_MEMCPY_DEVICE_TO_HOST), ACL_SUCCESS);
    EXPECT_FLOAT_EQ(actual.real, expected.real);
    EXPECT_FLOAT_EQ(actual.imag, expected.imag);
}

static void CheckFiniteNPanelLifecycleCase(
    aclblasHandle_t handle, int n, bool blockDiagonal, aclblasFillMode_t uplo, aclblasDiagType_t diag)
{
    SCOPED_TRACE(
        ::testing::Message() << "n=" << n << " uplo=" << uplo << " diag=" << diag
                             << " block_diagonal=" << blockDiagonal);
    const float poison = std::numeric_limits<float>::quiet_NaN();
    std::vector<aclblasComplex> matrix(static_cast<size_t>(n) * n, {poison, poison});
    std::vector<aclblasComplex> expected(n), rhs(n);
    for (int i = 0; i < n; ++i) {
        expected[i] = {0.5f + (i % 13) * 0.03125f, -0.25f + (i % 7) * 0.015625f};
    }
    for (int row = 0; row < n; ++row) {
        double real = 0.0, imag = 0.0;
        for (int col = 0; col < n; ++col) {
            if (uplo == ACLBLAS_UPPER ? col < row : col > row)
                continue;
            if (blockDiagonal && row / 64 != col / 64) {
                matrix[static_cast<size_t>(col) * n + row] = {0.0f, 0.0f};
                continue;
            }
            const int distance = std::abs(row - col);
            const float scale = diag == ACLBLAS_UNIT ? 1.0f : std::ldexp(1.0f, row % 9 - 4);
            const aclblasComplex value = distance == 0 ?
                                             aclblasComplex{scale, diag == ACLBLAS_UNIT ? 0.0f : scale * 0.125f} :
                                             aclblasComplex{
                                                 scale * (distance == 1 ? (std::min(row, col) % 2 ? -1.25f : -0.5f) :
                                                                          ((row + col) % 5 - 2) * 0.00006103515625f),
                                                 scale * ((row + 2 * col) % 7 - 3) * 0.000030517578125f};
            if (distance != 0 || diag != ACLBLAS_UNIT) {
                matrix[static_cast<size_t>(col) * n + row] = value;
            }
            AccumulateKnownProduct(real, imag, value, expected[col]);
        }
        rhs[row] = {static_cast<float>(real), static_cast<float>(imag)};
    }
    ASSERT_EQ(
        aclblasCtrsv_npu(handle, uplo, ACLBLAS_OP_N, diag, n, matrix.data(), n, rhs.data(), 1), ACLBLAS_STATUS_SUCCESS);
    ASSERT_NO_FATAL_FAILURE(CheckFiniteResult(rhs, expected));
}

TEST_F(CtrsvArch22Test, FiniteNPanelLifecycle)
{
    for (const int n : {512, 576, 1024, 2048, 3072, 4096}) {
        for (const bool blockDiagonal : {false, true}) {
            for (const auto uplo : {ACLBLAS_UPPER, ACLBLAS_LOWER}) {
                for (const auto diag : {ACLBLAS_UNIT, ACLBLAS_NON_UNIT}) {
                    ASSERT_NO_FATAL_FAILURE(CheckFiniteNPanelLifecycleCase(handle_, n, blockDiagonal, uplo, diag));
                }
            }
        }
    }
}

TEST_F(CtrsvArch22Test, ResidentUnitPanelLifecycle)
{
    for (const int n : {512, 2048, 4096}) {
        for (const auto uplo : {ACLBLAS_UPPER, ACLBLAS_LOWER}) {
            for (const auto trans : {ACLBLAS_OP_T, ACLBLAS_OP_C}) {
                SCOPED_TRACE(::testing::Message() << "n=" << n << " uplo=" << uplo << " trans=" << trans);
                const float poison = std::numeric_limits<float>::quiet_NaN();
                std::vector<aclblasComplex> matrix(static_cast<size_t>(n) * n, {poison, poison});
                std::vector<aclblasComplex> expected(n), rhs(n);
                for (int i = 0; i < n; ++i) {
                    expected[i] = {0.5f + (i % 13) * 0.03125f, -0.25f + (i % 7) * 0.015625f};
                }
                const bool upper = uplo != ACLBLAS_UPPER;
                for (int row = 0; row < n; ++row) {
                    double real = expected[row].real, imag = expected[row].imag;
                    for (int col = upper ? row + 1 : 0; col < (upper ? n : row); ++col) {
                        const aclblasComplex a{0.001f, trans == ACLBLAS_OP_C ? -0.001f : 0.001f};
                        matrix[static_cast<size_t>(row) * n + col] = {0.001f, 0.001f};
                        AccumulateKnownProduct(real, imag, a, expected[col]);
                    }
                    rhs[row] = {static_cast<float>(real), static_cast<float>(imag)};
                }
                ASSERT_EQ(
                    aclblasCtrsv_npu(handle_, uplo, trans, ACLBLAS_UNIT, n, matrix.data(), n, rhs.data(), 1),
                    ACLBLAS_STATUS_SUCCESS);
                ASSERT_NO_FATAL_FAILURE(CheckFiniteResult(rhs, expected));
            }
        }
    }
}

static void CheckPreparedTailPanelLifecycleCase(
    aclblasHandle_t handle, int n, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag)
{
    SCOPED_TRACE(::testing::Message() << "n=" << n << " uplo=" << uplo << " trans=" << trans << " diag=" << diag);
    const float poison = std::numeric_limits<float>::quiet_NaN();
    std::vector<aclblasComplex> matrix(static_cast<size_t>(n) * n, {poison, poison});
    std::vector<aclblasComplex> expected(n), rhs(n);
    const bool upper = (uplo == ACLBLAS_UPPER) != (trans != ACLBLAS_OP_N);
    for (int i = 0; i < n; ++i) {
        expected[i] = {0.75f + (i % 11) * 0.03125f, -0.125f + (i % 7) * 0.015625f};
    }
    for (int row = 0; row < n; ++row) {
        double real = 0.0, imag = 0.0;
        for (int col = 0; col < n; ++col) {
            if (upper ? col < row : col > row)
                continue;
            const int distance = std::abs(row - col);
            const float scale = diag == ACLBLAS_UNIT ? 1.0f : std::ldexp(1.0f, row % 9 - 4);
            const aclblasComplex value =
                distance == 0 ? aclblasComplex{scale, diag == ACLBLAS_UNIT ? 0.0f : scale * 0.125f} :
                                aclblasComplex{
                                    scale * (distance == 1 ? -0.25f : ((row + col) % 5 - 2) * 0.00006103515625f),
                                    scale * ((row + 2 * col) % 7 - 3) * 0.000030517578125f};
            if (distance != 0 || diag != ACLBLAS_UNIT) {
                const size_t offset =
                    trans == ACLBLAS_OP_N ? static_cast<size_t>(col) * n + row : static_cast<size_t>(row) * n + col;
                matrix[offset] = {value.real, trans == ACLBLAS_OP_C ? -value.imag : value.imag};
            }
            AccumulateKnownProduct(real, imag, value, expected[col]);
        }
        rhs[row] = {static_cast<float>(real), static_cast<float>(imag)};
    }
    ASSERT_EQ(aclblasCtrsv_npu(handle, uplo, trans, diag, n, matrix.data(), n, rhs.data(), 1), ACLBLAS_STATUS_SUCCESS);
    ASSERT_NO_FATAL_FAILURE(CheckFiniteResult(rhs, expected));
}

TEST_F(CtrsvArch22Test, PreparedTailPanelLifecycle)
{
    for (const int n : {513, 514, 515, 543, 575, 1000, 2561, 4095}) {
        for (const auto uplo : {ACLBLAS_UPPER, ACLBLAS_LOWER}) {
            for (const auto trans : {ACLBLAS_OP_N, ACLBLAS_OP_T, ACLBLAS_OP_C}) {
                for (const auto diag : {ACLBLAS_UNIT, ACLBLAS_NON_UNIT}) {
                    ASSERT_NO_FATAL_FAILURE(CheckPreparedTailPanelLifecycleCase(handle_, n, uplo, trans, diag));
                }
            }
        }
    }
}

static void CheckGroupedDependencyBoundariesCase(
    aclblasHandle_t handle, int n, aclblasFillMode_t uplo, aclblasOperation_t trans)
{
    SCOPED_TRACE(::testing::Message() << "n=" << n << " uplo=" << uplo << " trans=" << trans);
    std::vector<aclblasComplex> matrix(static_cast<size_t>(n) * n, {0.0f, 0.0f});
    std::vector<aclblasComplex> expected(n), rhs(n, {0.0f, 0.0f});
    for (int i = 0; i < n; ++i) {
        expected[i] = {1.0f + static_cast<float>(i % 7) * 0.03125f, -0.25f + static_cast<float>(i % 5) * 0.015625f};
    }
    for (int row = 0; row < n; ++row) {
        for (const int distance : {0, 1, 64, 256}) {
            const int col = uplo == ACLBLAS_UPPER ? row - distance : row + distance;
            if (col < 0 || col >= n)
                continue;
            const aclblasComplex coefficient = distance == 0 ?
                                                   aclblasComplex{1.25f, 0.125f} :
                                                   aclblasComplex{distance == 1 ? -0.5f : 0.0625f, 0.03125f};
            matrix[static_cast<size_t>(row) * n + col] = {
                coefficient.real, trans == ACLBLAS_OP_C ? -coefficient.imag : coefficient.imag};
            rhs[row].real += coefficient.real * expected[col].real - coefficient.imag * expected[col].imag;
            rhs[row].imag += coefficient.real * expected[col].imag + coefficient.imag * expected[col].real;
        }
    }
    ASSERT_EQ(
        aclblasCtrsv_npu(handle, uplo, trans, ACLBLAS_NON_UNIT, n, matrix.data(), n, rhs.data(), 1),
        ACLBLAS_STATUS_SUCCESS);
    ASSERT_NO_FATAL_FAILURE(CheckFiniteResult(rhs, expected));
}

TEST_F(CtrsvArch22Test, GroupedDependencyBoundaries)
{
    // Known solutions exercise dependencies within a panel, across native
    // panels, and across groups without the CSV harness's diagonal boost.
    for (const int n : {2048, 4096}) {
        for (const auto uplo : {ACLBLAS_UPPER, ACLBLAS_LOWER}) {
            for (const auto trans : {ACLBLAS_OP_T, ACLBLAS_OP_C}) {
                ASSERT_NO_FATAL_FAILURE(CheckGroupedDependencyBoundariesCase(handle_, n, uplo, trans));
            }
        }
    }
}

static void CheckPreparedRowScalingCase(
    aclblasHandle_t handle, int n, int lda, aclblasFillMode_t uplo, aclblasOperation_t trans)
{
    SCOPED_TRACE(::testing::Message() << "n=" << n << " lda=" << lda << " uplo=" << uplo << " trans=" << trans);
    std::vector<aclblasComplex> matrix(static_cast<size_t>(lda) * n, {0.0f, 0.0f});
    std::vector<aclblasComplex> expected(n), rhs(n);
    for (int i = 0; i < n; ++i) {
        expected[i] = {0.75f + static_cast<float>(i % 11) * 0.03125f, -0.125f + static_cast<float>(i % 7) * 0.015625f};
    }
    for (int row = 0; row < n; ++row) {
        const float scale = std::ldexp(1.0f, row % 15 - 7);
        double real = 0.0, imag = 0.0;
        for (int col = 0; col < n; ++col) {
            if (uplo == ACLBLAS_UPPER ? col > row : col < row)
                continue;
            const int distance = std::abs(row - col);
            if (row / 64 != col / 64 && distance != 64 && distance != 256)
                continue;
            const aclblasComplex coefficient =
                distance == 0 ?
                    aclblasComplex{scale * (1.0f + (row % 3) * 0.125f), scale * (row % 2 ? -0.25f : 0.125f)} :
                    aclblasComplex{scale * (distance == 1 ? -0.25f : 0.00390625f), scale * 0.001953125f};
            matrix[static_cast<size_t>(row) * lda + col] = {
                coefficient.real, trans == ACLBLAS_OP_C ? -coefficient.imag : coefficient.imag};
            AccumulateKnownProduct(real, imag, coefficient, expected[col]);
        }
        rhs[row] = {static_cast<float>(real), static_cast<float>(imag)};
    }
    ASSERT_EQ(
        aclblasCtrsv_npu(handle, uplo, trans, ACLBLAS_NON_UNIT, n, matrix.data(), lda, rhs.data(), 1),
        ACLBLAS_STATUS_SUCCESS);
    ASSERT_NO_FATAL_FAILURE(CheckFiniteResult(rhs, expected));
}

TEST_F(CtrsvArch22Test, PreparedRowScaling)
{
    // Vary D by four orders of magnitude while keeping D^-1 A well conditioned.
    // Dense native blocks exercise every inverse coefficient; extra bands
    // exercise the second resident slot and cross-panel publication.
    for (const int n : {2048, 4096}) {
        for (const int lda : {n, n + 8}) {
            for (const auto uplo : {ACLBLAS_UPPER, ACLBLAS_LOWER}) {
                for (const auto trans : {ACLBLAS_OP_T, ACLBLAS_OP_C}) {
                    ASSERT_NO_FATAL_FAILURE(CheckPreparedRowScalingCase(handle_, n, lda, uplo, trans));
                }
            }
        }
    }
}

static std::vector<CtrsvParam> LoadCtrsvCases()
{
    const char* csvPath = std::getenv("CTRSV_TEST_CSV");
    return GetCasesFromCsv<CtrsvParam>(csvPath == nullptr ? ReplaceFileExtension2Csv(__FILE__) : csvPath);
}

INSTANTIATE_TEST_SUITE_P(Trsv, CtrsvArch22Test, ::testing::ValuesIn(LoadCtrsvCases()), PrintCaseInfoString<CtrsvParam>);

static void BenchmarkCtrsv(
    aclblasHandle_t handle, aclrtStream stream, const CtrsvParam& p, int n, std::vector<aclblasComplex>& aHost, int lda,
    std::vector<aclblasComplex>& xHost)
{
    const char* warmupEnv = std::getenv("CTRSV_PERF_WARMUP");
    const char* iterationsEnv = std::getenv("CTRSV_PERF_ITERATIONS");
    const int warmup = warmupEnv == nullptr ? 10 : std::atoi(warmupEnv);
    const int iterations = iterationsEnv == nullptr ? 60 : std::atoi(iterationsEnv);
    double averageUs = 0.0;
    ASSERT_EQ(
        aclblasCtrsv_npu_benchmark(
            handle, stream, p.uplo, p.trans, p.diag, n, aHost.data(), lda, xHost.data(), static_cast<int>(p.incx),
            warmup, iterations, &averageUs),
        ACLBLAS_STATUS_SUCCESS);
    std::cout << "[PERF] case=" << p.caseName << " n=" << n << " average_us=" << averageUs << " warmup=" << warmup
              << " iterations=" << iterations << " timing=batched_e2e" << std::endl;
}

static void CheckCsvResult(
    const CtrsvParam& p, int n, int stride, const std::vector<aclblasComplex>& xHost,
    const std::vector<aclblasComplex>& golden)
{
    std::vector<aclblasComplex> outLogical(n), goldLogical(n);
    for (int i = 0; i < n; ++i) {
        const int pos = p.incx < 0 ? (n - 1 - i) * stride : i * stride;
        outLogical[i] = xHost[pos];
        goldLogical[i] = golden[pos];
    }
    if (p.caseName == "TC_EX_0154" || p.caseName == "TC_EX_0184" || p.caseName == "TC_EX_0330" ||
        p.caseName == "TC_EX_0340") {
        for (int i = 0; i < n; i += 64) {
            std::cout << "[BLOCK_SAMPLE] case=" << p.caseName << " index=" << i << " output=(" << outLogical[i].real
                      << "," << outLogical[i].imag << ") golden=(" << goldLogical[i].real << "," << goldLogical[i].imag
                      << ")" << std::endl;
        }
    }
    VerifyConfig cfg;
    cfg.mode = PrecisionMode::MERE_MARE;
    EXPECT_TRUE(Verifier::verifyMereMareComplexFloat(
        outLogical.data(), goldLogical.data(), n, cfg.mereThreshold, cfg.mareMultiplier, cfg.epsilonForRel,
        p.caseName));
}

TEST_P(CtrsvArch22Test, CsvDriven)
{
    const auto& p = GetParam();
    const int n = static_cast<int>(p.n);
    const int allocN = std::max(1, n);
    const int lda = std::max(allocN, static_cast<int>(p.lda));
    auto aHost = makeBlasComplexMatrix(allocN, allocN, lda, p.A, p.randomSeed);
    const int stride = std::abs(static_cast<int>(p.incx));
    const size_t xCount = n > 0 ? static_cast<size_t>(1 + (n - 1) * stride) : 1;
    auto xHost = makeBlasComplexMatrix(1, static_cast<int>(xCount), static_cast<int>(xCount), p.x, p.randomSeed + 37);
    auto golden = xHost;
    if (n > 0 && !aHost.empty()) {
        const float boost = std::max(5.0f, static_cast<float>(n));
        for (int i = 0; i < n; ++i) {
            auto& d = aHost[i + i * lda];
            d.real += d.real >= 0.0f ? boost : -boost;
        }
    }
    const bool performanceCase = p.caseName.rfind("TC_PF_", 0U) == 0U;
    if (performanceCase) {
        ASSERT_NO_FATAL_FAILURE(BenchmarkCtrsv(handle_, stream_, p, n, aHost, lda, xHost));
        return;
    }
    if (p.expectResult != ACLBLAS_STATUS_SUCCESS || n == 0) {
        const auto ret = aclblasCtrsv_npu(
            CtrsvArch22Test::handle_, p.uplo, p.trans, p.diag, p.n, aHost.empty() ? nullptr : aHost.data(), p.lda,
            n == 0 ? nullptr : xHost.data(), p.incx);
        EXPECT_EQ(ret, p.expectResult);
        return;
    }
    auto ret = aclblasCtrsv_npu(
        CtrsvArch22Test::handle_, p.uplo, p.trans, p.diag, p.n, aHost.data(), p.lda, xHost.data(), p.incx);
    ASSERT_EQ(ret, ACLBLAS_STATUS_SUCCESS);
    aclblasCtrsv_cpu(p.uplo, p.trans, p.diag, n, aHost.data(), lda, golden.data(), static_cast<int>(p.incx));
    CheckCsvResult(p, n, stride, xHost, golden);
}
