/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>
#include "blas_test.h"
#include "ctrsv_param.h"
#include "ctrsv_npu_wrapper.h"

namespace {
void ClearStoredTriangle(std::vector<aclblasComplex>& matrix, int n, int lda, aclblasFillMode_t uplo)
{
    for (int col = 0; col < n; ++col) {
        for (int row = 0; row < n; ++row) {
            if (uplo == ACLBLAS_UPPER ? row <= col : row >= col)
                matrix[static_cast<size_t>(col) * lda + row] = {0.0f, 0.0f};
        }
    }
}

void CheckDiagonalRange(
    aclblasHandle_t handle, int n, int lda, int incx, int exponent, aclblasFillMode_t uplo, aclblasOperation_t trans,
    bool complexDiagonal)
{
    SCOPED_TRACE(
        ::testing::Message() << "n=" << n << " lda=" << lda << " incx=" << incx << " exponent=" << exponent
                             << " uplo=" << uplo << " trans=" << trans);
    const float poison = std::numeric_limits<float>::quiet_NaN();
    const float scale = std::ldexp(1.0f, exponent);
    const size_t stride = incx < 0 ? static_cast<size_t>(-static_cast<int64_t>(incx)) : static_cast<size_t>(incx);
    if (stride == 0U) {
        ADD_FAILURE() << "incx must be nonzero";
        return;
    }
    const size_t span = static_cast<size_t>(n - 1) * stride + 1U;
    std::vector<aclblasComplex> matrix(static_cast<size_t>(lda) * n, {poison, poison});
    std::vector<aclblasComplex> rhs(span, {123.0f, -456.0f});
    ClearStoredTriangle(matrix, n, lda, uplo);
    for (int col = 0; col < n; ++col) {
        matrix[static_cast<size_t>(col) * lda + col] = {scale, complexDiagonal ? scale : 0.0f};
        rhs[static_cast<size_t>(col) * stride] = {
            scale, complexDiagonal ? (trans == ACLBLAS_OP_C ? -scale : scale) : 0.0f};
    }
    ASSERT_EQ(
        aclblasCtrsv_npu(handle, uplo, trans, ACLBLAS_NON_UNIT, n, matrix.data(), lda, rhs.data(), incx),
        ACLBLAS_STATUS_SUCCESS);
    for (size_t i = 0; i < span; ++i) {
        if (i % stride == 0U) {
            ASSERT_FLOAT_EQ(rhs[i].real, 1.0f) << i;
            ASSERT_FLOAT_EQ(rhs[i].imag, 0.0f) << i;
        } else {
            ASSERT_FLOAT_EQ(rhs[i].real, 123.0f) << i;
            ASSERT_FLOAT_EQ(rhs[i].imag, -456.0f) << i;
        }
    }
}

void CheckRangeModes(aclblasHandle_t handle, int n, int lda, int incx)
{
    for (auto uplo : {ACLBLAS_UPPER, ACLBLAS_LOWER}) {
        for (auto trans : {ACLBLAS_OP_N, ACLBLAS_OP_T, ACLBLAS_OP_C}) {
            for (int exponent : {-100, 80, 127}) {
                ASSERT_NO_FATAL_FAILURE(CheckDiagonalRange(handle, n, lda, incx, exponent, uplo, trans, false));
                ASSERT_NO_FATAL_FAILURE(CheckDiagonalRange(handle, n, lda, incx, exponent, uplo, trans, true));
            }
        }
    }
}

aclblasComplex ExpectedLane(int lane)
{
    return {static_cast<float>(lane % 11 + 1) / 8.0f, static_cast<float>(lane % 7 - 3) / 16.0f};
}

void CheckStridedDependency(aclblasHandle_t handle, int n, int incx, aclblasFillMode_t uplo, aclblasOperation_t trans)
{
    SCOPED_TRACE(::testing::Message() << "n=" << n << " incx=" << incx << " uplo=" << uplo << " trans=" << trans);
    const int lda = n + 5;
    const size_t stride = incx < 0 ? static_cast<size_t>(-static_cast<int64_t>(incx)) : static_cast<size_t>(incx);
    if (stride == 0U) {
        ADD_FAILURE() << "incx must be nonzero";
        return;
    }
    const bool upper = (uplo == ACLBLAS_UPPER) != (trans != ACLBLAS_OP_N);
    const float poison = std::numeric_limits<float>::quiet_NaN();
    std::vector<aclblasComplex> matrix(static_cast<size_t>(lda) * n, {poison, poison});
    std::vector<aclblasComplex> rhs(static_cast<size_t>(n - 1) * stride + 1U, {123.0f, -456.0f});
    ClearStoredTriangle(matrix, n, lda, uplo);
    for (int row = 0; row < n; ++row) {
        const auto expected = ExpectedLane(row);
        aclblasComplex b{2.0f * expected.real, 2.0f * expected.imag};
        matrix[static_cast<size_t>(row) * lda + row] = {2.0f, 0.0f};
        const int dependency = upper ? row + 1 : row - 1;
        if (dependency >= 0 && dependency < n) {
            const auto x = ExpectedLane(dependency);
            b.real += 0.25f * x.real - 0.125f * x.imag;
            b.imag += 0.25f * x.imag + 0.125f * x.real;
            const size_t offset = trans == ACLBLAS_OP_N ? static_cast<size_t>(dependency) * lda + row :
                                                          static_cast<size_t>(row) * lda + dependency;
            matrix[offset] = {0.25f, trans == ACLBLAS_OP_C ? -0.125f : 0.125f};
        }
        const size_t position = static_cast<size_t>(incx < 0 ? n - 1 - row : row) * stride;
        rhs[position] = b;
    }
    ASSERT_EQ(
        aclblasCtrsv_npu(handle, uplo, trans, ACLBLAS_NON_UNIT, n, matrix.data(), lda, rhs.data(), incx),
        ACLBLAS_STATUS_SUCCESS);
    for (int row = 0; row < n; ++row) {
        const auto expected = ExpectedLane(row);
        const size_t position = static_cast<size_t>(incx < 0 ? n - 1 - row : row) * stride;
        ASSERT_FLOAT_EQ(rhs[position].real, expected.real) << row;
        ASSERT_FLOAT_EQ(rhs[position].imag, expected.imag) << row;
    }
    for (size_t position = 0; position < rhs.size(); ++position) {
        if (position % stride != 0U) {
            ASSERT_FLOAT_EQ(rhs[position].real, 123.0f) << position;
            ASSERT_FLOAT_EQ(rhs[position].imag, -456.0f) << position;
        }
    }
}
} // namespace

class CtrsvRangeTest : public BlasTest<CtrsvParam> {};

TEST_F(CtrsvRangeTest, ContiguousDispatchBoundaries)
{
    for (int n : {7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129})
        ASSERT_NO_FATAL_FAILURE(CheckRangeModes(handle_, n, n, 1));
}

TEST_F(CtrsvRangeTest, StridedAndPaddedBoundaries)
{
    for (int n : {63, 64, 65, 127, 128, 129}) {
        for (int incx : {-3, -1, 1, 3})
            ASSERT_NO_FATAL_FAILURE(CheckRangeModes(handle_, n, n + 5, incx));
    }
}

TEST_F(CtrsvRangeTest, StridedDependenciesAcrossDmaTiles)
{
    for (int n : {63, 64, 65, 127, 128, 129}) {
        for (int incx : {-3, -1, 1, 3}) {
            for (auto uplo : {ACLBLAS_UPPER, ACLBLAS_LOWER}) {
                for (auto trans : {ACLBLAS_OP_N, ACLBLAS_OP_T, ACLBLAS_OP_C})
                    ASSERT_NO_FATAL_FAILURE(CheckStridedDependency(handle_, n, incx, uplo, trans));
            }
        }
    }
}
