/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <climits>
#include <cstring>
#include <vector>
#include <securec.h>
#include "blas_test.h"
#include "cher2_golden.h"
#include "cher2_npu_wrapper.h"

namespace {

size_t Cher2VecSize(int n, int inc)
{
    return (n > 0 && inc != 0 && inc != INT_MIN) ? (1ULL + static_cast<uint64_t>(n - 1) * std::abs(inc)) : 0U;
}

bool Cher2ValidStructure(const Cher2Param& p)
{
    return p.n >= 0 && p.incx != 0 && p.incy != 0 && p.incx != INT_MIN && p.incy != INT_MIN &&
           p.lda >= std::max(1, p.n) && !p.nullAlpha && (p.uplo == ACLBLAS_UPPER || p.uplo == ACLBLAS_LOWER);
}

void Cher2InitMatrix(const Cher2Param& p, std::vector<aclblasComplex>& a)
{
    uint32_t sentinelBits = 0x7fc12345U;
    float sentinel = 0.0f;
    ASSERT_EQ(EOK, memcpy_s(&sentinel, sizeof(sentinel), &sentinelBits, sizeof(sentinelBits)));
    for (int col = 0; col < p.n; ++col) {
        for (int row = 0; row < p.lda; ++row) {
            const size_t idx = static_cast<size_t>(col) * p.lda + row;
            const bool selected = row < p.n && (p.uplo == ACLBLAS_UPPER ? row <= col : row >= col);
            if (!selected) {
                a[idx] = {sentinel, sentinel};
            } else if (row == col) {
                a[idx].imag = sentinel;
            }
        }
    }
}

void VerifyCher2Matrix(
    const Cher2Param& p, const std::vector<aclblasComplex>& actual, const std::vector<aclblasComplex>& expected,
    const std::vector<aclblasComplex>& original)
{
    cher2_test::Accuracy realAcc;
    cher2_test::Accuracy imagAcc;
    for (int col = 0; col < p.n; ++col) {
        for (int row = 0; row < p.lda; ++row) {
            const size_t idx = static_cast<size_t>(col) * p.lda + row;
            const bool selected = row < p.n && (p.uplo == ACLBLAS_UPPER ? row <= col : row >= col);
            if (!selected) {
                ASSERT_EQ(std::memcmp(&actual[idx], &original[idx], sizeof(aclblasComplex)), 0) << p.caseName;
                continue;
            }
            if (row == col) {
                uint32_t imagBits = 0;
                ASSERT_EQ(EOK, memcpy_s(&imagBits, sizeof(imagBits), &actual[idx].imag, sizeof(imagBits)));
                EXPECT_EQ(imagBits, 0U) << p.caseName;
            }
            realAcc.Add(actual[idx].real, expected[idx].real);
            imagAcc.Add(actual[idx].imag, expected[idx].imag);
        }
    }
    if (!realAcc.Pass() || !imagAcc.Pass()) {
        realAcc.Print(p.caseName, "real");
        imagAcc.Print(p.caseName, "imag");
    }
    EXPECT_TRUE(realAcc.Pass()) << p.caseName;
    EXPECT_TRUE(imagAcc.Pass()) << p.caseName;
}

} // namespace

class Cher2Test : public BlasTest<Cher2Param> {};

TEST_F(Cher2Test, NullHandle)
{
    const aclblasComplex alpha{1.0f, 0.0f};
    aclblasComplex v{1.0f, 0.0f};
    EXPECT_EQ(
        static_cast<int>(aclblasCher2(nullptr, ACLBLAS_UPPER, 1, &alpha, &v, 1, &v, 1, &v, 1)),
        static_cast<int>(ACLBLAS_STATUS_HANDLE_IS_NULLPTR));
}

INSTANTIATE_TEST_SUITE_P(
    Cher2, Cher2Test, ::testing::ValuesIn(GetCasesFromCsv<Cher2Param>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<Cher2Param>);

void ReadAndCheckCher2Buffers(
    const Cher2Param& p, const cher2_arch22_test::Cher2NpuWrapper& dX, const cher2_arch22_test::Cher2NpuWrapper& dY,
    const cher2_arch22_test::Cher2NpuWrapper& dA, const std::vector<aclblasComplex>& x,
    const std::vector<aclblasComplex>& y, std::vector<aclblasComplex>& a)
{
    std::vector<aclblasComplex> xReadback = x;
    std::vector<aclblasComplex> yReadback = y;
    std::string err;
    ASSERT_TRUE(dX.Read(xReadback, err)) << p.caseName << " dX: " << err;
    ASSERT_TRUE(dY.Read(yReadback, err)) << p.caseName << " dY: " << err;
    ASSERT_TRUE(dA.Read(a, err)) << p.caseName << " dA: " << err;
    if (!x.empty()) {
        EXPECT_EQ(std::memcmp(xReadback.data(), x.data(), x.size() * sizeof(aclblasComplex)), 0) << p.caseName;
    }
    if (!y.empty()) {
        EXPECT_EQ(std::memcmp(yReadback.data(), y.data(), y.size() * sizeof(aclblasComplex)), 0) << p.caseName;
    }
}

TEST_P(Cher2Test, CsvDriven)
{
    const auto& p = GetParam();
    const bool valid = Cher2ValidStructure(p);
    std::vector<aclblasComplex> x = cher2_test::MakeInput(Cher2VecSize(p.n, p.incx), p.xFill, p.randomSeed, p.normal);
    std::vector<aclblasComplex> y =
        cher2_test::MakeInput(Cher2VecSize(p.n, p.incy), p.yFill, p.randomSeed + 100U, p.normal);
    const size_t aElems =
        valid ? static_cast<size_t>(p.n) * static_cast<size_t>(p.lda) : (p.aFill == "NULLPTR" ? 0 : 1);
    std::vector<aclblasComplex> a = cher2_test::MakeInput(aElems, p.aFill, p.randomSeed + 200U, p.normal);
    if (valid && !a.empty()) {
        Cher2InitMatrix(p, a);
    }
    const std::vector<aclblasComplex> original = a;

    const size_t guard = 68;
    cher2_arch22_test::Cher2NpuWrapper dX(x, guard);
    cher2_arch22_test::Cher2NpuWrapper dY(y, guard);
    cher2_arch22_test::Cher2NpuWrapper dA(a, guard);
    ASSERT_TRUE(dX.Ok() && dY.Ok() && dA.Ok())
        << p.caseName << ": " << dX.Error() << " " << dY.Error() << " " << dA.Error();
    const aclblasComplex* alphaPtr = p.nullAlpha ? nullptr : &p.alpha;
    const aclblasStatus_t st =
        aclblasCher2(Cher2Test::handle_, p.uplo, p.n, alphaPtr, dX.Data(), p.incx, dY.Data(), p.incy, dA.Data(), p.lda);
    EXPECT_EQ(static_cast<int>(st), static_cast<int>(p.expectResult)) << p.caseName;
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return;
    }
    ASSERT_EQ(aclrtSynchronizeStream(Cher2Test::stream_), ACL_SUCCESS);
    ReadAndCheckCher2Buffers(p, dX, dY, dA, x, y, a);
    if (HasFatalFailure()) {
        return;
    }
    if (p.n == 0 || (p.alpha.real == 0.0f && p.alpha.imag == 0.0f)) {
        if (!a.empty()) {
            EXPECT_EQ(std::memcmp(a.data(), original.data(), a.size() * sizeof(aclblasComplex)), 0) << p.caseName;
        }
        return;
    }
    ASSERT_FALSE(a.empty()) << p.caseName;
    std::vector<aclblasComplex> expected = original;
    Cher2Golden(p, x.data(), y.data(), expected.data());
    VerifyCher2Matrix(p, a, expected, original);
}
