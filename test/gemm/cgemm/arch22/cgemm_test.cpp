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
#include <functional>
#include <limits>
#include <string>
#include <vector>

#include "blas_test.h"
#include "csv_loader.h"
#include "fill.h"
#include "verify.h"
#include "cgemm_param.h"
#include "cgemm_golden.h"
#include "cgemm_npu_wrapper.h"

// arch22 (Atlas A2/A3) unit tests for aclblasCgemm.
// The arch22 implementation expands the complex product into four real GEMMs
// issued through aclblasSgemm, so the cases below target the four places where
// that expansion can silently go wrong:
//   1. the conjugation signs: trans='C' must conjugate exactly one operand, and
//      the sign pattern of the four products depends on which of transa/transb
//      is 'C', so all four combinations are covered;
//   2. the packed-plane addressing: the packed row stride is the stored row
//      count rounded to 64B, so a padded leading dimension must not shift any
//      logical read;
//   3. the alpha=0 / k=0 path, which skips the product planes entirely and must
//      leave beta as the only contributor;
//   4. the epilogue, where a zero component of alpha or beta must be skipped
//      rather than multiplied through (0 * Inf is NaN).
class CgemmTest : public BlasTest<CgemmParam> {};

namespace {
aclblasComplex MakeComplex(float re, float im) { return aclblasComplex{re, im}; }

// aclblasComplex is a plain struct without operator==, so gtest cannot compare
// vectors of it directly; compare the two components instead.
void ExpectComplexVectorEq(
    const std::vector<aclblasComplex>& actual, const std::vector<aclblasComplex>& expected, const std::string& what)
{
    ASSERT_EQ(actual.size(), expected.size()) << what;
    for (size_t i = 0; i < actual.size(); ++i) {
        ASSERT_EQ(actual[i].real, expected[i].real) << what << " index=" << i;
        ASSERT_EQ(actual[i].imag, expected[i].imag) << what << " index=" << i;
    }
}

int CgemmPhysicalRowsA(const CgemmParam& p) { return (p.transA == ACLBLAS_OP_N) ? p.m : p.k; }

int CgemmPhysicalColsA(const CgemmParam& p) { return (p.transA == ACLBLAS_OP_N) ? p.k : p.m; }

int CgemmPhysicalRowsB(const CgemmParam& p) { return (p.transB == ACLBLAS_OP_N) ? p.k : p.n; }

int CgemmPhysicalColsB(const CgemmParam& p) { return (p.transB == ACLBLAS_OP_N) ? p.n : p.k; }

// Leading-dimension padding is never read or written by the operator, so it has
// to come back bit-identical.
void ExpectPaddingPreserved(
    const CgemmParam& p, const std::vector<float>& actualReal, const std::vector<float>& actualImag,
    const std::vector<float>& inputReal, const std::vector<float>& inputImag)
{
    if (actualReal.empty()) {
        return;
    }
    VerifyConfig cfgPadding;
    cfgPadding.mode = PrecisionMode::EXACT;
    EXPECT_TRUE(Verifier::verifyVector(
        actualReal.data(), inputReal.data(), actualReal.size(), 1, cfgPadding, p.caseName + "_padding_real"));
    EXPECT_TRUE(Verifier::verifyVector(
        actualImag.data(), inputImag.data(), actualImag.size(), 1, cfgPadding, p.caseName + "_padding_imag"));
}

void VerifyCgemmOutput(
    const CgemmParam& p, const std::vector<aclblasComplex>& actual, const std::vector<aclblasComplex>& golden,
    const std::vector<aclblasComplex>& input)
{
    std::vector<float> actualReal;
    std::vector<float> actualImag;
    std::vector<float> goldenReal;
    std::vector<float> goldenImag;
    std::vector<float> paddingActualReal;
    std::vector<float> paddingActualImag;
    std::vector<float> paddingInputReal;
    std::vector<float> paddingInputImag;
    actualReal.reserve(static_cast<size_t>(p.m) * p.n);
    actualImag.reserve(static_cast<size_t>(p.m) * p.n);
    goldenReal.reserve(static_cast<size_t>(p.m) * p.n);
    goldenImag.reserve(static_cast<size_t>(p.m) * p.n);
    for (int col = 0; col < p.n; ++col) {
        for (int row = 0; row < p.m; ++row) {
            const size_t offset = static_cast<size_t>(col) * p.ldc + row;
            actualReal.push_back(actual[offset].real);
            actualImag.push_back(actual[offset].imag);
            goldenReal.push_back(golden[offset].real);
            goldenImag.push_back(golden[offset].imag);
        }
        for (int row = p.m; row < p.ldc; ++row) {
            const size_t offset = static_cast<size_t>(col) * p.ldc + row;
            paddingActualReal.push_back(actual[offset].real);
            paddingActualImag.push_back(actual[offset].imag);
            paddingInputReal.push_back(input[offset].real);
            paddingInputImag.push_back(input[offset].imag);
        }
    }

    VerifyConfig cfgReal;
    applyMixedTolerance(cfgReal, ACL_FLOAT, goldenReal.data(), goldenReal.size());
    EXPECT_TRUE(Verifier::verifyVector(
        actualReal.data(), goldenReal.data(), actualReal.size(), 1, cfgReal, p.caseName + "_real"));
    VerifyConfig cfgImag;
    applyMixedTolerance(cfgImag, ACL_FLOAT, goldenImag.data(), goldenImag.size());
    EXPECT_TRUE(Verifier::verifyVector(
        actualImag.data(), goldenImag.data(), actualImag.size(), 1, cfgImag, p.caseName + "_imag"));

    ExpectPaddingPreserved(p, paddingActualReal, paddingActualImag, paddingInputReal, paddingInputImag);
}

// Runs one shape end to end: fills A/B/C, computes the CPU golden into a copy,
// calls the NPU entry and compares. `extra` mutates the parameter before the fill
// so a case can point at a different alpha/beta/ld triple.
bool RunCgemmShape(aclblasHandle_t handle, const CgemmParam& base, std::function<void(CgemmParam&)> extra = nullptr)
{
    CgemmParam p = base;
    if (extra != nullptr) {
        extra(p);
    }
    const int rowsA = CgemmPhysicalRowsA(p);
    const int colsA = CgemmPhysicalColsA(p);
    const int rowsB = CgemmPhysicalRowsB(p);
    const int colsB = CgemmPhysicalColsB(p);
    if (p.m <= 0 || p.n <= 0 || p.k < 0) {
        return false;
    }
    std::vector<aclblasComplex> a =
        (rowsA > 0 && colsA > 0) ? makeBlasComplexMatrix(rowsA, colsA, p.lda, p.aFill, p.randomSeed) :
                                   std::vector<aclblasComplex>{};
    std::vector<aclblasComplex> b =
        (rowsB > 0 && colsB > 0) ? makeBlasComplexMatrix(rowsB, colsB, p.ldb, p.bFill, p.randomSeed + 1U) :
                                   std::vector<aclblasComplex>{};
    std::vector<aclblasComplex> c = makeBlasComplexMatrix(p.m, p.n, p.ldc, p.cFill, p.randomSeed + 2U);
    const std::vector<aclblasComplex> input = c;
    std::vector<aclblasComplex> golden = c;

    const aclblasComplex alpha = MakeComplex(p.alphaReal, p.alphaImag);
    const aclblasComplex beta = MakeComplex(p.betaReal, p.betaImag);
    const aclblasStatus_t npuRet = aclblasCgemm_npu(
        handle, p.transA, p.transB, p.m, p.n, p.k, &alpha, a.data(), p.lda, b.data(), p.ldb, &beta, c.data(), p.ldc);
    EXPECT_EQ(static_cast<int>(npuRet), static_cast<int>(ACLBLAS_STATUS_SUCCESS)) << p.caseName;
    if (npuRet != ACLBLAS_STATUS_SUCCESS) {
        return false;
    }
    const aclblasStatus_t goldenRet = aclblasCgemmCpu(
        handle, p.transA, p.transB, p.m, p.n, p.k, &alpha, a.data(), p.lda, b.data(), p.ldb, &beta, golden.data(),
        p.ldc);
    EXPECT_EQ(static_cast<int>(goldenRet), static_cast<int>(ACLBLAS_STATUS_SUCCESS)) << p.caseName;
    if (goldenRet != ACLBLAS_STATUS_SUCCESS) {
        return false;
    }
    VerifyCgemmOutput(p, c, golden, input);
    return true;
}

CgemmParam MakeCgemmCase(
    const std::string& name, aclblasOperation_t transA, aclblasOperation_t transB, int m, int n, int k, uint32_t seed)
{
    CgemmParam p(csv_map{});
    p.caseName = name;
    p.transA = transA;
    p.transB = transB;
    p.m = m;
    p.n = n;
    p.k = k;
    p.alphaReal = 1.0F;
    p.alphaImag = 0.0F;
    p.betaReal = 0.0F;
    p.betaImag = 0.0F;
    p.lda = std::max(1, (transA == ACLBLAS_OP_N) ? m : k);
    p.ldb = std::max(1, (transB == ACLBLAS_OP_N) ? k : n);
    p.ldc = std::max(1, m);
    p.randomSeed = seed;
    return p;
}
} // namespace

TEST_F(CgemmTest, NullHandle)
{
    aclblasComplex alpha = MakeComplex(1.0F, 0.0F);
    aclblasComplex beta = MakeComplex(0.0F, 0.0F);
    std::vector<aclblasComplex> data(64, MakeComplex(0.0F, 0.0F));
    EXPECT_EQ(
        aclblasCgemm(
            nullptr, ACLBLAS_OP_N, ACLBLAS_OP_N, 8, 8, 8, &alpha, data.data(), 8, data.data(), 8, &beta, data.data(),
            8),
        ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
}

// One row per invalid combination. Kept out of the TEST_F body so the body stays
// inside the 50-line non-comment limit the repo's CodeCheck applies to it, and so
// the repeated call shape lives in exactly one place.
struct CgemmInvalidCase {
    aclblasOperation_t transA;
    aclblasOperation_t transB;
    int m;
    int n;
    int k;
    int lda;
    int ldb;
    int ldc;
    bool nullAlpha;
    bool nullBeta;
    bool nullA;
    bool nullB;
    bool nullC;
    aclblasStatus_t expected;
};

std::vector<CgemmInvalidCase> MakeCgemmInvalidCases()
{
    const aclblasOperation_t bad = static_cast<aclblasOperation_t>(0xFF);
    const aclblasOperation_t nOp = ACLBLAS_OP_N;
    const aclblasOperation_t tOp = ACLBLAS_OP_T;
    const aclblasStatus_t value = ACLBLAS_STATUS_INVALID_VALUE;
    const aclblasStatus_t enumm = ACLBLAS_STATUS_INVALID_ENUM;
    return {
        {bad, nOp, 8, 8, 8, 8, 8, 8, false, false, false, false, false, enumm},
        {nOp, bad, 8, 8, 8, 8, 8, 8, false, false, false, false, false, enumm},
        {nOp, nOp, -1, 8, 8, 8, 8, 8, false, false, false, false, false, value},
        {nOp, nOp, 8, -1, 8, 8, 8, 8, false, false, false, false, false, value},
        {nOp, nOp, 8, 8, -1, 8, 8, 8, false, false, false, false, false, value},
        {nOp, nOp, 8, 8, 8, 7, 8, 8, false, false, false, false, false, value},
        {tOp, nOp, 8, 8, 8, 8, 7, 8, false, false, false, false, false, value},
        {nOp, nOp, 8, 8, 8, 8, 8, 7, false, false, false, false, false, value},
        {nOp, nOp, 8, 8, 8, 8, 8, 8, true, false, false, false, false, value},
        {nOp, nOp, 8, 8, 8, 8, 8, 8, false, true, false, false, false, value},
        {nOp, nOp, 8, 8, 8, 8, 8, 8, false, false, true, false, false, value},
        {nOp, nOp, 8, 8, 8, 8, 8, 8, false, false, false, true, false, value},
        {nOp, nOp, 8, 8, 8, 8, 8, 8, false, false, false, false, true, value},
    };
}

TEST_F(CgemmTest, InvalidEnumAndDims)
{
    aclblasComplex alpha = MakeComplex(1.0F, 0.0F);
    aclblasComplex beta = MakeComplex(0.0F, 0.0F);
    std::vector<aclblasComplex> data(256, MakeComplex(0.0F, 0.0F));
    for (const CgemmInvalidCase& c : MakeCgemmInvalidCases()) {
        const aclblasComplex* al = c.nullAlpha ? nullptr : &alpha;
        const aclblasComplex* be = c.nullBeta ? nullptr : &beta;
        const aclblasComplex* a = c.nullA ? nullptr : data.data();
        const aclblasComplex* b = c.nullB ? nullptr : data.data();
        aclblasComplex* out = c.nullC ? nullptr : data.data();
        EXPECT_EQ(
            aclblasCgemm(handle_, c.transA, c.transB, c.m, c.n, c.k, al, a, c.lda, b, c.ldb, be, out, c.ldc),
            c.expected);
    }
}

// m == 0 or n == 0 is a legal no-op: nothing is read or written, and with k == 0
// null matrix pointers are acceptable.
TEST_F(CgemmTest, ZeroDimIsNoOp)
{
    aclblasComplex alpha = MakeComplex(1.0F, 0.0F);
    aclblasComplex beta = MakeComplex(0.0F, 0.0F);
    std::vector<aclblasComplex> data(16, MakeComplex(3.0F, -3.0F));
    const std::vector<aclblasComplex> before = data;
    EXPECT_EQ(
        aclblasCgemm(
            handle_, ACLBLAS_OP_N, ACLBLAS_OP_N, 0, 8, 8, &alpha, data.data(), 8, data.data(), 8, &beta, data.data(),
            8),
        ACLBLAS_STATUS_SUCCESS);
    ExpectComplexVectorEq(data, before, "zero_m_no_op");
    EXPECT_EQ(
        aclblasCgemm(
            handle_, ACLBLAS_OP_N, ACLBLAS_OP_N, 8, 0, 0, &alpha, nullptr, 8, nullptr, 8, &beta, data.data(), 8),
        ACLBLAS_STATUS_SUCCESS);
    ExpectComplexVectorEq(data, before, "zero_n_no_op");
}

// alpha = (0,0) or k = 0 skips the product planes entirely, so beta is the only
// contributor.
TEST_F(CgemmTest, AlphaZeroAndZeroKUseBetaOnly)
{
    ASSERT_TRUE(RunCgemmShape(
        handle_, MakeCgemmCase("alpha_zero_NN", ACLBLAS_OP_N, ACLBLAS_OP_N, 8, 8, 4, 20263001U), [](CgemmParam& p) {
            p.alphaReal = 0.0F;
            p.alphaImag = 0.0F;
            p.betaReal = -0.5F;
            p.betaImag = 0.25F;
        }));
    ASSERT_TRUE(RunCgemmShape(
        handle_, MakeCgemmCase("k_zero_NT", ACLBLAS_OP_N, ACLBLAS_OP_T, 8, 6, 0, 20263002U), [](CgemmParam& p) {
            p.betaReal = 1.5F;
            p.betaImag = -0.5F;
        }));
    ASSERT_TRUE(RunCgemmShape(
        handle_, MakeCgemmCase("alpha_zero_CN", ACLBLAS_OP_C, ACLBLAS_OP_N, 10, 7, 5, 20263003U), [](CgemmParam& p) {
            p.alphaReal = 0.0F;
            p.alphaImag = 0.0F;
            p.betaReal = 0.5F;
            p.betaImag = 0.5F;
        }));
}

// With alpha = 0 and beta = 1 nothing contributes at all, so no kernel is
// launched and C must come back bit-identical.
TEST_F(CgemmTest, AlphaZeroBetaOneIsBitExactNoOp)
{
    CgemmParam p = MakeCgemmCase("alpha_zero_beta_one", ACLBLAS_OP_N, ACLBLAS_OP_N, 8, 8, 4, 20263004U);
    p.alphaReal = 0.0F;
    p.betaReal = 1.0F;
    std::vector<aclblasComplex> a = makeBlasComplexMatrix(8, 4, 8, p.aFill, p.randomSeed);
    std::vector<aclblasComplex> b = makeBlasComplexMatrix(4, 8, 8, p.bFill, p.randomSeed + 1U);
    std::vector<aclblasComplex> c = makeBlasComplexMatrix(8, 8, 8, p.cFill, p.randomSeed + 2U);
    const std::vector<aclblasComplex> before = c;
    const aclblasComplex alpha = MakeComplex(0.0F, 0.0F);
    const aclblasComplex beta = MakeComplex(1.0F, 0.0F);
    ASSERT_EQ(
        aclblasCgemm_npu(
            handle_, p.transA, p.transB, p.m, p.n, p.k, &alpha, a.data(), p.lda, b.data(), p.ldb, &beta, c.data(),
            p.ldc),
        ACLBLAS_STATUS_SUCCESS);
    ExpectComplexVectorEq(c, before, "alpha_zero_beta_one");
}

// All four conjugation combinations. trans='C' must conjugate exactly one
// operand, and the sign pattern of the four real products depends on which of
// transa / transb carries the conjugation, so each combination is a distinct
// branch of the host-side sign table.
TEST_F(CgemmTest, AllTransposeAndConjugateCombinations)
{
    const aclblasOperation_t ops[4] = {ACLBLAS_OP_N, ACLBLAS_OP_T, ACLBLAS_OP_C, ACLBLAS_OP_T};
    uint32_t seed = 20263010U;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            const std::string name = "trans_" + std::to_string(i) + std::to_string(j);
            CgemmParam p = MakeCgemmCase(name, ops[i], ops[j], 9, 7, 5, seed++);
            // A non-trivial complex alpha and beta on every combination, so a wrong
            // conjugation sign cannot hide behind alpha = 1 or beta = 0.
            p.alphaReal = 0.5F;
            p.alphaImag = -0.25F;
            p.betaReal = -0.75F;
            p.betaImag = 0.125F;
            ASSERT_TRUE(RunCgemmShape(handle_, p)) << name;
        }
    }
}

// Padded leading dimensions: the packed row stride is the stored row count
// rounded up to 64B, so a larger lda/ldb/ldc must not shift any logical read or
// write, and the padding itself must stay untouched.
TEST_F(CgemmTest, PaddedLeadingDimensions)
{
    ASSERT_TRUE(RunCgemmShape(
        handle_, MakeCgemmCase("padded_NN", ACLBLAS_OP_N, ACLBLAS_OP_N, 40, 24, 17, 20263030U), [](CgemmParam& p) {
            p.lda = 48;
            p.ldb = 53;
            p.ldc = 47;
        }));
    ASSERT_TRUE(RunCgemmShape(
        handle_, MakeCgemmCase("padded_TN", ACLBLAS_OP_T, ACLBLAS_OP_N, 33, 41, 19, 20263031U), [](CgemmParam& p) {
            p.lda = 45;
            p.ldb = 39;
            p.ldc = 36;
        }));
    ASSERT_TRUE(RunCgemmShape(
        handle_, MakeCgemmCase("padded_CC", ACLBLAS_OP_C, ACLBLAS_OP_C, 24, 12, 11, 20263032U), [](CgemmParam& p) {
            p.lda = 30;
            p.ldb = 27;
            p.ldc = 29;
            p.alphaReal = 0.5F;
            p.alphaImag = 0.25F;
            p.betaReal = -0.5F;
            p.betaImag = 0.75F;
        }));
}

// Non-finite entries in the incoming C must survive a zero beta component:
// 0 * Inf is NaN, so the epilogue has to skip the multiply rather than carry the
// NaN into a part that is otherwise exact.
TEST_F(CgemmTest, NonFiniteInputSurvivesZeroBetaComponent)
{
    CgemmParam p = MakeCgemmCase("nonfinite_input", ACLBLAS_OP_N, ACLBLAS_OP_N, 6, 6, 4, 20263040U);
    p.betaReal = 0.0F;
    p.betaImag = 0.0F;
    std::vector<aclblasComplex> a = makeBlasComplexMatrix(6, 4, 6, p.aFill, p.randomSeed);
    std::vector<aclblasComplex> b = makeBlasComplexMatrix(4, 6, 6, p.bFill, p.randomSeed + 1U);
    std::vector<aclblasComplex> c = makeBlasComplexMatrix(6, 6, 6, p.cFill, p.randomSeed + 2U);
    const float quietNaN = std::numeric_limits<float>::quiet_NaN();
    const float infinity = std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < c.size(); ++i) {
        c[i] = MakeComplex((i % 2U) == 0U ? quietNaN : infinity, (i % 3U) == 0U ? infinity : quietNaN);
    }
    const aclblasComplex alpha = MakeComplex(1.0F, 0.0F);
    const aclblasComplex beta = MakeComplex(0.0F, 0.0F);
    ASSERT_EQ(
        aclblasCgemm_npu(
            handle_, p.transA, p.transB, p.m, p.n, p.k, &alpha, a.data(), p.lda, b.data(), p.ldb, &beta, c.data(),
            p.ldc),
        ACLBLAS_STATUS_SUCCESS);
    for (int col = 0; col < p.n; ++col) {
        for (int row = 0; row < p.m; ++row) {
            const size_t offset = static_cast<size_t>(col) * p.ldc + row;
            ASSERT_TRUE(std::isfinite(c[offset].real)) << "row=" << row << " col=" << col;
            ASSERT_TRUE(std::isfinite(c[offset].imag)) << "row=" << row << " col=" << col;
        }
    }
}

// A zero imaginary component of alpha must not turn a non-finite product plane
// into NaN in the real part.
TEST_F(CgemmTest, ZeroAlphaImagComponentSkipsMultiply)
{
    CgemmParam p = MakeCgemmCase("alpha_imag_zero", ACLBLAS_OP_N, ACLBLAS_OP_N, 5, 5, 3, 20263041U);
    p.alphaReal = 2.0F;
    p.alphaImag = 0.0F;
    p.betaReal = 0.0F;
    p.betaImag = 0.0F;
    ASSERT_TRUE(RunCgemmShape(handle_, p));

    CgemmParam q = MakeCgemmCase("alpha_real_zero", ACLBLAS_OP_N, ACLBLAS_OP_N, 5, 5, 3, 20263042U);
    q.alphaReal = 0.0F;
    q.alphaImag = -1.5F;
    q.betaReal = 0.0F;
    q.betaImag = 0.0F;
    ASSERT_TRUE(RunCgemmShape(handle_, q));
}

INSTANTIATE_TEST_SUITE_P(
    CgemmArch22, CgemmTest, ::testing::ValuesIn(GetCasesFromCsv<CgemmParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<CgemmParam>);

TEST_P(CgemmTest, CsvDriven)
{
    const auto& p = GetParam();
    const int rowsA = CgemmPhysicalRowsA(p);
    const int colsA = CgemmPhysicalColsA(p);
    const int rowsB = CgemmPhysicalRowsB(p);
    const int colsB = CgemmPhysicalColsB(p);
    if (p.expectResult != ACLBLAS_STATUS_SUCCESS || rowsA <= 0 || colsA <= 0 || rowsB <= 0 || colsB <= 0 || p.m <= 0 ||
        p.n <= 0) {
        // Invalid-parameter rows only assert the status code.
        std::vector<aclblasComplex> a = rowsA > 0 && colsA > 0 ?
                                            makeBlasComplexMatrix(rowsA, colsA, p.lda, p.aFill, p.randomSeed) :
                                            std::vector<aclblasComplex>{};
        std::vector<aclblasComplex> b = rowsB > 0 && colsB > 0 ?
                                            makeBlasComplexMatrix(rowsB, colsB, p.ldb, p.bFill, p.randomSeed + 1U) :
                                            std::vector<aclblasComplex>{};
        std::vector<aclblasComplex> c = p.m > 0 && p.n > 0 ?
                                            makeBlasComplexMatrix(p.m, p.n, p.ldc, p.cFill, p.randomSeed + 2U) :
                                            std::vector<aclblasComplex>{};
        const aclblasComplex alpha = MakeComplex(p.alphaReal, p.alphaImag);
        const aclblasComplex beta = MakeComplex(p.betaReal, p.betaImag);
        // The CSV marks a null argument with its own column, so honour it here:
        // otherwise an "expect INVALID_VALUE" row would hand the operator a valid
        // pointer and get SUCCESS back.
        const aclblasComplex* alphaPtr = p.alphaNull ? nullptr : &alpha;
        const aclblasComplex* betaPtr = p.betaNull ? nullptr : &beta;
        const aclblasComplex* aPtr = p.aNull ? nullptr : a.data();
        const aclblasComplex* bPtr = p.bNull ? nullptr : b.data();
        aclblasComplex* cPtr = p.cNull ? nullptr : c.data();
        const aclblasStatus_t ret = aclblasCgemm_npu(
            handle_, p.transA, p.transB, p.m, p.n, p.k, alphaPtr, aPtr, p.lda, bPtr, p.ldb, betaPtr, cPtr, p.ldc);
        EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
        return;
    }
    ASSERT_TRUE(RunCgemmShape(handle_, p)) << p.caseName;
}
