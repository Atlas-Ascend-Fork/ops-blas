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
#include <complex>
#include <vector>

#include "verify.h"
#include "blas_test.h"
#include "csv_loader.h"
#include "fill.h"
#include "cgeam_param.h"
#include "../../geam_golden.h"
#include "cgeam_npu_wrapper.h"

#include "gtest/gtest.h"

// Generate a column-major complex matrix: m rows, n cols, column stride lda.
// Returns interleaved float vector of size lda * n * 2.
static inline std::vector<float> makeComplexMatrixCM(int m, int n, int lda, const BlasFillMode& fill, uint32_t seed)
{
    const size_t storageSize = static_cast<size_t>(lda) * n * 2;
    std::vector<float> data(storageSize, 0.0f);

    if (fill.method == BlasFillMode::M_VALUE) {
        for (size_t i = 0; i < storageSize; i++)
            data[i] = fill.val1;
        return data;
    }

    std::mt19937 rngReal(seed ? seed : 42);
    std::mt19937 rngImag((seed ? seed : 42) + 2000);
    auto genReal = createGenerator(fill, rngReal);
    auto genImag = createGenerator(fill, rngImag);

    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            size_t idx = (static_cast<size_t>(j) * lda + i) * 2;
            size_t flatIdx = static_cast<size_t>(j) * m + i;
            data[idx] = genReal->at(flatIdx);
            data[idx + 1] = genImag->at(flatIdx);
        }
    }
    return data;
}

// -- Test fixture ------------------------------------------------------------
class CgeamArch35Test : public BlasTest<CgeamParam> {};

// Helper: 4x4 complex matrix as interleaved float vector
static inline std::vector<float> c4x4(float v = 1.0f) { return std::vector<float>(4 * 4 * 2, v); }

// -- TEST_F: null handle (not in CSV) ---------------------------------------
TEST_F(CgeamArch35Test, NullHandle)
{
    aclblasComplex alpha = {1.0f, 0.0f}, beta = {1.0f, 0.0f};
    auto A = c4x4(), B = c4x4(), C = c4x4(0.0f);
    EXPECT_EQ(
        aclblasCgeam(
            nullptr, ACLBLAS_OP_N, ACLBLAS_OP_N, 4, 4, &alpha, reinterpret_cast<const aclblasComplex*>(A.data()), 4,
            &beta, reinterpret_cast<const aclblasComplex*>(B.data()), 4, reinterpret_cast<aclblasComplex*>(C.data()),
            4),
        ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
}

// -- TEST_F: A=nullptr (host validates and returns error) --------------------
TEST_F(CgeamArch35Test, NullA)
{
    aclblasComplex alpha = {1.0f, 0.0f}, beta = {1.0f, 0.0f};
    auto B = c4x4(), C = c4x4(0.0f);
    EXPECT_EQ(
        aclblasCgeam(
            CgeamArch35Test::handle_, ACLBLAS_OP_N, ACLBLAS_OP_N, 4, 4, &alpha, nullptr, 4, &beta,
            reinterpret_cast<const aclblasComplex*>(B.data()), 4, reinterpret_cast<aclblasComplex*>(C.data()), 4),
        ACLBLAS_STATUS_INVALID_VALUE);
}

// -- TEST_F: B=nullptr (host validates and returns error) --------------------
TEST_F(CgeamArch35Test, NullB)
{
    aclblasComplex alpha = {1.0f, 0.0f}, beta = {1.0f, 0.0f};
    auto A = c4x4(), C = c4x4(0.0f);
    EXPECT_EQ(
        aclblasCgeam(
            CgeamArch35Test::handle_, ACLBLAS_OP_N, ACLBLAS_OP_N, 4, 4, &alpha,
            reinterpret_cast<const aclblasComplex*>(A.data()), 4, &beta, nullptr, 4,
            reinterpret_cast<aclblasComplex*>(C.data()), 4),
        ACLBLAS_STATUS_INVALID_VALUE);
}

// -- TEST_F: C=nullptr (host validates and returns error) --------------------
TEST_F(CgeamArch35Test, NullC)
{
    aclblasComplex alpha = {1.0f, 0.0f}, beta = {1.0f, 0.0f};
    auto A = c4x4(), B = c4x4();
    EXPECT_EQ(
        aclblasCgeam(
            CgeamArch35Test::handle_, ACLBLAS_OP_N, ACLBLAS_OP_N, 4, 4, &alpha,
            reinterpret_cast<const aclblasComplex*>(A.data()), 4, &beta,
            reinterpret_cast<const aclblasComplex*>(B.data()), 4, nullptr, 4),
        ACLBLAS_STATUS_INVALID_VALUE);
}

// -- TEST_F: alpha=nullptr returns INVALID_VALUE (host validates) ------------
TEST_F(CgeamArch35Test, NullAlpha)
{
    aclblasComplex beta = {1.0f, 0.0f};
    auto A = c4x4(2.0f), B = c4x4(3.0f), C = c4x4(kBlasSentinel);
    EXPECT_EQ(
        aclblasCgeam(
            CgeamArch35Test::handle_, ACLBLAS_OP_N, ACLBLAS_OP_N, 4, 4, nullptr,
            reinterpret_cast<aclblasComplex*>(A.data()), 4, &beta, reinterpret_cast<aclblasComplex*>(B.data()), 4,
            reinterpret_cast<aclblasComplex*>(C.data()), 4),
        ACLBLAS_STATUS_INVALID_VALUE);
}

// -- TEST_F: beta=nullptr returns INVALID_VALUE (host validates) -------------
TEST_F(CgeamArch35Test, NullBeta)
{
    aclblasComplex alpha = {1.0f, 0.0f};
    auto A = c4x4(2.0f), B = c4x4(3.0f), C = c4x4(kBlasSentinel);
    EXPECT_EQ(
        aclblasCgeam(
            CgeamArch35Test::handle_, ACLBLAS_OP_N, ACLBLAS_OP_N, 4, 4, &alpha,
            reinterpret_cast<aclblasComplex*>(A.data()), 4, nullptr, reinterpret_cast<aclblasComplex*>(B.data()), 4,
            reinterpret_cast<aclblasComplex*>(C.data()), 4),
        ACLBLAS_STATUS_INVALID_VALUE);
}

// -- TEST_F: transa invalid 0xFF (host validates) ----------------------------
TEST_F(CgeamArch35Test, TransaInvalid)
{
    aclblasComplex alpha = {1.0f, 0.0f}, beta = {1.0f, 0.0f};
    auto A = c4x4(), B = c4x4(), C = c4x4(0.0f);
    EXPECT_EQ(
        aclblasCgeam(
            CgeamArch35Test::handle_, static_cast<aclblasOperation_t>(0xFF), ACLBLAS_OP_N, 4, 4, &alpha,
            reinterpret_cast<const aclblasComplex*>(A.data()), 4, &beta,
            reinterpret_cast<const aclblasComplex*>(B.data()), 4, reinterpret_cast<aclblasComplex*>(C.data()), 4),
        ACLBLAS_STATUS_INVALID_ENUM);
}

// -- TEST_F: transb invalid 0xFF (host validates) ----------------------------
TEST_F(CgeamArch35Test, TransbInvalid)
{
    aclblasComplex alpha = {1.0f, 0.0f}, beta = {1.0f, 0.0f};
    auto A = c4x4(), B = c4x4(), C = c4x4(0.0f);
    EXPECT_EQ(
        aclblasCgeam(
            CgeamArch35Test::handle_, ACLBLAS_OP_N, static_cast<aclblasOperation_t>(0xFF), 4, 4, &alpha,
            reinterpret_cast<const aclblasComplex*>(A.data()), 4, &beta,
            reinterpret_cast<const aclblasComplex*>(B.data()), 4, reinterpret_cast<aclblasComplex*>(C.data()), 4),
        ACLBLAS_STATUS_INVALID_ENUM);
}

// -- TEST_F: m<0 (host validates and returns error) -------------------------
TEST_F(CgeamArch35Test, MNegative)
{
    aclblasComplex alpha = {1.0f, 0.0f}, beta = {1.0f, 0.0f};
    EXPECT_EQ(
        aclblasCgeam(
            CgeamArch35Test::handle_, ACLBLAS_OP_N, ACLBLAS_OP_N, -1, 4, &alpha, nullptr, 1, &beta, nullptr, 1, nullptr,
            1),
        ACLBLAS_STATUS_INVALID_VALUE);
}

// -- TEST_F: n<0 (host validates and returns error) -------------------------
TEST_F(CgeamArch35Test, NNegative)
{
    aclblasComplex alpha = {1.0f, 0.0f}, beta = {1.0f, 0.0f};
    EXPECT_EQ(
        aclblasCgeam(
            CgeamArch35Test::handle_, ACLBLAS_OP_N, ACLBLAS_OP_N, 4, -1, &alpha, nullptr, 4, &beta, nullptr, 4, nullptr,
            4),
        ACLBLAS_STATUS_INVALID_VALUE);
}

// -- TEST_F: zero-size operation succeeds without dereferencing data pointers -
TEST_F(CgeamArch35Test, ZeroDimNullDataPointers)
{
    EXPECT_EQ(
        aclblasCgeam(
            CgeamArch35Test::handle_, ACLBLAS_OP_N, ACLBLAS_OP_N, 0, 4, nullptr, nullptr, 1, nullptr, nullptr, 1,
            nullptr, 1),
        ACLBLAS_STATUS_SUCCESS);
}

// -- TEST_F: lda too small when transa=N (host validates) --------------------
TEST_F(CgeamArch35Test, LdaTooSmallN)
{
    aclblasComplex alpha = {1.0f, 0.0f}, beta = {1.0f, 0.0f};
    auto A = c4x4(), B = c4x4(), C = c4x4(0.0f);
    EXPECT_EQ(
        aclblasCgeam(
            CgeamArch35Test::handle_, ACLBLAS_OP_N, ACLBLAS_OP_N, 4, 4, &alpha,
            reinterpret_cast<const aclblasComplex*>(A.data()), 3, &beta,
            reinterpret_cast<const aclblasComplex*>(B.data()), 4, reinterpret_cast<aclblasComplex*>(C.data()), 4),
        ACLBLAS_STATUS_INVALID_VALUE);
}

// -- TEST_F: ldc too small (host validates and returns error) ----------------
TEST_F(CgeamArch35Test, LdcTooSmall)
{
    aclblasComplex alpha = {1.0f, 0.0f}, beta = {1.0f, 0.0f};
    std::vector<float> A(8 * 4 * 2, 1.0f), B(8 * 4 * 2, 1.0f), C(8 * 4 * 2, 0.0f);
    EXPECT_EQ(
        aclblasCgeam(
            CgeamArch35Test::handle_, ACLBLAS_OP_N, ACLBLAS_OP_N, 8, 4, &alpha,
            reinterpret_cast<const aclblasComplex*>(A.data()), 8, &beta,
            reinterpret_cast<const aclblasComplex*>(B.data()), 8, reinterpret_cast<aclblasComplex*>(C.data()), 7),
        ACLBLAS_STATUS_INVALID_VALUE);
}

// -- TEST_F: in-place C==A with transa!=N (host validates) -------------------
TEST_F(CgeamArch35Test, InplaceATransInvalid)
{
    aclblasComplex alpha = {1.0f, 0.0f}, beta = {1.0f, 0.0f};
    std::vector<float> A(8 * 8 * 2, 1.0f), B(8 * 8 * 2, 1.0f);
    EXPECT_EQ(
        aclblasCgeam(
            CgeamArch35Test::handle_, ACLBLAS_OP_T, ACLBLAS_OP_N, 8, 8, &alpha,
            reinterpret_cast<const aclblasComplex*>(A.data()), 8, &beta,
            reinterpret_cast<const aclblasComplex*>(B.data()), 8, reinterpret_cast<aclblasComplex*>(A.data()), 8),
        ACLBLAS_STATUS_INVALID_VALUE);
}

// -- TEST_F: in-place C==B with transb!=N (host validates) -------------------
TEST_F(CgeamArch35Test, InplaceBTransInvalid)
{
    aclblasComplex alpha = {1.0f, 0.0f}, beta = {1.0f, 0.0f};
    std::vector<float> A(8 * 8 * 2, 1.0f), B(8 * 8 * 2, 1.0f);
    EXPECT_EQ(
        aclblasCgeam(
            CgeamArch35Test::handle_, ACLBLAS_OP_N, ACLBLAS_OP_T, 8, 8, &alpha,
            reinterpret_cast<const aclblasComplex*>(A.data()), 8, &beta,
            reinterpret_cast<const aclblasComplex*>(B.data()), 8, reinterpret_cast<aclblasComplex*>(B.data()), 8),
        ACLBLAS_STATUS_INVALID_VALUE);
}

// -- TEST_F: in-place C==A with lda!=ldc (host validates) --------------------
TEST_F(CgeamArch35Test, InplaceALdaNeLdc)
{
    aclblasComplex alpha = {1.0f, 0.0f}, beta = {1.0f, 0.0f};
    std::vector<float> A(10 * 8 * 2, 1.0f), B(10 * 8 * 2, 1.0f);
    EXPECT_EQ(
        aclblasCgeam(
            CgeamArch35Test::handle_, ACLBLAS_OP_N, ACLBLAS_OP_N, 8, 8, &alpha,
            reinterpret_cast<const aclblasComplex*>(A.data()), 10, &beta,
            reinterpret_cast<const aclblasComplex*>(B.data()), 8, reinterpret_cast<aclblasComplex*>(A.data()), 8),
        ACLBLAS_STATUS_INVALID_VALUE);
}

// -- TEST_F: in-place C==B with ldb!=ldc (host validates) --------------------
TEST_F(CgeamArch35Test, InplaceBLdbNeLdc)
{
    aclblasComplex alpha = {1.0f, 0.0f}, beta = {1.0f, 0.0f};
    std::vector<float> A(10 * 8 * 2, 1.0f), B(10 * 8 * 2, 1.0f);
    EXPECT_EQ(
        aclblasCgeam(
            CgeamArch35Test::handle_, ACLBLAS_OP_N, ACLBLAS_OP_N, 8, 8, &alpha,
            reinterpret_cast<const aclblasComplex*>(A.data()), 8, &beta,
            reinterpret_cast<const aclblasComplex*>(B.data()), 10, reinterpret_cast<aclblasComplex*>(B.data()), 8),
        ACLBLAS_STATUS_INVALID_VALUE);
}

// -- TEST_F: alpha=nullptr with A=nullptr returns INVALID_VALUE -------------
TEST_F(CgeamArch35Test, NullAlphaAllowsNullA)
{
    aclblasComplex beta = {1.0f, 0.0f};
    auto B = c4x4(), C = c4x4(kBlasSentinel);
    EXPECT_EQ(
        aclblasCgeam(
            CgeamArch35Test::handle_, ACLBLAS_OP_N, ACLBLAS_OP_N, 4, 4, nullptr, nullptr, 4, &beta,
            reinterpret_cast<aclblasComplex*>(B.data()), 4, reinterpret_cast<aclblasComplex*>(C.data()), 4),
        ACLBLAS_STATUS_INVALID_VALUE);
}

// -- CSV parameterised test suite -------------------------------------------
INSTANTIATE_TEST_SUITE_P(
    Cgeam, CgeamArch35Test, ::testing::ValuesIn(GetCasesFromCsv<CgeamParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<CgeamParam>);

// -- Helper: generate A or B input complex matrix -----------------------------
static std::vector<float> CgeamGenerateInput(const CgeamParam& p, bool isA, int extraSeed)
{
    int nullFlag = isA ? p.nullA : p.nullB;
    if (nullFlag != 0 || p.m <= 0 || p.n <= 0) {
        return {};
    }
    aclblasOperation_t trans = isA ? p.transa : p.transb;
    int ld = isA ? p.lda : p.ldb;
    const BlasFillMode& fill = isA ? p.aFill : p.bFill;
    int rows = (trans == ACLBLAS_OP_N) ? p.m : p.n;
    int cols = (trans == ACLBLAS_OP_N) ? p.n : p.m;
    const int storageLd = std::max(ld, rows);
    return makeComplexMatrixCM(rows, cols, storageLd, fill, p.randomSeed + extraSeed);
}

// -- Helper: prepare C buffer (in-place aware, complex) -----------------------
static aclblasComplex* CgeamPrepareC(const CgeamParam& p, const aclblasComplex* aPtr,
                                     const aclblasComplex* bPtr, std::vector<float>& cHost)
{
    if (p.inplace == 1 && aPtr) {
        return const_cast<aclblasComplex*>(aPtr);
    }
    if (p.inplace == 2 && bPtr) {
        return const_cast<aclblasComplex*>(bPtr);
    }
    if (p.nullC == 0 && p.m > 0 && p.n > 0) {
        cHost.assign(static_cast<size_t>(p.ldc) * static_cast<size_t>(p.n) * 2, kBlasSentinel);
    }
    return cHost.empty() ? nullptr : reinterpret_cast<aclblasComplex*>(cHost.data());
}

// -- Helper: compute golden reference (complex) --------------------------------
static std::vector<float> CgeamComputeGolden(
    const CgeamParam& p, aclblasComplex alpha, aclblasComplex beta,
    const aclblasComplex* aPtr, const aclblasComplex* bPtr, aclblasComplex* cPtr,
    const std::vector<float>& aHost, const std::vector<float>& bHost, const std::vector<float>& cHost)
{
    std::vector<float> goldenC;
    if (p.expectResult != ACLBLAS_STATUS_SUCCESS || p.m <= 0 || p.n <= 0 || !cPtr) {
        return goldenC;
    }
    size_t goldenSize = cHost.empty() ? (p.inplace == 1 ? aHost.size() :
                                         p.inplace == 2 ? bHost.size() : 0) :
                                        cHost.size();
    goldenC.assign(goldenSize, kBlasSentinel);
    aclblasGeam_cpu<std::complex<float>>(
        p.transa, p.transb, static_cast<std::size_t>(p.m), static_cast<std::size_t>(p.n),
        *reinterpret_cast<const std::complex<float>*>(&alpha), reinterpret_cast<const std::complex<float>*>(aPtr),
        static_cast<std::size_t>(p.lda), *reinterpret_cast<const std::complex<float>*>(&beta),
        reinterpret_cast<const std::complex<float>*>(bPtr), static_cast<std::size_t>(p.ldb),
        reinterpret_cast<std::complex<float>*>(goldenC.data()), static_cast<std::size_t>(p.ldc));
    return goldenC;
}

// -- Helper: verify output precision (complex) ---------------------------------
static void CgeamVerifyOutput(
    const CgeamParam& p, const std::vector<float>& goldenC,
    const std::vector<float>& aHost, const std::vector<float>& bHost, const std::vector<float>& cHost)
{
    const float* outPtr = nullptr;
    if (p.inplace == 1) {
        outPtr = aHost.data();
    } else if (p.inplace == 2) {
        outPtr = bHost.data();
    } else {
        outPtr = cHost.data();
    }

    if (outPtr == nullptr || goldenC.empty()) {
        return;
    }

    const size_t logicalCount = static_cast<size_t>(p.m) * p.n;
    std::vector<float> outReal(logicalCount);
    std::vector<float> outImag(logicalCount);
    std::vector<float> goldReal(logicalCount);
    std::vector<float> goldImag(logicalCount);
    for (int col = 0; col < p.n; ++col) {
        for (int row = 0; row < p.m; ++row) {
            const size_t logicalIndex = static_cast<size_t>(col) * p.m + row;
            const size_t storageIndex = (static_cast<size_t>(col) * p.ldc + row) * 2;
            outReal[logicalIndex] = outPtr[storageIndex];
            outImag[logicalIndex] = outPtr[storageIndex + 1];
            goldReal[logicalIndex] = goldenC[storageIndex];
            goldImag[logicalIndex] = goldenC[storageIndex + 1];
        }
    }

    VerifyConfig cfg;
    applyMixedTolerance(cfg, ACL_FLOAT, goldReal.data(), logicalCount);
    EXPECT_TRUE(Verifier::verifyVector(
        outReal.data(), goldReal.data(), logicalCount, 1, cfg, p.caseName + "_real"));
    EXPECT_TRUE(Verifier::verifyVector(
        outImag.data(), goldImag.data(), logicalCount, 1, cfg, p.caseName + "_imag"));
}

// -- TEST_P: 5-step CSV-driven flow -----------------------------------------
TEST_P(CgeamArch35Test, CsvDriven)
{
    const auto& p = GetParam();
    aclblasComplex alpha = {p.alphaFill.val1, p.alphaFill.val2};
    aclblasComplex beta = {p.betaFill.val1, p.betaFill.val2};

    std::vector<float> aHost = CgeamGenerateInput(p, true, 0);
    std::vector<float> bHost = CgeamGenerateInput(p, false, 1);
    const aclblasComplex* aPtr = aHost.empty() ? nullptr : reinterpret_cast<const aclblasComplex*>(aHost.data());
    const aclblasComplex* bPtr = bHost.empty() ? nullptr : reinterpret_cast<const aclblasComplex*>(bHost.data());

    std::vector<float> cHost;
    aclblasComplex* cPtr = CgeamPrepareC(p, aPtr, bPtr, cHost);

    std::vector<float> goldenC = CgeamComputeGolden(p, alpha, beta, aPtr, bPtr, cPtr, aHost, bHost, cHost);

    aclblasStatus_t ret = aclblasCgeam_npu(
        CgeamArch35Test::handle_, p.transa, p.transb, p.m, p.n, &alpha, aPtr, p.lda, &beta, bPtr, p.ldb, cPtr, p.ldc);
    EXPECT_EQ(ret, p.expectResult);
    if (p.expectResult != ACLBLAS_STATUS_SUCCESS) {
        return;
    }

    CgeamVerifyOutput(p, goldenC, aHost, bHost, cHost);
}
