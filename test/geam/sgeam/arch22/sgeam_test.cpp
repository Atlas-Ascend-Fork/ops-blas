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
#include <vector>

#include "verify.h"
#include "blas_test.h"
#include "csv_loader.h"
#include "fill.h"
#include "sgeam_param.h"
#include "../../geam_golden.h"
#include "sgeam_npu_wrapper.h"

#include "gtest/gtest.h"

// -- Test fixture ------------------------------------------------------------
class SgeamTest : public BlasTest<SgeamParam> {};

// -- TEST_F: null handle (not in CSV) ---------------------------------------
TEST_F(SgeamTest, NullHandle)
{
    float alpha = 1.0f;
    float beta = 1.0f;
    std::vector<float> A(4 * 4, 1.0f);
    std::vector<float> B(4 * 4, 1.0f);
    std::vector<float> C(4 * 4, 0.0f);
    EXPECT_EQ(
        aclblasSgeam(nullptr, ACLBLAS_OP_N, ACLBLAS_OP_N, 4, 4, &alpha, A.data(), 4, &beta, B.data(), 4, C.data(), 4),
        ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
}

// -- TEST_F: A=nullptr (host validates and returns error) --------------------
TEST_F(SgeamTest, NullA)
{
    float alpha = 1.0f;
    float beta = 1.0f;
    std::vector<float> B(4 * 4, 1.0f);
    std::vector<float> C(4 * 4, 0.0f);
    EXPECT_EQ(
        aclblasSgeam(
            SgeamTest::handle_, ACLBLAS_OP_N, ACLBLAS_OP_N, 4, 4, &alpha, nullptr, 4, &beta, B.data(), 4,
            C.data(), 4),
        ACLBLAS_STATUS_INVALID_VALUE);
}

// -- TEST_F: B=nullptr (host validates and returns error) --------------------
TEST_F(SgeamTest, NullB)
{
    float alpha = 1.0f;
    float beta = 1.0f;
    std::vector<float> A(4 * 4, 1.0f);
    std::vector<float> C(4 * 4, 0.0f);
    EXPECT_EQ(
        aclblasSgeam(
            SgeamTest::handle_, ACLBLAS_OP_N, ACLBLAS_OP_N, 4, 4, &alpha, A.data(), 4, &beta, nullptr, 4,
            C.data(), 4),
        ACLBLAS_STATUS_INVALID_VALUE);
}

// -- TEST_F: C=nullptr (host validates and returns error) --------------------
TEST_F(SgeamTest, NullC)
{
    float alpha = 1.0f;
    float beta = 1.0f;
    std::vector<float> A(4 * 4, 1.0f);
    std::vector<float> B(4 * 4, 1.0f);
    EXPECT_EQ(
        aclblasSgeam(
            SgeamTest::handle_, ACLBLAS_OP_N, ACLBLAS_OP_N, 4, 4, &alpha, A.data(), 4, &beta, B.data(), 4,
            nullptr, 4),
        ACLBLAS_STATUS_INVALID_VALUE);
}

// -- TEST_F: alpha=nullptr returns INVALID_VALUE (host validates) ------------
TEST_F(SgeamTest, NullAlpha)
{
    float beta = 1.0f;
    std::vector<float> A(4 * 4, 2.0f);
    std::vector<float> B(4 * 4, 3.0f);
    std::vector<float> C(4 * 4, kBlasSentinel);
    EXPECT_EQ(
        aclblasSgeam(
            SgeamTest::handle_, ACLBLAS_OP_N, ACLBLAS_OP_N, 4, 4, nullptr, A.data(), 4, &beta, B.data(), 4,
            C.data(), 4),
        ACLBLAS_STATUS_INVALID_VALUE);
}

// -- TEST_F: beta=nullptr returns INVALID_VALUE (host validates) -------------
TEST_F(SgeamTest, NullBeta)
{
    float alpha = 1.0f;
    std::vector<float> A(4 * 4, 2.0f);
    std::vector<float> B(4 * 4, 3.0f);
    std::vector<float> C(4 * 4, kBlasSentinel);
    EXPECT_EQ(
        aclblasSgeam(
            SgeamTest::handle_, ACLBLAS_OP_N, ACLBLAS_OP_N, 4, 4, &alpha, A.data(), 4, nullptr, B.data(), 4,
            C.data(), 4),
        ACLBLAS_STATUS_INVALID_VALUE);
}

// -- TEST_F: transa invalid 0xFF (host validates) ----------------------------
TEST_F(SgeamTest, TransaInvalid)
{
    float alpha = 1.0f;
    float beta = 1.0f;
    std::vector<float> A(4 * 4, 1.0f);
    std::vector<float> B(4 * 4, 1.0f);
    std::vector<float> C(4 * 4, 0.0f);
    EXPECT_EQ(
        aclblasSgeam(
            SgeamTest::handle_, static_cast<aclblasOperation_t>(0xFF), ACLBLAS_OP_N, 4, 4, &alpha, A.data(), 4,
            &beta, B.data(), 4, C.data(), 4),
        ACLBLAS_STATUS_INVALID_ENUM);
}

// -- TEST_F: transb invalid 0xFF (host validates) ----------------------------
TEST_F(SgeamTest, TransbInvalid)
{
    float alpha = 1.0f;
    float beta = 1.0f;
    std::vector<float> A(4 * 4, 1.0f);
    std::vector<float> B(4 * 4, 1.0f);
    std::vector<float> C(4 * 4, 0.0f);
    EXPECT_EQ(
        aclblasSgeam(
            SgeamTest::handle_, ACLBLAS_OP_N, static_cast<aclblasOperation_t>(0xFF), 4, 4, &alpha, A.data(), 4,
            &beta, B.data(), 4, C.data(), 4),
        ACLBLAS_STATUS_INVALID_ENUM);
}

// -- TEST_F: m<0 (host validates and returns error) -------------------------
TEST_F(SgeamTest, MNegative)
{
    float alpha = 1.0f;
    float beta = 1.0f;
    EXPECT_EQ(
        aclblasSgeam(
            SgeamTest::handle_, ACLBLAS_OP_N, ACLBLAS_OP_N, -1, 4, &alpha, nullptr, 1, &beta, nullptr, 1, nullptr,
            1),
        ACLBLAS_STATUS_INVALID_VALUE);
}

// -- TEST_F: n<0 (host validates and returns error) -------------------------
TEST_F(SgeamTest, NNegative)
{
    float alpha = 1.0f;
    float beta = 1.0f;
    EXPECT_EQ(
        aclblasSgeam(
            SgeamTest::handle_, ACLBLAS_OP_N, ACLBLAS_OP_N, 4, -1, &alpha, nullptr, 4, &beta, nullptr, 4, nullptr,
            4),
        ACLBLAS_STATUS_INVALID_VALUE);
}

// -- TEST_F: lda too small when transa=N (host validates) --------------------
TEST_F(SgeamTest, LdaTooSmallN)
{
    float alpha = 1.0f;
    float beta = 1.0f;
    std::vector<float> A(4 * 4, 1.0f);
    std::vector<float> B(4 * 4, 1.0f);
    std::vector<float> C(4 * 4, 0.0f);
    EXPECT_EQ(
        aclblasSgeam(
            SgeamTest::handle_, ACLBLAS_OP_N, ACLBLAS_OP_N, 4, 4, &alpha, A.data(), 3, &beta, B.data(), 4,
            C.data(), 4),
        ACLBLAS_STATUS_INVALID_VALUE);
}

// -- TEST_F: lda too small when transa=T (lda<n required) -------------------
TEST_F(SgeamTest, LdaTooSmallT)
{
    float alpha = 1.0f;
    float beta = 1.0f;
    std::vector<float> A(4 * 8, 1.0f);
    std::vector<float> B(4 * 8, 1.0f);
    std::vector<float> C(4 * 8, 0.0f);
    EXPECT_EQ(
        aclblasSgeam(
            SgeamTest::handle_, ACLBLAS_OP_T, ACLBLAS_OP_N, 4, 8, &alpha, A.data(), 7, &beta, B.data(), 8,
            C.data(), 4),
        ACLBLAS_STATUS_INVALID_VALUE);
}

// -- TEST_F: ldc too small (ldc < m) (host validates) -----------------------
TEST_F(SgeamTest, LdcTooSmall)
{
    float alpha = 1.0f;
    float beta = 1.0f;
    std::vector<float> A(8 * 4, 1.0f);
    std::vector<float> B(8 * 4, 1.0f);
    std::vector<float> C(8 * 4, 0.0f);
    EXPECT_EQ(
        aclblasSgeam(
            SgeamTest::handle_, ACLBLAS_OP_N, ACLBLAS_OP_N, 8, 4, &alpha, A.data(), 8, &beta, B.data(), 8,
            C.data(), 7),
        ACLBLAS_STATUS_INVALID_VALUE);
}

// -- TEST_F: in-place C==A with transa!=N (host validates) -------------------
TEST_F(SgeamTest, InplaceATransInvalid)
{
    float alpha = 1.0f;
    float beta = 1.0f;
    std::vector<float> A(8 * 8, 1.0f);
    std::vector<float> B(8 * 8, 1.0f);
    EXPECT_EQ(
        aclblasSgeam(
            SgeamTest::handle_, ACLBLAS_OP_T, ACLBLAS_OP_N, 8, 8, &alpha, A.data(), 8, &beta, B.data(), 8,
            A.data(), 8),
        ACLBLAS_STATUS_INVALID_VALUE);
}

// -- TEST_F: in-place C==B with transb!=N (host validates) -------------------
TEST_F(SgeamTest, InplaceBTransInvalid)
{
    float alpha = 1.0f;
    float beta = 1.0f;
    std::vector<float> A(8 * 8, 1.0f);
    std::vector<float> B(8 * 8, 1.0f);
    EXPECT_EQ(
        aclblasSgeam(
            SgeamTest::handle_, ACLBLAS_OP_N, ACLBLAS_OP_T, 8, 8, &alpha, A.data(), 8, &beta, B.data(), 8,
            B.data(), 8),
        ACLBLAS_STATUS_INVALID_VALUE);
}

// -- TEST_F: in-place C==A with lda!=ldc (host validates) --------------------
TEST_F(SgeamTest, InplaceALdaNeLdc)
{
    float alpha = 1.0f;
    float beta = 1.0f;
    std::vector<float> A(10 * 8, 1.0f);
    std::vector<float> B(10 * 8, 1.0f);
    EXPECT_EQ(
        aclblasSgeam(
            SgeamTest::handle_, ACLBLAS_OP_N, ACLBLAS_OP_N, 8, 8, &alpha, A.data(), 10, &beta, B.data(), 8,
            A.data(), 8),
        ACLBLAS_STATUS_INVALID_VALUE);
}

// -- TEST_F: in-place C==B with ldb!=ldc (host validates) --------------------
TEST_F(SgeamTest, InplaceBLdbNeLdc)
{
    float alpha = 1.0f;
    float beta = 1.0f;
    std::vector<float> A(10 * 8, 1.0f);
    std::vector<float> B(10 * 8, 1.0f);
    EXPECT_EQ(
        aclblasSgeam(
            SgeamTest::handle_, ACLBLAS_OP_N, ACLBLAS_OP_N, 8, 8, &alpha, A.data(), 8, &beta, B.data(), 10,
            B.data(), 8),
        ACLBLAS_STATUS_INVALID_VALUE);
}

// -- TEST_F: alpha=nullptr with A=nullptr returns INVALID_VALUE -------------
TEST_F(SgeamTest, NullAlphaAllowsNullA)
{
    float beta = 1.0f;
    std::vector<float> B(4 * 4, 1.0f);
    std::vector<float> C(4 * 4, kBlasSentinel);
    EXPECT_EQ(
        aclblasSgeam(
            SgeamTest::handle_, ACLBLAS_OP_N, ACLBLAS_OP_N, 4, 4, nullptr, nullptr, 4, &beta, B.data(), 4,
            C.data(), 4),
        ACLBLAS_STATUS_INVALID_VALUE);
}

// -- CSV parameterised test suite -------------------------------------------
INSTANTIATE_TEST_SUITE_P(
    Sgeam, SgeamTest, ::testing::ValuesIn(GetCasesFromCsv<SgeamParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<SgeamParam>);

// -- Helper: generate A or B input matrix ---------------------------------------
static std::vector<float> SgeamGenerateInput(const SgeamParam& p, bool isA, float extraSeed)
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
    return makeBlasMatrix(rows, cols, ld, fill, p.randomSeed + static_cast<uint32_t>(extraSeed));
}

// -- Helper: prepare C buffer (in-place aware) --------------------------------
static float* SgeamPrepareC(const SgeamParam& p, const float* aPtr, const float* bPtr, std::vector<float>& cHost)
{
    if (p.inplace == 1 && aPtr) {
        return const_cast<float*>(aPtr);
    }
    if (p.inplace == 2 && bPtr) {
        return const_cast<float*>(bPtr);
    }
    if (p.nullC == 0 && p.m > 0 && p.n > 0) {
        cHost.assign(static_cast<size_t>(p.ldc) * static_cast<size_t>(p.n), kBlasSentinel);
    }
    return cHost.empty() ? nullptr : cHost.data();
}

// -- Helper: compute golden reference ------------------------------------------
static std::vector<float> SgeamComputeGolden(
    const SgeamParam& p, float alpha, float beta, const float* aPtr, const float* bPtr, float* cPtr,
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
    aclblasGeam_cpu<float>(
        p.transa, p.transb, static_cast<std::size_t>(p.m), static_cast<std::size_t>(p.n), alpha, aPtr,
        static_cast<std::size_t>(p.lda), beta, bPtr, static_cast<std::size_t>(p.ldb), goldenC.data(),
        static_cast<std::size_t>(p.ldc));
    return goldenC;
}

// -- Helper: verify output precision ------------------------------------------
static void SgeamVerifyOutput(
    const SgeamParam& p, const std::vector<float>& goldenC,
    const std::vector<float>& aHost, const std::vector<float>& bHost, const std::vector<float>& cHost)
{
    VerifyConfig cfg;
    cfg.mode = PrecisionMode::MERE_MARE;
    cfg.mereThreshold = p.mereThreshold;
    cfg.mareMultiplier = p.mareMultiplier;
    const float* outPtr = nullptr;
    size_t outSize = 0;
    if (p.inplace == 1) {
        outPtr = aHost.data();
        outSize = aHost.size();
    } else if (p.inplace == 2) {
        outPtr = bHost.data();
        outSize = bHost.size();
    } else {
        outPtr = cHost.data();
        outSize = cHost.size();
    }
    if (outSize > 0) {
        EXPECT_TRUE(Verifier::verifyVector(outPtr, goldenC.data(), outSize, 1, cfg, p.caseName));
    }
}

// -- TEST_P: 5-step CSV-driven flow -----------------------------------------
TEST_P(SgeamTest, CsvDriven)
{
    const auto& p = GetParam();
    float alpha = p.alphaFill.val1;
    float beta = p.betaFill.val1;

    std::vector<float> aHost = SgeamGenerateInput(p, true, 0);
    std::vector<float> bHost = SgeamGenerateInput(p, false, 1);
    const float* aPtr = aHost.empty() ? nullptr : aHost.data();
    const float* bPtr = bHost.empty() ? nullptr : bHost.data();

    std::vector<float> cHost;
    float* cPtr = SgeamPrepareC(p, aPtr, bPtr, cHost);

    std::vector<float> goldenC = SgeamComputeGolden(p, alpha, beta, aPtr, bPtr, cPtr, aHost, bHost, cHost);

    aclblasStatus_t ret = aclblasSgeam_npu(
        SgeamTest::handle_, p.transa, p.transb, p.m, p.n, &alpha, aPtr, p.lda, &beta, bPtr, p.ldb, cPtr, p.ldc);
    EXPECT_EQ(static_cast<int>(ret), ACLBLAS_STATUS_SUCCESS);

    SgeamVerifyOutput(p, goldenC, aHost, bHost, cHost);
}
