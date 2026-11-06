/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>

#include "verify.h"
#include "fill.h"
#include "blas_test.h"
#include "csv_loader.h"
#include "ssymm_param.h"
#include "ssymm_golden.h"
#include "ssymm_npu_wrapper.h"

class SsymmArch35Test : public BlasTest<SsymmParam> {};

// ---------------------------------------------------------------------------
// L0 TEST_F cases — null pointers and validation-order quick returns
// ---------------------------------------------------------------------------

TEST_F(SsymmArch35Test, NullHandle)
{
    float alpha = 1.0f;
    float beta = 1.0f;
    std::vector<float> a(16, 0.0f);
    std::vector<float> b(16, 0.0f);
    std::vector<float> c(16, 0.0f);
    aclblasStatus_t ret = aclblasSsymm_npu(
        nullptr, ACLBLAS_SIDE_LEFT, ACLBLAS_LOWER, 4, 4, &alpha, a.data(), 4, b.data(), 4, &beta, c.data(), 4);
    EXPECT_EQ(ret, ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
}

TEST_F(SsymmArch35Test, NullA)
{
    float alpha = 1.0f;
    float beta = 1.0f;
    std::vector<float> b(16, 0.0f);
    std::vector<float> c(16, 0.0f);
    aclblasStatus_t ret = aclblasSsymm_npu(
        SsymmArch35Test::handle_, ACLBLAS_SIDE_LEFT, ACLBLAS_LOWER, 4, 4, &alpha, nullptr, 4, b.data(), 4, &beta,
        c.data(), 4);
    EXPECT_EQ(ret, ACLBLAS_STATUS_INVALID_VALUE);
}

TEST_F(SsymmArch35Test, NullB)
{
    float alpha = 1.0f;
    float beta = 1.0f;
    std::vector<float> a(16, 0.0f);
    std::vector<float> c(16, 0.0f);
    aclblasStatus_t ret = aclblasSsymm_npu(
        SsymmArch35Test::handle_, ACLBLAS_SIDE_LEFT, ACLBLAS_LOWER, 4, 4, &alpha, a.data(), 4, nullptr, 4, &beta,
        c.data(), 4);
    EXPECT_EQ(ret, ACLBLAS_STATUS_INVALID_VALUE);
}

TEST_F(SsymmArch35Test, NullC)
{
    float alpha = 1.0f;
    float beta = 1.0f;
    std::vector<float> a(16, 0.0f);
    std::vector<float> b(16, 0.0f);
    aclblasStatus_t ret = aclblasSsymm_npu(
        SsymmArch35Test::handle_, ACLBLAS_SIDE_LEFT, ACLBLAS_LOWER, 4, 4, &alpha, a.data(), 4, b.data(), 4, &beta,
        nullptr, 4);
    EXPECT_EQ(ret, ACLBLAS_STATUS_INVALID_VALUE);
}

// BLAS: beta==0 allows C to be NULL (C does not have to be a valid input).
TEST_F(SsymmArch35Test, NullCBetaZero)
{
    float alpha = 1.0f;
    float beta = 0.0f;
    std::vector<float> a(16, 0.0f);
    std::vector<float> b(16, 0.0f);
    aclblasStatus_t ret = aclblasSsymm_npu(
        SsymmArch35Test::handle_, ACLBLAS_SIDE_LEFT, ACLBLAS_LOWER, 4, 4, &alpha, a.data(), 4, b.data(), 4, &beta,
        nullptr, 4);
    EXPECT_EQ(ret, ACLBLAS_STATUS_SUCCESS);
}

// BLAS: alpha==0 allows A to be NULL (A does not need to be a valid input).
// With beta==1, C = beta*C = C (unchanged). Verify C is bit-exact unchanged.
TEST_F(SsymmArch35Test, AlphaZeroNullA)
{
    float alpha = 0.0f;
    float beta = 1.0f;
    const int m = 4;
    const int n = 4;
    const int lda = 4;
    const int ldb = 4;
    const int ldc = 4;
    std::vector<float> bHost = makeBlasMatrix(m, n, ldb, "VALUE_DIAG_1", 0);
    std::vector<float> cHost = makeBlasMatrix(m, n, ldc, "RANDOM_NORM_5_5", 44);
    std::vector<float> cResult = cHost;

    aclblasStatus_t ret = aclblasSsymm_npu(
        SsymmArch35Test::handle_, ACLBLAS_SIDE_LEFT, ACLBLAS_LOWER, m, n, &alpha, nullptr, lda, bHost.data(), ldb,
        &beta, cResult.data(), ldc);
    EXPECT_EQ(ret, ACLBLAS_STATUS_SUCCESS);

    const size_t cCount = static_cast<size_t>(ldc) * static_cast<size_t>(n);
    VerifyConfig cfg;
    cfg.mode = PrecisionMode::EXACT;
    EXPECT_TRUE(Verifier::verifyVector(cResult.data(), cHost.data(), cCount, 1, cfg, "AlphaZeroNullA"))
        << "AlphaZeroNullA: C should be unchanged when alpha==0 and beta==1";
}

// BLAS: alpha==0 allows B to be NULL (B does not need to be a valid input).
// With beta==1, C = beta*C = C (unchanged). Verify C is bit-exact unchanged.
TEST_F(SsymmArch35Test, AlphaZeroNullB)
{
    float alpha = 0.0f;
    float beta = 1.0f;
    const int m = 4;
    const int n = 4;
    const int lda = 4;
    const int ldb = 4;
    const int ldc = 4;
    std::vector<float> aHost = makeBlasMatrix(m, m, lda, "RANDOM_NORM_5_5", 42);
    std::vector<float> cHost = makeBlasMatrix(m, n, ldc, "RANDOM_NORM_5_5", 44);
    std::vector<float> cResult = cHost;

    aclblasStatus_t ret = aclblasSsymm_npu(
        SsymmArch35Test::handle_, ACLBLAS_SIDE_LEFT, ACLBLAS_LOWER, m, n, &alpha, aHost.data(), lda, nullptr, ldb,
        &beta, cResult.data(), ldc);
    EXPECT_EQ(ret, ACLBLAS_STATUS_SUCCESS);

    const size_t cCount = static_cast<size_t>(ldc) * static_cast<size_t>(n);
    VerifyConfig cfg;
    cfg.mode = PrecisionMode::EXACT;
    EXPECT_TRUE(Verifier::verifyVector(cResult.data(), cHost.data(), cCount, 1, cfg, "AlphaZeroNullB"))
        << "AlphaZeroNullB: C should be unchanged when alpha==0 and beta==1";
}

// Exposes Problem 2 fix: handle checked before m==0||n==0 quick return.
TEST_F(SsymmArch35Test, QuickReturnNullHandle)
{
    float alpha = 1.0f;
    float beta = 1.0f;
    aclblasStatus_t ret = aclblasSsymm_npu(
        nullptr, ACLBLAS_SIDE_LEFT, ACLBLAS_LOWER, 0, 0, &alpha, nullptr, 0, nullptr, 0, &beta, nullptr, 0);
    EXPECT_EQ(ret, ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
}

// Exposes Problem 2 fix: side enum checked before m==0 quick return.
TEST_F(SsymmArch35Test, QuickReturnInvalidSide)
{
    float alpha = 1.0f;
    float beta = 1.0f;
    std::vector<float> b(16, 0.0f);
    std::vector<float> c(16, 0.0f);
    aclblasSideMode_t invalidSide = static_cast<aclblasSideMode_t>(0xFF);
    aclblasStatus_t ret = aclblasSsymm_npu(
        SsymmArch35Test::handle_, invalidSide, ACLBLAS_LOWER, 0, 4, &alpha, nullptr, 0, b.data(), 4, &beta, c.data(),
        4);
    EXPECT_EQ(ret, ACLBLAS_STATUS_INVALID_ENUM);
}

// Exposes Problem 2 fix: uplo enum checked before n==0 quick return.
TEST_F(SsymmArch35Test, QuickReturnInvalidUplo)
{
    float alpha = 1.0f;
    float beta = 1.0f;
    std::vector<float> a(16, 0.0f);
    std::vector<float> b(16, 0.0f);
    aclblasFillMode_t invalidUplo = static_cast<aclblasFillMode_t>(0xFF);
    aclblasStatus_t ret = aclblasSsymm_npu(
        SsymmArch35Test::handle_, ACLBLAS_SIDE_LEFT, invalidUplo, 4, 0, &alpha, a.data(), 4, b.data(), 0, &beta,
        nullptr, 0);
    EXPECT_EQ(ret, ACLBLAS_STATUS_INVALID_ENUM);
}

// ---------------------------------------------------------------------------
// L0 TEST_F cases — device alpha/beta pointer mode
//   Exercises the aclrtPointerGetAttributes detection path: alpha/beta scalars
//   are copied to device memory and passed as device pointers. The host reads
//   them back via D2H memcpy. Cases cover both-device (SUCCESS), mixed mode
//   alpha-device/beta-host and alpha-host/beta-device (INVALID_VALUE, BLAS
//   unified pointer mode), and the alpha==0 device case (fast path via D2H).
// ---------------------------------------------------------------------------

// Shared driver for the both-device TEST_F cases: builds column-major matrices,
// runs the NPU op with alpha/beta as device pointers, and verifies against the
// CPU golden. alpha==0 → EXACT (host reads alpha=0 via D2H, fast path skips
// GEMM, result is bit-exact beta*C); otherwise mixed tolerance.
static void RunSsymmDeviceScaleCase(
    aclblasHandle handle, aclblasSideMode_t side, aclblasFillMode_t uplo, int m, int n, float alpha, float beta,
    bool alphaOnDevice, bool betaOnDevice, const char* caseName)
{
    const int aDim = (side == ACLBLAS_SIDE_LEFT) ? m : n;
    const int lda = aDim;
    const int ldb = m;
    const int ldc = m;

    std::vector<float> aHost = makeBlasMatrix(aDim, aDim, lda, "RANDOM_NORM_5_5", 42);
    std::vector<float> bHost = makeBlasMatrix(m, n, ldb, "RANDOM_NORM_5_5", 43);
    std::vector<float> cHost = makeBlasMatrix(m, n, ldc, "RANDOM_NORM_5_5", 44);
    std::vector<float> cResult = cHost;

    aclblasStatus_t ret = aclblasSsymm_npu(
        handle, side, uplo, m, n, &alpha, aHost.data(), lda, bHost.data(), ldb, &beta, cResult.data(), ldc,
        alphaOnDevice, betaOnDevice);
    ASSERT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_SUCCESS)) << caseName;

    std::vector<float> cGolden = cHost;
    // issue #820: golden 返回码必须断言检查，golden 失败时不得以输入初值做参考比对
    ASSERT_EQ(
        aclblasSsymm_cpu(
            handle, side, uplo, m, n, &alpha, aHost.data(), lda, bHost.data(), ldb, &beta, cGolden.data(), ldc),
        ACLBLAS_STATUS_SUCCESS);

    const size_t cCount = static_cast<size_t>(ldc) * static_cast<size_t>(n);
    VerifyConfig cfg;
    if (alpha == 0.0f) {
        cfg.mode = PrecisionMode::EXACT;
    } else {
        applyMixedTolerance(cfg, ACL_FLOAT, cGolden.data(), cCount);
    }
    EXPECT_TRUE(Verifier::verifyVector(cResult.data(), cGolden.data(), cCount, 1, cfg, caseName)) << caseName;
}

TEST_F(SsymmArch35Test, DeviceAlphaDeviceBeta)
{
    RunSsymmDeviceScaleCase(
        SsymmArch35Test::handle_, ACLBLAS_SIDE_LEFT, ACLBLAS_UPPER, 8, 6, 2.0f, 0.5f, /*alphaOnDevice=*/true,
        /*betaOnDevice=*/true, "DeviceAlphaDeviceBeta");
}

// Mixed pointer mode (alpha device + beta host) must be rejected.
TEST_F(SsymmArch35Test, DeviceAlphaHostBeta)
{
    float alpha = 1.5f;
    float beta = 1.0f;
    std::vector<float> a(64, 0.0f);
    std::vector<float> b(64, 0.0f);
    std::vector<float> c(64, 0.0f);
    aclblasStatus_t ret = aclblasSsymm_npu(
        SsymmArch35Test::handle_, ACLBLAS_SIDE_RIGHT, ACLBLAS_LOWER, 8, 8, &alpha, a.data(), 8, b.data(), 8, &beta,
        c.data(), 8,
        /*alphaOnDevice=*/true, /*betaOnDevice=*/false);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE));
}

// Mixed pointer mode (alpha host + beta device) must be rejected.
TEST_F(SsymmArch35Test, HostAlphaDeviceBeta)
{
    float alpha = 1.0f;
    float beta = 0.5f;
    std::vector<float> a(64, 0.0f);
    std::vector<float> b(64, 0.0f);
    std::vector<float> c(64, 0.0f);
    aclblasStatus_t ret = aclblasSsymm_npu(
        SsymmArch35Test::handle_, ACLBLAS_SIDE_LEFT, ACLBLAS_LOWER, 8, 8, &alpha, a.data(), 8, b.data(), 8, &beta,
        c.data(), 8,
        /*alphaOnDevice=*/false, /*betaOnDevice=*/true);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE));
}

// alpha==0 on device: host reads alpha=0 via D2H, fast path skips GEMM.
// beta==1 → C unchanged (EXACT).
TEST_F(SsymmArch35Test, DeviceAlphaZeroBetaOne)
{
    RunSsymmDeviceScaleCase(
        SsymmArch35Test::handle_, ACLBLAS_SIDE_LEFT, ACLBLAS_UPPER, 8, 8, 0.0f, 1.0f, /*alphaOnDevice=*/true,
        /*betaOnDevice=*/true, "DeviceAlphaZeroBetaOne");
}

// alpha==0 + beta==0.5, both device: host reads alpha=0 via D2H, fast path skips GEMM, C = 0.5*C (EXACT).
TEST_F(SsymmArch35Test, DeviceAlphaZeroBetaScale)
{
    RunSsymmDeviceScaleCase(
        SsymmArch35Test::handle_, ACLBLAS_SIDE_RIGHT, ACLBLAS_UPPER, 8, 8, 0.0f, 0.5f, /*alphaOnDevice=*/true,
        /*betaOnDevice=*/true, "DeviceAlphaZeroBetaScale");
}

INSTANTIATE_TEST_SUITE_P(
    Ssymm, SsymmArch35Test, ::testing::ValuesIn(GetCasesFromCsv<SsymmParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<SsymmParam>);

TEST_P(SsymmArch35Test, CsvDriven)
{
    const auto& p = GetParam();
    const int aDim = (p.side == ACLBLAS_SIDE_LEFT) ? p.m : p.n;

    std::vector<float> aHost = makeBlasMatrix(aDim, aDim, p.lda, p.aFill, p.randomSeed);
    std::vector<float> bHost = makeBlasMatrix(p.m, p.n, p.ldb, p.bFill, p.randomSeed + 1);
    std::vector<float> cHost = makeBlasMatrix(p.m, p.n, p.ldc, p.cFill, p.randomSeed + 2);
    std::vector<float> cResult = cHost;

    const float* alphaPtr = p.nullAlpha ? nullptr : &p.alpha;
    const float* betaPtr = p.nullBeta ? nullptr : &p.beta;
    const float* aPtr = aHost.empty() ? nullptr : aHost.data();
    const float* bPtr = bHost.empty() ? nullptr : bHost.data();
    float* cPtr = cResult.empty() ? nullptr : cResult.data();

    aclblasStatus_t ret = aclblasSsymm_npu(
        SsymmArch35Test::handle_, p.side, p.uplo, p.m, p.n, alphaPtr, aPtr, p.lda, bPtr, p.ldb, betaPtr, cPtr, p.ldc);

    if (p.expectResult != ACLBLAS_STATUS_SUCCESS) {
        EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
        return;
    }
    ASSERT_EQ(ret, ACLBLAS_STATUS_SUCCESS);

    std::vector<float> cGolden = cHost;
    float* goldPtr = cGolden.empty() ? nullptr : cGolden.data();

    // issue #820: golden 返回码必须断言检查，golden 失败时不得以输入初值做参考比对
    ASSERT_EQ(
        aclblasSsymm_cpu(
            SsymmArch35Test::handle_, p.side, p.uplo, p.m, p.n, alphaPtr, aPtr, p.lda, bPtr, p.ldb, betaPtr, goldPtr,
            p.ldc),
        ACLBLAS_STATUS_SUCCESS);

    if (p.m == 0 || p.n == 0)
        return;
    const size_t cCount = static_cast<size_t>(p.ldc) * static_cast<size_t>(p.n);

    // Use mixed tolerance (ops-precision-standard); alpha==0 → exact (C = beta*C, bit-reproducible).
    float alphaVal = p.alpha;
    VerifyConfig cfg;
    if (alphaVal == 0.0f) {
        cfg.mode = PrecisionMode::EXACT;
    } else {
        applyMixedTolerance(cfg, ACL_FLOAT, goldPtr, cCount);
    }
    EXPECT_TRUE(Verifier::verifyVector(cPtr, goldPtr, cCount, 1, cfg, p.caseName));
}
