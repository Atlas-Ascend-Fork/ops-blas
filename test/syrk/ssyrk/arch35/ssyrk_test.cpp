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
#include <vector>

#include "verify.h"
#include "blas_test.h"
#include "csv_loader.h"
#include "fill.h"
#include "host_scalar_utils.h"
#include "verify_uplo.h"
#include "ssyrk_param.h"
#include "ssyrk_golden.h"
#include "ssyrk_npu_wrapper.h"

// ═══════════════════════════════════════════════════════════════════════════════
// Test fixture
// ═══════════════════════════════════════════════════════════════════════════════

class SsyrkTest : public BlasTest<SsyrkParam> {};

// ── TEST_F: null handle (not CSV-driven) ──
TEST_F(SsyrkTest, NullHandle)
{
    float alpha = 1.0f;
    float beta = 0.0f;
    aclblasStatus_t ret =
        aclblasSsyrk_npu(nullptr, ACLBLAS_UPPER, ACLBLAS_OP_N, 4, 4, &alpha, nullptr, 4, &beta, nullptr, 4);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_HANDLE_IS_NULLPTR));
}

// ── TEST_F: host-scalar alpha/beta accepted (issue #752 SRK-02) ──
// The npu wrapper uploads scalars to device, so this calls the public API
// directly with host-side alpha/beta addresses: they must be dereferenced
// instead of receiving a misleading error from a failed D2H copy. Host golden
// for the UPPER triangle: C = alpha*A*A^T + beta*C.
static void BuildSsyrkHostGolden(
    int n, int k, int lda, int ldc, float alpha, float beta, const std::vector<float>& aHost,
    const std::vector<float>& cInit, std::vector<float>& golden)
{
    golden = cInit;
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            float acc = 0.0f;
            for (int t = 0; t < k; t++) {
                acc += aHost[static_cast<size_t>(t) * lda + i] * aHost[static_cast<size_t>(t) * lda + j];
            }
            const size_t idx = static_cast<size_t>(j) * ldc + i;
            golden[idx] = alpha * acc + beta * cInit[idx];
        }
    }
}

static bool RunSsyrkHostScalarShape(aclblasHandle_t handle, aclrtStream stream, int n, int k, int lda, int ldc)
{
    const float alpha = 2.0f;
    const float beta = 1.5f;
    std::vector<float> aHost(static_cast<size_t>(lda) * k);
    std::vector<float> cInit(static_cast<size_t>(ldc) * n);
    host_scalar::FillHostScalarPattern(aHost, cInit);

    std::vector<float> golden;
    BuildSsyrkHostGolden(n, k, lda, ldc, alpha, beta, aHost, cInit, golden);

    return host_scalar::RunOnDevice<float>(
        stream, aHost.size() * sizeof(float), 0, cInit.size() * sizeof(float), aHost.data(), nullptr, cInit.data(),
        [&](const void* dA, const void*, void* dC) {
            return aclblasSsyrk(
                handle, ACLBLAS_UPPER, ACLBLAS_OP_N, n, k, &alpha, static_cast<const float*>(dA), lda, &beta,
                static_cast<float*>(dC), ldc);
        },
        [&](const std::vector<float>& cOut) { host_scalar::VerifyUploNear(n, ldc, cOut, golden); });
}

TEST_F(SsyrkTest, HostScalarAlphaBeta)
{
    ASSERT_TRUE(RunSsyrkHostScalarShape(handle_, stream_, 4, 3, 4, 4));
    ASSERT_TRUE(RunSsyrkHostScalarShape(handle_, stream_, 72, 4, 72, 72));
}

// ── TEST_F: n=0 still validates scalars (issue #752 SRK-03) ──
// Validation runs before the n==0 quick return, so null alpha/beta must yield
// INVALID_VALUE rather than SUCCESS even for n=0.
TEST_F(SsyrkTest, ZeroDimValidatesScalars)
{
    float dummy = 0.0f;
    EXPECT_EQ(
        aclblasSsyrk(handle_, ACLBLAS_UPPER, ACLBLAS_OP_N, 0, 4, nullptr, &dummy, 4, &dummy, &dummy, 4),
        ACLBLAS_STATUS_INVALID_VALUE);
    EXPECT_EQ(
        aclblasSsyrk(handle_, ACLBLAS_UPPER, ACLBLAS_OP_N, 0, 4, &dummy, &dummy, 4, nullptr, &dummy, 4),
        ACLBLAS_STATUS_INVALID_VALUE);
}

INSTANTIATE_TEST_SUITE_P(
    Ssyrk, SsyrkTest, ::testing::ValuesIn(GetCasesFromCsv<SsyrkParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<SsyrkParam>);

// ═══════════════════════════════════════════════════════════════════════════════
// CSV-driven parameterised test (5-step flow)
//   1. Generate host data  2. Run NPU  3. Check return code
//   4. Run CPU golden       5. Verify precision
// ═══════════════════════════════════════════════════════════════════════════════

struct SsyrkHostData {
    std::vector<float> aHost;
    std::vector<float> cHost;
    std::vector<float> cGolden;
    const float* aPtr = nullptr;
    float* cPtr = nullptr;
    float* cGoldenPtr = nullptr;
    size_t cCount = 0;
};

static bool PrepareHostData(const SsyrkParam& p, SsyrkHostData& d)
{
    const int aRows = p.lda;
    const int aCols = (p.trans == ACLBLAS_OP_N) ? p.k : p.n;

    const size_t cBytes = static_cast<size_t>(p.ldc) * static_cast<size_t>(p.n) * sizeof(float);
    const size_t aBytes = static_cast<size_t>(p.lda) * static_cast<size_t>(aCols) * sizeof(float);
    constexpr size_t kHostMemLimit = 8ULL * 1024ULL * 1024ULL * 1024ULL;
    if (3 * cBytes + aBytes > kHostMemLimit) {
        std::cout << "[SKIP] host memory estimate (" << (3 * cBytes + aBytes) / (1024 * 1024)
                  << " MB) exceeds limit for n=" << p.n << ", ldc=" << p.ldc << std::endl;
        return false;
    }

    try {
        d.aHost = makeBlasMatrix(aRows, aCols, p.lda, p.aFill, p.randomSeed);
        d.cHost = makeBlasMatrix(p.n, p.n, p.ldc, p.cFill, p.randomSeed + 1);
    } catch (const std::bad_alloc&) {
        std::cout << "[SKIP] host memory allocation failed for n=" << p.n << ", k=" << p.k << std::endl;
        return false;
    }

    d.aPtr = (d.aHost.empty() || p.nullA) ? nullptr : d.aHost.data();
    d.cPtr = (d.cHost.empty() || p.nullC) ? nullptr : d.cHost.data();
    d.cCount = static_cast<size_t>(p.ldc) * static_cast<size_t>(p.n);

    if (d.cPtr != nullptr) {
        try {
            d.cGolden = d.cHost;
        } catch (const std::bad_alloc&) {
            std::cout << "[SKIP] host memory allocation failed for golden copy (n=" << p.n << ")" << std::endl;
            return false;
        }
        d.cGoldenPtr = d.cGolden.data();
    }
    return true;
}

TEST_P(SsyrkTest, CsvDriven)
{
    const auto& p = GetParam();
    SsyrkHostData d;
    if (!PrepareHostData(p, d)) {
        GTEST_SKIP() << "Skipped: host memory limit";
    }

    aclblasStatus_t ret = aclblasSsyrk_npu(
        SsyrkTest::handle_, p.uplo, p.trans, p.n, p.k, &p.alpha, d.aPtr, p.lda, &p.beta, d.cPtr, p.ldc);

    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
    if (p.expectResult != ACLBLAS_STATUS_SUCCESS || p.n == 0 || d.cPtr == nullptr) {
        return;
    }

    aclblasStatus_t goldenRet = aclblasSsyrk_cpu(
        SsyrkTest::handle_, p.uplo, p.trans, p.n, p.k, &p.alpha, d.aPtr, p.lda, &p.beta, d.cGoldenPtr, p.ldc);
    if (goldenRet != ACLBLAS_STATUS_SUCCESS) {
        EXPECT_EQ(goldenRet, ACLBLAS_STATUS_SUCCESS) << "golden computation failed";
        return;
    }

    VerifyUploTriangle(p, d.cPtr, d.cGoldenPtr, d.cCount);
}
