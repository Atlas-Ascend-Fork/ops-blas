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
#include <string>
#include <vector>

#include "verify.h"
#include "fill.h"
#include "blas_test.h"
#include "csv_loader.h"
#include "ssyr2k_param.h"
#include "host_scalar_utils.h"
#include "verify_uplo.h"
#include "ssyr2k_golden.h"
#include "ssyr2k_npu_wrapper.h"

class Ssyr2kArch35Test : public BlasTest<Ssyr2kParam> {};

TEST_F(Ssyr2kArch35Test, NullHandle)
{
    float alpha = 1.0f;
    float beta = 0.0f;
    std::vector<float> a(16, 0.0f);
    std::vector<float> b(16, 0.0f);
    std::vector<float> c(16, 0.0f);
    aclblasStatus_t ret = aclblasSsyr2k_npu(
        nullptr, ACLBLAS_UPPER, ACLBLAS_OP_N, 4, 4, &alpha, a.data(), 4, b.data(), 4, &beta, c.data(), 4);
    EXPECT_EQ(ret, ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
}

// ── TEST_F: host-scalar alpha/beta accepted (issue #759 SS2K-02) ──
// The npu wrapper uploads scalars to device, so this calls the public API
// directly with host-side alpha/beta addresses: they must be dereferenced
// instead of receiving a misleading error from a failed D2H copy. Host golden
// for the UPPER triangle: C = alpha*(A*B^T + B*A^T) + beta*C (trans=N).
static void BuildSsyr2kHostGolden(
    int n, int k, int lda, int ldb, int ldc, float alpha, float beta, const std::vector<float>& aHost,
    const std::vector<float>& bHost, const std::vector<float>& cInit, std::vector<float>& golden)
{
    golden = cInit;
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            float acc = 0.0f;
            for (int t = 0; t < k; t++) {
                acc += aHost[static_cast<size_t>(t) * lda + i] * bHost[static_cast<size_t>(t) * ldb + j] +
                       bHost[static_cast<size_t>(t) * ldb + i] * aHost[static_cast<size_t>(t) * lda + j];
            }
            const size_t idx = static_cast<size_t>(j) * ldc + i;
            golden[idx] = alpha * acc + beta * cInit[idx];
        }
    }
}

static bool RunSsyr2kHostScalarShape(aclblasHandle_t handle, aclrtStream stream, int n, int k, int ld)
{
    const float alpha = 2.0f;
    const float beta = 1.5f;
    std::vector<float> aHost(static_cast<size_t>(ld) * k);
    std::vector<float> bHost(static_cast<size_t>(ld) * k);
    std::vector<float> cInit(static_cast<size_t>(ld) * n);
    host_scalar::FillHostScalarPattern(aHost, bHost, cInit);

    std::vector<float> golden;
    BuildSsyr2kHostGolden(n, k, ld, ld, ld, alpha, beta, aHost, bHost, cInit, golden);

    return host_scalar::RunOnDevice<float>(
        stream, aHost.size() * sizeof(float), bHost.size() * sizeof(float), cInit.size() * sizeof(float), aHost.data(),
        bHost.data(), cInit.data(),
        [&](const void* dA, const void* dB, void* dC) {
            return aclblasSsyr2k(
                handle, ACLBLAS_UPPER, ACLBLAS_OP_N, n, k, &alpha, static_cast<const float*>(dA), ld,
                static_cast<const float*>(dB), ld, &beta, static_cast<float*>(dC), ld);
        },
        [&](const std::vector<float>& cOut) { host_scalar::VerifyUploNear(n, ld, cOut, golden); });
}

TEST_F(Ssyr2kArch35Test, HostScalarAlphaBeta)
{
    ASSERT_TRUE(RunSsyr2kHostScalarShape(handle_, stream_, 4, 3, 4));
    ASSERT_TRUE(RunSsyr2kHostScalarShape(handle_, stream_, 72, 4, 72));
}

// ── TEST_F: n=0 still validates scalars (issue #760 SS2K-03) ──
// Validation runs before the n==0 quick return, so null alpha/beta must yield
// INVALID_VALUE rather than SUCCESS even for n=0.
TEST_F(Ssyr2kArch35Test, ZeroDimValidatesScalars)
{
    float dummy = 0.0f;
    EXPECT_EQ(
        aclblasSsyr2k(handle_, ACLBLAS_UPPER, ACLBLAS_OP_N, 0, 4, nullptr, &dummy, 4, &dummy, 4, &dummy, &dummy, 4),
        ACLBLAS_STATUS_INVALID_VALUE);
    EXPECT_EQ(
        aclblasSsyr2k(handle_, ACLBLAS_UPPER, ACLBLAS_OP_N, 0, 4, &dummy, &dummy, 4, &dummy, 4, nullptr, &dummy, 4),
        ACLBLAS_STATUS_INVALID_VALUE);
}

INSTANTIATE_TEST_SUITE_P(
    Ssyr2k, Ssyr2kArch35Test, ::testing::ValuesIn(GetCasesFromCsv<Ssyr2kParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<Ssyr2kParam>);

TEST_P(Ssyr2kArch35Test, CsvDriven)
{
    const auto& p = GetParam();

    // Compute A/B matrix dimensions based on trans
    int aRows = (p.trans == ACLBLAS_OP_N) ? p.n : p.k;
    int aCols = (p.trans == ACLBLAS_OP_N) ? p.k : p.n;

    // Generate input data (column-major: need lda * numCols elements)
    const size_t aSize = static_cast<size_t>(p.lda) * static_cast<size_t>(aCols);
    const size_t bSize = static_cast<size_t>(p.ldb) * static_cast<size_t>(aCols);
    const size_t cSize = static_cast<size_t>(p.n) * static_cast<size_t>(p.ldc);

    std::vector<float> aHost = makeBlasArray(static_cast<int64_t>(aSize), p.fillA, p.randomSeed);
    std::vector<float> bHost = makeBlasArray(static_cast<int64_t>(bSize), p.fillB, p.randomSeed + 1);
    std::vector<float> cHost = makeBlasArray(static_cast<int64_t>(cSize), p.fillC, p.randomSeed + 2);

    // Save original C data for golden computation BEFORE NPU call modifies it
    std::vector<float> cGolden(cHost);

    const float* alphaPtr = p.nullAlpha ? nullptr : &p.alpha;
    const float* betaPtr = p.nullBeta ? nullptr : &p.beta;
    const float* aPtr = p.nullA ? nullptr : (aHost.empty() ? nullptr : aHost.data());
    const float* bPtr = p.nullB ? nullptr : (bHost.empty() ? nullptr : bHost.data());
    float* cPtr = p.nullC ? nullptr : (cHost.empty() ? nullptr : cHost.data());

    // For n <= 0 or k < 0 or other invalid param cases, pass through to operator
    // (npu_wrapper handles these by direct pass-through)
    if (p.n <= 0 || p.k < 0) {
        aclblasStatus_t ret = aclblasSsyr2k_npu(
            Ssyr2kArch35Test::handle_, p.uplo, p.trans, p.n, p.k, alphaPtr, aPtr, p.lda, bPtr, p.ldb, betaPtr, cPtr,
            p.ldc);
        EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
        return;
    }

    aclblasStatus_t ret = aclblasSsyr2k_npu(
        Ssyr2kArch35Test::handle_, p.uplo, p.trans, p.n, p.k, alphaPtr, aPtr, p.lda, bPtr, p.ldb, betaPtr, cPtr, p.ldc);

    if (p.expectResult != ACLBLAS_STATUS_SUCCESS) {
        EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
        return;
    }
    ASSERT_EQ(ret, ACLBLAS_STATUS_SUCCESS);

    // Compute golden (using original C data saved before NPU call)
    // cGolden was already copied from cHost BEFORE the NPU call modified cHost
    float* goldPtr = cGolden.empty() ? nullptr : cGolden.data();

    aclblasStatus_t goldenRet = aclblasSsyr2k_cpu(
        Ssyr2kArch35Test::handle_, p.uplo, p.trans, p.n, p.k, alphaPtr, aPtr, p.lda, bPtr, p.ldb, betaPtr, goldPtr,
        p.ldc);
    if (goldenRet != ACLBLAS_STATUS_SUCCESS) {
        EXPECT_EQ(goldenRet, ACLBLAS_STATUS_SUCCESS) << "golden computation failed";
        return;
    }

    const size_t cCount = cSize;
    VerifyUploTriangle(p, cPtr, goldPtr, cCount);
}
