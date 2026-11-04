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
#include "verify_uplo.h"
#include "ssyrkx_param.h"
#include "host_scalar_utils.h"
#include "ssyrkx_golden.h"
#include "ssyrkx_npu_wrapper.h"

class SsyrkxTest : public BlasTest<SsyrkxParam> {};

TEST_F(SsyrkxTest, NullHandle)
{
    float alpha = 1.0f;
    float beta = 0.0f;
    float dummy = 0.0f;
    aclblasStatus_t ret =
        aclblasSsyrkx_npu(nullptr, ACLBLAS_UPPER, ACLBLAS_OP_N, 4, 4, &alpha, &dummy, 4, &dummy, 4, &beta, &dummy, 4);
    EXPECT_EQ(ret, ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
}

// ── TEST_F: host-scalar alpha/beta accepted (issue #759 SX-02) ──
// The npu wrapper uploads scalars to device, so this calls the public API
// directly with host-side alpha/beta addresses: they must be dereferenced
// instead of receiving a misleading error from a failed D2H copy. Host golden
// for the UPPER triangle: C = alpha*A*B^T + beta*C (trans=N).
static void BuildSsyrkxHostGolden(
    int n, int k, int lda, int ldb, int ldc, float alpha, float beta, const std::vector<float>& aHost,
    const std::vector<float>& bHost, const std::vector<float>& cInit, std::vector<float>& golden)
{
    golden = cInit;
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            float acc = 0.0f;
            for (int t = 0; t < k; t++) {
                acc += aHost[static_cast<size_t>(t) * lda + i] * bHost[static_cast<size_t>(t) * ldb + j];
            }
            const size_t idx = static_cast<size_t>(j) * ldc + i;
            golden[idx] = alpha * acc + beta * cInit[idx];
        }
    }
}

static bool RunSsyrkxHostScalarShape(aclblasHandle_t handle, aclrtStream stream, int n, int k, int ld)
{
    const float alpha = 2.0f;
    const float beta = 1.5f;
    std::vector<float> aHost(static_cast<size_t>(ld) * k);
    std::vector<float> bHost(static_cast<size_t>(ld) * k);
    std::vector<float> cInit(static_cast<size_t>(ld) * n);
    host_scalar::FillHostScalarPattern(aHost, bHost, cInit);

    std::vector<float> golden;
    BuildSsyrkxHostGolden(n, k, ld, ld, ld, alpha, beta, aHost, bHost, cInit, golden);

    return host_scalar::RunOnDevice<float>(
        stream, aHost.size() * sizeof(float), bHost.size() * sizeof(float), cInit.size() * sizeof(float), aHost.data(),
        bHost.data(), cInit.data(),
        [&](const void* dA, const void* dB, void* dC) {
            return aclblasSsyrkx(
                handle, ACLBLAS_UPPER, ACLBLAS_OP_N, n, k, &alpha, static_cast<const float*>(dA), ld,
                static_cast<const float*>(dB), ld, &beta, static_cast<float*>(dC), ld);
        },
        [&](const std::vector<float>& cOut) { host_scalar::VerifyUploNear(n, ld, cOut, golden); });
}

TEST_F(SsyrkxTest, HostScalarAlphaBeta)
{
    ASSERT_TRUE(RunSsyrkxHostScalarShape(handle_, stream_, 4, 3, 4));
    ASSERT_TRUE(RunSsyrkxHostScalarShape(handle_, stream_, 72, 4, 72));
}

INSTANTIATE_TEST_SUITE_P(
    Ssyrkx, SsyrkxTest, ::testing::ValuesIn(GetCasesFromCsv<SsyrkxParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<SsyrkxParam>);

TEST_P(SsyrkxTest, CsvDriven)
{
    const auto& p = GetParam();

    // Determine matrix dimensions based on trans (A and B have same dimensions in syrkx)
    int aRows = (p.trans == ACLBLAS_OP_N) ? p.n : p.k;
    int aCols = (p.trans == ACLBLAS_OP_N) ? p.k : p.n;

    // Generate data using BlasFillMode from CSV
    uint32_t seedA = p.randomSeed;
    uint32_t seedB = p.randomSeed + 1;
    uint32_t seedC = p.randomSeed + 2;

    std::vector<float> aHost = makeBlasMatrix(aRows, aCols, p.lda, p.a, seedA);
    std::vector<float> bHost = makeBlasMatrix(aRows, aCols, p.ldb, p.b, seedB);
    std::vector<float> cHost = makeBlasMatrix(p.n, p.n, p.ldc, p.c, seedC);

    // Handle nullptr flags for error injection cases
    const float* aPtr = p.nullA ? nullptr : (aHost.empty() ? nullptr : aHost.data());
    const float* bPtr = p.nullB ? nullptr : (bHost.empty() ? nullptr : bHost.data());
    float* cPtr = p.nullC ? nullptr : (cHost.empty() ? nullptr : cHost.data());
    const float* alphaPtr = p.nullAlpha ? nullptr : &p.alpha;
    const float* betaPtr = p.nullBeta ? nullptr : &p.beta;

    // Save original C for golden's non-uplo triangle restoration
    std::vector<float> originalC;
    if (cPtr != nullptr && p.n > 0) {
        originalC.assign(cHost.begin(), cHost.end());
    }

    size_t cCount = (p.n > 0) ? static_cast<size_t>(p.ldc) * static_cast<size_t>(p.n) : 0;

    // Step 1: NPU execution
    aclblasStatus_t ret = aclblasSsyrkx_npu(
        SsyrkxTest::handle_, p.uplo, p.trans, p.n, p.k, alphaPtr, aPtr, p.lda, bPtr, p.ldb, betaPtr, cPtr, p.ldc);

    // Step 2: Check return code
    if (p.expectResult != ACLBLAS_STATUS_SUCCESS) {
        EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult))
            << "case=" << p.caseName << " expected=" << static_cast<int>(p.expectResult)
            << " got=" << static_cast<int>(ret);
        return;
    }
    ASSERT_EQ(ret, ACLBLAS_STATUS_SUCCESS) << "case=" << p.caseName;

    // Step 3: Quick-return case (n==0) — no computation, C unchanged
    if (p.n == 0) {
        return;
    }

    // Step 4: Golden computation
    std::vector<float> goldenC(originalC);

    aclblasStatus_t goldenRet = aclblasSsyrkx_cpu(
        SsyrkxTest::handle_, p.uplo, p.trans, p.n, p.k, alphaPtr, aPtr, p.lda, bPtr, p.ldb, betaPtr, goldenC.data(),
        p.ldc, originalC.data());

    ASSERT_EQ(goldenRet, ACLBLAS_STATUS_SUCCESS) << "golden failed for case=" << p.caseName;

    // Step 5: Verify uplo triangle + non-uplo triangle (unchanged from original)
    VerifyUploTriangle(p, cHost.data(), goldenC.data(), cCount);
}
