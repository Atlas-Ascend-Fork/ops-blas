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
#include "blas_test.h"
#include "csv_loader.h"
#include "device.h"
#include "trsv_param.h"
#include "trsv_golden.h"
#include "strsv_npu_wrapper.h"

class TrsvArch22Test : public BlasTest<TrsvParam> {};

TEST_F(TrsvArch22Test, NullHandle)
{
    // aclblasStrsv host checks A==nullptr || x==nullptr before accessing handle,
    // so null handle with null data returns ACLBLAS_STATUS_INVALID_VALUE.
    aclblasStatus_t ret =
        aclblasStrsv_npu(nullptr, ACLBLAS_LOWER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT, 5, nullptr, 5, nullptr, 1);
    EXPECT_EQ(ret, ACLBLAS_STATUS_INVALID_VALUE);
}

INSTANTIATE_TEST_SUITE_P(
    Trsv, TrsvArch22Test, ::testing::ValuesIn(GetCasesFromCsv<TrsvParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<TrsvParam>);

TEST_P(TrsvArch22Test, CsvDriven)
{
    const auto& p = GetParam();

    // Ensure n does not exceed lda for memory allocation
    const int64_t allocN = std::max(int64_t(1), std::abs(p.n));
    const int64_t allocLda = std::max(allocN, p.lda);

    // ── Step 1: Generate A matrix ──
    auto aHost = makeBlasArray(allocLda * allocN, p.A, p.randomSeed);
    // Strengthen diagonal to avoid near-singular triangular system
    if (!aHost.empty()) {
        for (int64_t i = 0; i < allocN; i++) {
            float& diag = aHost[i + i * allocLda];
            diag += (diag >= 0.0f ? 5.0f : -5.0f);
        }
    }

    // ── Step 2: Generate x vector (RHS) with stride ──
    const int ni = static_cast<int>(p.n);
    auto xHost = makeBlasStrided(ni, static_cast<int>(p.incx), p.x, p.randomSeed);

    // ── Step 3: Save golden copy before NPU modifies xHost in-place ──
    std::vector<float> golden = xHost;

    const bool needDevA = !aHost.empty() && p.n > 0;
    const bool needDevX = !xHost.empty() && p.n > 0;
    const size_t aBytes = needDevA ? aHost.size() * sizeof(float) : sizeof(float);
    const size_t xBytes = needDevX ? xHost.size() * sizeof(float) : sizeof(float);
    DeviceBuffer aDev(aBytes);
    DeviceBuffer xDev(xBytes);
    if (needDevA) {
        aDev.copyFromHost(aHost.data(), aBytes);
    }
    if (needDevX) {
        xDev.copyFromHost(xHost.data(), xBytes);
    }

    const float* aPtr = needDevA ? static_cast<const float*>(aDev.ptr()) : nullptr;
    float* xPtr = needDevX ? static_cast<float*>(xDev.ptr()) : nullptr;

    // ── Step 4: Execute NPU ──
    aclblasStatus_t ret =
        aclblasStrsv_npu(TrsvArch22Test::handle_, p.uplo, p.trans, p.diag, p.n, aPtr, p.lda, xPtr, p.incx);

    // ── Step 5: Verify ──
    if (p.expectResult != ACLBLAS_STATUS_SUCCESS) {
        EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
        return;
    }
    ASSERT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_SUCCESS))
        << "Unexpected NPU error code: " << static_cast<int>(ret);

    xDev.copyToHost(xHost.data(), xBytes);

    aclblasStrsv_cpu(TrsvArch22Test::handle_, p.uplo, p.trans, p.diag, p.n, aHost.data(), p.lda, golden.data(), p.incx);

    const int absIncx = std::abs(static_cast<int>(p.incx));
    const float* outPtr = (p.incx < 0 && ni > 0) ? xHost.data() + (ni - 1) * absIncx : xHost.data();
    const float* goldPtr = (p.incx < 0 && ni > 0) ? golden.data() + (ni - 1) * absIncx : golden.data();

    VerifyConfig cfg;
    cfg.mode = PrecisionMode::MERE_MARE;
    cfg.mereThreshold = p.mereThreshold;
    cfg.mareMultiplier = p.mareMultiplier;
    EXPECT_TRUE(Verifier::verifyVector(outPtr, goldPtr, static_cast<size_t>(ni), p.incx, cfg, p.caseName));
}
