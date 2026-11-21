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
#include <climits>

#include "verify.h"
#include "fill.h"
#include "blas_test.h"
#include "csv_loader.h"
#include "ssymm_param.h"
#include "ssymm_golden.h"
#include "ssymm_npu_wrapper.h"

class SsymmArch22Test : public BlasTest<SsymmParam> {};

// null handle: TEST_F (not CSV-driven), per blas-ST-develop convention
TEST_F(SsymmArch22Test, NullHandle)
{
    float alpha = 1.0f;
    float beta  = 1.0f;
    // Minimal dummy host buffers to satisfy non-null pointer check in the
    // direct kernel path; handle=nullptr is caught before any pointer deref.
    std::vector<float> a(16, 0.0f);
    std::vector<float> b(16, 0.0f);
    std::vector<float> c(16, 0.0f);
    aclblasStatus_t ret = aclblasSsymm_npu(
        nullptr, ACLBLAS_SIDE_LEFT, ACLBLAS_LOWER, 4, 4,
        &alpha, a.data(), 4, b.data(), 4, &beta, c.data(), 4);
    EXPECT_EQ(ret, ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
}

INSTANTIATE_TEST_SUITE_P(
    Ssymm, SsymmArch22Test,
    ::testing::ValuesIn(GetCasesFromCsv<SsymmParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<SsymmParam>);

// ---- Helper: generate host test data ----
struct SsymmHostData {
    std::vector<float> aHost;
    std::vector<float> bHost;
    std::vector<float> cHost;
    std::vector<float> cResult;
    const float* alphaPtr = nullptr;
    const float* betaPtr  = nullptr;
    const float* aPtr     = nullptr;
    const float* bPtr     = nullptr;
    float*       cPtr     = nullptr;
};

static SsymmHostData BuildSsymmHostData(const SsymmParam& p)
{
    constexpr int64_t kMaxSafeTestElems = 64LL * 1024 * 1024; // 256 MB per matrix
    auto safeElemCount = [](int64_t rows, int64_t cols) -> int64_t {
        if (rows <= 0 || cols <= 0) return 0;
        if (rows != 0 && cols > kMaxSafeTestElems / rows) return 0;
        return rows * cols;
    };

    const int64_t aDim = (p.side == ACLBLAS_SIDE_LEFT) ? p.m : p.n;
    const int64_t aElemCount = safeElemCount(aDim, p.lda);
    const int64_t bElemCount = safeElemCount(p.m, p.ldb);
    const int64_t cElemCount = safeElemCount(p.m, p.ldc);

    SsymmHostData d;
    if (aElemCount > 0) {
        d.aHost = makeBlasArray(aElemCount, "RANDOM:-10:10", p.randomSeed);
    }
    if (bElemCount > 0) {
        d.bHost = makeBlasArray(bElemCount, "RANDOM:-10:10", p.randomSeed + 1);
    }
    if (cElemCount > 0) {
        d.cHost   = makeBlasArray(cElemCount, "RANDOM:-10:10", p.randomSeed + 2);
        d.cResult = d.cHost;
    }

    d.alphaPtr = p.nullAlpha ? nullptr : &p.alpha;
    d.betaPtr  = p.nullBeta  ? nullptr : &p.beta;
    d.aPtr     = d.aHost.empty() ? nullptr : d.aHost.data();
    d.bPtr     = d.bHost.empty() ? nullptr : d.bHost.data();
    d.cPtr     = d.cResult.empty() ? nullptr : d.cResult.data();
    return d;
}

// ---- Helper: check for int overflow before API call ----
static bool SsymmParamsOverflowInt(const SsymmParam& p)
{
    return p.m > INT32_MAX || p.n > INT32_MAX || p.lda > INT32_MAX ||
           p.ldb > INT32_MAX || p.ldc > INT32_MAX;
}

TEST_P(SsymmArch22Test, CsvDriven)
{
    const auto& p = GetParam();
    SsymmHostData d = BuildSsymmHostData(p);

    // ---- Step 2: execute on NPU ----
    // Reject values that overflow int before calling the API (which takes int).
    if (SsymmParamsOverflowInt(p)) {
        EXPECT_EQ(static_cast<int>(p.expectResult), static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE));
        return;
    }
    aclblasStatus_t ret = aclblasSsymm_npu(
        SsymmArch22Test::handle_,
        p.side, p.uplo, p.m, p.n,
        d.alphaPtr, d.aPtr, p.lda,
        d.bPtr,     p.ldb,
        d.betaPtr,  d.cPtr, p.ldc);

    // ---- Step 3: verify error code; return early for non-SUCCESS cases ----
    if (p.expectResult != ACLBLAS_STATUS_SUCCESS) {
        EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
        return;
    }
    ASSERT_EQ(ret, ACLBLAS_STATUS_SUCCESS);

    // ---- Step 4: compute golden on CPU ----
    std::vector<float> cGolden = d.cHost;
    float* goldPtr = cGolden.empty() ? nullptr : cGolden.data();

    aclblasSsymm_cpu(
        SsymmArch22Test::handle_,
        p.side, p.uplo, p.m, p.n,
        d.alphaPtr, d.aPtr, p.lda,
        d.bPtr,     p.ldb,
        d.betaPtr,  goldPtr, p.ldc);

    // ---- Step 5: precision comparison (MERE_MARE, community standard) ----
    const size_t cCount = static_cast<size_t>(p.m) * static_cast<size_t>(p.ldc);
    if (cCount == 0) return;

    VerifyConfig cfg;
    cfg.mode           = PrecisionMode::MERE_MARE;
    cfg.mereThreshold  = (p.mereThreshold > 0.0) ? p.mereThreshold : (1.0 / 8192.0);
    cfg.mareMultiplier = (p.mareMultiplier > 0.0) ? p.mareMultiplier : 10.0;
    EXPECT_TRUE(Verifier::verifyVector(d.cPtr, goldPtr, cCount, 1, cfg, p.caseName));
}
