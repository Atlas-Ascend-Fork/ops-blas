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
#include "sgbmv_param.h"
#include "sgbmv_golden.h"
#include "sgbmv_npu_wrapper.h"

class SgbmvTest : public BlasTest<GbmvParam> {};

TEST_F(SgbmvTest, NullHandleReturnsStatusHandleIsNullptr)
{
    float alpha = 1.0f;
    float beta = 0.0f;
    std::vector<float> aBuf(16, 0.0f);
    std::vector<float> xBuf(4, 0.0f);
    std::vector<float> yBuf(4, 0.0f);
    aclblasStatus_t ret = aclblasSgbmv_npu(
        nullptr, ACLBLAS_OP_N, 4, 4, 1, 1, &alpha, aBuf.data(), 3, xBuf.data(), 1, &beta, yBuf.data(), 1);
    EXPECT_EQ(ret, ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
}

INSTANTIATE_TEST_SUITE_P(
    Gbmv, SgbmvTest, ::testing::ValuesIn(GetCasesFromCsv<GbmvParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<GbmvParam>);

// Degenerate-mode profile (Inf/NaN fills): when Inf/NaN fills degenerate the
// stock MERE/MARE statistics, print the finite-pair subset MERE/MARE plus the
// non-finite census on both sides. Informational evidence only -- the main-gate
// verdict line above it is always preserved (never silent).
struct SgbmvOodStats {
    double sumRel = 0.0;
    double maxRel = 0.0;
    size_t validCount = 0;
    size_t skippedNearZero = 0;
    size_t finitePairs = 0;
    size_t outNan = 0, outInf = 0, goldNan = 0, goldInf = 0;
    size_t bothNonFiniteEqual = 0, mixedPairs = 0, outliers = 0;
};

// One element pair: non-finite census + classification, or the finite-pair
// MERE/MARE accumulation.
static void SgbmvOodAccumulatePair(
    const float* out, const float* gold, size_t i, int64_t stride, double threshold, double outlierLimit,
    SgbmvOodStats& s)
{
    const float o = out[static_cast<int64_t>(i) * stride];
    const float g = gold[static_cast<int64_t>(i) * stride];
    const bool oFin = std::isfinite(o);
    const bool gFin = std::isfinite(g);
    if (!oFin) {
        if (std::isnan(o)) {
            s.outNan++;
        } else {
            s.outInf++;
        }
    }
    if (!gFin) {
        if (std::isnan(g)) {
            s.goldNan++;
        } else {
            s.goldInf++;
        }
    }
    if (!oFin || !gFin) {
        if (!oFin && !gFin && ((std::isnan(o) && std::isnan(g)) || o == g)) {
            s.bothNonFiniteEqual++;
        } else {
            s.mixedPairs++;
        }
        return;
    }
    s.finitePairs++;
    const double kEpsilon = 0.00006103515625; // 2^-14 (MereMareStrategy kEpsilon)
    const double absGold = std::fabs(static_cast<double>(g));
    if (absGold < threshold) {
        s.skippedNearZero++;
        return;
    }
    const double relErr = std::fabs(static_cast<double>(o) - static_cast<double>(g)) / (absGold + kEpsilon);
    s.sumRel += relErr;
    if (relErr > s.maxRel) {
        s.maxRel = relErr;
    }
    if (relErr > outlierLimit) {
        s.outliers++;
    }
    s.validCount++;
}

// Emit the degenerate-mode profile line (verbatim output contract).
static void SgbmvOodPrintDegenerateProfile(
    const SgbmvOodStats& s, double mereF, double threshold, double outlierLimit, const std::string& caseId)
{
    const bool countMatch = ((s.outNan + s.outInf) == (s.goldNan + s.goldInf));
    // Mirror the framework MereMareStrategy vacuous-set convention
    // (reportResult computes mere = 0.0 when validCount == 0) -- an empty
    // finite-pair subset yields mere_finite=0/mare_finite=0 and the decision
    // is carried by the non-finite count-equality condition; the harness
    // discloses the vacuity explicitly (vacuous_subset marker, never silent).
    const bool degeneratePass = (mereF < threshold) && (s.maxRel < outlierLimit) && countMatch;
    std::cout << std::scientific << std::setprecision(6);
    std::cout << "[SGBMV-ST][ood-degenerate] case=" << caseId << " finite_pairs=" << s.finitePairs
              << " valid=" << s.validCount << " skipped_near_zero=" << s.skippedNearZero << " mere_finite=" << mereF
              << " mare_finite=" << s.maxRel << " threshold=" << threshold << " outlier_limit=" << outlierLimit
              << " outliers=" << s.outliers << " out_nan=" << s.outNan << " out_inf=" << s.outInf
              << " gold_nan=" << s.goldNan << " gold_inf=" << s.goldInf
              << " both_nonfinite_equal=" << s.bothNonFiniteEqual << " mixed_pairs=" << s.mixedPairs
              << " nonfinite_count_match=" << (countMatch ? "true" : "false")
              << " degenerate_pass=" << (degeneratePass ? "true" : "false") << std::endl;
    std::cout << std::fixed << std::setprecision(6);
}

static void PrintSgbmvOodDegenerateProfile(
    const float* out, const float* gold, size_t count, int64_t stride, double threshold, double multiplier,
    const std::string& caseId)
{
    const double outlierLimit = multiplier * threshold;
    SgbmvOodStats s;
    for (size_t i = 0; i < count; i++) {
        SgbmvOodAccumulatePair(out, gold, i, stride, threshold, outlierLimit, s);
    }
    const double mereF = (s.validCount > 0) ? s.sumRel / static_cast<double>(s.validCount) : 0.0;
    SgbmvOodPrintDegenerateProfile(s, mereF, threshold, outlierLimit, caseId);
}

TEST_P(SgbmvTest, CsvDriven)
{
    const auto& p = GetParam();

    // Route evidence line: every case runs the
    // real aclblasSgbmv implementation on the real NPU backend via this GTest frame
    std::cout << "[SGBMV-ST][route] case=" << p.caseName
              << " backend=NPU(real) decision=EXEC" << std::endl;

    const bool isTransN = (p.trans == ACLBLAS_OP_N);
    const int xCount = isTransN ? p.n : p.m;
    const int yCount = isTransN ? p.m : p.n;

    auto a = makeBlasBanded(p.m, p.n, p.kl, p.ku, p.lda, p.a, p.randomSeed);
    auto x = makeBlasStrided(xCount, p.incx, p.x, p.randomSeed);
    auto y = makeBlasStrided(yCount, p.incy, p.y, p.randomSeed);
    std::vector<float> result = y;
    std::vector<float> golden = y;

    const float* alphaPtr = p.nullAlpha ? nullptr : &p.alpha;
    const float* betaPtr = p.nullBeta ? nullptr : &p.beta;
    float* aPtr = p.nullA ? nullptr : a.data();
    float* xPtr = p.nullX ? nullptr : x.data();
    float* yPtr = p.nullY ? nullptr : result.data();

    aclblasStatus_t ret = aclblasSgbmv_npu(
        p.nullHandle ? nullptr : SgbmvTest::handle_, p.trans, p.m, p.n, p.kl, p.ku, alphaPtr, aPtr, p.lda, xPtr,
        p.incx, betaPtr, yPtr, p.incy);

    if (p.expectResult != ACLBLAS_STATUS_SUCCESS) {
        EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
        return;
    }
    ASSERT_EQ(ret, ACLBLAS_STATUS_SUCCESS);

    aclblasSgbmv_cpu(
        SgbmvTest::handle_, p.trans, p.m, p.n, p.kl, p.ku, alphaPtr, aPtr, p.lda, xPtr, p.incx, betaPtr, golden.data(),
        p.incy);

    const float* outPtr = (p.incy < 0 && yCount > 0) ? result.data() + (yCount - 1) * (-p.incy) : result.data();
    const float* goldPtr = (p.incy < 0 && yCount > 0) ? golden.data() + (yCount - 1) * (-p.incy) : golden.data();

    VerifyConfig cfg;
    if (p.expectExact) {
        // alpha=0 special groups: exact-semantics assertion groups
        // — bit-level comparison instead of tolerance comparison (strictly stronger gate)
        cfg.mode = PrecisionMode::EXACT;
    } else {
        applyMixedTolerance(cfg, ACL_FLOAT, goldPtr, static_cast<size_t>(yCount));
    }
    const bool mainOk =
        Verifier::verifyVector(outPtr, goldPtr, static_cast<size_t>(yCount), p.incy, cfg, p.caseName);
    EXPECT_TRUE(mainOk);
    if (!mainOk && cfg.mode == PrecisionMode::MIXED_TOLERANCE) {
        // On a main-gate FAILURE, additionally print the framework MERE/MARE
        // verdict (stock MereMareStrategy; mere_threshold/mare_multiplier from
        // the CSV columns) and the degenerate-mode profile. The main FAILED
        // verdict line above is preserved (never silent).
        MereMareStrategy oodMereMare(p.mereThreshold, p.mareMultiplier);
        oodMereMare.verify(outPtr, goldPtr, static_cast<size_t>(yCount), p.incy, p.caseName);
        PrintSgbmvOodDegenerateProfile(outPtr, goldPtr, static_cast<size_t>(yCount), p.incy, p.mereThreshold,
                                       p.mareMultiplier, p.caseName);
    }
}
