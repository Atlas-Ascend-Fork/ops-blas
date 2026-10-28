/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "verify.h"
#include "blas_test.h"
#include "csv_loader.h"
#include "fill.h"
#include "csyr_param.h"
#include "csyr_golden.h"
#include "csyr_npu_wrapper.h"
#include "csyr_verify_ext.h"

// ─────────────────────────────────────────────────────────────────────────────
// csyr CSV-driven ST:
//   - four suites drive the 17-column case tables: the L0/L1/L2 layers
//     (18/68/40 cases) and csyr_test.csv (smoke regression, first 50 cases of
//     the task case pool for the shared PreSmoke pool);
//   - precision: spec double gate (matched_ratio >= 0.99 && max_abs_error <=
//     max(1e-2, 32*ULP), |actual-golden| <= atol + rtol*|golden|) AND the frame
//     MERE/MARE engineering gate from the CSV columns; real/imag judged
//     separately; NaN == NaN counts as matched;
//   - golden: streamed per-column (no full golden matrix),
//     association aligned to the kernel column-scalar fold (csyr_golden.h);
//   - G3: untouched-triangle bitwise snapshot assertion (untouched_check column);
//   - G4: n=4864/4865/4863 boundary cases live in the L2 CSV (see README.md).
// ─────────────────────────────────────────────────────────────────────────────

namespace {

using csyr_verify::CsyrToleranceGate;
using csyr_verify::CountUntouchedBitMismatch;
using csyr_verify::CountWholeBitMismatch;

CsyrToleranceGate MakeGate(const CsyrParam& p)
{
    CsyrToleranceGate g;
    if (p.mereThreshold > 0.0) {
        g.mereThreshold = p.mereThreshold;
    }
    if (p.mareMultiplier > 0.0) {
        g.mareMultiplier = p.mareMultiplier;
    }
    return g;
}

std::string CsyrDirFrom(const char* fileName)
{
    std::string p = fileName;
    const size_t slash = p.rfind('/');
    return (slash == std::string::npos) ? std::string("") : p.substr(0, slash + 1);
}

bool IsAlphaZero(float re, float im)
{
    return re == 0.0f && im == 0.0f;
}

void LogUntouchedMismatches(const std::string& caseId, const std::vector<csyr_verify::BitMismatchInfo>& reports)
{
    for (const auto& r : reports) {
        std::cout << "[" << caseId << "] untouched mismatch at (col=" << r.col << ", row=" << r.row << ") re "
                  << std::hex << r.beforeRe << "->" << r.afterRe << std::dec << " ("
                  << csyr_verify::ClassifyBitDiff(r.beforeRe, r.afterRe) << ") im " << std::hex << r.beforeIm << "->"
                  << r.afterIm << std::dec << " (" << csyr_verify::ClassifyBitDiff(r.beforeIm, r.afterIm) << ")"
                  << std::endl;
    }
}

// Streamed golden comparison over the uplo triangle (diagonal included).
// Returns false if either component gate fails.
bool StreamedGoldenCompare(const CsyrParam& p, const aclblasComplex& alphaVal, const aclblasComplex* xPtr,
    const std::vector<aclblasComplex>& aOrig, const std::vector<aclblasComplex>& aActual)
{
    CsyrToleranceGate gateRe = MakeGate(p);
    CsyrToleranceGate gateIm = MakeGate(p);
    std::vector<aclblasComplex> colGold(static_cast<size_t>(p.n));
    for (int j = 0; j < p.n; j++) {
        int iBegin = 0;
        int iEnd = -1;
        ComputeGoldenCsyrColumn(
            p.uplo, p.n, j, &alphaVal, xPtr, p.incx, aOrig.data(), p.lda, colGold.data(), &iBegin, &iEnd);
        for (int i = iBegin; i <= iEnd; i++) {
            const aclblasComplex& act = aActual[static_cast<size_t>(j) * p.lda + i];
            gateRe.Feed(act.real, colGold[static_cast<size_t>(i - iBegin)].real);
            gateIm.Feed(act.imag, colGold[static_cast<size_t>(i - iBegin)].imag);
        }
    }
    gateRe.Report(p.caseName, "real");
    gateIm.Report(p.caseName, "imag");
    return gateRe.Pass() && gateIm.Pass();
}

// ── performance phase: device-resident buffers, wall-clock sampling only ──
// Device buffers are owned by CsyrDeviceBuf (RAII): a fatal ASSERT_EQ return at
// any point below still frees every buffer allocated so far.
void RunPerfPhase(const CsyrParam& p, aclblasHandle_t handle)
{
    const bool smoke = p.isPerfSmoke;
    const int warmupIters = smoke ? 5 : 10;
    const int sampleIters = smoke ? 10 : 60;

    const size_t xLen = static_cast<size_t>(p.PhysicalXLen());
    const size_t aLen = p.PhysicalALen();
    std::vector<aclblasComplex> xHost = MakeCsyrComplexArray(static_cast<int64_t>(xLen), p.xToken, p.randomSeed);
    std::vector<aclblasComplex> aHost =
        MakeCsyrComplexArray(static_cast<int64_t>(aLen), p.aToken, p.randomSeed + 1U);

    aclblasComplex alphaVal;
    alphaVal.real = p.alphaReal;
    alphaVal.imag = p.alphaImag;

    void* dX = nullptr;
    void* dA = nullptr;
    ASSERT_EQ(CsyrAllocCopyToDevice(&dX, xHost.data(), xLen * sizeof(aclblasComplex)), ACLBLAS_STATUS_SUCCESS);
    const CsyrDeviceBuf dXOwner(dX, aclrtFree);
    ASSERT_EQ(CsyrAllocCopyToDevice(&dA, aHost.data(), aLen * sizeof(aclblasComplex)), ACLBLAS_STATUS_SUCCESS);
    const CsyrDeviceBuf dAOwner(dA, aclrtFree);

    aclblasStatus_t ret = ACLBLAS_STATUS_SUCCESS;
    for (int it = 0; it < warmupIters; it++) {
        ret = aclblasCsyr(handle, p.uplo, p.n, &alphaVal, static_cast<const aclblasComplex*>(dX), p.incx,
            static_cast<aclblasComplex*>(dA), p.lda);
        ASSERT_EQ(ret, ACLBLAS_STATUS_SUCCESS);
    }

    const auto tBegin = std::chrono::steady_clock::now();
    for (int it = 0; it < sampleIters; it++) {
        ret = aclblasCsyr(handle, p.uplo, p.n, &alphaVal, static_cast<const aclblasComplex*>(dX), p.incx,
            static_cast<aclblasComplex*>(dA), p.lda);
        ASSERT_EQ(ret, ACLBLAS_STATUS_SUCCESS);
    }
    aclrtSynchronizeDevice();
    const auto tEnd = std::chrono::steady_clock::now();

    const double avgUs = std::chrono::duration<double, std::micro>(tEnd - tBegin).count() / sampleIters;
    std::cout << "[PERF] " << p.caseName << " n=" << p.n << " uplo="
              << (p.uplo == ACLBLAS_UPPER ? "UPPER" : "LOWER") << " incx=" << p.incx << " avg_time=" << avgUs
              << " us (" << sampleIters << " samples, warmup " << warmupIters
              << (smoke ? ", smoke-no-gate)" : ")") << std::endl;
}

// Quick return must not touch A at all (bitwise).
void CheckQuickReturnUntouched(
    const CsyrParam& p, const std::vector<aclblasComplex>& aOrig, const std::vector<aclblasComplex>& aHost)
{
    const size_t mismatch = CountWholeBitMismatch(aOrig, aHost);
    EXPECT_EQ(mismatch, size_t(0)) << "[" << p.caseName << "] quick return modified A";
}

// ── determinism phase: repeat the same call 3 times from the same input ──
void RunDeterminismPhase(const CsyrParam& p, aclblasHandle_t handle, const aclblasComplex& alphaVal,
    const aclblasComplex* alphaPtr, const aclblasComplex* xPtr, size_t xLen,
    const std::vector<aclblasComplex>& aOrig, std::vector<aclblasComplex>& aHost)
{
    std::vector<aclblasComplex> firstRun;
    CsyrToleranceGate pairRe = MakeGate(p);
    CsyrToleranceGate pairIm = MakeGate(p);
    bool allGoldenPass = true;

    for (int run = 0; run < 3; run++) {
        aHost = aOrig; // restore input, bitwise (incl. run 0: the pre-block call already updated aHost)
        aclblasComplex* aPtr = aHost.empty() ? nullptr : aHost.data();
        const aclblasStatus_t st = aclblasCsyr_npu(handle, p.uplo, p.n, alphaPtr, xPtr, p.incx, aPtr, p.lda, xLen);
        ASSERT_EQ(st, ACLBLAS_STATUS_SUCCESS);

        const bool pass = StreamedGoldenCompare(p, alphaVal, xPtr, aOrig, aHost);
        allGoldenPass = allGoldenPass && pass;

        if (run == 0) {
            firstRun = aHost; // reference run for the pairwise comparison
            continue;
        }
        for (int j = 0; j < p.n; j++) {
            const int iBegin = (p.uplo == ACLBLAS_UPPER) ? 0 : j;
            const int iEnd = (p.uplo == ACLBLAS_UPPER) ? j : p.n - 1;
            for (int i = iBegin; i <= iEnd; i++) {
                const aclblasComplex& act = aHost[static_cast<size_t>(j) * p.lda + i];
                const aclblasComplex& ref = firstRun[static_cast<size_t>(j) * p.lda + i];
                pairRe.Feed(act.real, ref.real);
                pairIm.Feed(act.imag, ref.imag);
            }
        }
    }
    pairRe.Report(p.caseName, "real  run-repeatability");
    pairIm.Report(p.caseName, "imag  run-repeatability");
    EXPECT_TRUE(allGoldenPass);
    EXPECT_TRUE(pairRe.Pass());
    EXPECT_TRUE(pairIm.Pass());
}

// ── G3: untouched triangle must be bit-identical to the pre-call snapshot ──
void CheckUntouchedSnapshot(
    const CsyrParam& p, const std::vector<aclblasComplex>& snapshot, const std::vector<aclblasComplex>& aHost)
{
    std::vector<csyr_verify::BitMismatchInfo> reports;
    const size_t mismatch = CountUntouchedBitMismatch(p.uplo, p.n, p.lda, snapshot, aHost, 3, &reports);
    LogUntouchedMismatches(p.caseName, reports);
    EXPECT_EQ(mismatch, size_t(0)) << "[" << p.caseName << "] untouched triangle was written";
}

// Pre-call flags and baseline copies for the accuracy phases (aOrig feeds the
// golden/determinism/quick-return comparisons, snapshot feeds the G3 check).
struct CsyrAccPrep {
    bool quickReturn = false;
    bool needSnapshot = false;
    std::vector<aclblasComplex> aOrig;
    std::vector<aclblasComplex> snapshot;
};

CsyrAccPrep PrepareAccuracyBaselines(
    const CsyrParam& p, const aclblasComplex& alphaVal, const std::vector<aclblasComplex>& aHost)
{
    const bool quickReturn = (p.n == 0) || (!p.alphaNull && IsAlphaZero(alphaVal.real, alphaVal.imag));
    const bool needGolden = (p.expectResult == ACLBLAS_STATUS_SUCCESS) && (p.n > 0) && !p.alphaNull && !quickReturn;
    const bool needSnapshot = p.untouchedCheck && p.expectResult == ACLBLAS_STATUS_SUCCESS;
    const bool needBaseline =
        needGolden || p.isDeterminism || (quickReturn && p.expectResult == ACLBLAS_STATUS_SUCCESS);

    CsyrAccPrep prep;
    prep.quickReturn = quickReturn;
    prep.needSnapshot = needSnapshot;
    if (needBaseline) {
        prep.aOrig = aHost;
    }
    if (needSnapshot) {
        prep.snapshot = aHost;
    }
    return prep;
}

// ── accuracy / negative / snapshot / determinism phases ──
void RunAccuracyPhase(const CsyrParam& p, aclblasHandle_t handle)
{
    const size_t xLen = static_cast<size_t>(p.PhysicalXLen());
    const size_t aLen = p.PhysicalALen();
    std::vector<aclblasComplex> xHost = MakeCsyrComplexArray(static_cast<int64_t>(xLen), p.xToken, p.randomSeed);
    std::vector<aclblasComplex> aHost = MakeCsyrComplexArray(static_cast<int64_t>(aLen), p.aToken, p.randomSeed + 1U);

    const aclblasComplex* xPtr = xHost.empty() ? nullptr : xHost.data();
    aclblasComplex* aPtr = aHost.empty() ? nullptr : aHost.data();
    aclblasComplex alphaVal;
    alphaVal.real = p.alphaReal;
    alphaVal.imag = p.alphaImag;
    const aclblasComplex* alphaPtr = p.alphaNull ? nullptr : &alphaVal;

    const CsyrAccPrep prep = PrepareAccuracyBaselines(p, alphaVal, aHost);

    const aclblasStatus_t ret = aclblasCsyr_npu(handle, p.uplo, p.n, alphaPtr, xPtr, p.incx, aPtr, p.lda, xLen);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult))
        << "[" << p.caseName << "] status mismatch (got " << static_cast<int>(ret) << ")";
    if (ret != p.expectResult) {
        return;
    }
    if (p.expectResult != ACLBLAS_STATUS_SUCCESS) {
        return; // negative case: error code asserted above
    }
    if (p.n == 0) {
        return; // nothing to verify for an empty matrix
    }
    if (prep.quickReturn) {
        CheckQuickReturnUntouched(p, prep.aOrig, aHost);
        return;
    }
    if (p.isDeterminism) {
        RunDeterminismPhase(p, handle, alphaVal, alphaPtr, xPtr, xLen, prep.aOrig, aHost);
        return;
    }
    if (prep.needSnapshot) {
        CheckUntouchedSnapshot(p, prep.snapshot, aHost);
    }

    // ── streamed golden comparison: only the uplo triangle is ──
    //    compared; the untouched triangle is owned by the G3 snapshot above.
    const bool goldenPass = StreamedGoldenCompare(p, alphaVal, xPtr, prep.aOrig, aHost);
    EXPECT_TRUE(goldenPass);
}

} // namespace

class CsyrCsvTestBase : public BlasTest<CsyrParam> {
protected:
    void RunCase();
};

void CsyrCsvTestBase::RunCase()
{
    const CsyrParam& p = GetParam();

    if (p.isPerf) {
        RunPerfPhase(p, CsyrCsvTestBase::handle_);
        return;
    }

    const aclblasHandle_t handle = p.handleValid ? CsyrCsvTestBase::handle_ : nullptr;
    RunAccuracyPhase(p, handle);
}

class CsyrL0Test : public CsyrCsvTestBase {};
class CsyrL1Test : public CsyrCsvTestBase {};
class CsyrL2Test : public CsyrCsvTestBase {};
class CsyrFullTest : public CsyrCsvTestBase {}; // csyr_test.csv smoke regression (first 50 cases)

INSTANTIATE_TEST_SUITE_P(Csyr, CsyrL0Test,
    ::testing::ValuesIn(GetCasesFromCsv<CsyrParam>(CsyrDirFrom(__FILE__) + "csyr_l0_test_cases.csv")),
    PrintCaseInfoString<CsyrParam>);

INSTANTIATE_TEST_SUITE_P(Csyr, CsyrL1Test,
    ::testing::ValuesIn(GetCasesFromCsv<CsyrParam>(CsyrDirFrom(__FILE__) + "csyr_l1_test_cases.csv")),
    PrintCaseInfoString<CsyrParam>);

INSTANTIATE_TEST_SUITE_P(Csyr, CsyrL2Test,
    ::testing::ValuesIn(GetCasesFromCsv<CsyrParam>(CsyrDirFrom(__FILE__) + "csyr_l2_test_cases.csv")),
    PrintCaseInfoString<CsyrParam>);

INSTANTIATE_TEST_SUITE_P(Csyr, CsyrFullTest,
    ::testing::ValuesIn(GetCasesFromCsv<CsyrParam>(CsyrDirFrom(__FILE__) + "csyr_test.csv")),
    PrintCaseInfoString<CsyrParam>);

TEST_P(CsyrL0Test, CsvDriven)
{
    RunCase();
}

TEST_P(CsyrL1Test, CsvDriven)
{
    RunCase();
}

TEST_P(CsyrL2Test, CsvDriven)
{
    RunCase();
}

TEST_P(CsyrFullTest, CsvDriven)
{
    RunCase();
}
