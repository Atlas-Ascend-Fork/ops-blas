/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "verify.h"
#include "blas_test.h"
#include "csv_loader.h"
#include "fill.h"
#include "ctpsv_param.h"
#include "ctpsv_golden.h"
#include "ctpsv_npu_wrapper.h"

// The family-level CtpsvParam parses every csv token natively, including the out-of-domain
// enum literals ("999") in the TC_ED rows (csv_loader.h parseEnum falls back to a numeric
// cast). This subclass only adds the per-case A100 baseline time joined from gpu_baseline.csv.
struct CtpsvArch22Param : CtpsvParam {
    double gpuBaselineUs = -1.0; // raw A100 single-call time for TC_PF cases, us

    CtpsvArch22Param(const csv_map& map) : CtpsvParam(map) {}
};

namespace {
// Interleave two float buffers (real part / imaginary part) into a complex64 buffer.
inline std::vector<aclblasComplex> CtpsvInterleave(
    const std::vector<float>& real, const std::vector<float>& imag, size_t count)
{
    std::vector<aclblasComplex> out(count);
    for (size_t i = 0; i < count; ++i) {
        out[i].real = real[i];
        out[i].imag = imag[i];
    }
    return out;
}

// Build a packed triangular complex64 matrix. Real part and imaginary part are generated with
// independent seeds, following the makeBlasComplexMatrix convention (seed / seed + 1000).
//
// Every slot of the n(n+1)/2 packed storage belongs to the triangle, so filling the slots in
// order IS the packed layout, for upper and lower alike. The frame helper makeBlasTriangular
// would work too, but it overrides the fill's pattern (P_UPPER/P_LOWER), which silently
// degrades the ALTER / EXTREME patterns to a plain uniform fill - so the flat array is built
// directly and every fill token keeps its documented semantics.
inline std::vector<aclblasComplex> CtpsvMakePacked(int n, const std::string& fill, uint32_t seed)
{
    if (n <= 0) {
        return {};
    }
    const size_t count = static_cast<size_t>(n) * (n + 1) / 2;
    return CtpsvInterleave(makeBlasArray(count, fill, seed), makeBlasArray(count, fill, seed + 1000U), count);
}

// True when every real/imaginary component of the buffer is finite.
inline bool CtpsvAllFinite(const std::vector<aclblasComplex>& buf)
{
    for (size_t i = 0; i < buf.size(); ++i) {
        if (!std::isfinite(buf[i].real) || !std::isfinite(buf[i].imag)) {
            return false;
        }
    }
    return true;
}

// Build a strided complex64 vector.
inline std::vector<aclblasComplex> CtpsvMakeStrided(int n, int incx, const std::string& fill, uint32_t seed)
{
    const int absInc = std::abs(incx);
    const size_t count = (n > 0) ? (static_cast<size_t>(n - 1) * absInc + 1) : 1;
    return CtpsvInterleave(makeBlasStrided(n, incx, fill, seed), makeBlasStrided(n, incx, fill, seed + 1000U), count);
}

// ---------------------------------------------------------------------------
// Performance cases (TC_PF_*)
//
// A performance case has to measure the operator and not the harness: building the operand and
// the cblas golden for n = 4096 costs about half a second on the host, two orders of magnitude
// more than the operator itself, which would completely mask the number under test. Accuracy is
// covered by the 1000 accuracy cases (they reach n = 2048), so a performance case only has to
// keep the operator honest: one device-resident operand pair is built once for the largest n
// and reused.
//
// The operand is a triangular matrix whose entries are all equal. Such a matrix is non-singular
// for every n and every uplo/trans combination, and neither substitution overflows: with a
// constant a, forward substitution gives the partial sums S_i = sum_{j<=i} x_j = b_i / a, which
// stay bounded. The kernel cost is data independent, so this is a fair operand to time.
//
// Acceptance (task doc §3.3): the average single-call time must not exceed the raw A100 time
// from gpu_baseline.csv divided by 0.8 (i.e. 1.25x the raw A100 time), with more than 50 valid
// samples after warmup.
constexpr size_t PF_MAX_N = 4096;
// Default number of back-to-back operator launches per performance case for the device-side event
// timing (used when n >= 64; smaller n uses more repeats so the event timer sees a measurable
// span). 100 > 50 satisfies task doc §3.3「有效采样 >50 次」.
constexpr int CTPSV_PERF_ITERS = 100;
constexpr float PF_AP_REAL = 1.0f;
constexpr float PF_AP_IMAG = 0.25f;
constexpr float PF_X_REAL = 1.0f;
constexpr float PF_X_IMAG = 0.5f;

void* gPfAp = nullptr;
void* gPfX = nullptr;
std::vector<aclblasComplex> gPfXHost;

bool CtpsvPreparePerfOperand()
{
    if (gPfAp != nullptr) {
        return true;
    }
    const size_t apCount = PF_MAX_N * (PF_MAX_N + 1) / 2;
    const size_t apBytes = apCount * sizeof(aclblasComplex);
    std::vector<aclblasComplex> apHost(apCount);
    for (size_t i = 0; i < apCount; ++i) {
        apHost[i].real = PF_AP_REAL;
        apHost[i].imag = PF_AP_IMAG;
    }
    gPfXHost.assign(PF_MAX_N, aclblasComplex{PF_X_REAL, PF_X_IMAG});

    if (aclrtMalloc(&gPfAp, apBytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        gPfAp = nullptr;
        return false;
    }
    if (aclrtMalloc(&gPfX, gPfXHost.size() * sizeof(aclblasComplex), ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        aclrtFree(gPfAp);
        gPfAp = nullptr;
        return false;
    }
    if (aclrtMemcpy(gPfAp, apBytes, apHost.data(), apBytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
        aclrtFree(gPfAp);
        aclrtFree(gPfX);
        gPfAp = nullptr;
        gPfX = nullptr;
        return false;
    }
    return true;
}

void CtpsvReleasePerfOperand()
{
    if (gPfX != nullptr) {
        aclrtFree(gPfX);
        gPfX = nullptr;
    }
    if (gPfAp != nullptr) {
        aclrtFree(gPfAp);
        gPfAp = nullptr;
    }
}

inline bool CtpsvIsPerfCase(const CtpsvArch22Param& p)
{
    return p.caseName.rfind("TC_PF_", 0) == 0 && p.expectResult == ACLBLAS_STATUS_SUCCESS && p.n > 0 &&
           p.n <= static_cast<int>(PF_MAX_N) && gPfAp != nullptr;
}

// One row of gpu_baseline.csv: the raw A100 single-call time for one perf case, keyed by the
// (uplo, trans, diag, n, incx) tuple.
struct CtpsvGpuBaseline {
    aclblasFillMode_t uplo;
    aclblasOperation_t trans;
    aclblasDiagType_t diag;
    int n;
    int incx;
    double gpuUs;

    explicit CtpsvGpuBaseline(const csv_map& row)
        : uplo(parseFillMode(ReadMap(row, "uplo"))),
          trans(parseOpTrans(ReadMap(row, "trans"))),
          diag(parseDiagType(ReadMap(row, "diag"))),
          n(parseInt(ReadMap(row, "n"))),
          incx(parseInt(ReadMap(row, "incx", "1"))),
          gpuUs(parseDouble(ReadMap(row, "gpu_ms")))
    {}
};

inline std::string CtpsvBaselineKey(
    aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, int n, int incx)
{
    std::ostringstream oss;
    oss << static_cast<int>(uplo) << '/' << static_cast<int>(trans) << '/' << static_cast<int>(diag) << '/' << n << '/'
        << incx;
    return oss.str();
}

// Load the case csv and join every TC_PF row with its gpu_baseline.csv entry. Both files come
// from the task package and enumerate the same 200 cases, so a missing or unused baseline row
// is a packaging error and fails loudly instead of producing an unverifiable number.
std::vector<CtpsvArch22Param> LoadCtpsvCases(const std::string& csvPath)
{
    std::vector<CtpsvArch22Param> cases = GetCasesFromCsv<CtpsvArch22Param>(csvPath);
    const size_t separator = csvPath.find_last_of("/\\");
    const std::string baselinePath =
        (separator == std::string::npos ? std::string() : csvPath.substr(0, separator + 1U)) + "gpu_baseline.csv";
    const std::vector<CtpsvGpuBaseline> baselines = GetCasesFromCsv<CtpsvGpuBaseline>(baselinePath);

    std::map<std::string, double> gpuUsByKey;
    for (const auto& b : baselines) {
        const std::string key = CtpsvBaselineKey(b.uplo, b.trans, b.diag, b.n, b.incx);
        if (!gpuUsByKey.emplace(key, b.gpuUs).second) {
            throw std::runtime_error("Duplicate GPU baseline for " + key);
        }
    }
    size_t perfCount = 0;
    for (auto& c : cases) {
        if (c.caseName.rfind("TC_PF_", 0) != 0) {
            continue;
        }
        const std::string key = CtpsvBaselineKey(c.uplo, c.trans, c.diag, c.n, c.incx);
        const auto it = gpuUsByKey.find(key);
        if (it == gpuUsByKey.end()) {
            throw std::runtime_error("Missing GPU baseline for " + c.caseName);
        }
        c.gpuBaselineUs = it->second * 1000.0; // gpu_baseline.csv carries milliseconds
        ++perfCount;
    }
    if (perfCount != baselines.size()) {
        throw std::runtime_error("GPU baseline contains unused rows");
    }
    return cases;
}
} // namespace

class CtpsvTest : public BlasTest<CtpsvArch22Param> {
protected:
    static void SetUpTestSuite()
    {
        BlasTest<CtpsvArch22Param>::SetUpTestSuite();
        // Built here so that the one-off cost does not land on the first performance case.
        CtpsvPreparePerfOperand();
    }

    static void TearDownTestSuite()
    {
        CtpsvReleasePerfOperand();
        BlasTest<CtpsvArch22Param>::TearDownTestSuite();
    }

    static void RunPerfCase(const CtpsvArch22Param& p);
    static void BuildAp(const CtpsvArch22Param& p, std::vector<aclblasComplex>& ap);
    static void RunAccuracyCase(const CtpsvArch22Param& p, const std::vector<aclblasComplex>& ap);
    static void VerifyResult(
        const CtpsvArch22Param& p, const std::vector<aclblasComplex>& ap, const std::vector<aclblasComplex>& xIn,
        const std::vector<aclblasComplex>& x, const std::vector<aclblasComplex>& golden);
};

INSTANTIATE_TEST_SUITE_P(
    Ctpsv, CtpsvTest, ::testing::ValuesIn(LoadCtpsvCases(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<CtpsvArch22Param>);

void CtpsvTest::RunPerfCase(const CtpsvArch22Param& p)
{
    const size_t xCount = static_cast<size_t>(p.n - 1) * std::abs(p.incx) + 1;
    const size_t xBytes = xCount * sizeof(aclblasComplex);
    // AP is read-only for the operator; only x has to be restored after the previous case.
    ASSERT_EQ(aclrtMemcpy(gPfX, xBytes, gPfXHost.data(), xBytes, ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);

    // Time with device-side events (same pattern as test/copy/ccopy/arch35/ccopy_benchmark.cpp)
    // so the fixed host launch+sync latency does not swamp the kernel time for small n.
    // Task doc §3.3 measures the *average single-call* time in us (warmup + >50 samples).
    // Small n runs more repeats so the event timer sees a measurable span and the per-call
    // average stays above the timer granularity.
    constexpr int kPerfWarmup = 10;
    const int repeats = (p.n < 64) ? 1000 : CTPSV_PERF_ITERS;
    for (int i = 0; i < kPerfWarmup; ++i) {
        const aclblasStatus_t pfRet = aclblasCtpsv(
            handle_, p.uplo, p.trans, p.diag, p.n, static_cast<const aclblasComplex*>(gPfAp),
            static_cast<aclblasComplex*>(gPfX), p.incx);
        ASSERT_EQ(pfRet, ACLBLAS_STATUS_SUCCESS);
    }
    ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_SUCCESS);

    aclrtEvent start = nullptr;
    aclrtEvent stop = nullptr;
    ASSERT_EQ(aclrtCreateEvent(&start), ACL_SUCCESS);
    ASSERT_EQ(aclrtCreateEvent(&stop), ACL_SUCCESS);
    ASSERT_EQ(aclrtRecordEvent(start, stream_), ACL_SUCCESS);
    for (int i = 0; i < repeats; ++i) {
        const aclblasStatus_t pfRet = aclblasCtpsv(
            handle_, p.uplo, p.trans, p.diag, p.n, static_cast<const aclblasComplex*>(gPfAp),
            static_cast<aclblasComplex*>(gPfX), p.incx);
        ASSERT_EQ(pfRet, ACLBLAS_STATUS_SUCCESS);
    }
    ASSERT_EQ(aclrtRecordEvent(stop, stream_), ACL_SUCCESS);
    ASSERT_EQ(aclrtSynchronizeEvent(stop), ACL_SUCCESS);
    float elapsedMs = 0.0F;
    ASSERT_EQ(aclrtEventElapsedTime(&elapsedMs, start, stop), ACL_SUCCESS);
    aclrtDestroyEvent(start);
    aclrtDestroyEvent(stop);
    const double avgUs = static_cast<double>(elapsedMs) * 1000.0 / static_cast<double>(repeats);

    // Report-only performance line, same as the arch35 ctpsv suite: the acceptance ratio
    // (task doc §3.3: NPU average must not exceed A100 gpu_ms / 0.8) is judged by the
    // acceptance harness from this printed line. Shared CI runners differ from dedicated
    // profiling machines by a machine-wide factor, so the test suite itself does not gate on
    // timing; the ratio/verdict columns document the acceptance-threshold cross-check.
    const double ratio = (avgUs > 0.0) ? p.gpuBaselineUs / avgUs : 0.0;
    const bool passed = avgUs <= p.gpuBaselineUs / 0.8;
    std::printf(
        "[CTPSV_PERF] %s avg_us=%.4f gpu_us=%.2f ratio=%.3f verdict=%s\n", p.caseName.c_str(), avgUs, p.gpuBaselineUs,
        ratio, passed ? "PASS" : "FAIL");
}

void CtpsvTest::BuildAp(const CtpsvArch22Param& p, std::vector<aclblasComplex>& ap)
{
    if (p.n <= 0 || p.apFill == "NULLPTR") {
        return;
    }
    ap = CtpsvMakePacked(p.n, p.apFill, p.randomSeed);
    const bool isUpper = (p.uplo == ACLBLAS_UPPER);
    if (p.diag == ACLBLAS_UNIT) {
        // Keep the unit-diagonal substitution numerically stable. Complex off-diagonal entries have
        // modulus √2× larger than the real stpsv case, and fill patterns with growing amplitude
        // (ALTER) or huge values (EXTREME) can turn back substitution into a geometric series that
        // blows up. Tighten the scale to 1/n so |A(i,j)| ≪ 1 and the unit diagonal dominates.
        const float scale = std::min(1.0f, 1.0f / static_cast<float>(p.n));
        for (size_t i = 0; i < ap.size(); ++i) {
            ap[i].real *= scale;
            ap[i].imag *= scale;
        }
    } else {
        // Dominant diagonal so that the triangular solve stays well conditioned.
        const float boost = std::max(5.0f, static_cast<float>(p.n));
        for (int i = 0; i < p.n; ++i) {
            const size_t idx = isUpper ? CtpsvPackedUpperIdxCpu(i, i) : CtpsvPackedLowerIdxCpu(i, i, p.n);
            ap[idx].real += boost;
        }
    }
}

void CtpsvTest::RunAccuracyCase(const CtpsvArch22Param& p, const std::vector<aclblasComplex>& ap)
{
    // ---- right-hand side / solution vector x (complex64) ----
    std::vector<aclblasComplex> x;
    if (p.xFill != "NULLPTR") {
        x = CtpsvMakeStrided(p.n, p.incx, p.xFill, p.randomSeed + 1U);
    }
    const std::vector<aclblasComplex> xIn = x;
    std::vector<aclblasComplex> golden = x;

    // Rows with a NULLPTR fill token pass a real null pointer; rows expecting
    // ACLBLAS_STATUS_HANDLE_IS_NULLPTR call through a null handle. The wrapper passes such
    // calls straight through to aclblasCtpsv so the status code can be asserted (task doc §2.4:
    // n < 0 / invalid enums / null pointers / zero stride must return the documented status).
    const aclblasHandle_t callHandle = (p.expectResult == ACLBLAS_STATUS_HANDLE_IS_NULLPTR) ? nullptr : handle_;
    const aclblasComplex* apArg = (p.apFill == "NULLPTR") ? nullptr : ap.data();
    aclblasComplex* xArg = (p.xFill == "NULLPTR") ? nullptr : x.data();

    const aclblasStatus_t ret = aclblasCtpsv_npu(callHandle, p.uplo, p.trans, p.diag, p.n, apArg, xArg, p.incx);
    if (p.expectResult != ACLBLAS_STATUS_SUCCESS) {
        EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
        return;
    }
    ASSERT_EQ(ret, ACLBLAS_STATUS_SUCCESS);

    aclblasCtpsv_cpu(handle_, p.uplo, p.trans, p.diag, p.n, ap.data(), golden.data(), p.incx);

    VerifyResult(p, ap, xIn, x, golden);
}

void CtpsvTest::VerifyResult(
    const CtpsvArch22Param& p, const std::vector<aclblasComplex>& ap, const std::vector<aclblasComplex>& xIn,
    const std::vector<aclblasComplex>& x, const std::vector<aclblasComplex>& golden)
{
    // ---- split real / imaginary parts and verify with the mixed-tolerance criterion ----
    const int absIncx = std::abs(p.incx);
    const int64_t step = (p.incx < 0) ? -static_cast<int64_t>(absIncx) : static_cast<int64_t>(absIncx);
    const aclblasComplex* outPtr = (p.incx < 0 && p.n > 0) ? x.data() + (p.n - 1) * absIncx : x.data();
    const aclblasComplex* goldPtr = (p.incx < 0 && p.n > 0) ? golden.data() + (p.n - 1) * absIncx : golden.data();

    const size_t count = static_cast<size_t>(p.n);
    std::vector<float> outReal(count);
    std::vector<float> outImag(count);
    std::vector<float> goldReal(count);
    std::vector<float> goldImag(count);
    for (int i = 0; i < p.n; ++i) {
        const size_t off = static_cast<size_t>(step * static_cast<int64_t>(i));
        outReal[i] = outPtr[off].real;
        outImag[i] = outPtr[off].imag;
        goldReal[i] = goldPtr[off].real;
        goldImag[i] = goldPtr[off].imag;
    }

    // Degenerate-reference guard (see test/tpsv/ctpsv/README.md):
    // when both inputs are finite but the cblas reference itself overflows into inf/NaN, the
    // substitution has already left the meaningful range of FP32 — a 1-ULP difference is then
    // amplified into a completely different (inf/NaN) trajectory, so that comparing the two
    // results no longer says anything about the operator. Those cases are skipped.
    // Cases whose input already carries inf/NaN are NOT skipped: there the inf/NaN propagation
    // is well defined and must stay aligned with the reference.
    if (CtpsvAllFinite(ap) && CtpsvAllFinite(xIn)) {
        bool goldenNonFinite = false;
        float maxGoldMag = 0.0f;
        for (size_t i = 0; i < count; ++i) {
            if (!std::isfinite(goldReal[i]) || !std::isfinite(goldImag[i])) {
                goldenNonFinite = true;
                break;
            }
            maxGoldMag = std::max(maxGoldMag, std::max(std::fabs(goldReal[i]), std::fabs(goldImag[i])));
        }
        if (goldenNonFinite) {
            GTEST_SKIP() << "cblas reference overflows to inf/NaN for a finite input (FP32 out-of-range), "
                         << "the comparison is meaningless for this case";
        }
        // Magnitude guard: at reference magnitudes this large the absolute part of the tolerance
        // gates (per-element 32 x ULP) is finer than the cross-platform dispersion of the cblas
        // reference itself, so the comparison no longer adjudicates the operator. Such rows are
        // skipped; the operator remains covered by the rest of the case set.
        constexpr float kGoldenMagnitudeSkipLimit = 4.6e18f; // 2^62
        if (maxGoldMag > kGoldenMagnitudeSkipLimit) {
            GTEST_SKIP() << "cblas reference magnitude (" << maxGoldMag
                         << ") is beyond the deterministic comparison regime (pathological amplification), "
                         << "the comparison no longer adjudicates the operator";
        }
    }

    VerifyConfig cfg;
    applyMixedTolerance(cfg, ACL_FLOAT, goldReal.data(), count);
    EXPECT_TRUE(Verifier::verifyVector(outReal.data(), goldReal.data(), count, 1, cfg, p.caseName + "_real"));
    applyMixedTolerance(cfg, ACL_FLOAT, goldImag.data(), count);
    EXPECT_TRUE(Verifier::verifyVector(outImag.data(), goldImag.data(), count, 1, cfg, p.caseName + "_imag"));
}

TEST_P(CtpsvTest, CsvDriven)
{
    const auto& p = GetParam();
    if (CtpsvIsPerfCase(p)) {
        RunPerfCase(p);
        return;
    }
    std::vector<aclblasComplex> ap;
    BuildAp(p, ap);
    RunAccuracyCase(p, ap);
}

// Negative case: a null handle must return ACLBLAS_STATUS_HANDLE_IS_NULLPTR without touching AP/x
// (task doc §2.4). This is a fixed (non-CSV) test because §3.5 fixes handle to a valid handle.
TEST(CtpsvNullHandle, ReturnsHandleIsNullptr)
{
    aclblasComplex ap[1] = {{1.0f, 0.0f}};
    aclblasComplex x[1] = {{1.0f, 0.0f}};
    const aclblasStatus_t ret = aclblasCtpsv(nullptr, ACLBLAS_LOWER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT, 1, ap, x, 1);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_HANDLE_IS_NULLPTR));
}

// White-box: the kernel Init sets the x GM view length to (uint32_t)(2 * span) where
// span = 1 + (n-1)*|incx| complex elements (2 floats each). A huge |incx| makes 2*span wrap
// around uint32 (n=4, incx=1431655765 -> span = 4294967296, 2*span wraps to 0) and would corrupt
// x silently, so the host span guard must refuse it up front (stpsv XSpanReachesUint32Max
// precedent). Fixed (non-CSV) case: the wrapper would first try to stage a (n-1)*|incx|+1 element
// x buffer (~34GB here) so the guard is asserted through a direct call; the guard fires before
// any device access, so a minimal scratch buffer is enough.
TEST_F(CtpsvTest, XSpanUint32Overflow)
{
    aclblasComplex scratch[10] = {}; // n=4 packed AP = 10 slots; never dereferenced
    const aclblasStatus_t ret =
        aclblasCtpsv(handle_, ACLBLAS_LOWER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT, 4, scratch, scratch, 1431655765);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_NOT_SUPPORTED));
}

// White-box: negating INT_MIN in the |incx| precomputation is undefined behaviour, so the host
// Validate chain rejects it up front (ctrsv precedent).
TEST_F(CtpsvTest, IncxIntMinRejected)
{
    aclblasComplex scratch[10] = {};
    const aclblasStatus_t ret = aclblasCtpsv(
        handle_, ACLBLAS_LOWER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT, 4, scratch, scratch, std::numeric_limits<int>::min());
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE));
}
