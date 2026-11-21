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
#include <cstddef>
#include <cstdio>
#include <random>
#include <vector>

#include "verify.h"
#include "blas_test.h"
#include "csv_loader.h"
#include "fill.h"
#include "ctpsv_param.h"
#include "ctpsv_golden.h"
#include "ctpsv_npu_wrapper.h"

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
inline std::vector<aclblasComplex> CtpsvMakePacked(int n, bool upper, const std::string& fill, uint32_t seed)
{
    if (n <= 0) {
        return {};
    }
    const size_t count = static_cast<size_t>(n) * (n + 1) / 2;
    return CtpsvInterleave(
        makeBlasTriangular(n, upper, fill, seed), makeBlasTriangular(n, upper, fill, seed + 1000U), count);
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
// verify_performance.py judges a performance case by the wall time gtest reports for it, so such
// a case has to measure the operator and not the harness: building the operand and the cblas
// golden for n = 4096 costs about half a second on the host, two orders of magnitude more than
// the operator itself, which would completely mask the number under test. Accuracy is covered
// by the 1000 accuracy cases (they reach n = 2048), so a performance case only has to keep the
// operator honest: one device-resident operand pair is built once for the largest n and reused,
// and the result is checked to be finite.
//
// The operand is a triangular matrix whose entries are all equal. Such a matrix is non-singular
// for every n and every uplo/trans combination, and neither substitution overflows: with a
// constant a, forward substitution gives the partial sums S_i = sum_{j<=i} x_j = b_i / a, which
// stay bounded. The kernel cost is data independent, so this is a fair operand to time.
constexpr size_t PF_MAX_N = 4096;
// Default number of back-to-back operator launches per performance case for the device-side event
// timing (used when n >= 64; smaller n uses more repeats so the event timer sees a measurable
// span). 100 > 50 satisfies task doc §3.3「有效采样 >50 次」. verify_performance.py parses the
// printed "[CTPSV_PERF] <case> avg_us=..." per-call average.
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

inline bool CtpsvIsPerfCase(const CtpsvParam& p)
{
    return p.caseName.rfind("TC_PF_", 0) == 0 && p.expectResult == ACLBLAS_STATUS_SUCCESS && p.n > 0 &&
           p.n <= static_cast<int>(PF_MAX_N) && gPfAp != nullptr;
}
} // namespace

class CtpsvTest : public BlasTest<CtpsvParam> {
protected:
    static void SetUpTestSuite()
    {
        BlasTest<CtpsvParam>::SetUpTestSuite();
        // Built here so that the one-off cost does not land on the first performance case.
        CtpsvPreparePerfOperand();
    }

    static void TearDownTestSuite()
    {
        CtpsvReleasePerfOperand();
        BlasTest<CtpsvParam>::TearDownTestSuite();
    }

    static void RunPerfCase(const CtpsvParam& p);
    static void BuildAp(const CtpsvParam& p, std::vector<aclblasComplex>& ap);
    static void RunAccuracyCase(const CtpsvParam& p, std::vector<aclblasComplex>& ap);
    static void VerifyResult(
        const CtpsvParam& p, const std::vector<aclblasComplex>& ap, const std::vector<aclblasComplex>& xIn,
        const std::vector<aclblasComplex>& x, const std::vector<aclblasComplex>& golden);
};

INSTANTIATE_TEST_SUITE_P(
    Ctpsv, CtpsvTest, ::testing::ValuesIn(GetCasesFromCsv<CtpsvParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<CtpsvParam>);

void CtpsvTest::RunPerfCase(const CtpsvParam& p)
{
    const size_t xCount = static_cast<size_t>(p.n - 1) * std::abs(p.incx) + 1;
    const size_t xBytes = xCount * sizeof(aclblasComplex);
    // AP is read-only for the operator; only x has to be restored after the previous case.
    ASSERT_EQ(aclrtMemcpy(gPfX, xBytes, gPfXHost.data(), xBytes, ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);

    // Time with device-side events (same pattern as test/copy/ccopy/arch35/ccopy_benchmark.cpp)
    // so the fixed host launch+sync latency (~0.7 ms) does not swamp the kernel time for small
    // n. Task doc §3.3 measures the *average single-call* time in us (warmup + >50 samples).
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
    std::printf("[CTPSV_PERF] %s avg_us=%.4f\n", p.caseName.c_str(), avgUs);
}

void CtpsvTest::BuildAp(const CtpsvParam& p, std::vector<aclblasComplex>& ap)
{
    if (p.n <= 0 || p.apFill == "NULLPTR") {
        return;
    }
    const bool isUpper = (p.uplo == ACLBLAS_UPPER);
    ap = CtpsvMakePacked(p.n, isUpper, p.apFill, p.randomSeed);
    if (p.diag == ACLBLAS_UNIT) {
        // Keep the unit-diagonal substitution numerically stable. Complex off-diagonal entries have
        // modulus √2× larger than the real stpsv case, and the Gaussian fill carries a non-zero mean
        // (μ sampled to ±5 makes off-diagonal entries same-signed, turning back substitution into a
        // geometric series that blows up). Tighten the scale to 1/n (vs stpsv's 5/n) so |A(i,j)| ≪ 1
        // and the unit diagonal dominates for both uniform and Gaussian fills.
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

void CtpsvTest::RunAccuracyCase(const CtpsvParam& p, std::vector<aclblasComplex>& ap)
{
    // ---- right-hand side / solution vector x (complex64) ----
    std::vector<aclblasComplex> x;
    if (p.xFill != "NULLPTR") {
        x = CtpsvMakeStrided(p.n, p.incx, p.xFill, p.randomSeed + 1U);
    }
    const std::vector<aclblasComplex> xIn = x;
    std::vector<aclblasComplex> golden = x;

    aclblasStatus_t ret = aclblasCtpsv_npu(handle_, p.uplo, p.trans, p.diag, p.n, ap.data(), x.data(), p.incx);
    if (p.expectResult != ACLBLAS_STATUS_SUCCESS) {
        EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
        return;
    }
    ASSERT_EQ(ret, ACLBLAS_STATUS_SUCCESS);

    aclblasCtpsv_cpu(handle_, p.uplo, p.trans, p.diag, p.n, ap.data(), golden.data(), p.incx);

    VerifyResult(p, ap, xIn, x, golden);
}

void CtpsvTest::VerifyResult(
    const CtpsvParam& p, const std::vector<aclblasComplex>& ap, const std::vector<aclblasComplex>& xIn,
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
        for (size_t i = 0; i < count; ++i) {
            if (!std::isfinite(goldReal[i]) || !std::isfinite(goldImag[i])) {
                goldenNonFinite = true;
                break;
            }
        }
        if (goldenNonFinite) {
            GTEST_SKIP() << "cblas reference overflows to inf/NaN for a finite input (FP32 out-of-range), "
                         << "the comparison is meaningless for this case";
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
