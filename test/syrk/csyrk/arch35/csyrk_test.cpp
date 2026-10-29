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
#include <functional>
#include <cmath>
#include <cstdlib>
#include <vector>

#include "verify.h"
#include "blas_test.h"
#include "csv_loader.h"
#include "fill.h"
#include "csyrk_param.h"
#include "csyrk_golden.h"
#include "csyrk_npu_wrapper.h"

// ═══════════════════════════════════════════════════════════════════════════════
// Complex uplo-triangle verification helpers
//   1. uplo triangle: split real/imag, apply ACL_FLOAT mixed tolerance separately
//   2. non-uplo triangle: compare against the input-old C (EXACT, must be untouched)
// ═══════════════════════════════════════════════════════════════════════════════

// Collect uplo / non-uplo element pairs from the NPU output and golden buffers.
// `isUplo` follows the BLAS uplo convention: UPPER ⇒ i <= j, LOWER ⇒ i >= j.
// `goldPtr` doubles as the "input old" reference for the non-uplo triangle
// because cblas_csyrk only writes the uplo triangle.
template <typename Param>
static inline void CollectUploNonUploElements(
    const Param& p, const aclblasComplex* cPtr, const aclblasComplex* goldPtr, std::vector<float>& npuUploRe,
    std::vector<float>& npuUploIm, std::vector<float>& goldUploRe, std::vector<float>& goldUploIm,
    std::vector<float>& npuNonRe, std::vector<float>& npuNonIm, std::vector<float>& oldNonRe,
    std::vector<float>& oldNonIm)
{
    for (int j = 0; j < p.n; j++) {
        for (int i = 0; i < p.n; i++) {
            size_t idx = static_cast<size_t>(i) + static_cast<size_t>(j) * p.ldc;
            bool isUplo = (p.uplo == ACLBLAS_UPPER) ? (i <= j) : (i >= j);
            if (isUplo) {
                npuUploRe.push_back(cPtr[idx].real);
                npuUploIm.push_back(cPtr[idx].imag);
                goldUploRe.push_back(goldPtr[idx].real);
                goldUploIm.push_back(goldPtr[idx].imag);
            } else {
                npuNonRe.push_back(cPtr[idx].real);
                npuNonIm.push_back(cPtr[idx].imag);
                oldNonRe.push_back(goldPtr[idx].real);
                oldNonIm.push_back(goldPtr[idx].imag);
            }
        }
    }
}

// Verify uplo triangle precision — real and imag parts use ACL_FLOAT mixed tolerance.
static inline void VerifyUploPrecision(
    const std::string& caseName, const float* npuUploRe, const float* goldUploRe, size_t uploCount,
    const float* npuUploIm, const float* goldUploIm)
{
    VerifyConfig cfgRe;
    applyMixedTolerance(cfgRe, ACL_FLOAT, goldUploRe, uploCount);
    EXPECT_TRUE(Verifier::verifyVector(npuUploRe, goldUploRe, uploCount, 1, cfgRe, caseName + "_uplo_real"));

    VerifyConfig cfgIm;
    applyMixedTolerance(cfgIm, ACL_FLOAT, goldUploIm, uploCount);
    EXPECT_TRUE(Verifier::verifyVector(npuUploIm, goldUploIm, uploCount, 1, cfgIm, caseName + "_uplo_imag"));
}

// Verify non-uplo triangle — must equal input old (EXACT, verifies no pollution).
static inline void VerifyNonUploExact(
    const std::string& caseName, const float* npuNonRe, const float* oldNonRe, size_t nonCount, const float* npuNonIm,
    const float* oldNonIm)
{
    if (nonCount == 0) {
        return;
    }
    VerifyConfig cfgNonRe;
    cfgNonRe.mode = PrecisionMode::EXACT;
    EXPECT_TRUE(Verifier::verifyVector(npuNonRe, oldNonRe, nonCount, 1, cfgNonRe, caseName + "_nonuplo_real"));
    VerifyConfig cfgNonIm;
    cfgNonIm.mode = PrecisionMode::EXACT;
    EXPECT_TRUE(Verifier::verifyVector(npuNonIm, oldNonIm, nonCount, 1, cfgNonIm, caseName + "_nonuplo_imag"));
}

// Full complex uplo-triangle verification.
template <typename Param>
static inline void VerifyUploTriangleComplex(const Param& p, const aclblasComplex* cPtr, const aclblasComplex* goldPtr)
{
    if (p.n <= 0) {
        return;
    }

    std::vector<float> npuUploRe;
    std::vector<float> npuUploIm;
    std::vector<float> goldUploRe;
    std::vector<float> goldUploIm;
    std::vector<float> npuNonRe;
    std::vector<float> npuNonIm;
    std::vector<float> oldNonRe;
    std::vector<float> oldNonIm;
    CollectUploNonUploElements(
        p, cPtr, goldPtr, npuUploRe, npuUploIm, goldUploRe, goldUploIm, npuNonRe, npuNonIm, oldNonRe, oldNonIm);

    VerifyUploPrecision(
        p.caseName, npuUploRe.data(), goldUploRe.data(), npuUploRe.size(), npuUploIm.data(), goldUploIm.data());

    VerifyNonUploExact(p.caseName, npuNonRe.data(), oldNonRe.data(), npuNonRe.size(), npuNonIm.data(), oldNonIm.data());
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test fixture
// ═══════════════════════════════════════════════════════════════════════════════

class CsyrkTest : public BlasTest<CsyrkParam> {};

// ── TEST_F: null handle (not CSV-driven) ──
TEST_F(CsyrkTest, NullHandle)
{
    aclblasComplex alpha{1.0f, 0.0f};
    aclblasComplex beta{0.0f, 0.0f};
    aclblasStatus_t ret =
        aclblasCsyrk_npu(nullptr, ACLBLAS_UPPER, ACLBLAS_OP_N, 4, 4, &alpha, nullptr, 4, &beta, nullptr, 4);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_HANDLE_IS_NULLPTR));
}

// ── TEST_F: host-scalar alpha/beta accepted (issue #642) ──
// The npu wrapper uploads scalars to device, so this calls the public API
// directly with host-side alpha/beta addresses: they must be dereferenced
// instead of receiving a misleading INTERNAL_ERROR from a failed D2H copy.
// Shapes cover both the n<=64 direct path (skipped for host scalars, !722
// review) and the n>64 deint/cube/combine path carrying scalars via tiling.
// Host golden for the UPPER triangle: C = alpha*A*A^T + beta*C (no conjugation).
static void BuildCsyrkHostGolden(
    int n, int k, int lda, int ldc, const aclblasComplex& alpha, const aclblasComplex& beta,
    const std::vector<aclblasComplex>& aHost, const std::vector<aclblasComplex>& cInit,
    std::vector<aclblasComplex>& golden)
{
    golden = cInit;
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            float accRe = 0.0f;
            float accIm = 0.0f;
            for (int t = 0; t < k; t++) {
                const aclblasComplex& av = aHost[static_cast<size_t>(t) * lda + i];
                const aclblasComplex& bv = aHost[static_cast<size_t>(t) * lda + j];
                accRe += av.real * bv.real - av.imag * bv.imag;
                accIm += av.real * bv.imag + av.imag * bv.real;
            }
            const size_t idx = static_cast<size_t>(j) * ldc + i;
            golden[idx].real =
                alpha.real * accRe - alpha.imag * accIm + beta.real * cInit[idx].real - beta.imag * cInit[idx].imag;
            golden[idx].imag =
                alpha.real * accIm + alpha.imag * accRe + beta.real * cInit[idx].imag + beta.imag * cInit[idx].real;
        }
    }
}

static void VerifyCsyrkHostScalarOutput(
    int n, int ldc, const std::vector<aclblasComplex>& golden, const std::vector<aclblasComplex>& cOut)
{
    constexpr float eps = 1e-4f;
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            const size_t idx = static_cast<size_t>(j) * ldc + i;
            EXPECT_NEAR(cOut[idx].real, golden[idx].real, eps * (1.0f + std::abs(golden[idx].real)));
            EXPECT_NEAR(cOut[idx].imag, golden[idx].imag, eps * (1.0f + std::abs(golden[idx].imag)));
        }
    }
}

static bool RunCsyrkHostScalarShape(aclblasHandle_t handle, aclrtStream stream, int n, int k, int lda, int ldc)
{
    const aclblasComplex alpha{2.0f, 0.0f};
    const aclblasComplex beta{1.5f, 0.0f};
    std::vector<aclblasComplex> aHost(static_cast<size_t>(lda) * k);
    std::vector<aclblasComplex> cInit(static_cast<size_t>(ldc) * n);
    for (size_t i = 0; i < aHost.size(); i++) {
        aHost[i] = {0.5f + static_cast<float>(i), 0.25f + static_cast<float>(i)};
    }
    for (size_t i = 0; i < cInit.size(); i++) {
        cInit[i] = {1.0f + static_cast<float>(i), 2.0f + static_cast<float>(i)};
    }

    std::vector<aclblasComplex> golden;
    BuildCsyrkHostGolden(n, k, lda, ldc, alpha, beta, aHost, cInit, golden);

    const size_t aBytes = aHost.size() * sizeof(aclblasComplex);
    const size_t cBytes = cInit.size() * sizeof(aclblasComplex);
    void* dA = nullptr;
    void* dC = nullptr;
    bool ok = aclrtMalloc(&dA, aBytes, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS &&
              aclrtMalloc(&dC, cBytes, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS;
    if (ok) {
        ok = aclrtMemcpy(dA, aBytes, aHost.data(), aBytes, ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS &&
             aclrtMemcpy(dC, cBytes, cInit.data(), cBytes, ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS;
    }
    if (ok) {
        const aclblasStatus_t ret = aclblasCsyrk(
            handle, ACLBLAS_UPPER, ACLBLAS_OP_N, n, k, &alpha, static_cast<const aclblasComplex*>(dA), lda, &beta,
            static_cast<aclblasComplex*>(dC), ldc);
        EXPECT_EQ(ret, ACLBLAS_STATUS_SUCCESS);
        ok = ret == ACLBLAS_STATUS_SUCCESS && aclrtSynchronizeStream(stream) == ACL_SUCCESS;
    }
    if (ok) {
        std::vector<aclblasComplex> cOut(cInit.size());
        ok = aclrtMemcpy(cOut.data(), cBytes, dC, cBytes, ACL_MEMCPY_DEVICE_TO_HOST) == ACL_SUCCESS;
        EXPECT_TRUE(ok);
        if (ok) {
            VerifyCsyrkHostScalarOutput(n, ldc, golden, cOut);
        }
    }
    if (dA != nullptr) {
        aclrtFree(dA);
    }
    if (dC != nullptr) {
        aclrtFree(dC);
    }
    return ok;
}

TEST_F(CsyrkTest, HostScalarAlphaBeta)
{
    ASSERT_TRUE(RunCsyrkHostScalarShape(handle_, stream_, 4, 3, 4, 4));
    ASSERT_TRUE(RunCsyrkHostScalarShape(handle_, stream_, 72, 4, 72, 72));
}

INSTANTIATE_TEST_SUITE_P(
    Csyrk, CsyrkTest, ::testing::ValuesIn(GetCasesFromCsv<CsyrkParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<CsyrkParam>);

// ═══════════════════════════════════════════════════════════════════════════════
// CSV-driven parameterised test (5-step flow)
//   1. Generate host data  2. Run NPU  3. Check return code
//   4. Run CPU golden       5. Verify precision
// ═══════════════════════════════════════════════════════════════════════════════

struct CsyrkHostData {
    std::vector<aclblasComplex> aHost;
    std::vector<aclblasComplex> cHost;
    std::vector<aclblasComplex> cGolden;
    const aclblasComplex* aPtr = nullptr;
    aclblasComplex* cPtr = nullptr;
    aclblasComplex* cGoldenPtr = nullptr;
    size_t cCount = 0;
};

static bool PrepareHostData(const CsyrkParam& p, CsyrkHostData& d)
{
    const int aRows = p.lda;
    const int aCols = (p.trans == ACLBLAS_OP_N) ? p.k : p.n;

    const size_t cBytes = static_cast<size_t>(p.ldc) * static_cast<size_t>(p.n) * sizeof(aclblasComplex);
    const size_t aBytes = static_cast<size_t>(p.lda) * static_cast<size_t>(aCols) * sizeof(aclblasComplex);
    // 3 copies of C (npu output + golden + input-old reference) + A.
    constexpr size_t kHostMemLimit = 8ULL * 1024ULL * 1024ULL * 1024ULL;
    if (3 * cBytes + aBytes > kHostMemLimit) {
        std::cout << "[SKIP] host memory estimate (" << (3 * cBytes + aBytes) / (1024 * 1024)
                  << " MB) exceeds limit for n=" << p.n << ", ldc=" << p.ldc << std::endl;
        return false;
    }

    try {
        d.aHost = makeBlasComplexMatrix(aRows, aCols, p.lda, p.aFill, p.randomSeed);
        d.cHost = makeBlasComplexMatrix(p.n, p.n, p.ldc, p.cFill, p.randomSeed + 1U);
    } catch (const std::bad_alloc&) {
        std::cout << "[SKIP] host memory allocation failed for n=" << p.n << ", k=" << p.k << std::endl;
        return false;
    }

    d.aPtr = (d.aHost.empty() || p.nullA) ? nullptr : d.aHost.data();
    d.cPtr = (d.cHost.empty() || p.nullC) ? nullptr : d.cHost.data();
    d.cCount = static_cast<size_t>(p.ldc) * static_cast<size_t>(p.n);

    if (d.cPtr != nullptr) {
        try {
            d.cGolden = d.cHost; // golden starts from input-old C; non-uplo triangle stays untouched
        } catch (const std::bad_alloc&) {
            std::cout << "[SKIP] host memory allocation failed for golden copy (n=" << p.n << ")" << std::endl;
            return false;
        }
        d.cGoldenPtr = d.cGolden.data();
    }
    return true;
}

// Runs the call `perfRepeat` times and returns the last status (nullAlpha/
// nullBeta exercise the API's INVALID_VALUE path; the values are then unused).
static aclblasStatus_t RunNpuCalls(aclblasHandle_t handle, const CsyrkParam& p, CsyrkHostData& d, int perfRepeat)
{
    aclblasStatus_t ret = ACLBLAS_STATUS_SUCCESS;
    for (int it = 0; it < perfRepeat; it++) {
        ret = aclblasCsyrk_npu(
            handle, p.uplo, p.trans, p.n, p.k, &p.alpha, d.aPtr, p.lda, &p.beta, d.cPtr, p.ldc, p.nullAlpha,
            p.nullBeta);
    }
    return ret;
}

// Device buffers for the perf path: A/C/alpha/beta allocated once and reused so
// the measured time excludes malloc/H2D/D2H. Returns false when alloc failed
// (callers treat it as a skip).
struct CsyrkPerfBufs {
    void* dA = nullptr;
    void* dC = nullptr;
    void* dAlpha = nullptr;
    void* dBeta = nullptr;
};

static void FreeCsyrkPerfBufs(CsyrkPerfBufs& b)
{
    if (b.dA) {
        aclrtFree(b.dA);
    }
    if (b.dC) {
        aclrtFree(b.dC);
    }
    if (b.dAlpha) {
        aclrtFree(b.dAlpha);
    }
    if (b.dBeta) {
        aclrtFree(b.dBeta);
    }
}

static bool AllocCsyrkPerfBufs(const CsyrkParam& p, CsyrkHostData& d, CsyrkPerfBufs& b)
{
    const int aRows = (p.trans == ACLBLAS_OP_N) ? p.lda : p.k;
    const int aCols = (p.trans == ACLBLAS_OP_N) ? p.k : p.n;
    const size_t aBytes = static_cast<size_t>(aRows) * aCols * sizeof(aclblasComplex);
    const size_t cBytes = static_cast<size_t>(p.ldc) * p.n * sizeof(aclblasComplex);
    constexpr size_t scalarBytes = sizeof(aclblasComplex);
    auto allocCopy = [](void*& dev, const void* host, size_t bytes) -> bool {
        if (host == nullptr || bytes == 0) {
            return true;
        }
        if (aclrtMalloc(&dev, bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
            return false;
        }
        return aclrtMemcpy(dev, bytes, host, bytes, ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS;
    };
    if (!allocCopy(b.dA, d.aPtr, aBytes) || !allocCopy(b.dC, d.cPtr, cBytes) ||
        !allocCopy(b.dAlpha, &p.alpha, scalarBytes) || !allocCopy(b.dBeta, &p.beta, scalarBytes)) {
        FreeCsyrkPerfBufs(b);
        return false;
    }
    return true;
}

// One timed sample: warm up, then measure `perfIters` calls. win0/win1 are epoch
// microseconds so the acceptance script can attribute msprof kernel records to
// this case (kernels whose Task Start Time falls in [win0, win1] belong here).
static double TimeCsyrkKernel(const std::function<void()>& runKernel, int64_t& win0, int64_t& win1)
{
    constexpr int warmIters = 10;
    constexpr int perfIters = 50;
    for (int it = 0; it < warmIters; it++) {
        runKernel();
    }
    win0 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch())
               .count();
    auto t0 = std::chrono::steady_clock::now();
    for (int it = 0; it < perfIters; it++) {
        runKernel();
    }
    auto t1 = std::chrono::steady_clock::now();
    win1 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch())
               .count();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / perfIters;
}

// Performance cases (TC_PF_*): the acceptance script reads the GTest wall clock,
// and a full golden run is O(n^3) Netlib and dominates the measurement (n=2048
// golden ~20 s vs kernel ~3 ms), so skip the precision compare and only run the
// NPU path. Device buffers are allocated once and reused so the measured time is
// the kernel path (malloc/H2D/D2H excluded).
static void RunPerfCase(aclblasHandle_t handle, aclrtStream stream, const CsyrkParam& p, CsyrkHostData& d)
{
    CsyrkPerfBufs bufs;
    if (!AllocCsyrkPerfBufs(p, d, bufs)) {
        GTEST_SKIP() << "Skipped: device alloc failed";
    }
    auto runKernel = [&]() {
        aclblasCsyrk(
            handle, p.uplo, p.trans, p.n, p.k, static_cast<const aclblasComplex*>(bufs.dAlpha),
            static_cast<const aclblasComplex*>(bufs.dA), p.lda, static_cast<const aclblasComplex*>(bufs.dBeta),
            static_cast<aclblasComplex*>(bufs.dC), p.ldc);
        // issue #605: 同步失败时记录失败（ADD_FAILURE 使整个用例 FAIL，不再产出有效性能结论）
        if (aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
            ADD_FAILURE() << "csyrk perf aclrtSynchronizeStream failed";
            return;
        }
    };
    int64_t win0 = 0;
    int64_t win1 = 0;
    double msPerIter = TimeCsyrkKernel(runKernel, win0, win1);
    FreeCsyrkPerfBufs(bufs);
    std::cout << "[PERF] " << p.caseName << " n=" << p.n << " k=" << p.k << " avg_ms=" << msPerIter << " (iters=50)"
              << " win_us=" << win0 << "-" << win1 << std::endl;
}

// Accuracy case: run the CPU golden and compare the uplo triangle.
static void RunAccuracyCase(aclblasHandle_t handle, const CsyrkParam& p, CsyrkHostData& d)
{
    aclblasStatus_t goldenRet =
        aclblasCsyrk_cpu(handle, p.uplo, p.trans, p.n, p.k, &p.alpha, d.aPtr, p.lda, &p.beta, d.cGoldenPtr, p.ldc);
    if (goldenRet != ACLBLAS_STATUS_SUCCESS) {
        EXPECT_EQ(goldenRet, ACLBLAS_STATUS_SUCCESS) << "golden computation failed";
        return;
    }
    VerifyUploTriangleComplex(p, d.cPtr, d.cGoldenPtr);
}

TEST_P(CsyrkTest, CsvDriven)
{
    const auto& p = GetParam();
    CsyrkHostData d;
    if (!PrepareHostData(p, d)) {
        GTEST_SKIP() << "Skipped: host memory limit";
    }
    int perfRepeat = 1;
    if (const char* pr = std::getenv("CSYRK_PERF_REPEAT")) {
        perfRepeat = std::max(1, std::atoi(pr));
    }
    aclblasStatus_t ret = RunNpuCalls(CsyrkTest::handle_, p, d, perfRepeat);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
    if (p.expectResult != ACLBLAS_STATUS_SUCCESS || p.n == 0 || d.cPtr == nullptr) {
        return;
    }
    if (p.caseName.rfind("TC_PF_", 0) == 0) {
        RunPerfCase(CsyrkTest::handle_, CsyrkTest::stream_, p, d);
        return;
    }
    RunAccuracyCase(CsyrkTest::handle_, p, d);
}
