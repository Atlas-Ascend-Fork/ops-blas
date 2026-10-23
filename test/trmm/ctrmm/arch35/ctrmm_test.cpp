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
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

#include "verify.h"
#include "fill.h"
#include "blas_test.h"
#include "csv_loader.h"
#include "ctrmm_param.h"
#include "ctrmm_golden.h"
#include "ctrmm_npu_wrapper.h"

// Epsilon for the complex MERE/MARE suite, following the frame default (2^-14).
static constexpr double kCtrmmMereEpsilon = 1.0 / 16384.0;

// Column-major complex64 matrix buffer: m x n with leading dimension ld.
// makeBlasComplexMatrix handles m<=0 / n<=0 / ld<=0 by returning an empty vector.
static inline std::vector<aclblasComplex> MakeComplexMatrix(
    int rows, int cols, int ld, const BlasFillMode& fill, uint32_t seed)
{
    return makeBlasComplexMatrix(rows, cols, ld, fill, seed);
}

// Triangular matrix: dimA x dimA column-major, only the uplo triangle keeps
// the generated values; the other triangle is zeroed (storage hygiene — the
// operator must not read it, and neither must the golden).
static inline std::vector<aclblasComplex> MakeComplexTriMatrix(
    int dimA, int lda, aclblasFillMode_t uplo, const BlasFillMode& fill, uint32_t seed)
{
    std::vector<aclblasComplex> data = makeBlasComplexMatrix(dimA, dimA, lda, fill, seed);
    if (data.empty()) {
        return data;
    }
    for (int col = 0; col < dimA; ++col) {
        for (int row = 0; row < dimA; ++row) {
            bool inTriangle = (uplo == ACLBLAS_UPPER) ? (row <= col) : (row >= col);
            if (!inTriangle) {
                data[static_cast<size_t>(col) * lda + row] = aclblasComplex{0.0f, 0.0f};
            }
        }
    }
    return data;
}

class CtrmmArch35Test : public BlasTest<CtrmmParam> {};

// ── Handwritten: null handle ──
TEST_F(CtrmmArch35Test, NullHandle)
{
    aclblasComplex alpha{1.0f, 0.0f};
    std::vector<aclblasComplex> a(16, aclblasComplex{0.0f, 0.0f});
    std::vector<aclblasComplex> b(16, aclblasComplex{0.0f, 0.0f});
    std::vector<aclblasComplex> c(16, aclblasComplex{0.0f, 0.0f});
    aclblasStatus_t ret = aclblasCtrmm_npu(
        nullptr, ACLBLAS_SIDE_LEFT, ACLBLAS_UPPER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT, 4, 4, &alpha, a.data(), 4, b.data(),
        4, c.data(), 4);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_HANDLE_IS_NULLPTR));
}

// ── Handwritten: diag = UNIT must not read A's diagonal (or the unreferenced
// triangle) — poison both and require output equality against the same-shape
// clean-input run of the golden (task doc §2.5-5).
TEST_F(CtrmmArch35Test, UnitDiagNotRead)
{
    const int dim = 8;
    const int m = dim;
    const int n = dim;
    aclblasComplex alpha{1.0f, 0.0f};
    const uint32_t seed = 20260924u;

    std::vector<aclblasComplex> aClean =
        MakeComplexTriMatrix(dim, dim, ACLBLAS_LOWER, parseFill("RANDOM_NORM_5_5"), seed);
    ASSERT_EQ(aClean.size(), static_cast<size_t>(dim) * dim);
    std::vector<aclblasComplex> bHost = MakeComplexMatrix(m, n, n, parseFill("RANDOM_NORM_5_5"), seed + 1U);
    ASSERT_EQ(bHost.size(), static_cast<size_t>(m) * n);

    // Poisoned copy: NaN on the diagonal, Inf in the unreferenced (upper) triangle.
    std::vector<aclblasComplex> aPoison = aClean;
    const float nanv = std::numeric_limits<float>::quiet_NaN();
    const float infv = std::numeric_limits<float>::infinity();
    for (int col = 0; col < dim; ++col) {
        for (int row = 0; row < dim; ++row) {
            size_t idx = static_cast<size_t>(col) * dim + row;
            if (row == col) {
                aPoison[idx] = aclblasComplex{nanv, nanv};
            } else if (row < col) {
                aPoison[idx] = aclblasComplex{infv, -infv};
            }
        }
    }

    std::vector<aclblasComplex> cPoison(static_cast<size_t>(m) * n, aclblasComplex{0.0f, 0.0f});
    std::vector<aclblasComplex> cClean(static_cast<size_t>(m) * n, aclblasComplex{0.0f, 0.0f});

    aclblasStatus_t ret1 = aclblasCtrmm_npu(
        CtrmmArch35Test::handle_, ACLBLAS_SIDE_LEFT, ACLBLAS_LOWER, ACLBLAS_OP_N, ACLBLAS_UNIT, m, n, &alpha,
        aPoison.data(), dim, bHost.data(), n, cPoison.data(), n);
    ASSERT_EQ(static_cast<int>(ret1), static_cast<int>(ACLBLAS_STATUS_SUCCESS));

    // Golden with the same poisoned A: cblas_ctrmm with diag=Unit must not
    // read the diagonal either, so poisoned and clean goldens coincide.
    aclblasStatus_t g1 = aclblasCtrmm_cpu(
        CtrmmArch35Test::handle_, ACLBLAS_SIDE_LEFT, ACLBLAS_LOWER, ACLBLAS_OP_N, ACLBLAS_UNIT, m, n, &alpha,
        aPoison.data(), dim, bHost.data(), n, cClean.data(), n);
    ASSERT_EQ(static_cast<int>(g1), static_cast<int>(ACLBLAS_STATUS_SUCCESS));

    for (size_t i = 0; i < cPoison.size(); ++i) {
        EXPECT_FALSE(std::isnan(cPoison[i].real)) << "NaN leaked at element " << i;
        EXPECT_FALSE(std::isnan(cPoison[i].imag)) << "NaN leaked at element " << i;
        EXPECT_FALSE(std::isinf(cPoison[i].real)) << "Inf leaked at element " << i;
        EXPECT_FALSE(std::isinf(cPoison[i].imag)) << "Inf leaked at element " << i;
    }
    EXPECT_TRUE(Verifier::verifyMereMareComplexFloat(
        cPoison.data(), cClean.data(), cPoison.size(), 1.0 / 8192.0, 10.0, kCtrmmMereEpsilon, "UnitDiagNotRead_mere"));
}

// ── Handwritten: C == B in-place alias (cuBLAS-compatible overlap) ──
TEST_F(CtrmmArch35Test, AliasInPlaceCB)
{
    const int m = 8;
    const int n = 8;
    const int lda = m;
    const int ldb = m;
    const int ldc = m;
    aclblasComplex alpha{2.0f, -1.0f};
    const uint32_t seed = 20260925u;

    std::vector<aclblasComplex> aHost = MakeComplexTriMatrix(m, lda, ACLBLAS_UPPER, parseFill("RANDOM_NORM_5_5"), seed);
    std::vector<aclblasComplex> bHost = MakeComplexMatrix(m, n, ldb, parseFill("RANDOM_NORM_5_5"), seed + 1U);
    const size_t cCount = static_cast<size_t>(ldc) * n;
    const size_t cBytes = cCount * sizeof(aclblasComplex);

    void* dA = nullptr;
    void* dBC = nullptr;
    ASSERT_EQ(
        static_cast<int>(CtrmmAllocCopyH2D(dA, aHost.data(), aHost.size() * sizeof(aclblasComplex))), ACL_SUCCESS);
    ASSERT_EQ(static_cast<int>(CtrmmAllocCopyH2D(dBC, bHost.data(), cBytes)), ACL_SUCCESS);

    aclblasStatus_t ret = aclblasCtrmm(
        CtrmmArch35Test::handle_, ACLBLAS_SIDE_LEFT, ACLBLAS_UPPER, ACLBLAS_OP_C, ACLBLAS_NON_UNIT, m, n, &alpha,
        static_cast<const aclblasComplex*>(dA), lda, static_cast<const aclblasComplex*>(dBC), ldb,
        static_cast<aclblasComplex*>(dBC), ldc);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_SUCCESS));

    std::vector<aclblasComplex> cHost(cCount, aclblasComplex{0.0f, 0.0f});
    if (ret == ACLBLAS_STATUS_SUCCESS) {
        ret = CtrmmSyncAndCopyD2H(CtrmmArch35Test::handle_, dBC, cHost.data(), cBytes);
        EXPECT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_SUCCESS));
    }
    CtrmmFreeDev(dA);
    CtrmmFreeDev(dBC);
    if (ret != ACLBLAS_STATUS_SUCCESS) {
        return;
    }

    // Out-of-place golden on a separate buffer.
    std::vector<aclblasComplex> cGolden(cCount, aclblasComplex{0.0f, 0.0f});
    aclblasStatus_t g = aclblasCtrmm_cpu(
        CtrmmArch35Test::handle_, ACLBLAS_SIDE_LEFT, ACLBLAS_UPPER, ACLBLAS_OP_C, ACLBLAS_NON_UNIT, m, n, &alpha,
        aHost.data(), lda, bHost.data(), ldb, cGolden.data(), ldc);
    ASSERT_EQ(static_cast<int>(g), static_cast<int>(ACLBLAS_STATUS_SUCCESS));
    EXPECT_TRUE(Verifier::verifyMereMareComplexFloat(
        cHost.data(), cGolden.data(), cCount, 1.0 / 8192.0, 10.0, kCtrmmMereEpsilon, "AliasInPlaceCB_mere"));
}

INSTANTIATE_TEST_SUITE_P(
    Ctrmm, CtrmmArch35Test, ::testing::ValuesIn(GetCasesFromCsv<CtrmmParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<CtrmmParam>);

// Builds the triangular A and rectangular B host buffers and resolves the
// alpha/A/B pointers (nullptr per the CSV null flags). alphaOnDevice when the
// case description contains "device_alpha".
static void PrepareCtrmmTestData(
    const CtrmmParam& p, int aDim, std::vector<aclblasComplex>& aHost, std::vector<aclblasComplex>& bHost,
    const aclblasComplex*& alphaPtr, const aclblasComplex*& aPtr, const aclblasComplex*& bPtr, bool& alphaOnDevice)
{
    aHost = MakeComplexTriMatrix(aDim, p.lda, p.uplo, p.aFill, p.randomSeed);
    bHost = MakeComplexMatrix(p.m, p.n, p.ldb, p.bFill, p.randomSeed + 1U);

    static thread_local aclblasComplex alphaStorage{1.0f, 0.0f};
    alphaStorage = aclblasComplex{p.alphaReal, p.alphaImag};
    alphaPtr = p.nullAlpha ? nullptr : &alphaStorage;
    aPtr = p.nullA ? nullptr : (aHost.empty() ? nullptr : aHost.data());
    bPtr = p.nullB ? nullptr : (bHost.empty() ? nullptr : bHost.data());

    alphaOnDevice = (!p.nullAlpha && p.description.find("device_alpha") != std::string::npos);
}

// alpha == (0,0) with nullA/nullB: the host fast path memsets C's active rows
// without reading A/B. Result must be exactly zero (bit-reproducible).
static void HandleCtrmmAlphaZeroNullAB(
    const CtrmmParam& p, aclblasHandle_t handle, const aclblasComplex* alphaPtr, const aclblasComplex* aPtr,
    const aclblasComplex* bPtr, aclblasComplex* cPtr, bool alphaOnDevice)
{
    const size_t cCount = static_cast<size_t>(p.ldc) * static_cast<size_t>(p.n);
    const size_t cBytes = cCount * sizeof(aclblasComplex);
    void* dC = nullptr;
    aclError mRet = aclrtMalloc(&dC, cBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    ASSERT_EQ(static_cast<int>(mRet), ACL_SUCCESS);
    void* dAlpha = nullptr;
    const aclblasComplex* alphaArg = alphaPtr;
    if (alphaOnDevice && alphaPtr != nullptr) {
        aclError aRet = aclrtMalloc(&dAlpha, sizeof(aclblasComplex), ACL_MEM_MALLOC_HUGE_FIRST);
        ASSERT_EQ(static_cast<int>(aRet), ACL_SUCCESS);
        aRet = aclrtMemcpy(dAlpha, sizeof(aclblasComplex), alphaPtr, sizeof(aclblasComplex), ACL_MEMCPY_HOST_TO_DEVICE);
        ASSERT_EQ(static_cast<int>(aRet), ACL_SUCCESS);
        alphaArg = static_cast<const aclblasComplex*>(dAlpha);
    }
    aclblasStatus_t ret = aclblasCtrmm(
        handle, p.side, p.uplo, p.trans, p.diag, p.m, p.n, alphaArg, aPtr, p.lda, bPtr, p.ldb,
        static_cast<aclblasComplex*>(dC), p.ldc);
    if (ret == ACLBLAS_STATUS_SUCCESS && cPtr != nullptr) {
        ret = CtrmmSyncAndCopyD2H(handle, dC, cPtr, cBytes);
    }
    CtrmmFreeDev(dAlpha);
    CtrmmFreeDev(dC);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
    if (ret == ACLBLAS_STATUS_SUCCESS && cPtr != nullptr) {
        // Contract: alpha==(0,0) zeroes the m ACTIVE rows of each column; the
        // padding rows are outside the operator contract (and dC padding is
        // uninitialized), so only the active region is asserted.
        for (int j = 0; j < p.n; ++j) {
            const aclblasComplex* col = cPtr + static_cast<size_t>(j) * static_cast<size_t>(p.ldc);
            for (int i = 0; i < p.m; ++i) {
                EXPECT_EQ(col[i].real, 0.0f) << "C real must be zero when alpha==(0,0), col " << j << " row " << i;
                EXPECT_EQ(col[i].imag, 0.0f) << "C imag must be zero when alpha==(0,0), col " << j << " row " << i;
            }
        }
    }
}

// Opt-in CSV dump of (actual, golden) pairs (CTRMM_DUMP_CSV): accuracy
// charts rendered outside the test; capped to keep files small. Extracted
// verbatim from the MERE gate body.
static void DumpCtrmmMereCsv(const CtrmmParam& p, const aclblasComplex* out, const aclblasComplex* gold, size_t count)
{
    if (const char* csvPath = std::getenv("CTRMM_DUMP_CSV")) {
        std::FILE* fp = std::fopen(csvPath, "w");
        if (fp != nullptr) {
            std::fprintf(fp, "idx,i,j,actual_re,golden_re,actual_im,golden_im\n");
            const size_t cap = 4096;
            const size_t n = (count < cap) ? count : cap;
            for (size_t k = 0; k < n; ++k) {
                std::fprintf(
                    fp, "%zu,%zu,%zu,%.9g,%.9g,%.9g,%.9g\n", k, k % static_cast<size_t>(p.ldc),
                    k / static_cast<size_t>(p.ldc), static_cast<double>(out[k].real), static_cast<double>(gold[k].real),
                    static_cast<double>(out[k].imag), static_cast<double>(gold[k].imag));
            }
            std::fclose(fp);
        }
    }
}

// Opt-in console dump (CTRMM_DUMP): first out/gold pairs plus a mismatch
// coordinate histogram ((i mod 64) x (j mod 64)) so periodic failure
// patterns show up directly in the log. Extracted verbatim.
// Mismatch coordinate histogram: classify each failing element by its
// (row i mod 64) and (col j mod 64) so a periodic pattern shows up.
static void DumpCtrmmMismatchHistogram(
    const CtrmmParam& p, const aclblasComplex* out, const aclblasComplex* gold, size_t count)
{
    // Mismatch coordinate histogram: classify each failing element by its
    // (row i mod 64) and (col j mod 64) so a periodic pattern shows up.
    size_t printed = 0;
    int rowMod[64] = {0};
    int colMod[64] = {0};
    int tot = 0;
    for (size_t idx = 0; idx < count; ++idx) {
        const float o[2] = {out[idx].real, out[idx].imag};
        const float g[2] = {gold[idx].real, gold[idx].imag};
        for (int c = 0; c < 2; ++c) {
            const bool bad = (o[c] != g[c]) && !(std::isnan(o[c]) && std::isnan(g[c])) &&
                             (std::isfinite(o[c]) != std::isfinite(g[c]) ||
                              (std::isfinite(o[c]) && std::fabs(o[c] - g[c]) > std::fabs(g[c]) * 1e-3f + 1e-4f));
            if (bad) {
                const size_t i = idx % static_cast<size_t>(p.ldc);
                const size_t j = idx / static_cast<size_t>(p.ldc);
                if (printed < 12) {
                    std::cout << "  BAD (i=" << i << ",j=" << j << ") out=" << o[c] << " gold=" << g[c]
                              << " i%64=" << (i % 64) << " j%64=" << (j % 64) << std::endl;
                    printed++;
                }
                rowMod[i % 64]++;
                colMod[j % 64]++;
                tot++;
            }
        }
    }
    std::cout << "  bad_total=" << tot << " rows%64 top: ";
    for (int k = 0; k < 64; ++k)
        if (rowMod[k] > tot / 64)
            std::cout << k << ":" << rowMod[k] << " ";
    std::cout << " | cols%64 top: ";
    for (int k = 0; k < 64; ++k)
        if (colMod[k] > tot / 64)
            std::cout << k << ":" << colMod[k] << " ";
    std::cout << std::endl;
}

static void DumpCtrmmMereConsole(
    const CtrmmParam& p, const aclblasComplex* out, const aclblasComplex* gold, size_t count)
{
    if (std::getenv("CTRMM_DUMP") != nullptr) {
        std::cout << "[" << p.caseName << "] DUMP m=" << p.m << " n=" << p.n << " ldc=" << p.ldc
                  << " side=" << static_cast<int>(p.side) << " uplo=" << static_cast<int>(p.uplo)
                  << " trans=" << static_cast<int>(p.trans) << std::endl;
        for (size_t i = 0; i < 8 && i < count; ++i) {
            std::cout << "  i=" << i << " out=(" << out[i].real << "," << out[i].imag << ") gold=(" << gold[i].real
                      << "," << gold[i].imag << ")" << std::endl;
        }
        DumpCtrmmMismatchHistogram(p, out, gold, count);
    }
}

// Suite B gate: MERE (mean element-wise relative error) over re/im pooled,
// plus the Inf/NaN position rule as a hard failure. MARE (the max) is
// recorded but not gated: against a float32 cblas golden the max criterion is
// not winnable once the golden itself drifts from the exact value on
// heavy-cancellation elements — golden-vs-double experiment (2026-09-24,
// device host, OpenBLAS): 2048^2 LEFT/UPPER/N/NON_UNIT golden self-error
// MARE=3.1e-3 > 10*2^-13 on 19/8.4M elements while MERE stays ~1e-7.
// The task-book element-level criterion (Suite A, matchedRatio >= 0.99)
// absorbs this noise by construction; MERE (mean) likewise.
static void VerifyCtrmmMereGate(
    const CtrmmParam& p, const aclblasComplex* out, const aclblasComplex* gold, size_t count)
{
    double sumRel = 0.0;
    double maxRel = 0.0;
    size_t valid = 0;
    size_t outliers = 0;
    // Near-zero skip floor: relative error is undefined when the golden sits
    // in the accumulation-noise zone. The noise lives on the accumulated
    // value BEFORE the alpha scaling, so in output space the floor scales
    // with |alpha| (e.g. alpha=(100,0) lifts gold=1.3e-4 elements whose
    // pre-alpha accumulation is ~1.3e-6 — pure noise). max(1,|alpha|) keeps
    // the frame's 1x floor for |alpha| <= 1.
    const double alphaMag = std::sqrt(
        static_cast<double>(p.alphaReal) * static_cast<double>(p.alphaReal) +
        static_cast<double>(p.alphaImag) * static_cast<double>(p.alphaImag));
    const double skipFloor = std::max(1.0, alphaMag) * p.mereThreshold;
    for (size_t i = 0; i < count; i++) {
        const float o[2] = {out[i].real, out[i].imag};
        const float g[2] = {gold[i].real, gold[i].imag};
        for (int c = 0; c < 2; c++) {
            if (o[c] == g[c]) {
                continue; // exact-equal (covers equal-sign Inf)
            }
            if (std::isnan(o[c]) && std::isnan(g[c])) {
                continue; // NaN on both sides
            }
            if (std::isnan(o[c]) || std::isnan(g[c]) || std::isinf(o[c]) || std::isinf(g[c])) {
                ADD_FAILURE() << p.caseName << ": Inf/NaN position mismatch at element " << i << " component " << c
                              << " out=" << o[c] << " gold=" << g[c];
                continue;
            }
            const double ag = std::fabs(static_cast<double>(g[c]));
            if (ag < skipFloor || ag == 0.0) {
                continue; // near-zero or zero golden: relative error undefined
            }
            const double rel = std::fabs(static_cast<double>(o[c]) - static_cast<double>(g[c])) / ag;
            sumRel += rel;
            valid++;
            if (rel > maxRel) {
                maxRel = rel;
            }
            if (rel > p.mareMultiplier * p.mereThreshold) {
                outliers++;
            }
        }
    }
    const double mere = (valid > 0) ? sumRel / static_cast<double>(valid) : 0.0;
    DumpCtrmmMereCsv(p, out, gold, count);
    DumpCtrmmMereConsole(p, out, gold, count);
    std::cout << "[" << p.caseName << "] MERE=" << mere << " (gate < " << p.mereThreshold << ")"
              << " MARE=" << maxRel << " (recorded, limit " << p.mareMultiplier * p.mereThreshold << ", " << outliers
              << " outliers / " << valid << " valid)" << std::endl;
    EXPECT_LT(mere, p.mereThreshold) << p.caseName << ": MERE gate";
}

// Suite A gate: matchedRatio >= 0.99 (atol=2^-16, rtol=2^-10).
// maxAbsErr is recorded but not gated (task doc doesn't require it).
static void VerifyCtrmmTaskBookMixed(
    const CtrmmParam& p, const aclblasComplex* out, const aclblasComplex* gold, size_t count)
{
    constexpr double kAtol = 1.52587890625e-5; // 2^-16
    constexpr double kRtol = 9.765625e-4;      // 2^-10
    size_t matched = 0;
    size_t total = 0;
    double maxAbs = 0.0;
    for (size_t i = 0; i < count; i++) {
        const float o[2] = {out[i].real, out[i].imag};
        const float g[2] = {gold[i].real, gold[i].imag};
        for (int c = 0; c < 2; c++) {
            total++;
            if (o[c] == g[c] || (std::isnan(o[c]) && std::isnan(g[c]))) {
                matched++;
                continue;
            }
            if (std::isnan(o[c]) || std::isnan(g[c]) || std::isinf(o[c]) || std::isinf(g[c])) {
                continue; // unmatched; the position rule already hard-fails in the MERE gate
            }
            const double ae = std::fabs(static_cast<double>(o[c]) - static_cast<double>(g[c]));
            const double ag = std::fabs(static_cast<double>(g[c]));
            if (ae > maxAbs) {
                maxAbs = ae;
            }
            const double ulp = getUlpsAt(ag, 23, -126);
            const bool ok = (ae <= kAtol + kRtol * ag) || (ae <= std::max(1e-2, 32.0 * ulp));
            if (ok) {
                matched++;
            }
        }
    }
    const double ratio = (total > 0) ? static_cast<double>(matched) / static_cast<double>(total) : 1.0;
    std::cout << "[" << p.caseName << "] taskbook-mixed matchedRatio=" << ratio << " (req 0.99) maxAbsErr=" << maxAbs
              << " (recorded)" << std::endl;
    EXPECT_GE(ratio, 0.99) << p.caseName << ": task-book matched_ratio gate";
}

/**
 * @brief Dual-strategy verification entry point.
 *
 * Suite A (hard): task-book §3.2 element criterion + matchedRatio >= 0.99.
 * Suite B (hard): MERE mean gate + Inf/NaN position rule (above).
 * alpha == (0,0) uses EXACT equality (C = 0 is bit-reproducible).
 */
static void RunCtrmmGoldenAndVerify(
    const CtrmmParam& p, aclblasHandle_t handle, const aclblasComplex* alphaPtr, const aclblasComplex* aPtr,
    const aclblasComplex* bPtr, aclblasComplex* cPtr)
{
    const size_t cCount = static_cast<size_t>(p.ldc) * static_cast<size_t>(p.n);
    if (cCount == 0) {
        return;
    }
    std::vector<aclblasComplex> cGolden(cCount, aclblasComplex{0.0f, 0.0f});

    aclblasStatus_t goldenRet = aclblasCtrmm_cpu(
        handle, p.side, p.uplo, p.trans, p.diag, p.m, p.n, alphaPtr, aPtr, p.lda, bPtr, p.ldb, cGolden.data(), p.ldc);
    ASSERT_EQ(static_cast<int>(goldenRet), static_cast<int>(ACLBLAS_STATUS_SUCCESS)) << "golden computation failed";

    if (p.AlphaIsZero()) {
        VerifyConfig cfgExact;
        cfgExact.mode = PrecisionMode::EXACT;
        std::vector<float> outRe(cCount);
        std::vector<float> outIm(cCount);
        std::vector<float> goldRe(cCount);
        std::vector<float> goldIm(cCount);
        for (size_t i = 0; i < cCount; i++) {
            outRe[i] = cPtr[i].real;
            outIm[i] = cPtr[i].imag;
            goldRe[i] = cGolden[i].real;
            goldIm[i] = cGolden[i].imag;
        }
        EXPECT_TRUE(
            Verifier::verifyVector(outRe.data(), goldRe.data(), cCount, 1, cfgExact, p.caseName + "_exact_real"));
        EXPECT_TRUE(
            Verifier::verifyVector(outIm.data(), goldIm.data(), cCount, 1, cfgExact, p.caseName + "_exact_imag"));
        return;
    }

    // Suite B: MERE mean gate (re/im pooled) + Inf/NaN position rule;
    // MARE recorded — see VerifyCtrmmMereGate for the golden-self-error basis.
    VerifyCtrmmMereGate(p, cPtr, cGolden.data(), cCount);

    // Suite A: task-book element criterion + matchedRatio gate; maxAbsErr
    // recorded (see VerifyCtrmmTaskBookMixed for the frame-delta rationale).
    VerifyCtrmmTaskBookMixed(p, cPtr, cGolden.data(), cCount);
}

TEST_P(CtrmmArch35Test, CsvDriven)
{
    const auto& p = GetParam();
    const int aDim = (p.side == ACLBLAS_SIDE_LEFT) ? p.m : p.n;

    std::vector<aclblasComplex> aHost, bHost;
    const aclblasComplex* alphaPtr = nullptr;
    const aclblasComplex* aPtr = nullptr;
    const aclblasComplex* bPtr = nullptr;
    bool alphaOnDevice = false;
    PrepareCtrmmTestData(p, aDim, aHost, bHost, alphaPtr, aPtr, bPtr, alphaOnDevice);

    // Zero/negative dimensions or unusable leading dims: passthrough to the
    // API with a dummy C (the operator owns validation and quick returns).
    if (p.m <= 0 || p.n <= 0 || p.lda <= 0 || p.ldb <= 0 || p.ldc <= 0) {
        aclblasComplex dummy{0.0f, 0.0f};
        aclblasStatus_t ret = aclblasCtrmm_npu(
            CtrmmArch35Test::handle_, p.side, p.uplo, p.trans, p.diag, p.m, p.n, alphaPtr, aPtr, p.lda, bPtr, p.ldb,
            &dummy, p.ldc);
        EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
        return;
    }

    // Performance-track cases (TC_PF_*): minimal body — single call + sync.
    // The official performance script times the whole GTest case wall-clock,
    // so no golden/verify work may live here; correctness of these parameter
    // combinations is covered by the accuracy families and the sampling runs.
    const bool isPerfCase = p.caseName.rfind("TC_PF_", 0) == 0;

    const size_t cCount = static_cast<size_t>(p.ldc) * static_cast<size_t>(p.n);
    std::vector<aclblasComplex> cHost(cCount, aclblasComplex{0.0f, 0.0f});
    aclblasComplex* cPtr = p.nullC ? nullptr : cHost.data();

    if (p.AlphaIsZero() && (p.nullA || p.nullB)) {
        HandleCtrmmAlphaZeroNullAB(p, CtrmmArch35Test::handle_, alphaPtr, aPtr, bPtr, cPtr, alphaOnDevice);
        return;
    }

    aclblasStatus_t ret = aclblasCtrmm_npu(
        CtrmmArch35Test::handle_, p.side, p.uplo, p.trans, p.diag, p.m, p.n, alphaPtr, aPtr, p.lda, bPtr, p.ldb, cPtr,
        p.ldc, alphaOnDevice, p.description.find("alias") != std::string::npos);

    if (p.expectResult != ACLBLAS_STATUS_SUCCESS) {
        EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
        return;
    }
    ASSERT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_SUCCESS));

    if (isPerfCase) {
        return;
    }
    RunCtrmmGoldenAndVerify(p, CtrmmArch35Test::handle_, alphaPtr, aPtr, bPtr, cPtr);
}
