/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

#include "verify.h"
#include "blas_test.h"
#include "csv_loader.h"
#include "fill.h"
#include "verify_uplo.h"
#include "cherk_param.h"
#include "cherk_golden.h"
#include "cherk_npu_wrapper.h"

// ═══════════════════════════════════════════════════════════════════════════════
// Complex uplo-triangle verification helpers (测试方案 §7.3 / §7.4)
// ═══════════════════════════════════════════════════════════════════════════════

// Verify Hermitian diagonal: when beta==0 output is pure A·A^H, diagonal must be real.
template <typename Param>
static inline void VerifyDiagonalHermitian(const Param& p, const aclblasComplex* cPtr)
{
    if (p.beta != 0.0f) {
        return;
    }
    double maxDiagImag = 0.0;
    for (int i = 0; i < p.n; i++) {
        size_t idx = static_cast<size_t>(i) + static_cast<size_t>(i) * p.ldc;
        double im = std::abs(static_cast<double>(cPtr[idx].imag));
        if (im > maxDiagImag) {
            maxDiagImag = im;
        }
    }
    // ACL_FLOAT atol = 2^-16 ≈ 1.5259e-5 (测试方案 §7.2)
    EXPECT_LE(maxDiagImag, 1.52587890625e-5)
        << "[" << p.caseName << "] Hermitian diagonal imag not zero (max=" << maxDiagImag << ")";
}

// ═══════════════════════════════════════════════════════════════════════════════
// Complex uplo-triangle verification (测试方案 §7.3 / §7.4)
//   1. uplo triangle: split real/imag, apply ACL_FLOAT mixed tolerance separately
//   2. non-uplo triangle: compare against the input-old C (EXACT, must be untouched)
//   3. Hermitian diagonal: |C[i][i].imag| <= atol when beta==0 (pure A·A^H output)
// `goldPtr` already carries the input-old values in its non-uplo triangle
// because cblas_cherk only writes the uplo triangle — so it doubles as the
// "input old" reference for the non-uplo check.
// ═══════════════════════════════════════════════════════════════════════════════
template <typename Param>
static VerifyConfig MakeTriangleVerifyConfig(const Param& p, bool mixed, size_t uploCount)
{
    VerifyConfig cfg;
    if (!mixed) {
        cfg.mode = PrecisionMode::EXACT;
        return cfg;
    }

    applyMixedTolerance(cfg, ACL_FLOAT, nullptr, uploCount);
    // Cube/vector reduction order differs from Netlib's scalar accumulation;
    // retain the existing k-dependent bound.
    cfg.mixedMaxAbsErrorLimit = std::max(cfg.mixedMaxAbsErrorLimit, 1.0e-4 * static_cast<double>(p.k));
    return cfg;
}

template <typename Param, typename Component, typename Consume>
static void VisitTriangleElements(
    const Param& p, const aclblasComplex* cPtr, const aclblasComplex* goldPtr, bool uplo, Component component,
    Consume consume)
{
    for (int j = 0; j < p.n; ++j) {
        for (int i = 0; i < p.n; ++i) {
            const bool isUplo = (p.uplo == ACLBLAS_UPPER) ? (i <= j) : (i >= j);
            if (isUplo != uplo) {
                continue;
            }
            const size_t idx = static_cast<size_t>(i) + static_cast<size_t>(j) * p.ldc;
            consume(component(cPtr[idx]), component(goldPtr[idx]));
        }
    }
}

template <typename Param, typename Component>
static void VerifyTrianglePart(
    const Param& p, const aclblasComplex* cPtr, const aclblasComplex* goldPtr, const char* suffix, bool uplo,
    Component component, bool mixed, size_t uploCount, size_t nonUploCount)
{
    const size_t count = uplo ? uploCount : nonUploCount;
    if (count == 0U) {
        return;
    }
    const VerifyConfig cfg = MakeTriangleVerifyConfig(p, mixed, uploCount);
    EXPECT_TRUE(Verifier::verifyEach(
        [&](auto consume) { VisitTriangleElements(p, cPtr, goldPtr, uplo, component, consume); }, count, cfg,
        p.caseName + suffix));
}

template <typename Param>
static inline void VerifyUploTriangleComplex(
    const Param& p, const aclblasComplex* cPtr, const aclblasComplex* goldPtr, size_t /*cCount*/)
{
    if (p.n <= 0) {
        return;
    }

    const size_t uploCount = static_cast<size_t>(p.n) * static_cast<size_t>(p.n + 1) / 2U;
    const size_t nonUploCount = static_cast<size_t>(p.n) * static_cast<size_t>(p.n - 1) / 2U;

    VerifyTrianglePart(
        p, cPtr, goldPtr, "_uplo_real", true, [](const aclblasComplex& value) { return value.real; }, true, uploCount,
        nonUploCount);
    VerifyTrianglePart(
        p, cPtr, goldPtr, "_uplo_imag", true, [](const aclblasComplex& value) { return value.imag; }, true, uploCount,
        nonUploCount);
    VerifyTrianglePart(
        p, cPtr, goldPtr, "_nonuplo_real", false, [](const aclblasComplex& value) { return value.real; }, false,
        uploCount, nonUploCount);
    VerifyTrianglePart(
        p, cPtr, goldPtr, "_nonuplo_imag", false, [](const aclblasComplex& value) { return value.imag; }, false,
        uploCount, nonUploCount);

    VerifyDiagonalHermitian(p, cPtr);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test fixture
// ═══════════════════════════════════════════════════════════════════════════════

class CherkArch35Test : public BlasTest<CherkParam> {};

static bool HasCasePrefix(const CherkParam& p, const char* prefix) { return p.caseName.rfind(prefix, 0U) == 0U; }

static std::vector<CherkParam> LoadCherkCorrectnessCases()
{
    const bool fullRegression = std::getenv("CHERK_FULL_REGRESSION") != nullptr;
    std::string csvPath = ReplaceFileExtension2Csv(__BASE_FILE__);
    // Only arch22 has a separate smoke/full CSV pair.
    if (fullRegression && csvPath.find("arch22") != std::string::npos) {
        csvPath.insert(csvPath.size() - 4U, "_full");
    }
    std::vector<CherkParam> selectedCases;
    for (const CherkParam& p : GetCasesFromCsv<CherkParam>(csvPath)) {
        if (!fullRegression && HasCasePrefix(p, "TC_PF_")) {
            continue;
        }

        selectedCases.push_back(p);
    }
    return selectedCases;
}

// ── TEST_F: null handle (not CSV-driven) ──
TEST_F(CherkArch35Test, NullHandle)
{
    float alpha = 1.0f;
    float beta = 0.0f;
    aclblasStatus_t ret =
        aclblasCherk_npu(nullptr, ACLBLAS_UPPER, ACLBLAS_OP_N, 4, 4, &alpha, nullptr, 4, &beta, nullptr, 4);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_HANDLE_IS_NULLPTR));
}

INSTANTIATE_TEST_SUITE_P(
    Cherk, CherkArch35Test, ::testing::ValuesIn(LoadCherkCorrectnessCases()), PrintCaseInfoString<CherkParam>);

// ═══════════════════════════════════════════════════════════════════════════════
// CSV-driven parameterised test (5-step flow)
//   1. Generate host data  2. Run NPU  3. Check return code
//   4. Run CPU golden       5. Verify precision
// ═══════════════════════════════════════════════════════════════════════════════

struct CherkHostData {
    std::vector<aclblasComplex> aHost;
    std::vector<aclblasComplex> cHost;
    std::vector<aclblasComplex> cGolden;
    const aclblasComplex* aPtr = nullptr;
    aclblasComplex* cPtr = nullptr;
    aclblasComplex* cGoldenPtr = nullptr;
    size_t cCount = 0;
};

static bool PrepareHostData(const CherkParam& p, CherkHostData& d, bool prepareGolden)
{
    const int aRows = p.lda;
    const int aCols = (p.trans == ACLBLAS_OP_N) ? p.k : p.n;

    const size_t cBytes = static_cast<size_t>(p.ldc) * static_cast<size_t>(p.n) * sizeof(aclblasComplex);
    const size_t aBytes = static_cast<size_t>(p.lda) * static_cast<size_t>(aCols) * sizeof(aclblasComplex);
    const size_t cCopies = prepareGolden ? 2U : 1U;
    constexpr size_t kHostMemLimit = 8ULL * 1024ULL * 1024ULL * 1024ULL; // §8 遗留问题 4
    if (cCopies * cBytes + aBytes > kHostMemLimit) {
        std::cout << "[SKIP] host memory estimate (" << (cCopies * cBytes + aBytes) / (1024 * 1024)
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

    if (prepareGolden && d.cPtr != nullptr) {
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

TEST_P(CherkArch35Test, CsvDriven)
{
    const auto& p = GetParam();
    CherkHostData d;
    if (!PrepareHostData(p, d, true)) {
        GTEST_SKIP() << "Skipped: host memory limit";
    }

    // nullAlpha: wrapper forwards nullptr for alpha so the API's INVALID_VALUE
    // path is exercised. The alpha value in p.alpha is irrelevant in that case.
    aclblasStatus_t ret = aclblasCherk_npu(
        CherkArch35Test::handle_, p.uplo, p.trans, p.n, p.k, &p.alpha, d.aPtr, p.lda, p.nullBeta ? nullptr : &p.beta,
        d.cPtr, p.ldc, p.nullAlpha);

    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
    if (p.expectResult != ACLBLAS_STATUS_SUCCESS || p.n == 0 || d.cPtr == nullptr) {
        return;
    }

    aclblasStatus_t goldenRet = aclblasCherk_cpu(
        CherkArch35Test::handle_, p.uplo, p.trans, p.n, p.k, &p.alpha, d.aPtr, p.lda, &p.beta, d.cGoldenPtr, p.ldc);
    if (goldenRet != ACLBLAS_STATUS_SUCCESS) {
        EXPECT_EQ(goldenRet, ACLBLAS_STATUS_SUCCESS) << "golden computation failed";
        return;
    }

    VerifyUploTriangleComplex(p, d.cPtr, d.cGoldenPtr, d.cCount);
}
