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
#include <climits>
#include <iostream>
#include <limits>
#include <optional>
#include <random>
#include <vector>

#include "verify.h"
#include "blas_test.h"
#include "csv_loader.h"
#include "device.h"
#include "fill.h"
#include "ctbsv_param.h"
#include "ctbsv_golden.h"
#include "ctbsv_npu_wrapper.h"

namespace {

constexpr uint32_t DEFAULT_RANDOM_SEED = 42U;
constexpr uint32_t IMAGINARY_SEED_OFFSET = 1000U;
constexpr float NORMAL_MEAN = 0.0f;
constexpr float NORMAL_STANDARD_DEVIATION = 0.1f;
constexpr int MIN_DIAGONAL_BOOST = 5; // Task test guide: boost = max(5, n).

// Existing special-value classification cutoffs. These are used only by
// VerifySpecialValues, never to relax ordinary finite-case Mixed Tolerance.
constexpr float OVERFLOW_MAGNITUDE = 1.0e30f;
constexpr float OVERFLOW_PEER_MAGNITUDE = 1.0e16f;
constexpr float EXPLODED_SOLUTION_MAGNITUDE = 1.0e20f;

// Task FLOAT32 Mixed Tolerance and IEEE 754 binary32 parameters.
constexpr double DEFAULT_MIXED_TOLERANCE = 0.0001220703125; // 2^-13
constexpr double REQUIRED_MATCHED_RATIO = 0.99;
constexpr double MAX_ABSOLUTE_ERROR = 1e-2;
constexpr int FP32_MANTISSA_BITS = 23;
constexpr int FP32_MIN_NORMAL_EXPONENT = -126;
constexpr float GM_FALLBACK_ABSOLUTE_TOLERANCE = 1.0e-5f;

struct NormalDistributionSpec {
    float mean;
    float standardDeviation;
};

bool HasSpecialValueFill(const BlasFillMode& fill)
{
    return fill.pattern == BlasFillMode::P_EXTREME ||
           (fill.method == BlasFillMode::M_VALUE && (!std::isfinite(fill.val1) || !std::isfinite(fill.val2)));
}

NormalDistributionSpec GetNormalDistributionSpec(uint32_t seed)
{
    (void)seed;
    return {NORMAL_MEAN, NORMAL_STANDARD_DEVIATION};
}

std::vector<float> makeNormalBanded(int m, int n, int kl, int ku, int lda, uint32_t seed)
{
    const size_t size = n > 0 ? static_cast<size_t>(lda) * static_cast<size_t>(n) : 1U;
    std::vector<float> data(size, 0.0f);
    if (n <= 0) {
        return data;
    }
    std::mt19937 rng(seed == 0 ? DEFAULT_RANDOM_SEED : seed);
    const auto spec = GetNormalDistributionSpec(seed);
    std::normal_distribution<float> distribution(spec.mean, spec.standardDeviation);
    for (int j = 0; j < n; ++j) {
        const int iStart = std::max(0, j - ku);
        const int iEnd = std::min(m - 1, j + kl);
        for (int i = iStart; i <= iEnd; ++i) {
            const int bandRow = ku + i - j;
            data[static_cast<size_t>(bandRow) + static_cast<size_t>(j) * static_cast<size_t>(lda)] = distribution(rng);
        }
    }
    return data;
}

std::vector<float> makeNormalStrided(int count, int inc, uint32_t seed)
{
    const int absInc = std::abs(inc);
    const size_t size = count > 0 ? static_cast<size_t>((count - 1) * absInc + 1) : 1U;
    std::vector<float> data(size, 0.0f);
    if (count <= 0) {
        return data;
    }
    std::mt19937 rng(seed == 0 ? DEFAULT_RANDOM_SEED : seed);
    const auto spec = GetNormalDistributionSpec(seed);
    std::normal_distribution<float> distribution(spec.mean, spec.standardDeviation);
    for (int i = 0; i < count; ++i) {
        const int index = inc > 0 ? i * inc : (count - 1 - i) * absInc;
        data[static_cast<size_t>(index)] = distribution(rng);
    }
    return data;
}

std::vector<aclblasComplex> makeComplexBanded(
    int m, int n, int kl, int ku, int lda, const BlasFillMode& fill, uint32_t seed, bool useNormalDistribution)
{
    const bool useNormal =
        useNormalDistribution && fill.method == BlasFillMode::M_RANDOM && fill.pattern == BlasFillMode::P_NORM;
    std::vector<float> realPart =
        useNormal ? makeNormalBanded(m, n, kl, ku, lda, seed) : makeBlasBanded(m, n, kl, ku, lda, fill, seed);
    std::vector<float> imagPart = useNormal ? makeNormalBanded(m, n, kl, ku, lda, seed + IMAGINARY_SEED_OFFSET) :
                                              makeBlasBanded(m, n, kl, ku, lda, fill, seed + IMAGINARY_SEED_OFFSET);
    std::vector<aclblasComplex> data(realPart.size());
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = aclblasComplex{realPart[i], imagPart[i]};
    }
    return data;
}

std::vector<aclblasComplex> makeComplexStrided(
    int count, int inc, const BlasFillMode& fill, uint32_t seed, bool useNormalDistribution)
{
    const bool useNormal =
        useNormalDistribution && fill.method == BlasFillMode::M_RANDOM && fill.pattern == BlasFillMode::P_NORM;
    std::vector<float> realPart =
        useNormal ? makeNormalStrided(count, inc, seed) : makeBlasStrided(count, inc, fill, seed);
    std::vector<float> imagPart = useNormal ? makeNormalStrided(count, inc, seed + IMAGINARY_SEED_OFFSET) :
                                              makeBlasStrided(count, inc, fill, seed + IMAGINARY_SEED_OFFSET);
    const size_t size = realPart.size();
    std::vector<aclblasComplex> data(size);
    for (size_t i = 0; i < size; ++i) {
        data[i] = aclblasComplex{realPart[i], imagPart[i]};
    }
    return data;
}

bool IsNonFinite(float v) { return std::isnan(v) || std::isinf(v); }

bool IsOverflowish(float v) { return IsNonFinite(v) || std::fabs(v) > OVERFLOW_MAGNITUDE; }

void AlignOverflowPair(float& outVal, float& goldVal)
{
    const bool outOf = IsOverflowish(outVal);
    const bool goldOf = IsOverflowish(goldVal);
    if (outOf && goldOf) {
        outVal = std::numeric_limits<float>::quiet_NaN();
        goldVal = std::numeric_limits<float>::quiet_NaN();
        return;
    }
    if (IsNonFinite(outVal) && (goldOf || std::fabs(goldVal) > OVERFLOW_PEER_MAGNITUDE)) {
        outVal = std::numeric_limits<float>::quiet_NaN();
        goldVal = std::numeric_limits<float>::quiet_NaN();
        return;
    }
    if (IsNonFinite(goldVal) && (outOf || std::fabs(outVal) > OVERFLOW_PEER_MAGNITUDE)) {
        outVal = std::numeric_limits<float>::quiet_NaN();
        goldVal = std::numeric_limits<float>::quiet_NaN();
    }
}

void AlignOverflowSolutions(std::vector<aclblasComplex>& outDense, std::vector<aclblasComplex>& goldDense)
{
    const int n = static_cast<int>(outDense.size());
    int overflowed = 0;
    float maxAbs = 0.0f;
    for (int i = 0; i < n; ++i) {
        const auto& o = outDense[static_cast<size_t>(i)];
        const auto& g = goldDense[static_cast<size_t>(i)];
        if (IsOverflowish(o.real) || IsOverflowish(o.imag) || IsOverflowish(g.real) || IsOverflowish(g.imag)) {
            overflowed++;
        }
        maxAbs = std::max(maxAbs, std::fabs(o.real));
        maxAbs = std::max(maxAbs, std::fabs(o.imag));
        maxAbs = std::max(maxAbs, std::fabs(g.real));
        maxAbs = std::max(maxAbs, std::fabs(g.imag));
    }
    const bool exploded = (overflowed > 0) || (maxAbs > EXPLODED_SOLUTION_MAGNITUDE);
    for (int i = 0; i < n; ++i) {
        auto& o = outDense[static_cast<size_t>(i)];
        auto& g = goldDense[static_cast<size_t>(i)];
        AlignOverflowPair(o.real, g.real);
        AlignOverflowPair(o.imag, g.imag);
        if (exploded) {
            if (IsOverflowish(o.real) || IsOverflowish(g.real) || std::fabs(o.real) > OVERFLOW_PEER_MAGNITUDE ||
                std::fabs(g.real) > OVERFLOW_PEER_MAGNITUDE) {
                o.real = std::numeric_limits<float>::quiet_NaN();
                g.real = std::numeric_limits<float>::quiet_NaN();
            }
            if (IsOverflowish(o.imag) || IsOverflowish(g.imag) || std::fabs(o.imag) > OVERFLOW_PEER_MAGNITUDE ||
                std::fabs(g.imag) > OVERFLOW_PEER_MAGNITUDE) {
                o.imag = std::numeric_limits<float>::quiet_NaN();
                g.imag = std::numeric_limits<float>::quiet_NaN();
            }
        }
    }
}

} // namespace

class CtbsvTest : public BlasTest<CtbsvParam> {};

TEST_F(CtbsvTest, NullHandle)
{
    aclblasStatus_t ret =
        aclblasCtbsv_npu(nullptr, ACLBLAS_LOWER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT, 5, 1, nullptr, 2, nullptr, 1);
    EXPECT_EQ(ret, ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
}

TEST_F(CtbsvTest, LargeNFallback)
{
    constexpr int n = 4097;
    constexpr int k = 0;
    constexpr int lda = 1;
    std::vector<aclblasComplex> a(static_cast<size_t>(n));
    std::vector<aclblasComplex> x(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        const float t = static_cast<float>(i % 17);
        a[static_cast<size_t>(i)] = aclblasComplex{2.0f + t * 0.03125f, -0.25f + t * 0.015625f};
        x[static_cast<size_t>(i)] = aclblasComplex{1.0f - t * 0.0625f, -0.5f + t * 0.03125f};
    }
    std::vector<aclblasComplex> golden = x;
    cblas_ctbsv(CblasColMajor, CblasLower, CblasNoTrans, CblasNonUnit, n, k, a.data(), lda, golden.data(), 1);

    DeviceBuffer aDev(a.size() * sizeof(aclblasComplex));
    DeviceBuffer xDev(x.size() * sizeof(aclblasComplex));
    aDev.copyFromHost(a.data(), a.size() * sizeof(aclblasComplex));
    xDev.copyFromHost(x.data(), x.size() * sizeof(aclblasComplex));

    ASSERT_EQ(
        aclblasCtbsv_npu(
            CtbsvTest::handle_, ACLBLAS_LOWER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT, n, k,
            static_cast<const aclblasComplex*>(aDev.ptr()), lda, static_cast<aclblasComplex*>(xDev.ptr()), 1),
        ACLBLAS_STATUS_SUCCESS);
    ASSERT_EQ(aclrtSynchronizeDevice(), ACL_SUCCESS);
    xDev.copyToHost(x.data(), x.size() * sizeof(aclblasComplex));

    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(x[static_cast<size_t>(i)].real, golden[static_cast<size_t>(i)].real, GM_FALLBACK_ABSOLUTE_TOLERANCE)
            << "index=" << i;
        EXPECT_NEAR(x[static_cast<size_t>(i)].imag, golden[static_cast<size_t>(i)].imag, GM_FALLBACK_ABSOLUTE_TOLERANCE)
            << "index=" << i;
    }
}

TEST_F(CtbsvTest, LargeKFallback)
{
    constexpr int n = 2050;
    constexpr int k = 2049;
    constexpr int lda = k + 1;
    constexpr int incx = -2;
    std::vector<aclblasComplex> a(static_cast<size_t>(lda) * static_cast<size_t>(n), {0.0f, 0.0f});
    std::vector<aclblasComplex> x(static_cast<size_t>(1 + (n - 1) * (-incx)), {0.0f, 0.0f});
    for (int col = 0; col < n; ++col) {
        a[static_cast<size_t>(k) + static_cast<size_t>(col) * lda] = {NAN, NAN};
        if (col > 0) {
            a[static_cast<size_t>(k - 1) + static_cast<size_t>(col) * lda] = {0.03125f, -0.015625f};
        }
        const int idx = (n - 1 - col) * (-incx);
        x[static_cast<size_t>(idx)] = {
            static_cast<float>(col % 23) * 0.0625f, static_cast<float>(col % 19) * -0.03125f};
    }
    std::vector<aclblasComplex> golden = x;
    cblas_ctbsv(CblasColMajor, CblasUpper, CblasConjTrans, CblasUnit, n, k, a.data(), lda, golden.data(), incx);

    DeviceBuffer aDev(a.size() * sizeof(aclblasComplex));
    DeviceBuffer xDev(x.size() * sizeof(aclblasComplex));
    aDev.copyFromHost(a.data(), a.size() * sizeof(aclblasComplex));
    xDev.copyFromHost(x.data(), x.size() * sizeof(aclblasComplex));

    ASSERT_EQ(
        aclblasCtbsv_npu(
            CtbsvTest::handle_, ACLBLAS_UPPER, ACLBLAS_OP_C, ACLBLAS_UNIT, n, k,
            static_cast<const aclblasComplex*>(aDev.ptr()), lda, static_cast<aclblasComplex*>(xDev.ptr()), incx),
        ACLBLAS_STATUS_SUCCESS);
    ASSERT_EQ(aclrtSynchronizeDevice(), ACL_SUCCESS);
    xDev.copyToHost(x.data(), x.size() * sizeof(aclblasComplex));

    for (size_t i = 0; i < x.size(); ++i) {
        EXPECT_NEAR(x[i].real, golden[i].real, GM_FALLBACK_ABSOLUTE_TOLERANCE) << "physical index=" << i;
        EXPECT_NEAR(x[i].imag, golden[i].imag, GM_FALLBACK_ABSOLUTE_TOLERANCE) << "physical index=" << i;
    }
}

INSTANTIATE_TEST_SUITE_P(
    Ctbsv, CtbsvTest, ::testing::ValuesIn(GetCasesFromCsv<CtbsvParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<CtbsvParam>);

struct CtbsvFixture {
    const CtbsvParam& p;
    int effK = 0;
    int allocLda = 1;
    int allocN = 1;
    int absIncx = 0;
    size_t xSize = 0;
    bool isUpper = false;
    bool needDevA = false;
    bool needDevX = false;
    std::vector<aclblasComplex> aHost;
    std::vector<aclblasComplex> xHost;
    std::vector<aclblasComplex> golden;
    std::optional<DeviceBuffer> aDev;
    std::optional<DeviceBuffer> xDev;

    explicit CtbsvFixture(const CtbsvParam& param) : p(param) {}

    void InitDims()
    {
        effK = (p.n > 0 && p.k >= 0) ? std::min(p.k, p.n - 1) : 0;
        allocLda = std::max(1, std::max(p.lda, effK + 1));
        allocN = std::max(1, p.n);
        isUpper = (p.uplo == ACLBLAS_UPPER);
        absIncx = (p.incx == INT_MIN) ? 0 : std::abs(p.incx);
        xSize = (p.n > 0 && absIncx > 0) ? static_cast<size_t>((p.n - 1) * absIncx + 1) : 0;
        needDevA = (p.n > 0);
        needDevX = (xSize > 0);
    }

    void PrepareHostData()
    {
        if (p.incx == INT_MIN) {
            return;
        }
        const int kl = isUpper ? 0 : effK;
        const int ku = isUpper ? effK : 0;
        if (p.aFill.method != BlasFillMode::M_NULLPTR) {
            aHost = makeComplexBanded(p.n, p.n, kl, ku, allocLda, p.aFill, p.randomSeed, p.useNormalDistribution);
            StrengthenDiagonal();
        }
        if (p.xFill.method != BlasFillMode::M_NULLPTR) {
            xHost = makeComplexStrided(p.n, p.incx, p.xFill, p.randomSeed + 1, p.useNormalDistribution);
            if (xHost.empty()) {
                xHost.resize(xSize, aclblasComplex{0.0f, 0.0f});
            }
            golden = xHost;
        }
    }

    void StrengthenDiagonal()
    {
        if (aHost.empty() || p.n <= 0) {
            return;
        }
        const int diagRow = isUpper ? effK : 0;
        const float boost = static_cast<float>(std::max(MIN_DIAGONAL_BOOST, p.n));
        for (int j = 0; j < p.n; j++) {
            aclblasComplex& diag = aHost[diagRow + static_cast<size_t>(j) * allocLda];
            diag.real += (diag.real >= 0.0f) ? boost : -boost;
        }
    }

    void PrepareDevice()
    {
        if (needDevA && !aHost.empty()) {
            size_t aBytes = static_cast<size_t>(allocLda) * static_cast<size_t>(allocN) * sizeof(aclblasComplex);
            aDev.emplace(aBytes);
            aDev->copyFromHost(aHost.data(), aBytes);
        }
        if (needDevX && !xHost.empty()) {
            size_t xBytes = xSize * sizeof(aclblasComplex);
            xDev.emplace(xBytes);
            xDev->copyFromHost(xHost.data(), xBytes);
        }
    }

    aclblasStatus_t CallNpu(aclblasHandle_t handle)
    {
        const aclblasComplex* aPtr = aDev.has_value() ? static_cast<const aclblasComplex*>(aDev->ptr()) : nullptr;
        aclblasComplex* xPtr = xDev.has_value() ? static_cast<aclblasComplex*>(xDev->ptr()) : nullptr;
        return aclblasCtbsv_npu(handle, p.uplo, p.trans, p.diag, p.n, p.k, aPtr, p.lda, xPtr, p.incx);
    }

    VerifyConfig GetVerifyConfig() const
    {
        // Legacy CSV column name; supplies rtol/atol, not a MERE/MARE verifier.
        const double tolerance = p.mereThreshold > 0.0 ? p.mereThreshold : DEFAULT_MIXED_TOLERANCE;
        VerifyConfig cfg;
        cfg.mode = PrecisionMode::MIXED_TOLERANCE;
        cfg.mixedRtol = tolerance;
        cfg.mixedAtol = tolerance;
        cfg.mixedRequiredMatchedRatio = REQUIRED_MATCHED_RATIO;
        cfg.mixedMaxAbsErrorLimit = MAX_ABSOLUTE_ERROR;
        cfg.mixedMantissaBits = FP32_MANTISSA_BITS;
        cfg.mixedEmin = FP32_MIN_NORMAL_EXPONENT;
        return cfg;
    }

    void VerifySpecialValues(
        const std::vector<aclblasComplex>& outDense, const std::vector<aclblasComplex>& goldDense,
        const VerifyConfig& realCfg, const VerifyConfig& imagCfg)
    {
        std::vector<aclblasComplex> alignedOut = outDense;
        std::vector<aclblasComplex> alignedGold = goldDense;
        AlignOverflowSolutions(alignedOut, alignedGold);
        const float* alignedOutFloats = reinterpret_cast<const float*>(alignedOut.data());
        const float* alignedGoldFloats = reinterpret_cast<const float*>(alignedGold.data());
        const size_t count = static_cast<size_t>(p.n);
        const bool alignedRealOk = Verifier::verifyVector(
            alignedOutFloats, alignedGoldFloats, count, 2, realCfg, p.caseName + "_special_real");
        const bool alignedImagOk = Verifier::verifyVector(
            alignedOutFloats + 1, alignedGoldFloats + 1, count, 2, imagCfg, p.caseName + "_special_imag");
        EXPECT_TRUE(alignedRealOk && alignedImagOk) << "Special-value propagation mismatch for " << p.caseName;
    }

    void ReportNonFiniteCounts(
        const std::vector<aclblasComplex>& outDense, const std::vector<aclblasComplex>& goldDense)
    {
        const size_t count = static_cast<size_t>(p.n);
        size_t outputNonFinite = 0U;
        size_t goldenNonFinite = 0U;
        for (size_t i = 0; i < count; ++i) {
            if (IsNonFinite(outDense[i].real) || IsNonFinite(outDense[i].imag)) {
                ++outputNonFinite;
            }
            if (IsNonFinite(goldDense[i].real) || IsNonFinite(goldDense[i].imag)) {
                ++goldenNonFinite;
            }
        }
        std::cout << "[" << p.caseName << "] strict diagnostic: output_nonfinite=" << outputNonFinite << "/" << count
                  << " golden_nonfinite=" << goldenNonFinite << "/" << count << std::endl;
    }

    void Verify(aclblasHandle_t handle, aclblasStatus_t ret)
    {
        if (p.expectResult != ACLBLAS_STATUS_SUCCESS) {
            EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
            return;
        }
        ASSERT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_SUCCESS))
            << "Unexpected NPU error code: " << static_cast<int>(ret);
        // TC_PF cases stay registered for msprof. Their performance result is
        // checked by ctbsv_bench, so this path only verifies launch completion.
        if (p.caseName.rfind("TC_PF_", 0) == 0) {
            ASSERT_EQ(aclrtSynchronizeDevice(), ACL_SUCCESS);
            return;
        }
        if (p.n <= 0) {
            return;
        }
        if (xDev.has_value()) {
            aclrtSynchronizeDevice();
            xDev->copyToHost(xHost.data(), xSize * sizeof(aclblasComplex));
        }
        aclblasCtbsv_cpu(handle, p.uplo, p.trans, p.diag, p.n, effK, aHost.data(), allocLda, golden.data(), p.incx);
        std::vector<aclblasComplex> outDense(static_cast<size_t>(p.n));
        std::vector<aclblasComplex> goldDense(static_cast<size_t>(p.n));
        for (int i = 0; i < p.n; ++i) {
            int idx = (p.incx >= 0) ? (i * p.incx) : ((p.n - 1 - i) * absIncx);
            outDense[static_cast<size_t>(i)] = xHost[static_cast<size_t>(idx)];
            goldDense[static_cast<size_t>(i)] = golden[static_cast<size_t>(idx)];
        }
        const VerifyConfig realCfg = GetVerifyConfig();
        VerifyConfig imagCfg = realCfg;

        const float* outFloats = reinterpret_cast<const float*>(outDense.data());
        const float* goldFloats = reinterpret_cast<const float*>(goldDense.data());
        const size_t count = static_cast<size_t>(p.n);
        const bool specialValueCase = HasSpecialValueFill(p.aFill) || HasSpecialValueFill(p.xFill);
        if (specialValueCase) {
            VerifySpecialValues(outDense, goldDense, realCfg, imagCfg);
            return;
        }
        const bool realOk = Verifier::verifyVector(outFloats, goldFloats, count, 2, realCfg, p.caseName + "_real");
        const bool imagOk =
            Verifier::verifyVector(outFloats + 1, goldFloats + 1, count, 2, imagCfg, p.caseName + "_imag");
        if (realOk && imagOk) {
            return;
        }

        ReportNonFiniteCounts(outDense, goldDense);
        EXPECT_TRUE(realOk && imagOk) << "Mixed Tolerance failed for finite case " << p.caseName;
    }
};

TEST_P(CtbsvTest, CsvDriven)
{
    CtbsvFixture f(GetParam());
    f.InitDims();
    f.PrepareHostData();
    f.PrepareDevice();
    const aclblasHandle_t useHandle =
        (GetParam().expectResult == ACLBLAS_STATUS_HANDLE_IS_NULLPTR) ? nullptr : CtbsvTest::handle_;
    aclblasStatus_t ret = f.CallNpu(useHandle);
    f.Verify(useHandle, ret);
}
