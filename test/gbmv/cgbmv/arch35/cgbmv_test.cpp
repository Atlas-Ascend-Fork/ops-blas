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
#include <complex>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "verify.h"
#include "blas_test.h"
#include "csv_loader.h"
#include "fill.h"
#include "cgbmv_param.h"
#include "cgbmv_golden.h"
#include "cgbmv_npu_wrapper.h"

namespace {

// Positions the operator must never read (outside the band, lda padding, x stride holes) hold NaN, so any
// stray read turns the result into NaN. y stride holes hold a sentinel that must survive bit-exactly.
constexpr aclblasComplex POISON = {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN()};
constexpr aclblasComplex Y_HOLE = {-999.0f, 999.0f};

size_t LogicalIndex(int i, int count, int inc)
{
    if (inc > 0) {
        return static_cast<size_t>(i) * static_cast<size_t>(inc);
    }
    return static_cast<size_t>(count - 1 - i) * static_cast<size_t>(inc == 0 ? 1 : -inc);
}

// Real and imaginary parts are independent samples: consecutive generator values, or two normal draws.
std::vector<aclblasComplex> GenerateComplex(size_t count, const CgbmvFill& fill, uint32_t seed)
{
    std::mt19937 rng(seed != 0 ? seed : 42U);
    std::vector<aclblasComplex> v(count);
    if (fill.gauss) {
        std::normal_distribution<float> dist(fill.mu, fill.sigma);
        for (auto& e : v) {
            e.real = dist(rng);
            e.imag = dist(rng);
        }
        return v;
    }
    auto gen = createGenerator(fill.mode, rng);
    for (size_t i = 0; i < count; ++i) {
        v[i].real = gen->at(2 * i);
        v[i].imag = gen->at(2 * i + 1);
    }
    return v;
}

// Band elements are filled from `fill`; everything else (outside the band, lda padding) is POISON.
std::vector<aclblasComplex> MakeBandedA(int m, int n, int kl, int ku, int lda, const CgbmvFill& fill, uint32_t seed)
{
    if (IsNullFill(fill)) {
        return {};
    }
    if (m <= 0 || n <= 0 || lda <= 0) {
        return {POISON};
    }
    std::vector<aclblasComplex> a(static_cast<size_t>(lda) * static_cast<size_t>(n), POISON);
    std::vector<size_t> bandPos;
    for (int j = 0; j < n; ++j) {
        int iStart = std::max(0, j - ku);
        int iEnd = std::min(m - 1, j + kl);
        for (int i = iStart; i <= iEnd; ++i) {
            int bandRow = ku + i - j;
            if (bandRow >= 0 && bandRow < lda) {
                bandPos.push_back(static_cast<size_t>(bandRow) + static_cast<size_t>(j) * lda);
            }
        }
    }
    std::vector<aclblasComplex> vals = GenerateComplex(bandPos.size(), fill, seed);
    for (size_t k = 0; k < bandPos.size(); ++k) {
        a[bandPos[k]] = vals[k];
    }
    return a;
}

std::vector<aclblasComplex> MakeStrided(
    int count, int inc, const CgbmvFill& fill, uint32_t seed, const aclblasComplex& hole)
{
    if (IsNullFill(fill)) {
        return {};
    }
    size_t absInc = static_cast<size_t>(inc == 0 ? 1 : std::abs(inc));
    size_t size = (count > 0) ? (static_cast<size_t>(count) - 1) * absInc + 1 : 1;
    std::vector<aclblasComplex> v(size, hole);
    if (count <= 0) {
        return v;
    }
    std::vector<aclblasComplex> vals = GenerateComplex(static_cast<size_t>(count), fill, seed);
    for (int i = 0; i < count; ++i) {
        v[LogicalIndex(i, count, inc)] = vals[static_cast<size_t>(i)];
    }
    return v;
}

struct CgbmvHostData {
    std::vector<aclblasComplex> a;
    std::vector<aclblasComplex> x;
    std::vector<aclblasComplex> yIn;
    std::vector<aclblasComplex> yOut;
    aclblasComplex alpha{};
    aclblasComplex beta{};
    int yCount = 0;
};

CgbmvHostData PrepareHostData(const CgbmvParam& p)
{
    CgbmvHostData d;
    bool isTransN = (p.trans == ACLBLAS_OP_N);
    int xCount = isTransN ? p.n : p.m;
    d.yCount = isTransN ? p.m : p.n;
    d.a = MakeBandedA(p.m, p.n, p.kl, p.ku, p.lda, p.aFill, p.randomSeed);
    d.x = MakeStrided(xCount, p.incx, p.xFill, p.randomSeed + 1U, POISON);
    d.yIn = MakeStrided(d.yCount, p.incy, p.yFill, p.randomSeed + 2U, Y_HOLE);
    d.yOut = d.yIn;
    d.alpha = {p.alphaReal, p.alphaImag};
    d.beta = {p.betaReal, p.betaImag};
    return d;
}

bool BitwiseEqual(const aclblasComplex& lhs, const aclblasComplex& rhs)
{
    return std::memcmp(&lhs, &rhs, sizeof(aclblasComplex)) == 0;
}

// Every y element outside the logical positions (stride holes) must be left untouched.
bool StrideHolesUnchanged(
    const std::vector<aclblasComplex>& before, const std::vector<aclblasComplex>& after, int count, int inc)
{
    std::vector<bool> logical(before.size(), false);
    for (int i = 0; i < count; ++i) {
        logical[LogicalIndex(i, count, inc)] = true;
    }
    for (size_t k = 0; k < before.size(); ++k) {
        if (!logical[k] && !BitwiseEqual(before[k], after[k])) {
            std::cout << "stride hole " << k << " was modified" << std::endl;
            return false;
        }
    }
    return true;
}

struct ComponentSplit {
    std::vector<float> out;
    std::vector<float> gold;
    size_t nonFiniteMismatch = 0;
};

// Non-finite values have distinct semantics: NaN matches NaN; infinity must also match its sign.
bool NonFiniteMatches(double actual, double reference)
{
    return std::isnan(reference) ? std::isnan(actual) : actual == reference;
}

// Split one component of the logical y elements. Finite reference values go through the precision standard;
// special values are checked separately without treating NaN and the two infinities as interchangeable.
ComponentSplit SplitComponent(
    const std::vector<aclblasComplex>& out, const std::vector<aclblasComplex>& gold, int count, int inc, bool imag)
{
    ComponentSplit s;
    for (int i = 0; i < count; ++i) {
        size_t idx = LogicalIndex(i, count, inc);
        float o = imag ? out[idx].imag : out[idx].real;
        float g = imag ? gold[idx].imag : gold[idx].real;
        if (!std::isfinite(g)) {
            s.nonFiniteMismatch += NonFiniteMatches(o, g) ? 0U : 1U;
            continue;
        }
        s.out.push_back(o);
        s.gold.push_back(g);
    }
    return s;
}

bool VerifyComponent(const ComponentSplit& s, const std::string& name)
{
    if (s.nonFiniteMismatch != 0) {
        std::cout << "[" << name << "] FAILED: " << s.nonFiniteMismatch
                  << " elements disagree with golden's NaN/Inf classification or infinity sign" << std::endl;
        return false;
    }
    if (s.out.empty()) {
        return true;
    }
    // Task spec 3.2 standard per component (MIXED_TOLERANCE: rtol 2^-10, atol 2^-16, matched ratio >= 0.99,
    // per-element limit max(1e-2, 32 ULP)); the CSV mere_threshold / mare_multiplier columns are not used.
    VerifyConfig cfg;
    applyMixedTolerance(cfg, ACL_FLOAT, s.gold.data(), s.gold.size());
    return Verifier::verifyVector(s.out.data(), s.gold.data(), s.out.size(), 1, cfg, name);
}

bool VerifyOutput(
    const std::vector<aclblasComplex>& out, const std::vector<aclblasComplex>& gold, int count, int inc,
    const std::string& caseName)
{
    bool realPass = VerifyComponent(SplitComponent(out, gold, count, inc, false), caseName + "_real");
    bool imagPass = VerifyComponent(SplitComponent(out, gold, count, inc, true), caseName + "_imag");
    return realPass && imagPass;
}

const aclblasComplex* DataOrNull(const std::vector<aclblasComplex>& v) { return v.empty() ? nullptr : v.data(); }

aclblasComplex* DataOrNull(std::vector<aclblasComplex>& v) { return v.empty() ? nullptr : v.data(); }

// ".../cgbmv_test.csv" -> ".../cgbmv_test_supplement.csv"
std::string SupplementCsvPath(const std::string& taskCsv)
{
    const std::string ext = ".csv";
    return taskCsv.substr(0, taskCsv.size() - ext.size()) + "_supplement" + ext;
}

// FP64 diagnostic (the verdict is unchanged): y is recomputed in double precision with cblas_zgbmv on the same
// inputs. NPU and FP32 golden results are each compared with that high-precision reference. Printed for failing
// CSV cases, and for all CSV cases when CGBMV_FP64_STATS=1. "tol" is the error in units of the task's per-element
// tolerance max(1e-2, 32 ULP); "ULP" is taken over elements with |reference| >= 1 (near-zero results make ULPs
// meaningless).
struct ErrStat {
    double npuTol = 0.0;
    double goldTol = 0.0;
    double npuUlp = 0.0;
    double goldUlp = 0.0;
    double npuUlpAtFail = 0.0;
    double goldUlpAtFail = 0.0;
    int failures = 0;
    int referenceNonFinite = 0;
    int referenceOutOfRange = 0;
    int npuCompared = 0;
    int goldCompared = 0;
    int npuNonFiniteMismatch = 0;
    int goldNonFiniteMismatch = 0;
    int npuMatched = 0;
    int goldMatched = 0;
};

double UlpOf(double v)
{
    float f = static_cast<float>(std::fabs(v));
    // nextafter(FLT_MAX, +Inf) would produce an infinite tolerance for a finite FP32 reference.
    int exponent = 0;
    if (f == 0.0f) {
        return std::ldexp(1.0, -149);
    }
    std::frexp(static_cast<double>(f), &exponent);
    return std::ldexp(1.0, std::max(exponent - 1, -126) - 23);
}

// UlpOf is strictly positive and tol is at least 1e-2, so neither denominator can be zero; the explicit test
// keeps every division below safe on its own.
double RatioTo(double diff, double unit) { return unit > 0.0 ? diff / unit : std::numeric_limits<double>::infinity(); }

std::vector<std::complex<double>> Widen(const std::vector<aclblasComplex>& v)
{
    std::vector<std::complex<double>> w(v.size());
    for (size_t i = 0; i < v.size(); ++i) {
        w[i] = {v[i].real, v[i].imag};
    }
    return w;
}

ErrStat CompareWithExact(
    const CgbmvHostData& d, int incy, const std::vector<aclblasComplex>& golden,
    const std::vector<std::complex<double>>& exact, bool imag)
{
    ErrStat st;
    for (int i = 0; i < d.yCount; ++i) {
        size_t k = LogicalIndex(i, d.yCount, incy);
        double e = imag ? exact[k].imag() : exact[k].real();
        double o = imag ? d.yOut[k].imag : d.yOut[k].real;
        double g = imag ? golden[k].imag : golden[k].real;
        if (!std::isfinite(e)) {
            ++st.referenceNonFinite;
            st.npuNonFiniteMismatch += NonFiniteMatches(o, e) ? 0 : 1;
            st.goldNonFiniteMismatch += NonFiniteMatches(g, e) ? 0 : 1;
            continue;
        }
        if (std::fabs(e) > std::numeric_limits<float>::max()) {
            ++st.referenceOutOfRange; // a finite FP64 result need not be representable in FP32
            continue;
        }
        double ulp = UlpOf(e);
        double tol = std::max(1.0e-2, 32.0 * ulp);
        double matchTol = std::ldexp(1.0, -16) + std::ldexp(1.0, -10) * std::fabs(e);
        if (std::isfinite(o)) {
            ++st.npuCompared;
            st.npuMatched += std::fabs(o - e) <= matchTol ? 1 : 0;
            st.npuTol = std::max(st.npuTol, RatioTo(std::fabs(o - e), tol));
            if (std::fabs(e) >= 1.0) {
                st.npuUlp = std::max(st.npuUlp, RatioTo(std::fabs(o - e), ulp));
            }
        } else {
            ++st.npuNonFiniteMismatch;
            st.npuTol = std::numeric_limits<double>::infinity();
        }
        if (std::isfinite(g)) {
            ++st.goldCompared;
            st.goldMatched += std::fabs(g - e) <= matchTol ? 1 : 0;
            st.goldTol = std::max(st.goldTol, RatioTo(std::fabs(g - e), tol));
            if (std::fabs(e) >= 1.0) {
                st.goldUlp = std::max(st.goldUlp, RatioTo(std::fabs(g - e), ulp));
            }
        } else {
            ++st.goldNonFiniteMismatch;
            st.goldTol = std::numeric_limits<double>::infinity();
        }
        if (std::isfinite(o) && std::isfinite(g) && std::fabs(o - g) > std::max(1.0e-2, 32.0 * UlpOf(g))) {
            ++st.failures;
            st.npuUlpAtFail = std::max(st.npuUlpAtFail, RatioTo(std::fabs(o - e), ulp));
            st.goldUlpAtFail = std::max(st.goldUlpAtFail, RatioTo(std::fabs(g - e), ulp));
        }
    }
    return st;
}

bool Fp64StatsEnabled()
{
    static const bool enabled = [] {
        const char* v = std::getenv("CGBMV_FP64_STATS");
        return v != nullptr && std::string(v) == "1";
    }();
    return enabled;
}

void DiagnoseAgainstFp64(const CgbmvParam& p, const CgbmvHostData& d, const std::vector<aclblasComplex>& golden)
{
    std::vector<std::complex<double>> a = Widen(d.a);
    std::vector<std::complex<double>> x = Widen(d.x);
    std::vector<std::complex<double>> exact = Widen(d.yIn);
    const std::complex<double> alpha(p.alphaReal, p.alphaImag);
    const std::complex<double> beta(p.betaReal, p.betaImag);
    cblas_zgbmv(
        CblasColMajor, ToCblasOp(p.trans), p.m, p.n, p.kl, p.ku, &alpha, a.data(), p.lda, x.data(), p.incx, &beta,
        exact.data(), p.incy);
    for (bool imag : {false, true}) {
        ErrStat st = CompareWithExact(d, p.incy, golden, exact, imag);
        std::cout << "[" << p.caseName << (imag ? "_imag" : "_real") << "] FP64 reference: tol NPU " << st.npuTol
                  << " golden " << st.goldTol << "; ULP NPU " << st.npuUlp << " golden " << st.goldUlp
                  << "; over-limit elements " << st.failures << " (ULP NPU " << st.npuUlpAtFail << " golden "
                  << st.goldUlpAtFail << "); finite-compared NPU " << st.npuCompared << " golden " << st.goldCompared
                  << "; reference-nonfinite " << st.referenceNonFinite << "; reference-out-of-fp32-range "
                  << st.referenceOutOfRange << "; nonfinite-mismatch NPU " << st.npuNonFiniteMismatch << " golden "
                  << st.goldNonFiniteMismatch << "; matched NPU " << st.npuMatched << " golden " << st.goldMatched
                  << " / representable-reference " << d.yCount - st.referenceNonFinite - st.referenceOutOfRange
                  << std::endl;
    }
}

} // namespace

class CgbmvTest : public BlasTest<CgbmvParam> {};

TEST(CgbmvVerifierTest, NonFiniteClassificationAndSign)
{
    const float inf = std::numeric_limits<float>::infinity();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    for (float g : {inf, -inf, nan}) {
        for (float o : {0.0f, inf, -inf, nan}) {
            bool expected = std::isnan(g) ? std::isnan(o) : o == g;
            ComponentSplit s = SplitComponent({{o, o}}, {{g, g}}, 1, 1, false);
            EXPECT_EQ(s.nonFiniteMismatch == 0, expected);
            s = SplitComponent({{o, o}}, {{g, g}}, 1, 1, true);
            EXPECT_EQ(s.nonFiniteMismatch == 0, expected);
        }
    }
}

TEST(CgbmvVerifierTest, Fp64DiagnosticCountsNonFiniteOutputs)
{
    CgbmvHostData d;
    d.yCount = 1;
    d.yOut = {POISON};
    ErrStat st = CompareWithExact(d, 1, {{1.0f, 1.0f}}, {{1.0, 1.0}}, false);
    EXPECT_EQ(st.npuCompared, 0);
    EXPECT_EQ(st.goldCompared, 1);
    EXPECT_EQ(st.npuNonFiniteMismatch, 1);
    EXPECT_EQ(st.goldNonFiniteMismatch, 0);
    EXPECT_TRUE(std::isinf(st.npuTol));
    EXPECT_EQ(st.goldMatched, 1);
    st = CompareWithExact(d, 1, {{1.0f, 1.0f}}, {{1.0e100, 1.0e100}}, false);
    EXPECT_EQ(st.referenceOutOfRange, 1);
    EXPECT_EQ(st.npuCompared, 0);
    EXPECT_DOUBLE_EQ(UlpOf(std::numeric_limits<float>::max()), std::ldexp(1.0, 104));
}

namespace {

struct OverflowShape {
    int n;
    int secondCol;
    int incx;
    int incy;
    float value;
    float beta;
};

// Preserve cancellation with beta*y when a parallel partial sum overflows before the final store.
void RunOverflowCancellation(aclblasHandle_t handle, const OverflowShape& s)
{
    SCOPED_TRACE("n=" + std::to_string(s.n) + " secondCol=" + std::to_string(s.secondCol));
    const int lda = s.n;
    std::vector<aclblasComplex> a(static_cast<size_t>(lda) * s.n, POISON);
    for (int j = 0; j < s.n; ++j) {
        a[static_cast<size_t>(j) * lda + s.n - 1 - j] = {(j == 0 || j == s.secondCol) ? s.value : 0.0f, 0.0f};
    }
    std::vector<aclblasComplex> x(1 + static_cast<size_t>(s.n - 1) * std::abs(s.incx), POISON);
    for (int j = 0; j < s.n; ++j) {
        x[LogicalIndex(j, s.n, s.incx)] = {1.0f, 0.0f};
    }
    std::vector<aclblasComplex> y{{-3.0e38f, 0.0f}};
    std::vector<aclblasComplex> golden = y;
    const aclblasComplex alpha{1.0f, 0.0f};
    const aclblasComplex beta{s.beta, 0.0f};
    ASSERT_EQ(
        aclblasCgbmv_cpu(
            handle, ACLBLAS_OP_N, 1, s.n, 0, s.n - 1, &alpha, a.data(), lda, x.data(), s.incx, &beta, golden.data(),
            s.incy),
        ACLBLAS_STATUS_SUCCESS);
    ASSERT_TRUE(std::isfinite(golden[0].real));
    ASSERT_EQ(
        aclblasCgbmv_npu(
            handle, ACLBLAS_OP_N, 1, s.n, 0, s.n - 1, &alpha, a.data(), lda, x.data(), s.incx, &beta, y.data(), s.incy),
        ACLBLAS_STATUS_SUCCESS);
    std::cout << "Overflow regression: actual=(" << y[0].real << "," << y[0].imag << ") golden=(" << golden[0].real
              << "," << golden[0].imag << ")" << std::endl;
    EXPECT_TRUE(VerifyOutput(y, golden, 1, s.incy, "OverflowCancellation"));
}

} // namespace

TEST_F(CgbmvTest, NShortFiniteResultSurvivesOverflow)
{
    // n = 2 keeps the column window below CGBMV_N_STEPPED_MIN_COLS, so this runs the short-slice VF.
    RunOverflowCancellation(handle_, {2, 1, 1, 1, 2.0e38f, 1.0f});
}

TEST_F(CgbmvTest, NLongFiniteResultSurvivesOverflow)
{
    RunOverflowCancellation(handle_, {512, 16, -2, -3, 2.0e38f, 1.0f});
}

TEST_F(CgbmvTest, NFiniteResultSurvivesOverflowWithinSlice)
{
    RunOverflowCancellation(handle_, {512, 1, 2, 3, 2.0e38f, 1.0f});
}

TEST_F(CgbmvTest, NFiniteResultSurvivesOverflowGeneralBeta)
{
    RunOverflowCancellation(handle_, {2, 1, 1, 1, 1.8e38f, 0.5f});
}

namespace {

// One call on the NPU and on the golden with the same inputs; y is judged with the strict NaN/Inf rule.
void RunAgainstGolden(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, int kl, int ku, aclblasComplex alpha,
    std::vector<aclblasComplex> a, int lda, std::vector<aclblasComplex> x, int incx, aclblasComplex beta,
    std::vector<aclblasComplex> y, int incy, const std::string& name)
{
    SCOPED_TRACE(name);
    const std::vector<aclblasComplex> yIn = y;
    std::vector<aclblasComplex> golden = y;
    const int yLen = (trans == ACLBLAS_OP_N) ? m : n;
    ASSERT_EQ(
        aclblasCgbmv_cpu(
            handle, trans, m, n, kl, ku, &alpha, a.data(), lda, x.data(), incx, &beta, golden.data(), incy),
        ACLBLAS_STATUS_SUCCESS);
    ASSERT_EQ(
        aclblasCgbmv_npu(handle, trans, m, n, kl, ku, &alpha, a.data(), lda, x.data(), incx, &beta, y.data(), incy),
        ACLBLAS_STATUS_SUCCESS);
    for (int i = 0; i < yLen; ++i) {
        size_t k = LogicalIndex(i, yLen, incy);
        std::cout << name << " y[" << i << "]: actual=(" << y[k].real << "," << y[k].imag << ") golden=("
                  << golden[k].real << "," << golden[k].imag << ")" << std::endl;
    }
    EXPECT_TRUE(StrideHolesUnchanged(yIn, y, yLen, incy));
    bool ok = VerifyOutput(y, golden, yLen, incy, name);
    if (!ok || Fp64StatsEnabled()) {
        CgbmvParam p(csv_map{});
        p.caseName = name;
        p.trans = trans;
        p.m = m;
        p.n = n;
        p.kl = kl;
        p.ku = ku;
        p.lda = lda;
        p.incx = incx;
        p.incy = incy;
        p.alphaReal = alpha.real;
        p.alphaImag = alpha.imag;
        p.betaReal = beta.real;
        p.betaImag = beta.imag;
        CgbmvHostData d{a, x, yIn, y, alpha, beta, yLen};
        DiagnoseAgainstFp64(p, d, golden);
    }
    EXPECT_TRUE(ok); // FP64 is diagnostic only; keep the original Golden verdict, including overflow differences.
}

} // namespace

// (1+0i)*(Inf+Inf*i) is (Inf, Inf) in the reference but NaN with the plain complex product.
TEST_F(CgbmvTest, NComplexInfinityPropagation)
{
    const float inf = std::numeric_limits<float>::infinity();
    RunAgainstGolden(
        handle_, ACLBLAS_OP_N, 1, 1, 0, 0, {1.0f, 0.0f}, {{1.0f, 0.0f}}, 1, {{inf, inf}}, 1, {0.0f, 0.0f}, {POISON}, 1,
        "NInfX");
}

TEST_F(CgbmvTest, TCComplexInfinityPropagation)
{
    const float inf = std::numeric_limits<float>::infinity();
    const aclblasComplex one{1.0f, 0.0f};
    const aclblasComplex zero{0.0f, 0.0f};
    for (aclblasOperation_t trans : {ACLBLAS_OP_T, ACLBLAS_OP_C}) {
        const std::string tag = (trans == ACLBLAS_OP_T) ? "T" : "C";
        RunAgainstGolden(handle_, trans, 1, 1, 0, 0, one, {one}, 1, {{inf, inf}}, 1, zero, {POISON}, 1, tag + "InfX");
        // four band rows reduced by four lanes, one of the terms infinite
        RunAgainstGolden(
            handle_, trans, 4, 1, 3, 0, one, {one, one, one, one}, 4, {{inf, inf}, one, one, one}, 1, zero, {POISON}, 1,
            tag + "InfXInGroup");
    }
}

// T/C: the lanes' partial sums overflow (Inf + -Inf = NaN) while the reference's sequential sum stays finite.
TEST_F(CgbmvTest, TCFiniteResultSurvivesOverflow)
{
    const float big = 2.0e38f;
    const aclblasComplex one{1.0f, 0.0f};
    const aclblasComplex zero{0.0f, 0.0f};
    for (aclblasOperation_t trans : {ACLBLAS_OP_T, ACLBLAS_OP_C}) {
        const std::string tag = (trans == ACLBLAS_OP_T) ? "T" : "C";
        // four rows, one per lane: the shuffle adds rows 0 + 2 and 1 + 3 first
        RunAgainstGolden(
            handle_, trans, 4, 1, 3, 0, one, {{big, 0.0f}, {-big, 0.0f}, {big, 0.0f}, {-big, 0.0f}}, 4,
            {one, one, one, one}, 1, zero, {POISON}, 1, tag + "OverflowInShuffle");
        // 64 rows over 32 lanes: lane 0 adds rows 0 and 32, lane 1 rows 1 and 33; negative incx
        std::vector<aclblasComplex> a(64, zero);
        a[0] = a[32] = {big, 0.0f};
        a[1] = a[33] = {-big, 0.0f};
        const int incx = -2;
        std::vector<aclblasComplex> x(1 + 63 * 2, POISON);
        for (int i = 0; i < 64; ++i) {
            x[LogicalIndex(i, 64, incx)] = one;
        }
        RunAgainstGolden(
            handle_, trans, 64, 1, 63, 0, one, a, 64, x, incx, one, {{0.5f, 0.0f}}, 1, tag + "OverflowWithinLane");
    }
}

// beta * y with an infinite y: (2+0i)*(Inf+Inf*i) is (Inf, Inf) in the reference, NaN with the plain product.
TEST_F(CgbmvTest, BetaTimesInfiniteY)
{
    const float inf = std::numeric_limits<float>::infinity();
    const aclblasComplex yInf{inf, inf};
    const aclblasComplex beta{2.0f, 0.0f};
    const aclblasComplex one{1.0f, 0.0f};
    const aclblasComplex zero{0.0f, 0.0f};
    // N: row 0 holds the band, rows 1 and 2 lie below it (y = beta * y only)
    RunAgainstGolden(
        handle_, ACLBLAS_OP_N, 3, 1, 0, 0, one, {one}, 1, {one}, 1, beta, {yInf, yInf, yInf}, 1, "NBetaInfY");
    RunAgainstGolden(
        handle_, ACLBLAS_OP_T, 1, 2, 0, 1, one, {POISON, one, one, POISON}, 2, {one}, 1, beta, {yInf, yInf}, 1,
        "TBetaInfY");
    // alpha = 0: y = beta * y without reading A or x
    RunAgainstGolden(
        handle_, ACLBLAS_OP_N, 2, 2, 0, 0, zero, {one, one}, 1, {one, one}, 1, beta, {yInf, yInf}, 1,
        "AlphaZeroBetaInfY");
}

namespace {

// Every column dot is finite; only adding finite beta*y overflows. Use two columns so negative incy is exercised.
void RunTCFinalAdditionOverflow(aclblasHandle_t handle, aclblasOperation_t trans, aclblasComplex beta, int incx)
{
    constexpr int rows = 64;
    constexpr int cols = 2;
    constexpr int lda = 65;
    constexpr int incy = -3;
    const aclblasComplex one{1.0f, 0.0f};
    const aclblasComplex zero{0.0f, 0.0f};
    std::vector<aclblasComplex> a(static_cast<size_t>(lda) * cols, POISON);
    for (int col = 0; col < cols; ++col) {
        for (int row = 0; row < rows; ++row) {
            a[static_cast<size_t>(col) * lda + 1 + row - col] = row == 0 ? aclblasComplex{2.0e38f, 0.0f} : zero;
        }
    }
    std::vector<aclblasComplex> x(1 + static_cast<size_t>(rows - 1) * std::abs(incx), POISON);
    for (int row = 0; row < rows; ++row) {
        x[LogicalIndex(row, rows, incx)] = one;
    }
    std::vector<aclblasComplex> y(1 + (cols - 1) * (-incy), Y_HOLE);
    // Both beta=(1,0) and beta=(0.5,0.5) give a finite real contribution of 2e38.
    const aclblasComplex yValue = beta.imag == 0.0f ? aclblasComplex{2.0e38f, 0.0f} : aclblasComplex{2.0e38f, -2.0e38f};
    for (int col = 0; col < cols; ++col) {
        y[LogicalIndex(col, cols, incy)] = yValue;
    }
    const std::string name = std::string(trans == ACLBLAS_OP_T ? "T" : "C") +
                             "FinalAdditionOverflow_betaImag=" + std::to_string(beta.imag) +
                             "_incx=" + std::to_string(incx);
    RunAgainstGolden(handle, trans, rows, cols, rows - 1, 1, one, a, lda, x, incx, beta, y, incy, name);
}

} // namespace

TEST_F(CgbmvTest, TCFinalAdditionOverflow)
{
    for (aclblasOperation_t trans : {ACLBLAS_OP_T, ACLBLAS_OP_C}) {
        for (aclblasComplex beta : {aclblasComplex{1.0f, 0.0f}, aclblasComplex{0.5f, 0.5f}}) {
            for (int incx : {1, -2}) {
                RunTCFinalAdditionOverflow(handle_, trans, beta, incx);
            }
        }
    }
}

TEST_F(CgbmvTest, NullHandle)
{
    std::vector<aclblasComplex> a(12);
    std::vector<aclblasComplex> x(4);
    std::vector<aclblasComplex> y(4);
    aclblasComplex alpha = {1.0f, 0.0f};
    aclblasComplex beta = {0.0f, 0.0f};
    aclblasStatus_t ret =
        aclblasCgbmv_npu(nullptr, ACLBLAS_OP_N, 4, 4, 1, 1, &alpha, a.data(), 3, x.data(), 1, &beta, y.data(), 1);
    EXPECT_EQ(ret, ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
}

// alpha = (0,0): y = beta * y must not read A or x, so NaN-filled A/x must not leak into y.
TEST_F(CgbmvTest, AlphaZeroDoesNotReadAX)
{
    const int m = 64;
    const int n = 48;
    const int kl = 5;
    const int ku = 7;
    const int lda = kl + ku + 1;
    std::vector<aclblasComplex> a(static_cast<size_t>(lda) * n, POISON);
    std::vector<aclblasComplex> x(n, POISON);
    std::vector<aclblasComplex> y(m);
    for (int i = 0; i < m; ++i) {
        y[i] = {static_cast<float>(i) * 0.25f, 1.0f - static_cast<float>(i) * 0.5f};
    }
    std::vector<aclblasComplex> golden = y;
    aclblasComplex alpha = {0.0f, 0.0f};
    aclblasComplex beta = {0.5f, -1.5f};
    ASSERT_EQ(
        aclblasCgbmv_npu(
            CgbmvTest::handle_, ACLBLAS_OP_N, m, n, kl, ku, &alpha, a.data(), lda, x.data(), 1, &beta, y.data(), 1),
        ACLBLAS_STATUS_SUCCESS);
    ASSERT_EQ(
        aclblasCgbmv_cpu(
            CgbmvTest::handle_, ACLBLAS_OP_N, m, n, kl, ku, &alpha, a.data(), lda, x.data(), 1, &beta, golden.data(),
            1),
        ACLBLAS_STATUS_SUCCESS);
    EXPECT_TRUE(VerifyOutput(y, golden, m, 1, "AlphaZeroDoesNotReadAX"));
}

// ---------------------------------------------------------------------------------------------------------------
// alpha / beta in device memory (task spec: alpha and beta may live in Host or Device memory)
// ---------------------------------------------------------------------------------------------------------------
namespace {

struct ScalarCase {
    aclblasOperation_t trans;
    aclblasComplex alpha;
    aclblasComplex beta;
    bool alphaOnDevice;
    bool betaOnDevice;
    bool poisonAX; // fill all of A and x with NaN: any read of A or x becomes visible in y
    bool poisonY;  // fill the old y with NaN: checks that beta == 0 does not read y
};

// Runs one band problem (m=67, n=45, kl=6, ku=9, lda padded by 3, incx=-2, incy=3) and compares with golden.
bool RunScalarPlacementCase(aclblasHandle_t handle, const ScalarCase& c, const std::string& name)
{
    const int m = 67;
    const int n = 45;
    const int kl = 6;
    const int ku = 9;
    const int lda = kl + ku + 1 + 3;
    const int incx = -2;
    const int incy = 3;
    const bool isTransN = (c.trans == ACLBLAS_OP_N);
    const int yCount = isTransN ? m : n;
    const CgbmvFill rnd = ParseCgbmvFill("RANDOM_NORM_5_5");
    std::vector<aclblasComplex> a = MakeBandedA(m, n, kl, ku, lda, rnd, 20260914U);
    if (c.poisonAX) {
        std::fill(a.begin(), a.end(), POISON);
    }
    std::vector<aclblasComplex> x = MakeStrided(isTransN ? n : m, incx, rnd, 20260915U, POISON);
    if (c.poisonAX) {
        std::fill(x.begin(), x.end(), POISON);
    }
    std::vector<aclblasComplex> yIn = MakeStrided(yCount, incy, rnd, 20260916U, Y_HOLE);
    if (c.poisonY) {
        for (int i = 0; i < yCount; ++i) {
            yIn[LogicalIndex(i, yCount, incy)] = POISON;
        }
    }
    std::vector<aclblasComplex> yOut = yIn;
    std::vector<aclblasComplex> golden = yIn;
    CgbmvNpuOptions opt;
    opt.alphaOnDevice = c.alphaOnDevice;
    opt.betaOnDevice = c.betaOnDevice;
    aclblasStatus_t ret = aclblasCgbmv_npu(
        handle, c.trans, m, n, kl, ku, &c.alpha, a.data(), lda, x.data(), incx, &c.beta, yOut.data(), incy, &opt);
    if (ret != ACLBLAS_STATUS_SUCCESS) {
        std::cout << "[" << name << "] aclblasCgbmv returned " << static_cast<int>(ret) << std::endl;
        return false;
    }
    const aclblasStatus_t cpu = aclblasCgbmv_cpu(
        handle, c.trans, m, n, kl, ku, &c.alpha, a.data(), lda, x.data(), incx, &c.beta, golden.data(), incy);
    if (cpu != ACLBLAS_STATUS_SUCCESS) {
        std::cout << "[" << name << "] golden returned an error" << std::endl;
        return false;
    }
    bool holes = StrideHolesUnchanged(yIn, yOut, yCount, incy);
    bool isQuickReturn = c.alpha.real == 0.0f && c.alpha.imag == 0.0f && c.beta.real == 1.0f && c.beta.imag == 0.0f;
    if (isQuickReturn) {
        return holes && std::equal(yIn.begin(), yIn.end(), yOut.begin(), BitwiseEqual);
    }
    return holes && VerifyOutput(yOut, golden, yCount, incy, name);
}

const char* TransTag(aclblasOperation_t t) { return t == ACLBLAS_OP_N ? "N" : (t == ACLBLAS_OP_T ? "T" : "C"); }

} // namespace

// General alpha/beta with every placement combination and every op(A).
TEST_F(CgbmvTest, ScalarPlacementMatchesGolden)
{
    const aclblasComplex alpha = {0.75f, -1.25f};
    const aclblasComplex beta = {-0.5f, 0.3f};
    constexpr bool placements[3][2] = {{true, true}, {true, false}, {false, true}};
    for (aclblasOperation_t t : {ACLBLAS_OP_N, ACLBLAS_OP_T, ACLBLAS_OP_C}) {
        for (const auto& pl : placements) {
            std::string name = std::string("ScalarPlacement_") + TransTag(t) + (pl[0] ? "_alphaDev" : "_alphaHost") +
                               (pl[1] ? "_betaDev" : "_betaHost");
            ScalarCase c{t, alpha, beta, pl[0], pl[1], false, false};
            EXPECT_TRUE(RunScalarPlacementCase(CgbmvTest::handle_, c, name)) << name;
        }
    }
}

// Device alpha == 0 and beta == 1: quick return decided in the kernel, y stays bit-identical, A/x unread.
TEST_F(CgbmvTest, DeviceScalarQuickReturn)
{
    for (aclblasOperation_t t : {ACLBLAS_OP_N, ACLBLAS_OP_T}) {
        ScalarCase c{t, {0.0f, 0.0f}, {1.0f, 0.0f}, true, true, true, false};
        EXPECT_TRUE(RunScalarPlacementCase(CgbmvTest::handle_, c, std::string("QuickReturn_") + TransTag(t)));
    }
}

// Device alpha == 0: y = beta * y without reading A/x (A and x are all NaN).
TEST_F(CgbmvTest, DeviceAlphaZeroDoesNotReadAX)
{
    for (aclblasOperation_t t : {ACLBLAS_OP_N, ACLBLAS_OP_C}) {
        ScalarCase c{t, {0.0f, 0.0f}, {0.5f, -1.5f}, true, false, true, false};
        EXPECT_TRUE(RunScalarPlacementCase(CgbmvTest::handle_, c, std::string("DevAlphaZero_") + TransTag(t)));
    }
}

// Device beta == 0: the old y (all NaN) must not be read.
TEST_F(CgbmvTest, DeviceBetaZeroDoesNotReadY)
{
    for (aclblasOperation_t t : {ACLBLAS_OP_N, ACLBLAS_OP_T}) {
        ScalarCase c{t, {1.0f, 0.5f}, {0.0f, 0.0f}, false, true, false, true};
        EXPECT_TRUE(RunScalarPlacementCase(CgbmvTest::handle_, c, std::string("DevBetaZero_") + TransTag(t)));
    }
}

namespace {

// Runs trans N with unit strides on both backends and compares y with the golden; shared by the fixed-input
// trans-N cases below.
bool RunNUnitStrideCase(
    aclblasHandle_t handle, int m, int n, int kl, int ku, int lda, const aclblasComplex& alpha,
    const aclblasComplex& beta, const std::vector<aclblasComplex>& a, const std::vector<aclblasComplex>& x,
    std::vector<aclblasComplex>& y, const std::string& name)
{
    std::vector<aclblasComplex> golden = y;
    aclblasStatus_t npu =
        aclblasCgbmv_npu(handle, ACLBLAS_OP_N, m, n, kl, ku, &alpha, a.data(), lda, x.data(), 1, &beta, y.data(), 1);
    aclblasStatus_t cpu = aclblasCgbmv_cpu(
        handle, ACLBLAS_OP_N, m, n, kl, ku, &alpha, a.data(), lda, x.data(), 1, &beta, golden.data(), 1);
    if (npu != ACLBLAS_STATUS_SUCCESS || cpu != ACLBLAS_STATUS_SUCCESS) {
        std::cout << "[" << name << "] npu=" << static_cast<int>(npu) << " cpu=" << static_cast<int>(cpu) << std::endl;
        return false;
    }
    return VerifyOutput(y, golden, m, 1, name);
}

// trans N, alpha = (Inf, 0), m > n + kl: rows from n + kl on hold no band element, so as in the reference they only
// get beta * y and must stay finite; rows inside the band are non-finite in both results.
bool RunRowsBelowBandCase(aclblasHandle_t handle, int m, int n, int kl, int ku, const std::string& name)
{
    const int lda = kl + ku + 1;
    const CgbmvFill rnd = ParseCgbmvFill("RANDOM_NORM_5_5");
    std::vector<aclblasComplex> a = MakeBandedA(m, n, kl, ku, lda, rnd, 20260917U);
    std::vector<aclblasComplex> x = MakeStrided(n, 1, rnd, 20260918U, POISON);
    std::vector<aclblasComplex> y = MakeStrided(m, 1, rnd, 20260919U, Y_HOLE);
    const aclblasComplex alpha = {std::numeric_limits<float>::infinity(), 0.0f};
    const aclblasComplex beta = {0.5f, -1.5f};
    return RunNUnitStrideCase(handle, m, n, kl, ku, lda, alpha, beta, a, x, y, name);
}

} // namespace

// Rows below the band must not pick up alpha * 0 (NaN for an infinite alpha). Long-slice kernel: n = 300,
// window 300 -> 32 slices of 10 columns; short-slice kernel: n = 40. n + kl is not a multiple of 32 in both.
TEST_F(CgbmvTest, NRowsBelowBandSkipAlpha)
{
    EXPECT_TRUE(RunRowsBelowBandCase(CgbmvTest::handle_, 400, 300, 30, 240, "RowsBelowBand_long"));
    EXPECT_TRUE(RunRowsBelowBandCase(CgbmvTest::handle_, 120, 40, 5, 4, "RowsBelowBand_short"));
}

namespace {

// trans N on the long-slice kernel (301 x 301, kl = ku = 150: 32 slices of 10 columns) with A = 1e10, x = 1e30 and
// alpha = 1e-30. The reference scales x by alpha first (terms of 1e10, sums ~3e12); A * x alone (1e40) would
// overflow FP32, so the kernel must use the same order and match the finite golden.
bool RunAlphaScalesXFirstCase(aclblasHandle_t handle, const std::string& name)
{
    const int m = 301;
    const int n = 301;
    const int kl = 150;
    const int ku = 150;
    const int lda = kl + ku + 1;
    std::vector<aclblasComplex> a(static_cast<size_t>(lda) * n, POISON);
    for (int j = 0; j < n; ++j) {
        for (int i = std::max(0, j - ku); i <= std::min(m - 1, j + kl); ++i) {
            a[static_cast<size_t>(ku + i - j) + static_cast<size_t>(j) * lda] = {1.0e10f, 0.0f};
        }
    }
    std::vector<aclblasComplex> x(n, {1.0e30f, 0.0f});
    std::vector<aclblasComplex> y(m, {0.0f, 0.0f});
    const aclblasComplex alpha = {1.0e-30f, 0.0f};
    const aclblasComplex beta = {0.0f, 0.0f};
    return RunNUnitStrideCase(handle, m, n, kl, ku, lda, alpha, beta, a, x, y, name);
}

} // namespace

// Long-slice kernel must scale x by alpha before multiplying by A (reference order), or in-range data overflows.
TEST_F(CgbmvTest, NLongSliceScalesXByAlphaFirst)
{
    EXPECT_TRUE(RunAlphaScalesXFirstCase(CgbmvTest::handle_, "AlphaScalesXFirst"));
}

// Task-provided cases (cgbmv_test.csv, kept byte-identical to the task package).
INSTANTIATE_TEST_SUITE_P(
    Cgbmv, CgbmvTest, ::testing::ValuesIn(GetCasesFromCsv<CgbmvParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<CgbmvParam>);

// Supplementary cases for task spec 3.5 scenarios the task CSV does not cover (normal-distribution data,
// pure-imaginary / large alpha and beta, alignment offsets, normal-distribution mirror of the task cases);
// optional columns are described in cgbmv_param.h.
INSTANTIATE_TEST_SUITE_P(
    CgbmvSup, CgbmvTest,
    ::testing::ValuesIn(GetCasesFromCsv<CgbmvParam>(SupplementCsvPath(ReplaceFileExtension2Csv(__FILE__)))),
    PrintCaseInfoString<CgbmvParam>);

// Shared body for the CSV-driven cases (task CSV and supplementary CSV).
void RunCsvDrivenCase(aclblasHandle_t handle, const CgbmvParam& p)
{
    CgbmvHostData d = PrepareHostData(p);
    const aclblasComplex* alphaPtr = p.nullAlpha ? nullptr : &d.alpha;
    const aclblasComplex* betaPtr = p.nullBeta ? nullptr : &d.beta;

    CgbmvNpuOptions opt;
    opt.aAlignOffset = p.aAlignOffset;
    opt.xAlignOffset = p.xAlignOffset;
    opt.yAlignOffset = p.yAlignOffset;
    aclblasStatus_t ret = aclblasCgbmv_npu(
        handle, p.trans, p.m, p.n, p.kl, p.ku, alphaPtr, DataOrNull(d.a), p.lda, DataOrNull(d.x), p.incx, betaPtr,
        DataOrNull(d.yOut), p.incy, &opt);
    if (p.expectResult != ACLBLAS_STATUS_SUCCESS) {
        EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
        return;
    }
    ASSERT_EQ(ret, ACLBLAS_STATUS_SUCCESS);

    std::vector<aclblasComplex> golden = d.yIn;
    ASSERT_EQ(
        aclblasCgbmv_cpu(
            handle, p.trans, p.m, p.n, p.kl, p.ku, alphaPtr, DataOrNull(d.a), p.lda, DataOrNull(d.x), p.incx, betaPtr,
            DataOrNull(golden), p.incy),
        ACLBLAS_STATUS_SUCCESS);

    EXPECT_TRUE(StrideHolesUnchanged(d.yIn, d.yOut, d.yCount, p.incy)) << p.caseName;
    EXPECT_TRUE(opt.yPrefixIntact) << p.caseName << ": memory before y was written";
    if (p.m <= 0 || p.n <= 0) {
        EXPECT_TRUE(std::equal(d.yIn.begin(), d.yIn.end(), d.yOut.begin(), BitwiseEqual)) << p.caseName;
        return;
    }
    bool ok = VerifyOutput(d.yOut, golden, d.yCount, p.incy, p.caseName);
    if (!ok || Fp64StatsEnabled()) {
        DiagnoseAgainstFp64(p, d, golden); // extra evidence only; the verdict is the one judged above
    }
    EXPECT_TRUE(ok);
}

TEST_P(CgbmvTest, CsvDriven) { RunCsvDrivenCase(CgbmvTest::handle_, GetParam()); }
