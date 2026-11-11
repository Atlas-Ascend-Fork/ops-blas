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
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "verify.h"
#include "blas_test.h"
#include "csv_loader.h"
#include "chbmv_param.h"
#include "chbmv_golden.h"
#include "chbmv_npu_wrapper.h"

namespace {

std::vector<aclblasComplex> PackComplexVec(const std::vector<float>& src)
{
    size_t n = src.size() / 2;
    std::vector<aclblasComplex> dst(n);
    for (size_t i = 0; i < n; ++i) {
        dst[i].real = src[2 * i];
        dst[i].imag = src[2 * i + 1];
    }
    return dst;
}

// Section 3.2: Inf/NaN equivalence rule + FLOAT32 mixed tolerance
// (rtol = 2^-13, atol = 2^-13, max_abs <= max(1e-2, 32*ULP))
bool VerifyChbmvComponent(
    const std::vector<aclblasComplex>& actual, const std::vector<aclblasComplex>& golden, bool imagPart,
    const std::string& caseName)
{
    constexpr double kRtol = 0.0001220703125; // 2^-13
    constexpr double kAtol = 0.0001220703125; // 2^-13
    constexpr double kReqRatio = 0.99;
    constexpr double kMaxAbsFixed = 0.01;

    size_t finiteTotal = 0;
    size_t finiteMatched = 0;
    size_t nonFiniteTotal = 0;
    size_t nonFinitePassed = 0;
    bool maxErrorPassed = true;
    double maxAbsErr = 0.0;

    for (size_t i = 0; i < actual.size(); ++i) {
        float act = imagPart ? actual[i].imag : actual[i].real;
        float gld = imagPart ? golden[i].imag : golden[i].real;

        if (!std::isfinite(act) || !std::isfinite(gld)) {
            ++nonFiniteTotal;
            // Section 3.2: both non-finite (same NaN, same-sign Inf, or one NaN and one +-Inf) is equivalent
            if (!std::isfinite(act) && !std::isfinite(gld)) {
                if (std::isnan(act) || std::isnan(gld) || (act == gld)) {
                    ++nonFinitePassed;
                }
            }
            continue;
        }

        ++finiteTotal;
        double err = std::abs(static_cast<double>(act) - static_cast<double>(gld));
        maxAbsErr = std::max(maxAbsErr, err);
        double tol = kAtol + kRtol * std::abs(static_cast<double>(gld));
        if (err <= tol) {
            ++finiteMatched;
        }
        double ulp32 = 32.0 * getUlpsAt(static_cast<double>(gld), 23, -126);
        if (err > std::max(kMaxAbsFixed, ulp32)) {
            maxErrorPassed = false;
        }
    }

    double matchedRatio =
        (finiteTotal == 0) ? 1.0 : (static_cast<double>(finiteMatched) / static_cast<double>(finiteTotal));
    bool nonFiniteOk = (nonFinitePassed == nonFiniteTotal);
    bool passed = (matchedRatio >= kReqRatio) && maxErrorPassed && nonFiniteOk;
    if (!passed) {
        std::cout << "[" << caseName << (imagPart ? "_imag" : "_real") << "] FAILED: matchedRatio=" << matchedRatio
                  << " maxAbsErr=" << maxAbsErr << " nonFinite=" << nonFinitePassed << "/" << nonFiniteTotal
                  << std::endl;
    }
    return passed;
}

} // namespace

class ChbmvTest : public BlasTest<ChbmvParam> {
protected:
    static void* perfA_;
    static void* perfX_;
    static void* perfY_;
    static constexpr size_t kMaxPerfBytesA = 64ULL * 1024 * 1024;
    static constexpr size_t kMaxPerfBytesXY = 2ULL * 1024 * 1024;

    static void SetUpTestSuite()
    {
        BlasTest<ChbmvParam>::SetUpTestSuite();
        // The performance buffers are shared by every case, so a failed allocation must abort the
        // suite instead of leaving a null pointer for the later memset and kernel launches.
        ASSERT_EQ(aclrtMalloc(&perfA_, kMaxPerfBytesA, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
        ASSERT_EQ(aclrtMalloc(&perfX_, kMaxPerfBytesXY, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
        ASSERT_EQ(aclrtMalloc(&perfY_, kMaxPerfBytesXY, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
        ASSERT_EQ(aclrtMemset(perfA_, kMaxPerfBytesA, 0, kMaxPerfBytesA), ACL_SUCCESS);
        ASSERT_EQ(aclrtMemset(perfX_, kMaxPerfBytesXY, 0, kMaxPerfBytesXY), ACL_SUCCESS);
        ASSERT_EQ(aclrtMemset(perfY_, kMaxPerfBytesXY, 0, kMaxPerfBytesXY), ACL_SUCCESS);
    }

    static void TearDownTestSuite()
    {
        if (perfA_ != nullptr) {
            aclrtFree(perfA_);
            perfA_ = nullptr;
        }
        if (perfX_ != nullptr) {
            aclrtFree(perfX_);
            perfX_ = nullptr;
        }
        if (perfY_ != nullptr) {
            aclrtFree(perfY_);
            perfY_ = nullptr;
        }
        BlasTest<ChbmvParam>::TearDownTestSuite();
    }
};

void* ChbmvTest::perfA_ = nullptr;
void* ChbmvTest::perfX_ = nullptr;
void* ChbmvTest::perfY_ = nullptr;

TEST_F(ChbmvTest, NullHandle)
{
    aclblasComplex alpha{1.0f, 0.0f};
    aclblasComplex beta{0.0f, 0.0f};
    std::vector<aclblasComplex> a(30, {1.0f, 0.0f});
    std::vector<aclblasComplex> x(10, {1.0f, 0.0f});
    std::vector<aclblasComplex> y(10, {0.0f, 0.0f});
    aclblasStatus_t ret =
        aclblasChbmv_npu(nullptr, ACLBLAS_LOWER, 10, 2, &alpha, a.data(), 3, x.data(), 1, &beta, y.data(), 1);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_HANDLE_IS_NULLPTR));
}


TEST_F(ChbmvTest, NullHandlePrecedence)
{
    // The handle is validated before n, so a null handle is always reported as
    // ACLBLAS_STATUS_HANDLE_IS_NULLPTR even when another argument is invalid. This follows the
    // complex siblings cher2 / chpr2 rather than the n-first order of ssbmv.
    aclblasComplex alpha{1.0f, 0.0f};
    aclblasComplex beta{0.0f, 0.0f};
    aclblasComplex buf[4] = {{1.0f, 0.0f}, {1.0f, 0.0f}, {1.0f, 0.0f}, {1.0f, 0.0f}};
    const aclblasComplex* data = buf;
    aclblasComplex* out = buf;
    EXPECT_EQ(
        aclblasChbmv(nullptr, ACLBLAS_UPPER, -1, 0, &alpha, data, 1, data, 1, &beta, out, 1),
        ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
    EXPECT_EQ(
        aclblasChbmv(nullptr, static_cast<aclblasFillMode_t>(999), 4, 1, &alpha, data, 2, data, 1, &beta, out, 1),
        ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
    EXPECT_EQ(
        aclblasChbmv(nullptr, ACLBLAS_UPPER, 4, 1, &alpha, data, 2, data, 1, &beta, out, 1),
        ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
    EXPECT_EQ(
        aclblasChbmv(nullptr, ACLBLAS_UPPER, 0, 0, nullptr, nullptr, 1, nullptr, 1, nullptr, nullptr, 1),
        ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
    EXPECT_EQ(
        aclblasChbmv(handle_, ACLBLAS_UPPER, -1, 0, &alpha, data, 1, data, 1, &beta, out, 1),
        ACLBLAS_STATUS_INVALID_VALUE);
    EXPECT_EQ(
        aclblasChbmv(handle_, static_cast<aclblasFillMode_t>(999), 4, 1, &alpha, data, 2, data, 1, &beta, out, 1),
        ACLBLAS_STATUS_INVALID_VALUE);
    EXPECT_EQ(
        aclblasChbmv(handle_, ACLBLAS_UPPER, 0, 0, nullptr, nullptr, 1, nullptr, 1, nullptr, nullptr, 1),
        ACLBLAS_STATUS_SUCCESS);
}

TEST_F(ChbmvTest, DiagonalImaginaryPoisonAndBoundaries)
{
    const float nanVal = std::numeric_limits<float>::quiet_NaN();
    const float infVal = std::numeric_limits<float>::infinity();
    const float poisons[4] = {nanVal, infVal, -infVal, 999.0f};
    const int nList[8] = {16, 17, 31, 32, 33, 255, 256, 257};
    const int kList[8] = {0, 1, 7, 8, 15, 16, 32, 64};

    for (int idx = 0; idx < 8; ++idx) {
        int n = nList[idx];
        int k = std::min(n - 1, kList[idx]);
        int lda = k + 2;
        aclblasFillMode_t uplo = (idx % 2 == 0) ? ACLBLAS_UPPER : ACLBLAS_LOWER;
        aclblasComplex alpha = {1.25f, -0.75f};
        aclblasComplex beta = {-0.5f, 0.25f};
        size_t aSize = static_cast<size_t>(lda) * static_cast<size_t>(n);
        std::vector<aclblasComplex> aHost(aSize, {0.5f, -0.25f});
        std::vector<aclblasComplex> xHost(static_cast<size_t>(n), {1.0f, 0.5f});
        std::vector<aclblasComplex> yHost(static_cast<size_t>(n), {-1.0f, 2.0f});
        std::vector<aclblasComplex> yCpu = yHost;
                const aclblasStatus_t status1 = aclblasChbmv_cpu(
            handle_, uplo, n, k, &alpha, aHost.data(), lda, xHost.data(), 1, &beta, yCpu.data(), 1);
        ASSERT_EQ(status1, ACLBLAS_STATUS_SUCCESS);

        int diagRow = (uplo == ACLBLAS_UPPER) ? k : 0;
        for (int col = 0; col < n; ++col) {
            aHost[static_cast<size_t>(col) * static_cast<size_t>(lda) + static_cast<size_t>(diagRow)].imag =
                poisons[idx % 4];
        }
        std::vector<aclblasComplex> yNpu = yHost;
                const aclblasStatus_t status2 = aclblasChbmv_npu(
            handle_, uplo, n, k, &alpha, aHost.data(), lda, xHost.data(), 1, &beta, yNpu.data(), 1);
        ASSERT_EQ(status2, ACLBLAS_STATUS_SUCCESS);
        EXPECT_TRUE(VerifyChbmvComponent(yNpu, yCpu, false, "DiagPoison_" + std::to_string(idx)));
        EXPECT_TRUE(VerifyChbmvComponent(yNpu, yCpu, true, "DiagPoison_" + std::to_string(idx)));
    }
}

TEST_F(ChbmvTest, NormalDistributionAndRandomScalars)
{
    std::mt19937 gen(20261101U);
    std::uniform_real_distribution<float> uniformDist(-5.0f, 5.0f);
    std::uniform_real_distribution<float> sigmaDist(0.1f, 2.0f);
    for (int i = 0; i < 32; ++i) {
        const bool isNormal = (i % 2 == 1);
        const float mu = uniformDist(gen);
        const float sigma = sigmaDist(gen);
        std::normal_distribution<float> normDist(mu, sigma);
        auto sample = [&]() -> float { return isNormal ? normDist(gen) : uniformDist(gen); };

        int n = (i % 4 == 0) ? 257 : ((i % 4 == 1) ? 64 : ((i % 4 == 2) ? 33 : 15));
        int k = (i % 3 == 0) ? std::min(n - 1, 32) : ((i % 3 == 1) ? std::min(n - 1, 8) : 2);
        int lda = k + 1 + (i % 3);
        int incx = (i % 5 == 0) ? -2 : 1;
        int incy = (i % 5 == 1) ? -3 : 1;
        aclblasFillMode_t uplo = (i < 16) ? ACLBLAS_UPPER : ACLBLAS_LOWER;
        aclblasComplex alpha = {sample(), sample()};
        aclblasComplex beta = (i % 4 == 0) ? aclblasComplex{0.0f, 0.0f} : aclblasComplex{sample(), sample()};

        size_t aSize = static_cast<size_t>(lda) * static_cast<size_t>(n);
        size_t xSize = static_cast<size_t>(1 + (n - 1) * std::abs(incx));
        size_t ySize = static_cast<size_t>(1 + (n - 1) * std::abs(incy));
        std::vector<aclblasComplex> aHost(aSize);
        std::vector<aclblasComplex> xHost(xSize);
        std::vector<aclblasComplex> yHost(ySize);
        for (auto& v : aHost) v = {sample(), sample()};
        for (auto& v : xHost) v = {sample(), sample()};
        for (auto& v : yHost) v = {sample(), sample()};

        std::vector<aclblasComplex> yCpu = yHost;
        std::vector<aclblasComplex> yNpu = yHost;
                const aclblasStatus_t status3 = aclblasChbmv_cpu(
            handle_, uplo, n, k, &alpha, aHost.data(), lda, xHost.data(), incx, &beta, yCpu.data(), incy);
        ASSERT_EQ(status3, ACLBLAS_STATUS_SUCCESS);
                const aclblasStatus_t status4 = aclblasChbmv_npu(
            handle_, uplo, n, k, &alpha, aHost.data(), lda, xHost.data(), incx, &beta, yNpu.data(), incy);
        ASSERT_EQ(status4, ACLBLAS_STATUS_SUCCESS);

        const int64_t absIncy = std::abs(incy);
        std::vector<aclblasComplex> npuLogical(n);
        std::vector<aclblasComplex> cpuLogical(n);
        for (int r = 0; r < n; ++r) {
            int64_t idx = (incy > 0) ? (static_cast<int64_t>(r) * incy) : (static_cast<int64_t>(n - 1 - r) * absIncy);
            npuLogical[r] = yNpu[idx];
            cpuLogical[r] = yCpu[idx];
        }
        EXPECT_TRUE(VerifyChbmvComponent(npuLogical, cpuLogical, false, "TC_DIST_" + std::to_string(i)));
        EXPECT_TRUE(VerifyChbmvComponent(npuLogical, cpuLogical, true, "TC_DIST_" + std::to_string(i)));
    }
}


TEST_F(ChbmvTest, OverflowAndLargeWindowBoundary)
{
    // 1. Finite overflow counterexample: A=(1e38, 0), x=(4, 0), alpha=(0.25, 0), beta=(0, 0) -> (1e38, 0)
    {
        aclblasComplex alpha{0.25f, 0.0f};
        aclblasComplex beta{0.0f, 0.0f};
        std::vector<aclblasComplex> aHost = {{1.0e38f, 0.0f}};
        std::vector<aclblasComplex> xHost = {{4.0f, 0.0f}};
        std::vector<aclblasComplex> yCpu = {{0.0f, 0.0f}};
        std::vector<aclblasComplex> yNpu = {{0.0f, 0.0f}};
                const aclblasStatus_t status5 = aclblasChbmv_cpu(
            handle_, ACLBLAS_UPPER, 1, 0, &alpha, aHost.data(), 1, xHost.data(), 1, &beta, yCpu.data(), 1);
        ASSERT_EQ(status5, ACLBLAS_STATUS_SUCCESS);
                const aclblasStatus_t status6 = aclblasChbmv_npu(
            handle_, ACLBLAS_UPPER, 1, 0, &alpha, aHost.data(), 1, xHost.data(), 1, &beta, yNpu.data(), 1);
        ASSERT_EQ(status6, ACLBLAS_STATUS_SUCCESS);
        EXPECT_TRUE(std::isfinite(yNpu[0].real));
        EXPECT_TRUE(VerifyChbmvComponent(yNpu, yCpu, false, "OverflowPreScaleAlpha"));
        EXPECT_TRUE(VerifyChbmvComponent(yNpu, yCpu, true, "OverflowPreScaleAlpha"));
    }

    // 2. UB/GM window boundary n=8192 and n=8193 with k=4096: verify zero missed rows
    for (int n : {8192, 8193}) {
        int k = 4096;
        int lda = 4097;
        aclblasComplex alpha{1.0f, 0.0f};
        aclblasComplex beta{0.0f, 0.0f};
        std::vector<aclblasComplex> aHost(static_cast<size_t>(lda) * static_cast<size_t>(n), {0.0f, 0.0f});
        std::vector<aclblasComplex> xHost(static_cast<size_t>(n), {0.0f, 0.0f});
        std::vector<aclblasComplex> yNpu(static_cast<size_t>(n), {-7.0f, 3.0f});
                const aclblasStatus_t status7 = aclblasChbmv_npu(
            handle_, ACLBLAS_UPPER, n, k, &alpha, aHost.data(), lda, xHost.data(), 1, &beta, yNpu.data(), 1);
        ASSERT_EQ(status7, ACLBLAS_STATUS_SUCCESS);
        for (int i = 0; i < n; ++i) {
            ASSERT_EQ(yNpu[i].real, 0.0f) << "Missed row " << i << " for n=" << n;
            ASSERT_EQ(yNpu[i].imag, 0.0f) << "Missed row " << i << " for n=" << n;
        }
    }
}


TEST_F(ChbmvTest, LdaPaddingCoverage)
{
    // The task CSV only exercises lda == k + 1. Column stride must be lda, not k + 1, so cover
    // lda > k + 1 explicitly for both triangles and several paddings.
    const int nList[4] = {33, 64, 129, 257};
    const int kList[4] = {0, 3, 17, 64};
    for (int idx = 0; idx < 4; ++idx) {
        for (int pad = 1; pad <= 3; ++pad) {
            const int n = nList[idx];
            const int k = kList[idx];
            const int lda = k + 1 + pad;
            const aclblasFillMode_t uplo = (pad % 2 == 0) ? ACLBLAS_UPPER : ACLBLAS_LOWER;
            const aclblasComplex alpha{1.25f, -0.5f};
            const aclblasComplex beta{-0.25f, 0.75f};

            std::mt19937 gen(20261105U + static_cast<uint32_t>(idx * 8 + pad));
            std::uniform_real_distribution<float> dist(-4.0f, 4.0f);
            std::vector<aclblasComplex> aHost(static_cast<size_t>(lda) * n);
            std::vector<aclblasComplex> xHost(static_cast<size_t>(n));
            std::vector<aclblasComplex> yHost(static_cast<size_t>(n));
            for (auto& v : aHost) v = {dist(gen), dist(gen)};
            for (auto& v : xHost) v = {dist(gen), dist(gen)};
            for (auto& v : yHost) v = {dist(gen), dist(gen)};
            // Hermitian: zero the imaginary part of every stored diagonal element.
            const int diagRow = (uplo == ACLBLAS_UPPER) ? k : 0;
            for (int col = 0; col < n; ++col) {
                aHost[static_cast<size_t>(col) * lda + diagRow].imag = 0.0f;
            }

            std::vector<aclblasComplex> yCpu = yHost;
            std::vector<aclblasComplex> yNpu = yHost;
            const std::string tag = "LdaPad_" + std::to_string(n) + "_" + std::to_string(k) + "_" + std::to_string(lda);
            const aclblasStatus_t status8 = aclblasChbmv_cpu(
                handle_, uplo, n, k, &alpha, aHost.data(), lda, xHost.data(), 1, &beta, yCpu.data(), 1);
            ASSERT_EQ(status8, ACLBLAS_STATUS_SUCCESS) << tag;
            const aclblasStatus_t status9 = aclblasChbmv_npu(
                handle_, uplo, n, k, &alpha, aHost.data(), lda, xHost.data(), 1, &beta, yNpu.data(), 1);
            ASSERT_EQ(status9, ACLBLAS_STATUS_SUCCESS) << tag;
            EXPECT_TRUE(VerifyChbmvComponent(yNpu, yCpu, false, tag));
            EXPECT_TRUE(VerifyChbmvComponent(yNpu, yCpu, true, tag));
        }
    }
}

namespace {
std::vector<ChbmvParam> CsvCases();
} // namespace

INSTANTIATE_TEST_SUITE_P(
    Chbmv, ChbmvTest, ::testing::ValuesIn(CsvCases()), PrintCaseInfoString<ChbmvParam>);

namespace {

const char* const kRequiredColumns[] = {
    "case_name", "uplo", "n", "k", "alpha_real", "alpha_imag", "a_fill", "lda",
    "x_fill", "incx", "beta_real", "beta_imag", "y_fill", "incy", "expect_result",
};

void CheckCsvColumns(const std::string& path)
{
    std::ifstream source(path);
    if (!source.is_open()) {
        throw std::runtime_error("chbmv CSV cannot be opened: " + path);
    }
    std::string header;
    std::getline(source, header);
    if (!header.empty() && header.front() == static_cast<char>(0xEF)) {
        header.erase(0, 3);
    }
    for (const char* column : kRequiredColumns) {
        if (header.find(column) == std::string::npos) {
            throw std::runtime_error(std::string("chbmv CSV is missing column: ") + column);
        }
    }
    size_t rows = 0;
    std::string line;
    while (std::getline(source, line)) {
        if (!line.empty()) {
            ++rows;
        }
    }
    if (rows != 1200) {
        throw std::runtime_error("chbmv CSV must contain 1200 data rows, got " + std::to_string(rows));
    }
}

std::vector<ChbmvParam> AppendSupplement(std::vector<ChbmvParam> cases, const std::string& path)
{
    // The task package CSV fixes lda == k + 1 for every row, so the supplemental table adds the
    // legal lda > k + 1 padding shapes; a stride bug that wrongly used k + 1 stays invisible there.
    const std::string supplement = path.substr(0, path.size() - 4) + "_supplement.csv";
    std::ifstream probe(supplement);
    if (!probe.is_open()) {
        return cases;
    }
    probe.close();
    const auto extra = GetCasesFromCsv<ChbmvParam>(supplement);
    if (extra.empty()) {
        throw std::runtime_error("chbmv supplement CSV is empty: " + supplement);
    }
    cases.insert(cases.end(), extra.begin(), extra.end());
    return cases;
}

std::vector<ChbmvParam> CsvCases()
{
    const std::string path = ReplaceFileExtension2Csv(__FILE__);
    CheckCsvColumns(path);
    auto cases = GetCasesFromCsv<ChbmvParam>(path);
    if (cases.size() != 1200) {
        throw std::runtime_error("chbmv CSV must load 1200 cases, got " + std::to_string(cases.size()));
    }
    std::unordered_set<std::string> names;
    size_t performance = 0;
    for (const auto& item : cases) {
        if (item.caseName.empty() || !names.insert(item.caseName).second) {
            throw std::runtime_error("chbmv CSV contains an empty or duplicate case name");
        }
        performance += item.caseName.rfind("TC_PF_", 0) == 0 ? 1 : 0;
    }
    if (performance != 200 || cases.size() - performance != 1000) {
        throw std::runtime_error("chbmv CSV must hold 1000 accuracy and 200 performance cases");
    }
    return AppendSupplement(cases, path);
}

constexpr size_t kPerfCapBytesA = 64ULL * 1024 * 1024;
constexpr size_t kPerfCapBytesXY = 2ULL * 1024 * 1024;
constexpr int kPerfEventWarmup = 5;
constexpr int kPerfEventSamples = 55;

void CheckPerfCapacity(const ChbmvParam& p)
{
    const int64_t abytes = static_cast<int64_t>(p.lda) * p.n * static_cast<int64_t>(sizeof(aclblasComplex));
    const int64_t xspan = p.n > 0 ? static_cast<int64_t>(p.n - 1) * std::abs(static_cast<int64_t>(p.incx)) + 1 : 0;
    const int64_t yspan = p.n > 0 ? static_cast<int64_t>(p.n - 1) * std::abs(static_cast<int64_t>(p.incy)) + 1 : 0;
    const int64_t unit = static_cast<int64_t>(sizeof(aclblasComplex));
    ASSERT_LE(abytes, static_cast<int64_t>(kPerfCapBytesA)) << p.caseName;
    ASSERT_LE(xspan * unit, static_cast<int64_t>(kPerfCapBytesXY)) << p.caseName;
    ASSERT_LE(yspan * unit, static_cast<int64_t>(kPerfCapBytesXY)) << p.caseName;
}

aclblasStatus_t LaunchPerfCase(
    const ChbmvParam& p, aclblasHandle_t handle, void* bufA, void* bufX, void* bufY)
{
    return aclblasChbmv(
        handle, p.uplo, p.n, p.k, &p.alpha, static_cast<const aclblasComplex*>(bufA), p.lda,
        static_cast<const aclblasComplex*>(bufX), p.incx, &p.beta, static_cast<aclblasComplex*>(bufY), p.incy);
}

void RunPerfEventTiming(
    const ChbmvParam& p, aclblasHandle_t handle, aclrtStream stream, void* bufA, void* bufX, void* bufY)
{
    for (int i = 0; i < kPerfEventWarmup; ++i) {
        ASSERT_EQ(LaunchPerfCase(p, handle, bufA, bufX, bufY), ACLBLAS_STATUS_SUCCESS);
    }
    ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);
    aclrtEvent begin = nullptr;
    aclrtEvent end = nullptr;
    ASSERT_EQ(aclrtCreateEvent(&begin), ACL_SUCCESS);
    ASSERT_EQ(aclrtCreateEvent(&end), ACL_SUCCESS);
    float totalMs = 0.0f;
    for (int i = 0; i < kPerfEventSamples; ++i) {
        ASSERT_EQ(aclrtRecordEvent(begin, stream), ACL_SUCCESS);
        ASSERT_EQ(LaunchPerfCase(p, handle, bufA, bufX, bufY), ACLBLAS_STATUS_SUCCESS);
        ASSERT_EQ(aclrtRecordEvent(end, stream), ACL_SUCCESS);
        ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);
        float elapsed = 0.0f;
        ASSERT_EQ(aclrtEventElapsedTime(&elapsed, begin, end), ACL_SUCCESS);
        totalMs += elapsed;
    }
    aclrtDestroyEvent(begin);
    aclrtDestroyEvent(end);
    const double avgUs = (static_cast<double>(totalMs) / static_cast<double>(kPerfEventSamples)) * 1000.0;
    ASSERT_GT(avgUs, 0.0);
    std::cout << "[PERF_EVENT] " << p.caseName << " " << avgUs << std::endl;
}

void RunPerfOnce(const ChbmvParam& p, aclblasHandle_t handle, void* bufA, void* bufX, void* bufY)
{
    ASSERT_EQ(LaunchPerfCase(p, handle, bufA, bufX, bufY), ACLBLAS_STATUS_SUCCESS);
    ASSERT_EQ(aclrtSynchronizeDevice(), ACL_SUCCESS);
}

void RunAccuracyCase(const ChbmvParam& p, aclblasHandle_t handle)
{
    const int64_t aCount = (p.n > 0 && p.lda > 0) ? (static_cast<int64_t>(p.lda) * p.n) : 1;
    const int64_t xCount = (p.n > 0 && p.incx != 0) ? (static_cast<int64_t>(p.n - 1) * std::abs(p.incx) + 1) : 1;
    const int64_t yCount = (p.n > 0 && p.incy != 0) ? (static_cast<int64_t>(p.n - 1) * std::abs(p.incy) + 1) : 1;
    auto aHost = PackComplexVec(makeBlasArray(aCount * 2, p.a, p.randomSeed));
    auto xHost = PackComplexVec(makeBlasArray(xCount * 2, p.x, p.randomSeed + 1));
    auto yHost = PackComplexVec(makeBlasArray(yCount * 2, p.y, p.randomSeed + 2));
    const aclblasComplex* alphaPtr = p.nullAlpha ? nullptr : &p.alpha;
    const aclblasComplex* betaPtr = p.nullBeta ? nullptr : &p.beta;
    const aclblasComplex* aPtr = (p.a.method == BlasFillMode::M_NULLPTR) ? nullptr : aHost.data();
    const aclblasComplex* xPtr = (p.x.method == BlasFillMode::M_NULLPTR) ? nullptr : xHost.data();
    std::vector<aclblasComplex> yNpu = yHost;
    aclblasComplex* yNpuPtr = (p.y.method == BlasFillMode::M_NULLPTR) ? nullptr : yNpu.data();
    const aclblasStatus_t ret = aclblasChbmv_npu(
        handle, p.uplo, p.n, p.k, alphaPtr, aPtr, p.lda, xPtr, p.incx, betaPtr, yNpuPtr, p.incy);
    if (p.expectResult != ACLBLAS_STATUS_SUCCESS) {
        EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
        return;
    }
    ASSERT_EQ(ret, ACLBLAS_STATUS_SUCCESS);
    if (p.n == 0) {
        return;
    }
    std::vector<aclblasComplex> yCpu = yHost;
    ASSERT_EQ(
        aclblasChbmv_cpu(handle, p.uplo, p.n, p.k, alphaPtr, aPtr, p.lda, xPtr, p.incx, betaPtr, yCpu.data(), p.incy),
        ACLBLAS_STATUS_SUCCESS);
    const int64_t absIncy = std::abs(p.incy);
    std::vector<aclblasComplex> npuLogical(p.n);
    std::vector<aclblasComplex> cpuLogical(p.n);
    for (int i = 0; i < p.n; ++i) {
        const int64_t index =
            (p.incy > 0) ? (static_cast<int64_t>(i) * p.incy) : (static_cast<int64_t>(p.n - 1 - i) * absIncy);
        npuLogical[i] = yNpu[index];
        cpuLogical[i] = yCpu[index];
    }
    EXPECT_TRUE(VerifyChbmvComponent(npuLogical, cpuLogical, false, p.caseName));
    EXPECT_TRUE(VerifyChbmvComponent(npuLogical, cpuLogical, true, p.caseName));
}

} // namespace

TEST_P(ChbmvTest, CsvDriven)
{
    const auto& p = GetParam();
    if (p.IsPerformanceCase() && std::getenv("CHBMV_VERIFY_PF_ACC") == nullptr) {
        CheckPerfCapacity(p);
        if (::testing::Test::HasFatalFailure()) {
            return;
        }
        if (std::getenv("CHBMV_EVENT_TIMING") != nullptr) {
            RunPerfEventTiming(p, handle_, stream_, perfA_, perfX_, perfY_);
        } else {
            RunPerfOnce(p, handle_, perfA_, perfX_, perfY_);
        }
        return;
    }
    RunAccuracyCase(p, handle_);
}
