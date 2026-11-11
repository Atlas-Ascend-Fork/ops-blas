/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <climits>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include "blas_test.h"
#include "fill.h"
#include "chbmv_golden.h"
#include "allocation.h"

namespace {
using ChbmvTest = BlasTest<ChbmvParam>;
using chbmv_test::Allocation;
constexpr size_t GUARD = 4;
constexpr aclblasComplex SENTINEL{-999.0f, 777.0f};

std::vector<aclblasComplex> MakeComplex(size_t count, const std::string& fill, uint32_t seed)
{
    const auto re = makeBlasArray(static_cast<int64_t>(count), fill, seed);
    const auto im = makeBlasArray(static_cast<int64_t>(count), fill, seed + 1000U);
    std::vector<aclblasComplex> data(count);
    for (size_t i = 0; i < count; ++i) {
        data[i] = {re[i], im[i]};
    }
    return data;
}

std::vector<aclblasComplex> MakeVector(int n, int inc, const std::string& fill, uint32_t seed)
{
    const int64_t step = inc > 0 ? static_cast<int64_t>(inc) : -static_cast<int64_t>(inc);
    std::vector<aclblasComplex> result(static_cast<size_t>((n - 1LL) * step + 1), SENTINEL);
    auto logical = MakeComplex(static_cast<size_t>(n), fill, seed);
    for (int i = 0; i < n; ++i) {
        const int64_t pos = inc > 0 ? i * step : (n - 1LL - i) * step;
        result[static_cast<size_t>(pos)] = logical[static_cast<size_t>(i)];
    }
    return result;
}

std::vector<ChbmvParam> Cases()
{
    const char* caseDirectory = std::getenv("CHBMV_TEST_CASE_DIR");
    const std::string mainCsv = caseDirectory != nullptr && caseDirectory[0] != '\0' ?
                                    std::string(caseDirectory) + "/chbmv_test.csv" :
                                    ReplaceFileExtension2Csv(__FILE__);
    auto cases = GetCasesFromCsv<ChbmvParam>(mainCsv);
    const size_t mainCount = cases.size();
    const std::string directory = mainCsv.substr(0, mainCsv.find_last_of("/\\") + 1);
    const auto appendOptional = [&](const char* filename) {
        const std::string path = directory + filename;
        if (!std::ifstream(path).good()) {
            return size_t{0};
        }
        auto extra = GetCasesFromCsv<ChbmvParam>(path);
        const size_t count = extra.size();
        cases.insert(cases.end(), extra.begin(), extra.end());
        return count;
    };
    const size_t supplementCount = appendOptional("chbmv_supplement_test.csv");
    const size_t gaussianCount = appendOptional("chbmv_gaussian_mirror_test.csv");
    size_t numericCount = 0;
    size_t originalPfCount = 0;
    for (const auto& p : cases) {
        numericCount += p.n > 0 && p.expectResult == ACLBLAS_STATUS_SUCCESS ? 1U : 0U;
        originalPfCount += p.caseName.rfind("TC_PF_", 0) == 0 ? 1U : 0U;
    }
    std::cout << "[CASE_REGISTRY] main=" << mainCount << " supplement=" << supplementCount
              << " gaussian_mirror=" << gaussianCount << " csv_total=" << cases.size()
              << " numeric_compare_csv=" << numericCount << " original_pf=" << originalPfCount
              << " directed_ut_registered_separately=1" << std::endl;
    return cases;
}

bool Null(const std::string& token) { return token == "NULLPTR" || token == "null"; }

void Compare(
    const ChbmvParam& p, const std::vector<aclblasComplex>& actual, const std::vector<aclblasComplex>& expected)
{
    ChbmvReference::Stats re;
    ChbmvReference::Stats im;
    const int64_t step = p.incy > 0 ? static_cast<int64_t>(p.incy) : -static_cast<int64_t>(p.incy);
    for (int i = 0; i < p.n; ++i) {
        const int64_t pos = p.incy > 0 ? i * step : (p.n - 1LL - i) * step;
        re.Add(actual[static_cast<size_t>(pos)].real, expected[static_cast<size_t>(pos)].real);
        im.Add(actual[static_cast<size_t>(pos)].imag, expected[static_cast<size_t>(pos)].imag);
    }
    for (const auto& component : {std::make_pair("real", re), std::make_pair("imag", im)}) {
        const auto& s = component.second;
        std::cout << "[ACCURACY] case=" << p.caseName << " component=" << component.first << " finite=" << s.finite
                  << " matched=" << s.matched << " max_abs_error=" << s.maxError << " special=" << s.special
                  << " special_matched=" << s.specialMatched << " pass=" << s.Pass() << std::endl;
        EXPECT_TRUE(s.Pass()) << p.caseName << " " << component.first;
    }
}

void CheckOutputGuards(
    const ChbmvParam& p, const std::vector<aclblasComplex>& guarded, const std::vector<aclblasComplex>& original,
    const std::vector<aclblasComplex>& actual)
{
    for (size_t i = 0; i < GUARD; ++i) {
        EXPECT_EQ(std::memcmp(&guarded[i], &SENTINEL, sizeof(SENTINEL)), 0);
        EXPECT_EQ(std::memcmp(&guarded[GUARD + original.size() + i], &SENTINEL, sizeof(SENTINEL)), 0);
    }
    const int64_t absInc = p.incy > 0 ? static_cast<int64_t>(p.incy) : -static_cast<int64_t>(p.incy);
    for (size_t i = 0; i < actual.size(); ++i) {
        if (i % static_cast<size_t>(absInc) != 0) {
            EXPECT_EQ(std::memcmp(&actual[i], &original[i], sizeof(aclblasComplex)), 0);
        }
    }
}

template <typename Call>
void RecordPerformance(const ChbmvParam& p, aclrtStream stream, const Call& call)
{
    ASSERT_EQ(p.beta.real, 0.0f);
    ASSERT_EQ(p.beta.imag, 0.0f); // Repeated invocations preserve the PF input contract.
    for (int i = 0; i < 5; ++i) {
        ASSERT_EQ(call(), ACLBLAS_STATUS_SUCCESS);
    }
    ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);
    aclrtEvent start = nullptr;
    aclrtEvent stop = nullptr;
    ASSERT_EQ(aclrtCreateEvent(&start), ACL_SUCCESS);
    ASSERT_EQ(aclrtCreateEvent(&stop), ACL_SUCCESS);
    ASSERT_EQ(aclrtRecordEvent(start, stream), ACL_SUCCESS);
    for (int i = 0; i < 10; ++i) {
        ASSERT_EQ(call(), ACLBLAS_STATUS_SUCCESS);
    }
    ASSERT_EQ(aclrtRecordEvent(stop, stream), ACL_SUCCESS);
    ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);
    float milliseconds = 0;
    ASSERT_EQ(aclrtEventElapsedTime(&milliseconds, start, stop), ACL_SUCCESS);
    ASSERT_EQ(aclrtDestroyEvent(start), ACL_SUCCESS);
    ASSERT_EQ(aclrtDestroyEvent(stop), ACL_SUCCESS);
    std::cout << std::setprecision(9) << "[PERF] case=" << p.caseName << " n=" << p.n << " k=" << p.k
              << " avg_us=" << milliseconds * 1000.0f / 10.0f << " samples=10 warmup=5 method=aclrtEvent" << std::endl;
}

TEST_P(ChbmvTest, CsvDriven)
{
    const ChbmvParam& p = GetParam();
    const bool normal = p.n > 0 && p.expectResult == ACLBLAS_STATUS_SUCCESS;
    const bool perf = p.caseName.rfind("TC_PF_", 0) == 0;
    // Error and n=0 paths receive tiny buffers even when dimensions/strides would imply enormous allocations.
    auto a = normal ? MakeComplex(static_cast<size_t>(p.lda) * p.n, p.aFill, p.randomSeed + 4000U) :
                      std::vector<aclblasComplex>(1, SENTINEL);
    auto x = normal ? MakeVector(p.n, p.incx, p.xFill, p.randomSeed + 1U) : std::vector<aclblasComplex>(1, SENTINEL);
    auto original =
        normal ? MakeVector(p.n, p.incy, p.yFill, p.randomSeed + 2U) : std::vector<aclblasComplex>(1, SENTINEL);
    std::vector<aclblasComplex> guarded(original.size() + 2 * GUARD, SENTINEL);
    std::copy(original.begin(), original.end(), guarded.begin() + GUARD);
    Allocation da;
    Allocation dx;
    Allocation dy;
    ASSERT_EQ(da.Upload(a), ACL_SUCCESS);
    ASSERT_EQ(dx.Upload(x), ACL_SUCCESS);
    ASSERT_EQ(dy.Upload(guarded), ACL_SUCCESS);
    auto call = [&]() {
        return aclblasChbmv(
            handle_, p.uplo, p.n, p.k, p.nullAlpha ? nullptr : &p.alpha, Null(p.aFill) ? nullptr : da.Ptr(), p.lda,
            Null(p.xFill) ? nullptr : dx.Ptr(), p.incx, p.nullBeta ? nullptr : &p.beta,
            Null(p.yFill) ? nullptr : dy.Ptr() + GUARD, p.incy);
    };
    ASSERT_EQ(call(), p.expectResult) << p.caseName;
    ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_SUCCESS);
    if (!normal) {
        ASSERT_EQ(dy.Download(guarded), ACL_SUCCESS);
        for (size_t i = 0; i < guarded.size(); ++i) {
            EXPECT_EQ(std::memcmp(&guarded[i], &SENTINEL, sizeof(SENTINEL)), 0)
                << p.caseName << " changed y or its guard at " << i;
        }
        return;
    }
    ASSERT_EQ(dy.Download(guarded), ACL_SUCCESS);
    std::vector<aclblasComplex> actual(guarded.begin() + GUARD, guarded.end() - GUARD);
    const auto expected = ChbmvReference::Golden(p, a, x, original);
    Compare(p, actual, expected);
    CheckOutputGuards(p, guarded, original, actual);
    if (perf) {
        RecordPerformance(p, stream_, call);
    }
}

TEST_F(ChbmvTest, NullHandlePrecedesQuickReturn)
{
    EXPECT_EQ(
        aclblasChbmv(nullptr, ACLBLAS_UPPER, 0, 0, nullptr, nullptr, 1, nullptr, 1, nullptr, nullptr, 1),
        ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
}

void CheckDirected(
    aclblasHandle_t handle, aclrtStream stream, const ChbmvParam& p, const std::vector<aclblasComplex>& a,
    const std::vector<aclblasComplex>& x, const std::vector<aclblasComplex>& original)
{
    std::vector<aclblasComplex> guarded(original.size() + 2 * GUARD, SENTINEL);
    std::copy(original.begin(), original.end(), guarded.begin() + GUARD);
    Allocation da;
    Allocation dx;
    Allocation dy;
    ASSERT_EQ(da.Upload(a), ACL_SUCCESS);
    ASSERT_EQ(dx.Upload(x), ACL_SUCCESS);
    ASSERT_EQ(dy.Upload(guarded), ACL_SUCCESS);
    ASSERT_EQ(
        aclblasChbmv(
            handle, p.uplo, p.n, p.k, &p.alpha, da.Ptr(), p.lda, dx.Ptr(), p.incx, &p.beta, dy.Ptr() + GUARD, p.incy),
        ACLBLAS_STATUS_SUCCESS);
    ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);
    ASSERT_EQ(dy.Download(guarded), ACL_SUCCESS);
    std::vector<aclblasComplex> actual(guarded.begin() + GUARD, guarded.end() - GUARD);
    Compare(p, actual, ChbmvReference::Golden(p, a, x, original));
    for (size_t i = 0; i < GUARD; ++i) {
        EXPECT_EQ(std::memcmp(&guarded[i], &SENTINEL, sizeof(SENTINEL)), 0);
        EXPECT_EQ(std::memcmp(&guarded[GUARD + original.size() + i], &SENTINEL, sizeof(SENTINEL)), 0);
    }
    const int64_t step = p.incy > 0 ? p.incy : -static_cast<int64_t>(p.incy);
    for (size_t i = 0; i < actual.size(); ++i) {
        if (i % static_cast<size_t>(step) != 0) {
            EXPECT_EQ(std::memcmp(&actual[i], &original[i], sizeof(SENTINEL)), 0);
        }
    }
    std::vector<aclblasComplex> aAfter(a.size());
    std::vector<aclblasComplex> xAfter(x.size());
    ASSERT_EQ(da.Download(aAfter), ACL_SUCCESS);
    ASSERT_EQ(dx.Download(xAfter), ACL_SUCCESS);
    EXPECT_EQ(std::memcmp(aAfter.data(), a.data(), a.size() * sizeof(aclblasComplex)), 0);
    EXPECT_EQ(std::memcmp(xAfter.data(), x.data(), x.size() * sizeof(aclblasComplex)), 0);
}

TEST_F(ChbmvTest, ZeroDimensionDoesNotWrite)
{
    std::vector<aclblasComplex> data(2 * GUARD + 1, SENTINEL);
    Allocation dy;
    ASSERT_EQ(dy.Upload(data), ACL_SUCCESS);
    // n=0 must precede the remaining parameter checks, including null scalars.
    ASSERT_EQ(
        aclblasChbmv(
            handle_, static_cast<aclblasFillMode_t>(999), 0, -1, nullptr, nullptr, 0, nullptr, 0, nullptr,
            dy.Ptr() + GUARD, INT_MIN),
        ACLBLAS_STATUS_SUCCESS);
    ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_SUCCESS);
    ASSERT_EQ(dy.Download(data), ACL_SUCCESS);
    for (const auto& value : data) {
        EXPECT_EQ(std::memcmp(&value, &SENTINEL, sizeof(SENTINEL)), 0);
    }
}

TEST_F(ChbmvTest, DiagonalImaginaryAndUnusedCellsNaN)
{
    const float nan = std::numeric_limits<float>::quiet_NaN();
    for (const auto uplo : {ACLBLAS_UPPER, ACLBLAS_LOWER}) {
        // Exercise both the 64-row diagonal path and the wide-band row-dot path.
        for (const auto shape : {std::make_pair(7, 2), std::make_pair(67, 65)}) {
            ChbmvParam p(csv_map{{"case_name", "UT_DIAG_IMAG_UNUSED_NAN"}});
            p.uplo = uplo;
            p.n = shape.first;
            p.k = shape.second;
            p.lda = p.k + 3;
            p.alpha = {0.75f, -0.25f};
            p.beta = {0.0f, 0.0f};
            std::vector<aclblasComplex> a(static_cast<size_t>(p.lda) * p.n, {nan, nan});
            for (int j = 0; j < p.n; ++j) {
                const int first = uplo == ACLBLAS_UPPER ? std::max(0, j - p.k) : j;
                const int end = uplo == ACLBLAS_UPPER ? j : std::min(p.n - 1, j + p.k);
                for (int i = first; i <= end; ++i) {
                    const int bandRow = uplo == ACLBLAS_UPPER ? p.k + i - j : i - j;
                    a[static_cast<size_t>(j) * p.lda + bandRow] =
                        i == j ? aclblasComplex{2.0f + 0.125f * j, nan} :
                                 aclblasComplex{0.125f * (i + 1), -0.0625f * (j + 1)};
                }
            }
            const auto x = MakeVector(p.n, p.incx, "RANDOM_NORM_5_5", 2026111101U);
            const std::vector<aclblasComplex> original(p.n, {nan, nan});
            ASSERT_NO_FATAL_FAILURE(CheckDirected(handle_, stream_, p, a, x, original));
        }
    }
}

TEST_F(ChbmvTest, BetaOneMixedInfinityKeepsFiniteComponent)
{
    const float inf = std::numeric_limits<float>::infinity();
    const aclblasComplex logicalY[] = {{inf, 2.0f}, {3.0f, -inf}, {-inf, -4.0f}, {5.0f, inf}};
    const aclblasComplex logicalX[] = {{1.0f, 1.0f}, {2.0f, -2.0f}, {-3.0f, 3.0f}, {4.0f, -4.0f}};
    for (const auto uplo : {ACLBLAS_UPPER, ACLBLAS_LOWER}) {
        for (const auto strides : {std::make_pair(1, 1), std::make_pair(-2, 3)}) {
            ChbmvParam p(csv_map{{"case_name", "UT_BETA_ONE_MIXED_INF"}});
            p.uplo = uplo;
            p.n = 4;
            p.k = 0;
            p.lda = 1;
            p.incx = strides.first;
            p.incy = strides.second;
            p.alpha = {1.0f, 0.0f}; // Keep alpha nonzero so beta=1 is handled by the kernel.
            p.beta = {1.0f, 0.0f};
            const int sx = p.incx > 0 ? p.incx : -p.incx;
            std::vector<aclblasComplex> x(1 + 3 * sx, SENTINEL);
            std::vector<aclblasComplex> original(1 + 3 * p.incy, SENTINEL);
            for (int i = 0; i < p.n; ++i) {
                x[p.incx > 0 ? i * sx : (p.n - 1 - i) * sx] = logicalX[i];
                original[i * p.incy] = logicalY[i];
            }
            const std::vector<aclblasComplex> a(p.n, {2.0f, 0.0f});
            ASSERT_NO_FATAL_FAILURE(CheckDirected(handle_, stream_, p, a, x, original));
        }
    }
}

INSTANTIATE_TEST_SUITE_P(Chbmv, ChbmvTest, ::testing::ValuesIn(Cases()), PrintCaseInfoString<ChbmvParam>);
} // namespace
