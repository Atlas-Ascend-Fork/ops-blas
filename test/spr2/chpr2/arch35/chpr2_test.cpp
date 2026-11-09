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
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>
#include <cblas.h>
#include "blas_test.h"
#include "csv_loader.h"
#include "verify.h"
#include "../chpr2_param.h"
#include "chpr2_npu_wrapper.h"
#include "../chpr2_golden.h"

namespace {
std::vector<Case> Cases()
{
    const char* overridePath = std::getenv("CHPR2_CSV");
    std::string path = overridePath ? overridePath : std::string(__FILE__);
    if (!overridePath)
        path = path.substr(0, path.find_last_of("/\\") + 1) + "chpr2_test.csv";
    auto result = GetCasesFromCsv<Case>(path);
    constexpr size_t EXPECTED_CASES = 1200;
    constexpr size_t EXPECTED_BOUNDARY_CASES = 28;
    constexpr size_t EXPECTED_PERFORMANCE_CASES = 200;
    std::unordered_set<std::string> names;
    size_t boundaryCases = 0;
    size_t performanceCases = 0;
    for (const auto& p : result) {
        if (p.name.empty() || !names.insert(p.name).second)
            throw std::runtime_error("Empty or duplicate chpr2 case name: " + p.name);
        boundaryCases += p.name.compare(0, 6, "TC_ED_") == 0;
        performanceCases += p.name.compare(0, 6, "TC_PF_") == 0;
    }
    if (result.size() != EXPECTED_CASES || boundaryCases != EXPECTED_BOUNDARY_CASES ||
        performanceCases != EXPECTED_PERFORMANCE_CASES) {
        throw std::runtime_error(
            "Invalid chpr2 CSV case counts: total=" + std::to_string(result.size()) +
            ", boundary=" + std::to_string(boundaryCases) + ", performance=" + std::to_string(performanceCases));
    }
    // Additional random complex scalars, poisoned diagonal and quick-return coverage.
    std::mt19937 rng(20261028);
    std::uniform_real_distribution<float> uniform(-5, 5);
    std::uniform_real_distribution<float> sigma(0.1f, 2.0f);
    std::normal_distribution<float> normal(uniform(rng), sigma(rng));
    for (int k = 0; k < 32; ++k) {
        Case p;
        p.name = "TC_EXTRA_" + std::to_string(k);
        if (!names.insert(p.name).second)
            throw std::runtime_error("Duplicate chpr2 extra case name: " + p.name);
        p.n = 33;
        p.incx = k % 3 == 0 ? -3 : 1;
        p.incy = k % 3 == 1 ? -2 : 1;
        p.uplo = k % 2 ? ACLBLAS_UPPER : ACLBLAS_LOWER;
        p.alpha = k < 16 ? Complex{uniform(rng), uniform(rng)} : Complex{normal(rng), normal(rng)};
        p.seed = 20262000 + k;
        p.poisonDiagonal = true;
        p.xFill = p.yFill = p.apFill = "RANDOM_NORM_5_5";
        if (k == 0)
            p.alpha = {0, 0};
        if (k == 1)
            p.n = 0;
        if (k == 2) {
            p.alpha = {0, 0};
            p.yFill = "NULLPTR";
            p.expected = ACLBLAS_STATUS_INVALID_VALUE;
        }
        if (k == 3)
            p.xFill = p.yFill = "VALUE_NORM_0";
        result.push_back(p);
    }
    return result;
}

size_t VectorSize(int n, int inc)
{
    if (n <= 0 || inc == 0 || inc == INT_MIN)
        return 1;
    return 1 + static_cast<size_t>(n - 1) * std::abs(static_cast<int64_t>(inc));
}

int ProfileCalls()
{
    const char* value = std::getenv("CHPR2_PROFILE_CALLS");
    if (value == nullptr)
        return 10;
    char* end = nullptr;
    long calls = std::strtol(value, &end, 10);
    if (end == value || *end != '\0' || calls < 1 || calls > 1000)
        throw std::runtime_error("CHPR2_PROFILE_CALLS must be an integer from 1 to 1000");
    return static_cast<int>(calls);
}

std::vector<Complex> Fill(size_t size, const std::string& pattern, uint32_t seed)
{
    std::vector<Complex> out(size);
    if (pattern == "RANDOM_NORM_5_5" || pattern == "NULLPTR") {
        auto uniform = makeBlasArray(static_cast<int64_t>(size * 2), "RANDOM_NORM_5_5", seed);
        auto normal = makeBlasArray(static_cast<int64_t>(size * 2), "GAUSS_NORM_5_2", seed + 1);
        for (size_t i = 0; i < size; ++i) {
            const auto& values = i % 2 ? normal : uniform;
            out[i] = {values[2 * i], values[2 * i + 1]};
        }
    } else if (pattern == "RANDOM_ALTER" || pattern == "RANDOM_EXTREME") {
        // Preserve complex-specific paired signs and magnitudes for these cases.
        for (size_t i = 0; i < size; ++i) {
            if (pattern == "RANDOM_ALTER")
                out[i] = {i % 2 ? -1.0f : 1.0f, i % 2 ? 1.0f : -1.0f};
            else
                out[i] = {i % 2 ? 1e10f : -1e10f, i % 2 ? -1e-10f : 1e-10f};
        }
    } else {
        auto values = makeBlasArray(static_cast<int64_t>(size * 2), pattern, seed);
        if (values.size() != size * 2)
            throw std::runtime_error("Unknown fill token: " + pattern);
        for (size_t i = 0; i < size; ++i)
            out[i] = {values[2 * i], values[2 * i + 1]};
    }
    return out;
}

size_t Diagonal(int j, int n, aclblasFillMode_t uplo)
{
    return uplo == ACLBLAS_UPPER ? static_cast<size_t>(j) * (j + 1) / 2 + j :
                                   static_cast<size_t>(j) * (2LL * n - j + 1) / 2;
}

struct Stats {
    size_t finite = 0, matched = 0, nonfinite = 0, specialMatched = 0, limitFailures = 0;
    double maxError = 0;
    double Ratio() const { return finite == 0 ? 1.0 : static_cast<double>(matched) / finite; }
};

Stats CollectStats(const std::vector<Complex>& actual, const std::vector<Complex>& expected, bool imaginary)
{
    Stats s;
    for (size_t i = 0; i < actual.size(); ++i) {
        float a = imaginary ? actual[i].imag : actual[i].real;
        float g = imaginary ? expected[i].imag : expected[i].real;
        if (!std::isfinite(a) || !std::isfinite(g)) {
            ++s.nonfinite;
            if ((std::isnan(a) && std::isnan(g)) || (std::isinf(a) && std::isinf(g) && a == g))
                ++s.specialMatched;
            continue;
        }
        ++s.finite;
        double error = std::abs(static_cast<double>(a) - g);
        s.maxError = std::max(s.maxError, error);
        if (error <= std::ldexp(1.0, -13) * (1.0 + std::abs(static_cast<double>(g))))
            ++s.matched;
        float magnitude = std::abs(g);
        float adjacent = std::nextafter(magnitude, 0.0f);
        double ulp =
            magnitude == 0 ? std::numeric_limits<float>::denorm_min() : static_cast<double>(magnitude) - adjacent;
        if (error > std::max(1e-2, 32 * ulp))
            ++s.limitFailures;
    }
    return s;
}

bool VerifyComponent(
    const std::vector<Complex>& actual, const std::vector<Complex>& golden, bool imaginary, const Case& p)
{
    auto visit = [&](auto consume) {
        for (size_t i = 0; i < actual.size(); ++i) {
            float a = imaginary ? actual[i].imag : actual[i].real;
            float g = imaginary ? golden[i].imag : golden[i].real;
            if ((std::isnan(a) && std::isnan(g)) || (std::isinf(a) && std::isinf(g) && a == g))
                continue;
            consume(a, g);
        }
    };
    size_t compared = 0;
    visit([&](float, float) { ++compared; });
    const std::string name = p.name + (imaginary ? "_imag" : "_real");
    VerifyConfig mixed;
    mixed.mode = PrecisionMode::MIXED_TOLERANCE;
    mixed.mixedAtol = 1.0 / 8192.0;
    mixed.mixedRtol = 1.0 / 8192.0;
    mixed.mixedRequiredMatchedRatio = 0.99;
    mixed.mixedMaxAbsErrorLimit = 1e-2;
    VerifyConfig mere;
    mere.mode = PrecisionMode::MERE_MARE;
    mere.mereThreshold = p.mereThreshold;
    mere.mareMultiplier = p.mareMultiplier;
    bool mixedPass = Verifier::verifyEach(visit, compared, mixed, name + "_mixed");
    // CSV MERE/MARE is an additional engineering diagnostic; the task
    // acceptance criterion is the mixed tolerance above.
    const std::string diagnostic = name + "_ENGINEERING_DIAGNOSTIC_NOT_ACCEPTANCE";
    std::cout << "[" << diagnostic << "] MERE/MARE is diagnostic only, not an acceptance result" << std::endl;
    (void)Verifier::verifyEach(visit, compared, mere, diagnostic);
    return mixedPass;
}

void Record(const Case& p, aclblasStatus_t status, const Stats& real, const Stats& imag, bool passed)
{
    const char* path = std::getenv("CHPR2_RESULTS");
    if (!path)
        return;
    bool header = !std::ifstream(path).good();
    std::ofstream out(path, std::ios::app);
    if (!out)
        throw std::runtime_error("Cannot write CHPR2_RESULTS");
    if (header)
        out << "case_name,n,uplo,incx,incy,status,expected,real_ratio,real_max_abs,real_nonfinite,real_nonfinite_"
               "passed,imag_ratio,imag_max_abs,imag_nonfinite,imag_nonfinite_passed,passed,alpha_real,alpha_imag,alpha_"
               "null,x_fill,y_fill,ap_fill,seed,poison_diagonal\n";
    out << std::setprecision(12) << p.name << ',' << p.n << ',' << static_cast<int>(p.uplo) << ',' << p.incx << ','
        << p.incy << ',' << status << ',' << p.expected << ',' << real.Ratio() << ',' << real.maxError << ','
        << real.nonfinite << ',' << real.specialMatched << ',' << imag.Ratio() << ',' << imag.maxError << ','
        << imag.nonfinite << ',' << imag.specialMatched << ',' << (passed ? "PASS" : "FAIL") << ',' << p.alpha.real
        << ',' << p.alpha.imag << ',' << p.alphaNull << ',' << p.xFill << ',' << p.yFill << ',' << p.apFill << ','
        << p.seed << ',' << p.poisonDiagonal << '\n';
}
} // namespace

class Chpr2Arch35Test : public BlasTest<Case> {};

INSTANTIATE_TEST_SUITE_P(
    Chpr2, Chpr2Arch35Test, ::testing::ValuesIn(Cases()),
    [](const ::testing::TestParamInfo<Case>& info) { return info.param.name; });

TEST(Chpr2Validation, NullHandlePrecedesAll)
{
    EXPECT_EQ(
        aclblasChpr2(nullptr, static_cast<aclblasFillMode_t>(999), 0, nullptr, nullptr, INT_MIN, nullptr, 0, nullptr),
        ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
    EXPECT_EQ(
        aclblasChpr2(nullptr, ACLBLAS_UPPER, -1, nullptr, nullptr, 1, nullptr, 1, nullptr),
        ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
}

class Chpr2BoundaryTest : public BlasTest<int> {};

TEST_F(Chpr2BoundaryTest, ByteSpanAndLargeStride)
{
    // Only one element is allocated: overflow checks must happen before launching.
    std::vector<Complex> x{{2.0f, 3.0f}};
    std::vector<Complex> y{{-1.0f, 2.0f}};
    std::vector<Complex> ap{{7.0f, 19.0f}};
    Chpr2NpuWrapper dx(x);
    Chpr2NpuWrapper dy(y);
    Chpr2NpuWrapper da(ap);
    Complex alpha{0.5f, -0.25f};
    Complex zero{0.0f, 0.0f};
    EXPECT_EQ(
        aclblasChpr2(handle_, ACLBLAS_UPPER, INT_MAX, &alpha, dx.Data(), INT_MAX, dy.Data(), 1, da.Data()),
        ACLBLAS_STATUS_INVALID_VALUE);
    EXPECT_EQ(
        aclblasChpr2(handle_, ACLBLAS_LOWER, INT_MAX, &alpha, dx.Data(), 1, dy.Data(), -INT_MAX, da.Data()),
        ACLBLAS_STATUS_INVALID_VALUE);
    EXPECT_EQ(
        aclblasChpr2(handle_, ACLBLAS_UPPER, INT_MAX, &zero, dx.Data(), INT_MAX, dy.Data(), -INT_MAX, da.Data()),
        ACLBLAS_STATUS_SUCCESS);
    ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_SUCCESS);
    da.Read(true);
    ASSERT_EQ(
        aclblasChpr2(handle_, ACLBLAS_LOWER, 1, &alpha, dx.Data(), INT_MAX, dy.Data(), -INT_MAX, da.Data()),
        ACLBLAS_STATUS_SUCCESS);
    ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_SUCCESS);
    auto result = da.Read();
    auto golden = ap;
    Chpr2Golden(ACLBLAS_LOWER, 1, alpha, x, INT_MAX, y, -INT_MAX, golden);
    Case boundary;
    boundary.name = "large_stride";
    EXPECT_TRUE(VerifyComponent(result, golden, false, boundary));
    EXPECT_TRUE(VerifyComponent(result, golden, true, boundary));
    EXPECT_EQ(result[0].imag, 0.0f);
    dx.Read(true);
    dy.Read(true);
}

TEST_P(Chpr2Arch35Test, CsvDriven)
{
    const Case& p = GetParam();
    auto x = Fill(VectorSize(p.n, p.incx), p.xFill, p.seed);
    auto y = Fill(VectorSize(p.n, p.incy), p.yFill, p.seed + 1);
    const size_t length = p.n > 0 ? static_cast<size_t>(p.n) * (p.n + 1) / 2 : 1;
    auto ap = Fill(length, p.apFill, p.seed + 2);
    for (int j = 0; j < p.n; ++j)
        ap[Diagonal(j, p.n, p.uplo)].imag = p.poisonDiagonal ? 17.0f : 0.0f;
    Chpr2NpuWrapper dx(x), dy(y), da(ap);
    const bool profiling = std::getenv("CHPR2_PROFILE") != nullptr;
    aclblasStatus_t status = ACLBLAS_STATUS_SUCCESS;
    const int calls = profiling ? ProfileCalls() : 1;
    for (int repeat = 0; repeat < calls; ++repeat) {
        status = aclblasChpr2(
            handle_, p.uplo, p.n, p.alphaNull ? nullptr : &p.alpha, p.xFill == "NULLPTR" ? nullptr : dx.Data(), p.incx,
            p.yFill == "NULLPTR" ? nullptr : dy.Data(), p.incy, p.apFill == "NULLPTR" ? nullptr : da.Data());
        EXPECT_EQ(status, p.expected);
        ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_SUCCESS);
    }
    const bool noop = p.n <= 0 || (!p.alphaNull && p.alpha.real == 0 && p.alpha.imag == 0);
    auto actual = da.Read(noop || status != ACLBLAS_STATUS_SUCCESS);
    dx.Read(true);
    dy.Read(true);
    Stats real, imag;
    // msprof replays the captured kernel; its output must not be compared as a single update.
    if (status == ACLBLAS_STATUS_SUCCESS && p.expected == ACLBLAS_STATUS_SUCCESS && !noop && !profiling) {
        auto golden = ap;
        Chpr2Golden(p.uplo, p.n, p.alpha, x, p.incx, y, p.incy, golden);
        real = CollectStats(actual, golden, false);
        imag = CollectStats(actual, golden, true);
        EXPECT_TRUE(VerifyComponent(actual, golden, false, p));
        EXPECT_TRUE(VerifyComponent(actual, golden, true, p));
        for (int j = 0; j < p.n; ++j) {
            EXPECT_EQ(actual[Diagonal(j, p.n, p.uplo)].imag, 0.0f);
            EXPECT_EQ(golden[Diagonal(j, p.n, p.uplo)].imag, 0.0f);
        }
    }
    Record(p, status, real, imag, !HasFailure());
}
