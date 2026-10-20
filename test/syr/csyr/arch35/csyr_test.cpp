/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "blas_test.h"
#include "csyr_golden.h"
#include "csyr_npu_wrapper.h"
#include "csyr_param.h"
#include "fill.h"
#include "csv_loader.h"
#include "verify.h"

class CsyrArch35Test : public BlasTest<CsyrParam> {};

namespace {

std::vector<aclblasComplex> MakeComplexStrided(int count, int inc, const BlasFillMode& fill, uint32_t seed)
{
    const auto real = makeBlasStrided(count, inc, fill, seed);
    // Keep real/imaginary samples reproducible but use different seed streams.
    const auto imag = makeBlasStrided(count, inc, fill, seed + 1000u);
    std::vector<aclblasComplex> result(real.size());
    for (size_t i = 0; i < result.size(); ++i) {
        result[i] = {real[i], imag[i]};
    }
    return result;
}

bool SameBits(const aclblasComplex& lhs, const aclblasComplex& rhs)
{
    return std::memcmp(&lhs, &rhs, sizeof(aclblasComplex)) == 0;
}

void VerifyTriangleComponent(
    aclblasFillMode_t uplo, int n, int lda, const std::vector<aclblasComplex>& actual,
    const std::vector<aclblasComplex>& golden, bool imaginary, const std::string& caseName)
{
    const size_t count = static_cast<size_t>(n) * static_cast<size_t>(n + 1) / 2u;
    std::vector<float> actualValues;
    std::vector<float> goldenValues;
    actualValues.reserve(count);
    goldenValues.reserve(count);
    for (int col = 0; col < n; ++col) {
        const int rowStart = (uplo == ACLBLAS_UPPER) ? 0 : col;
        const int rowEnd = (uplo == ACLBLAS_UPPER) ? col + 1 : n;
        for (int row = rowStart; row < rowEnd; ++row) {
            const size_t index = static_cast<size_t>(col) * lda + row;
            actualValues.push_back(imaginary ? actual[index].imag : actual[index].real);
            goldenValues.push_back(imaginary ? golden[index].imag : golden[index].real);
        }
    }
    VerifyConfig config;
    applyMixedTolerance(config, ACL_FLOAT, goldenValues.data(), goldenValues.size());
    EXPECT_TRUE(
        Verifier::verifyVector(actualValues.data(), goldenValues.data(), actualValues.size(), 1, config, caseName));
}

void VerifyTriangle(
    aclblasFillMode_t uplo, int n, int lda, const std::vector<aclblasComplex>& actual,
    const std::vector<aclblasComplex>& golden, const std::string& caseName)
{
    VerifyTriangleComponent(uplo, n, lda, actual, golden, false, caseName + "_real");
    VerifyTriangleComponent(uplo, n, lda, actual, golden, true, caseName + "_imag");
}

uint32_t CountChangedOutsideTriangle(
    aclblasFillMode_t uplo, int n, int lda, const std::vector<aclblasComplex>& actual,
    const std::vector<aclblasComplex>& original)
{
    uint32_t changed = 0;
    for (int col = 0; col < n; ++col) {
        for (int row = 0; row < lda; ++row) {
            const bool active =
                row < n && ((uplo == ACLBLAS_UPPER && row <= col) || (uplo == ACLBLAS_LOWER && row >= col));
            const size_t index = static_cast<size_t>(col) * lda + row;
            if (!active && !SameBits(actual[index], original[index])) {
                ++changed;
            }
        }
    }
    return changed;
}

void VerifyAlphaSample(
    aclblasHandle_t handle, aclblasFillMode_t uplo, const aclblasComplex& alpha, uint32_t seed,
    bool alphaOnDevice = false)
{
    constexpr int n = 33;
    constexpr int lda = 35;
    const BlasFillMode fill = parseFill("RANDOM_NORM_5_5");
    std::vector<aclblasComplex> x = MakeComplexStrided(n, 1, fill, seed);
    std::vector<aclblasComplex> actual = makeBlasComplexMatrix(n, n, lda, fill, seed + 1u);
    std::vector<aclblasComplex> golden = actual;
    ASSERT_EQ(
        aclblasCsyr_npu(handle, uplo, n, &alpha, x.data(), 1, actual.data(), lda, {alphaOnDevice}),
        ACLBLAS_STATUS_SUCCESS);

    ASSERT_EQ(aclblasCsyr_cpu(handle, uplo, n, &alpha, x.data(), 1, golden.data(), lda), ACLBLAS_STATUS_SUCCESS);
    VerifyTriangle(uplo, n, lda, actual, golden, "alpha_distribution");
    EXPECT_EQ(CountChangedOutsideTriangle(uplo, n, lda, actual, golden), 0u);
}

} // namespace

TEST_F(CsyrArch35Test, NullHandle)
{
    aclblasComplex alpha{1.0f, 0.0f};
    aclblasComplex value{0.0f, 0.0f};
    EXPECT_EQ(aclblasCsyr(nullptr, ACLBLAS_UPPER, 1, &alpha, &value, 1, &value, 1), ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
}

TEST_F(CsyrArch35Test, HostZeroAllowsNullData)
{
    aclblasComplex alpha{0.0f, 0.0f};
    EXPECT_EQ(
        aclblasCsyr(CsyrArch35Test::handle_, ACLBLAS_UPPER, 4, &alpha, nullptr, 1, nullptr, 4), ACLBLAS_STATUS_SUCCESS);
}

TEST_F(CsyrArch35Test, DeviceZeroAllowsNullData)
{
    aclblasComplex alpha{0.0f, 0.0f};
    EXPECT_EQ(
        aclblasCsyr_npu(CsyrArch35Test::handle_, ACLBLAS_LOWER, 4, &alpha, nullptr, 1, nullptr, 4, {true}),
        ACLBLAS_STATUS_SUCCESS);
}

TEST_F(CsyrArch35Test, DeviceNonzeroRejectsNullData)
{
    aclblasComplex alpha{1.0f, -0.5f};
    aclblasComplex x[4] = {};
    aclblasComplex matrix[16] = {};
    EXPECT_EQ(
        aclblasCsyr_npu(CsyrArch35Test::handle_, ACLBLAS_UPPER, 4, &alpha, nullptr, 1, matrix, 4, {true}),
        ACLBLAS_STATUS_INVALID_VALUE);
    EXPECT_EQ(
        aclblasCsyr_npu(CsyrArch35Test::handle_, ACLBLAS_LOWER, 4, &alpha, x, 1, nullptr, 4, {true}),
        ACLBLAS_STATUS_INVALID_VALUE);
}

TEST_F(CsyrArch35Test, DeviceAlphaComputesComplexUpdate)
{
    aclblasComplex alpha{1.0f, 2.0f};
    aclblasComplex x[2] = {{1.0f, 1.0f}, {2.0f, -1.0f}};
    aclblasComplex matrix[4] = {};
    ASSERT_EQ(
        aclblasCsyr_npu(CsyrArch35Test::handle_, ACLBLAS_LOWER, 2, &alpha, x, 1, matrix, 2, {true}),
        ACLBLAS_STATUS_SUCCESS);
    EXPECT_FLOAT_EQ(matrix[0].real, -4.0f);
    EXPECT_FLOAT_EQ(matrix[0].imag, 2.0f);
    EXPECT_FLOAT_EQ(matrix[1].real, 1.0f);
    EXPECT_FLOAT_EQ(matrix[1].imag, 7.0f);
    EXPECT_FLOAT_EQ(matrix[3].real, 11.0f);
    EXPECT_FLOAT_EQ(matrix[3].imag, 2.0f);
    EXPECT_FLOAT_EQ(matrix[2].real, 0.0f);
    EXPECT_FLOAT_EQ(matrix[2].imag, 0.0f);

    VerifyAlphaSample(CsyrArch35Test::handle_, ACLBLAS_UPPER, alpha, 20260909u, true);
}

TEST_F(CsyrArch35Test, AlphaDistributionCoverage)
{
    std::mt19937 rng(20260908u);
    std::uniform_real_distribution<float> uniform(-5.0f, 5.0f);
    for (uint32_t sample = 0; sample < 50u; ++sample) {
        SCOPED_TRACE("uniform_alpha_sample=" + std::to_string(sample));
        const aclblasComplex alpha{uniform(rng), uniform(rng)};
        const aclblasFillMode_t uplo = (sample % 2u == 0u) ? ACLBLAS_UPPER : ACLBLAS_LOWER;
        VerifyAlphaSample(CsyrArch35Test::handle_, uplo, alpha, 20261000u + sample);
    }

    std::uniform_real_distribution<float> mean(-5.0f, 5.0f);
    std::uniform_real_distribution<float> sigma(0.1f, 2.0f);
    for (uint32_t sample = 0; sample < 50u; ++sample) {
        SCOPED_TRACE("normal_alpha_sample=" + std::to_string(sample));
        const float realMean = mean(rng);
        const float realSigma = sigma(rng);
        const float imagMean = mean(rng);
        const float imagSigma = sigma(rng);
        std::normal_distribution<float> realDistribution(realMean, realSigma);
        std::normal_distribution<float> imagDistribution(imagMean, imagSigma);
        const aclblasComplex alpha{realDistribution(rng), imagDistribution(rng)};
        const aclblasFillMode_t uplo = (sample % 2u == 0u) ? ACLBLAS_LOWER : ACLBLAS_UPPER;
        VerifyAlphaSample(CsyrArch35Test::handle_, uplo, alpha, 20261100u + sample);
    }
}

INSTANTIATE_TEST_SUITE_P(
    Csyr, CsyrArch35Test, ::testing::ValuesIn(GetCasesFromCsv<CsyrParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<CsyrParam>);

TEST_P(CsyrArch35Test, CsvDriven)
{
    const CsyrParam& p = GetParam();
    std::vector<aclblasComplex> xHost = MakeComplexStrided(p.n, p.incx, p.x, p.randomSeed);
    std::vector<aclblasComplex> aHost = makeBlasComplexMatrix(p.n, p.n, p.lda, p.a, p.randomSeed + 1u);
    std::vector<aclblasComplex> golden = aHost;

    const aclblasComplex* alphaPtr = p.alphaNull ? nullptr : &p.alpha;
    const aclblasComplex* xPtr = xHost.empty() ? nullptr : xHost.data();
    aclblasComplex* aPtr = aHost.empty() ? nullptr : aHost.data();
    aclblasStatus_t status =
        aclblasCsyr_npu(CsyrArch35Test::handle_, p.uplo, p.n, alphaPtr, xPtr, p.incx, aPtr, p.lda, {p.alphaOnDevice});
    ASSERT_EQ(status, p.expectResult);
    ASSERT_EQ(
        aclblasCsyr_cpu(handle_, p.uplo, p.n, alphaPtr, xPtr, p.incx, golden.empty() ? nullptr : golden.data(), p.lda),
        p.expectResult);
    if (status != ACLBLAS_STATUS_SUCCESS || p.n == 0 || p.alphaNull || aHost.empty()) {
        return;
    }

    VerifyTriangle(p.uplo, p.n, p.lda, aHost, golden, p.caseName);
    EXPECT_EQ(CountChangedOutsideTriangle(p.uplo, p.n, p.lda, aHost, golden), 0u);
}

namespace {

struct CsyrBoundaryCase {
    int n;
    int lda;
    int incx;
    aclblasFillMode_t uplo;
    aclblasComplex alpha;
    bool deviceAlpha;
    // 0: finite random; 1: Inf in x; 2: NaN in x; 3: Inf/NaN in A;
    // 4: finite products overflowing FP32; 5: subnormal/minimum-normal inputs.
    int pattern;
    size_t xGuard = CSYR_GUARD_BYTES;
    size_t aGuard = CSYR_GUARD_BYTES;
};

void FillBoundaryMatrix(
    const CsyrBoundaryCase& c, std::mt19937& rng, std::uniform_real_distribution<float>& distribution,
    std::vector<aclblasComplex>& actual)
{
    const float inf = std::numeric_limits<float>::infinity();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    for (int col = 0; col < c.n; ++col) {
        for (int row = 0; row < c.n; ++row) {
            if (!(c.uplo == ACLBLAS_UPPER ? row <= col : row >= col)) {
                continue;
            }
            const size_t index = static_cast<size_t>(col) * c.lda + row;
            actual[index] = {distribution(rng), distribution(rng)};
            if (c.pattern == 3) {
                if (index % 7 == 0) {
                    actual[index].real = inf;
                }
                if (index % 11 == 0) {
                    actual[index].imag = nan;
                }
            }
        }
    }
}

void MakeBoundaryData(
    const CsyrBoundaryCase& c, uint32_t seed, std::vector<aclblasComplex>& x, std::vector<aclblasComplex>& actual)
{
    const float inf = std::numeric_limits<float>::infinity();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> distribution(-5.0f, 5.0f);
    const int step = std::abs(c.incx);
    x.assign(static_cast<size_t>(c.n - 1) * step + 1, {99.0f, -99.0f});
    actual.assign(static_cast<size_t>(c.lda) * c.n, {37.0f, -41.0f});
    for (int i = 0; i < c.n; ++i) {
        aclblasComplex value{distribution(rng), distribution(rng)};
        if (c.pattern == 1) {
            if (i % 7 == 0) {
                value.real = (i % 2 == 0) ? -inf : inf;
            }
            if (i % 11 == 0) {
                value.imag = inf;
            }
        } else if (c.pattern == 2) {
            if (i % 7 == 0) {
                value.real = nan;
            }
            if (i % 11 == 0) {
                value.imag = nan;
            }
        } else if (c.pattern == 4) {
            value = {1e20f, (i % 2 == 0) ? 1e20f : -1e20f};
        } else if (c.pattern == 5) {
            value = {std::numeric_limits<float>::denorm_min(), -std::numeric_limits<float>::min()};
        }
        x[static_cast<size_t>(c.incx > 0 ? i : c.n - 1 - i) * step] = value;
    }
    FillBoundaryMatrix(c, rng, distribution, actual);
}

void RunCsyrBoundary(aclblasHandle_t handle, const CsyrBoundaryCase& c, uint32_t seed)
{
    SCOPED_TRACE(
        "n=" + std::to_string(c.n) + ",lda=" + std::to_string(c.lda) + ",incx=" + std::to_string(c.incx) +
        ",uplo=" + std::to_string(c.uplo) + ",pattern=" + std::to_string(c.pattern) + ",seed=" + std::to_string(seed) +
        ",xGuard=" + std::to_string(c.xGuard) + ",aGuard=" + std::to_string(c.aGuard));
    std::vector<aclblasComplex> x, actual;
    MakeBoundaryData(c, seed, x, actual);
    const auto original = actual;
    auto golden = actual;
    ASSERT_EQ(
        aclblasCsyr_cpu(handle, c.uplo, c.n, &c.alpha, x.data(), c.incx, golden.data(), c.lda), ACLBLAS_STATUS_SUCCESS);
    const CsyrNpuOptions options{c.deviceAlpha, c.xGuard, c.aGuard};
    ASSERT_EQ(
        aclblasCsyr_npu(handle, c.uplo, c.n, &c.alpha, x.data(), c.incx, actual.data(), c.lda, options),
        ACLBLAS_STATUS_SUCCESS);
    VerifyTriangle(c.uplo, c.n, c.lda, actual, golden, "boundary_" + std::to_string(seed));
    EXPECT_EQ(CountChangedOutsideTriangle(c.uplo, c.n, c.lda, actual, original), 0u);
}

} // namespace

TEST_F(CsyrArch35Test, BoundaryRouteCoverage)
{
    std::vector<CsyrBoundaryCase> cases;
    for (int n : {31, 32, 33, 511, 512, 513, 1023, 1024, 1025, 2047, 2048, 2049, 4096, 4097}) {
        for (auto uplo : {ACLBLAS_UPPER, ACLBLAS_LOWER}) {
            for (int pattern : {0, 1, 2}) {
                cases.push_back({n, n, 1, uplo, {1, 0}, false, pattern});
            }
            cases.push_back({n, n + 3, 1, uplo, {1, 0}, false, 1});
        }
    }
    for (int n : {32, 512, 1024, 2048, 4096, 4097}) {
        for (auto uplo : {ACLBLAS_UPPER, ACLBLAS_LOWER}) {
            for (bool device : {false, true}) {
                cases.push_back({n, n + 1, 1, uplo, {2, -3}, device, 1});
            }
        }
    }
    for (int n : {33, 512, 1024, 4097}) {
        for (auto uplo : {ACLBLAS_UPPER, ACLBLAS_LOWER}) {
            cases.push_back({n, n + 3, -2, uplo, {1, 0}, false, 1});
        }
    }
    for (bool device : {false, true}) {
        for (auto uplo : {ACLBLAS_UPPER, ACLBLAS_LOWER}) {
            cases.push_back({512, 515, 1, uplo, {0, -0.0f}, device, 1});
        }
    }
    ASSERT_EQ(cases.size(), 148u);
    for (size_t i = 0; i < cases.size(); ++i) {
        ASSERT_NO_FATAL_FAILURE(RunCsyrBoundary(handle_, cases[i], 93001u + i));
    }
    RecordProperty("boundary_subcases", cases.size());
}

TEST_F(CsyrArch35Test, BoundaryExtremeCoverage)
{
    const float inf = std::numeric_limits<float>::infinity();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    std::vector<CsyrBoundaryCase> cases;
    for (int n : {31, 32, 512, 513, 1024, 1025, 2048, 4096}) {
        for (auto uplo : {ACLBLAS_UPPER, ACLBLAS_LOWER}) {
            for (bool device : {false, true}) {
                for (auto alpha : {aclblasComplex{inf, 0}, aclblasComplex{0, -inf}, aclblasComplex{nan, 1}}) {
                    cases.push_back({n, n + 3, 1, uplo, alpha, device, 0});
                }
                cases.push_back({n, n + 3, 1, uplo, {1e20f, -1e20f}, device, 4});
            }
            for (int pattern : {3, 4, 5}) {
                cases.push_back({n, n, 1, uplo, {1, 0}, false, pattern});
            }
        }
    }
    ASSERT_EQ(cases.size(), 176u);
    for (size_t i = 0; i < cases.size(); ++i) {
        ASSERT_NO_FATAL_FAILURE(RunCsyrBoundary(handle_, cases[i], 94001u + i));
    }
    RecordProperty("boundary_subcases", cases.size());
}

TEST_F(CsyrArch35Test, BoundaryFourByteAlignment)
{
    std::vector<CsyrBoundaryCase> cases;
    // Cover x-only, A-only and joint offsets. All addresses satisfy alignof(aclblasComplex).
    static_assert(alignof(aclblasComplex) == 4, "Update the alignment test for a changed public ABI");
    for (int n : {31, 32, 512, 1024, 2048, 4096, 4097}) {
        for (auto uplo : {ACLBLAS_UPPER, ACLBLAS_LOWER}) {
            for (int padding : {0, 3}) {
                cases.push_back(
                    {n, n + padding, 1, uplo, {1, 0}, false, 0, CSYR_MISALIGNED_GUARD_BYTES, CSYR_GUARD_BYTES});
                cases.push_back(
                    {n, n + padding, 1, uplo, {1, 0}, false, 0, CSYR_GUARD_BYTES, CSYR_MISALIGNED_GUARD_BYTES});
                cases.push_back(
                    {n,
                     n + padding,
                     1,
                     uplo,
                     {1, 0},
                     false,
                     0,
                     CSYR_MISALIGNED_GUARD_BYTES,
                     CSYR_MISALIGNED_GUARD_BYTES});
            }
        }
    }
    ASSERT_EQ(cases.size(), 84u);
    for (size_t i = 0; i < cases.size(); ++i) {
        ASSERT_NO_FATAL_FAILURE(RunCsyrBoundary(handle_, cases[i], 95001u + i));
    }
    RecordProperty("boundary_subcases", cases.size());
}
