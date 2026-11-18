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
#include <cstring>
#include <cstdlib>
#include <iomanip>
#include <filesystem>
#include <fstream>
#include <limits>
#include <random>
#include <vector>
#include "ChemvArch35Test.h"
#include "chemv_param.h"
#include "chemv_cases.h"
#include "chemv_compare.h"
#include "chemv_golden.h"
#include "chemv_npu_wrapper.h"

namespace {

size_t StorageLength(int n, int inc)
{
    if (n <= 0 || inc == INT_MIN) {
        return 1;
    }
    return 1 + static_cast<size_t>(n - 1) * std::abs(static_cast<int64_t>(inc));
}

std::vector<aclblasComplex> MakeData(size_t size, const BlasFillMode& fill, uint32_t seed)
{
    if (fill.method == BlasFillMode::M_NULLPTR) {
        return {};
    }
    if (fill.method == BlasFillMode::M_RANDOM && fill.pattern == BlasFillMode::P_NORM && (seed & 1U) == 0) {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> mean(-5.0f, 5.0f);
        std::uniform_real_distribution<float> sigma(0.1f, 2.0f);
        const float sampledMean = mean(rng);
        const float sampledSigma = sigma(rng);
        std::normal_distribution<float> dist(sampledMean, sampledSigma);
        std::vector<aclblasComplex> values(size);
        for (auto& v : values) {
            v = {dist(rng), dist(rng)};
        }
        return values;
    }
    return makeBlasComplexMatrix(static_cast<int>(size), 1, static_cast<int>(size), fill, seed);
}

struct HostData {
    std::vector<aclblasComplex> a, x, y;
    explicit HostData(const ChemvParam& p)
    {
        a = MakeData(p.n > 0 && p.lda > 0 ? static_cast<size_t>(p.n) * p.lda : 1, p.a, p.randomSeed);
        x = MakeData(StorageLength(p.n, p.incx), p.x, p.randomSeed + 2);
        y = MakeData(StorageLength(p.n, p.incy), p.y, p.randomSeed + 4);
        const float nan = std::numeric_limits<float>::quiet_NaN();
        if (p.n > 0 && p.lda >= p.n && !a.empty()) {
            for (int col = 0; col < p.n; ++col) {
                for (int row = 0; row < p.lda; ++row) {
                    auto& v = a[static_cast<size_t>(col) * p.lda + row];
                    if (row >= p.n || (p.uplo == ACLBLAS_UPPER ? row > col : row < col)) {
                        v = {nan, nan};
                    } else if (row == col) {
                        v.imag = nan;
                    }
                }
            }
        }
    }
};

} // namespace

TEST_F(ChemvArch35Test, NullHandlePrecedesDimensions)
{
    for (int n : {-1, 0, 1}) {
        EXPECT_EQ(
            aclblasChemv(nullptr, ACLBLAS_UPPER, n, nullptr, nullptr, 0, nullptr, 0, nullptr, nullptr, 0),
            ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
    }
}

INSTANTIATE_TEST_SUITE_P(
    Chemv, ChemvArch35Test, ::testing::ValuesIn(GetChemvCasesFromCsv(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<ChemvParam>);

namespace {
void DumpCase(
    const ChemvParam& p, const HostData& h, const std::vector<aclblasComplex>& before,
    const std::vector<aclblasComplex>& golden)
{
    const char* directory = std::getenv("CHEMV_DUMP_DIR");
    if (directory == nullptr || *directory == '\0') {
        return;
    }
    const std::filesystem::path base = std::filesystem::path(directory) / p.caseName;
    std::filesystem::create_directories(base);
    const auto write = [&](const char* name, const std::vector<aclblasComplex>& values) {
        auto path = base;
        path.append(name);
        std::ofstream stream(path, std::ios::binary);
        stream.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(values[0]));
        if (!stream) {
            throw std::runtime_error("Cannot save Chemv diagnostic dump");
        }
    };
    write("a.bin", h.a);
    write("x.bin", h.x);
    write("before.bin", before);
    write("actual.bin", h.y);
    write("golden.bin", golden);
    std::ofstream metadata(base / "metadata.json");
    metadata << std::setprecision(std::numeric_limits<float>::max_digits10) << "{\"case\":\"" << p.caseName
             << "\",\"n\":" << p.n << ",\"lda\":" << p.lda << ",\"uplo\":" << p.uplo << ",\"incx\":" << p.incx
             << ",\"incy\":" << p.incy << ",\"alpha\":[" << p.alpha.real << ',' << p.alpha.imag << "]"
             << ",\"beta\":[" << p.beta.real << ',' << p.beta.imag << "]"
             << ",\"seed\":" << p.randomSeed << ",\"dtype\":\"complex64\"}";
    if (!metadata) {
        throw std::runtime_error("Cannot save Chemv diagnostic metadata");
    }
}

void CheckStrideHoles(const std::vector<aclblasComplex>& actual, const std::vector<aclblasComplex>& original, int incy)
{
    // Check holes in strided y byte-for-byte (including NaN payloads).
    const size_t stride = static_cast<size_t>(std::abs(static_cast<int64_t>(incy)));
    ASSERT_NE(stride, 0U);
    for (size_t i = 0; i < actual.size(); ++i) {
        if (i % stride != 0) {
            EXPECT_EQ(std::memcmp(&actual[i], &original[i], sizeof(actual[i])), 0) << "hole=" << i;
        }
    }
}

void RunCase(aclblasHandle_t handle, aclrtStream stream, const ChemvParam& p)
{
    HostData h(p);
    const auto original = h.y;
    chemv_test::DeviceData d(h.a, h.x, h.y);
    const auto invoke = [&]() { return d.Invoke(handle, p); };
    ASSERT_EQ(invoke(), p.expectResult);
    if (p.expectResult != ACLBLAS_STATUS_SUCCESS) {
        return;
    }
    ASSERT_EQ(d.CopyResult(stream, h.y), ACL_SUCCESS);
    if (p.n == 0) {
        return;
    }
    const bool quickReturn = p.alpha.real == 0 && p.alpha.imag == 0 && p.beta.real == 1 && p.beta.imag == 0;
    if (quickReturn) {
        ASSERT_EQ(h.y.size(), original.size());
        EXPECT_TRUE(h.y.empty() || std::memcmp(h.y.data(), original.data(), h.y.size() * sizeof(h.y[0])) == 0);
        return;
    }
    auto golden = original;
    ASSERT_EQ(
        chemv_test::aclblasChemv_cpu(
            handle, p.uplo, p.n, &p.alpha, h.a.empty() ? nullptr : h.a.data(), p.lda,
            h.x.empty() ? nullptr : h.x.data(), p.incx, &p.beta, golden.data(), p.incy),
        ACLBLAS_STATUS_SUCCESS);
    DumpCase(p, h, original, golden);
    EXPECT_TRUE(chemv_test::Compare(
        h.y.data(), golden.data(), p.n, p.incy, false, p.caseName, p.mereThreshold, p.mareMultiplier));
    EXPECT_TRUE(chemv_test::Compare(
        h.y.data(), golden.data(), p.n, p.incy, true, p.caseName, p.mereThreshold, p.mareMultiplier));
    CheckStrideHoles(h.y, original, p.incy);
    if (p.IsPerformanceCase() && std::getenv("CHEMV_PROFILE") != nullptr) {
        // One correctness launch above, then 5 warmups and 60 samples.
        // Kernel duration comes from msprof, never from host wall clock.
        for (int i = 0; i < 65; ++i) {
            ASSERT_EQ(invoke(), ACLBLAS_STATUS_SUCCESS);
        }
        ASSERT_EQ(chemv_test::DeviceData::Synchronize(stream), ACL_SUCCESS);
        std::cout << "[PROFILE] case=" << p.caseName << " correctness=1 warmup=5 samples=60" << std::endl;
    }
}

struct ScalarSample {
    aclblasComplex value;
    float realMean = 0.0f;
    float realSigma = 0.0f;
    float imagMean = 0.0f;
    float imagSigma = 0.0f;
};

ScalarSample MakeScalar(uint32_t seed, bool normal)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> uniform(-5.0f, 5.0f);
    ScalarSample sample{};
    if (normal) {
        std::uniform_real_distribution<float> sigma(0.1f, 2.0f);
        sample.realMean = uniform(rng);
        sample.realSigma = sigma(rng);
        sample.imagMean = uniform(rng);
        sample.imagSigma = sigma(rng);
        std::normal_distribution<float> real(sample.realMean, sample.realSigma);
        std::normal_distribution<float> imag(sample.imagMean, sample.imagSigma);
        sample.value = {real(rng), imag(rng)};
    } else {
        sample.value = {uniform(rng), uniform(rng)};
    }
    return sample;
}

void LogScalar(const char* label, const ScalarSample& sample, uint32_t seed, bool normal)
{
    std::cout << std::setprecision(std::numeric_limits<float>::max_digits10) << "[SCALAR] " << label << " seed=" << seed
              << " distribution=" << (normal ? "normal" : "uniform") << " real=" << sample.value.real
              << " imag=" << sample.value.imag;
    if (normal) {
        std::cout << " real_mu=" << sample.realMean << " real_sigma=" << sample.realSigma
                  << " imag_mu=" << sample.imagMean << " imag_sigma=" << sample.imagSigma;
    }
    std::cout << std::endl;
}
} // namespace

TEST_P(ChemvArch35Test, CsvDriven) { RunCase(handle_, stream_, GetParam()); }

TEST_F(ChemvArch35Test, RandomScalarDistributions)
{
    // Keep every supplied CSV/performance input unchanged. This additional test
    // exercises 128 real computations with alpha and beta independently sampled.
    constexpr int kSamples = 128;
    constexpr int kSizes[] = {1, 7, 16, 33, 65};
    constexpr int kIncrements[] = {1, -1, 2, -2, 3, -3};
    int uniformCount = 0;
    int normalCount = 0;
    for (int i = 0; i < kSamples; ++i) {
        const bool normal = (i & 1) != 0;
        normal ? ++normalCount : ++uniformCount;
        const uint32_t alphaSeed = 351000U + static_cast<uint32_t>(i);
        const uint32_t betaSeed = 951000U + static_cast<uint32_t>(i);
        const auto alpha = MakeScalar(alphaSeed, normal);
        const auto beta = MakeScalar(betaSeed, normal);
        const int n = kSizes[i % 5];
        ChemvParam p(csv_map{
            {"case_name", "TC_RS_" + std::to_string(i)},
            {"uplo", (i / 2) % 2 == 0 ? "ACLBLAS_UPPER" : "ACLBLAS_LOWER"},
            {"n", std::to_string(n)},
            {"lda", std::to_string(n + 1 + (i % 3) * 7)},
            {"incx", std::to_string(kIncrements[(i / 2) % 6])},
            {"incy", std::to_string(kIncrements[(i / 12) % 6])},
            {"alpha_real", "1"},
            {"alpha_imag", "0"},
            {"beta_real", "0"},
            {"beta_imag", "0"},
            {"a_fill", "RANDOM_NORM_5_5"},
            {"x_fill", "RANDOM_NORM_5_5"},
            {"y_fill", "RANDOM_NORM_5_5"},
            {"mere_threshold", "0.00012207031"},
            {"mare_multiplier", "10"},
            {"random_seed", std::to_string(260000U + static_cast<uint32_t>(i))}});
        p.alpha = alpha.value;
        p.beta = beta.value;
        std::cout << "[SUPPLEMENTAL] case=" << p.caseName << " n=" << p.n << " uplo=" << p.uplo << " lda=" << p.lda
                  << " incx=" << p.incx << " incy=" << p.incy << std::endl;
        LogScalar("alpha", alpha, alphaSeed, normal);
        LogScalar("beta", beta, betaSeed, normal);
        ASSERT_NO_FATAL_FAILURE(RunCase(handle_, stream_, p));
    }
    EXPECT_EQ(uniformCount, 64);
    EXPECT_EQ(normalCount, 64);
}
