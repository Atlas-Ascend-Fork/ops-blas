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
#include <cstring>
#include <iomanip>
#include <limits>
#include <random>
#include <set>
#include "blas_test.h"
#include "chpr_param.h"
#include "chpr_npu_wrapper.h"
#include "cblas_compat.h"
#include "verify.h"

namespace {
using namespace chpr35;
std::vector<ChprParam> LoadChprArch35Cases(const std::string& defaultPath)
{
    const char* path = std::getenv("CHPR_TEST_CSV");
    auto cases = GetCasesFromCsv<ChprParam>(path == nullptr ? defaultPath : path);
    std::set<std::string> ids;
    size_t perf = 0;
    for (const auto& p : cases) {
        if (!ids.insert(p.caseName).second) {
            throw std::runtime_error("Duplicate CHPR case: " + p.caseName);
        }
        perf += p.caseName.rfind("TC_PF_", 0) == 0;
    }
    if (cases.size() != 1200 || perf != 200) {
        throw std::runtime_error("CHPR requires unchanged 1200-case CSV (1000 accuracy + 200 performance)");
    }
    return cases;
}
std::vector<aclblasComplex> MakeComplex(size_t count, const BlasFillMode& fill, uint32_t seed)
{
    auto re = makeBlasArray(static_cast<int64_t>(count), fill, seed);
    auto im = makeBlasArray(static_cast<int64_t>(count), fill, seed + 1000U);
    std::vector<aclblasComplex> data(count);
    for (size_t i = 0; i < count; ++i) {
        data[i] = {re[i], im[i]};
    }
    return data;
}

bool CompareComponent(
    const std::string& id, const char* component, const std::vector<aclblasComplex>& out,
    const std::vector<aclblasComplex>& golden, bool imag, double mereThreshold, double mareMultiplier)
{
    constexpr double tol = 1.0 / 8192;
    size_t finite = 0, matched = 0, special = 0, specialMatched = 0, boundFailures = 0;
    double maxAbs = 0;
    for (size_t i = 0; i < out.size(); ++i) {
        const float a = imag ? out[i].imag : out[i].real;
        const float g = imag ? golden[i].imag : golden[i].real;
        if (!std::isfinite(a) || !std::isfinite(g)) {
            ++special;
            specialMatched +=
                (!std::isfinite(a) && !std::isfinite(g) &&
                 (std::isnan(a) || std::isnan(g) || std::signbit(a) == std::signbit(g)));
            continue;
        }
        ++finite;
        const double error = std::abs(static_cast<double>(a) - g);
        matched += error <= tol + tol * std::abs(static_cast<double>(g));
        maxAbs = std::max(maxAbs, error);
        // The 32-ULP alternative applies at each reference value, never to a case-wide magnitude.
        boundFailures += error > std::max(0.01, 32 * getUlpsAt(g, 23, -126));
    }
    const double ratio = finite == 0 ? 1.0 : static_cast<double>(matched) / finite;
    const bool pass = ratio >= 0.99 && boundFailures == 0 && special == specialMatched;
    std::cout << std::setprecision(17) << "CHPR_METRIC {\"case\":\"" << id << "\",\"component\":\"" << component
              << "\",\"finite\":" << finite << ",\"matched\":" << matched << ",\"matched_ratio\":" << ratio
              << ",\"max_abs_error\":" << maxAbs << ",\"bound_failures\":" << boundFailures
              << ",\"nonfinite\":" << special << ",\"nonfinite_matched\":" << specialMatched
              << ",\"pass\":" << (pass ? "true" : "false") << "}" << std::endl;
    // Supplementary framework MERE/MARE reporting; the task mixed standard above controls GTest.
    if (mereThreshold > 0 && mareMultiplier > 0) {
        MereMareStrategy supplementary(mereThreshold, mareMultiplier);
        supplementary.verifyEach(
            [&](auto consume) {
                for (size_t i = 0; i < out.size(); ++i) {
                    consume(imag ? out[i].imag : out[i].real, imag ? golden[i].imag : golden[i].real);
                }
            },
            out.size(), id + "_" + component + "_supplementary");
    }
    return pass;
}

void RunValid(
    aclblasHandle_t handle, aclrtStream stream, const std::string& id, aclblasFillMode_t uplo, int n, float alpha,
    int incx, const std::vector<aclblasComplex>& x, const std::vector<aclblasComplex>& ap, int calls = 1,
    double mere = 0, double mare = 0)
{
    DeviceBuffer dx(x), dap(ap);
    auto golden = ap;
    for (int i = 0; i < calls; ++i) {
        ASSERT_EQ(aclblasChpr(handle, uplo, n, &alpha, dx.get(), incx, dap.get()), ACLBLAS_STATUS_SUCCESS);
        cblas_chpr(CblasColMajor, ToCblasUplo(uplo), n, alpha, x.data(), incx, golden.data());
    }
    CheckAcl(aclrtSynchronizeStream(stream));
    const auto out = dap.Read();
    const auto xAfter = dx.Read();
    EXPECT_EQ(std::memcmp(x.data(), xAfter.data(), x.size() * sizeof(aclblasComplex)), 0) << "x is read-only";
    if (alpha == 0.0f) {
        EXPECT_EQ(std::memcmp(ap.data(), out.data(), ap.size() * sizeof(aclblasComplex)), 0) << "exact no-op";
    } else {
        for (int j = 0; j < n; ++j) {
            EXPECT_EQ(out[DiagonalIndex(j, n, uplo)].imag, 0.0f) << "diagonal " << j;
        }
    }
    EXPECT_TRUE(CompareComponent(id, "real", out, golden, false, mere, mare));
    EXPECT_TRUE(CompareComponent(id, "imag", out, golden, true, mere, mare));
}

void RunSeededDistributions(aclblasHandle_t handle, aclrtStream stream)
{
    // Eight uniform and eight normal cases: independently sampled complex components and real alpha.
    std::mt19937 rng(20261028U);
    std::uniform_real_distribution<float> uniform(-5, 5), sigma(0.1f, 2);
    for (int c = 0; c < 16; ++c) {
        const bool normal = c >= 8;
        std::normal_distribution<float> gaussian(uniform(rng), sigma(rng));
        auto sample = [&]() { return normal ? gaussian(rng) : uniform(rng); };
        const int n = 7 + c * 2, incx = (c % 2 == 0 ? 1 : -1) * (1 + c % 7);
        const auto uplo = c % 2 == 0 ? ACLBLAS_UPPER : ACLBLAS_LOWER;
        const float alpha = sample();
        std::vector<aclblasComplex> x(VectorSize(n, incx)), ap(PackedSize(n));
        for (auto& v : x) {
            v = {sample(), sample()};
        }
        for (auto& v : ap) {
            v = {sample(), sample()};
        }
        for (int j = 0; j < n; ++j) {
            ap[DiagonalIndex(j, n, uplo)].imag = j % 2 == 0 ? NAN : 12345.0f;
        }
        // Netlib negative-stride origin is (1-n)*incx, covered by the reference call.
        x[static_cast<size_t>(n / 2) * static_cast<size_t>(std::abs(incx))] = {0, 0};
        RunValid(handle, stream, "SUP_dist_" + std::to_string(c), uplo, n, alpha, incx, x, ap, 3);
        RunValid(handle, stream, "SUP_alpha0_" + std::to_string(c), uplo, n, -0.0f, incx, x, ap);
    }
}
} // namespace

class ChprArch35Test : public BlasTest<ChprParam> {};
INSTANTIATE_TEST_SUITE_P(
    Chpr, ChprArch35Test, ::testing::ValuesIn(LoadChprArch35Cases(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<ChprParam>);

TEST_P(ChprArch35Test, CsvDriven)
{
    const auto& p = GetParam();
    const bool badShape =
        p.n <= 0 || p.incx == 0 || p.incx == INT_MIN || (p.uplo != ACLBLAS_UPPER && p.uplo != ACLBLAS_LOWER);
    const bool xNull = p.x.method == BlasFillMode::M_NULLPTR;
    const bool apNull = p.ap.method == BlasFillMode::M_NULLPTR;
    if (badShape || p.alphaNull || xNull || apNull) {
        // Validation calls do not allocate using invalid n/incx; non-null placeholders must never be accessed.
        aclblasComplex sentinel{17.0f, -23.0f};
        const auto before = sentinel;
        const auto status = aclblasChpr(
            handle_, p.uplo, p.n, p.alphaNull ? nullptr : &p.alpha, xNull ? nullptr : &sentinel, p.incx,
            apNull ? nullptr : &sentinel);
        EXPECT_EQ(status, p.expectResult);
        CheckAcl(aclrtSynchronizeStream(stream_));
        EXPECT_EQ(std::memcmp(&sentinel, &before, sizeof(sentinel)), 0);
        std::cout << "CHPR_STATUS {\"case\":\"" << p.caseName << "\",\"actual\":" << status
                  << ",\"expected\":" << p.expectResult << "}" << std::endl;
        return;
    }
    ASSERT_EQ(p.expectResult, ACLBLAS_STATUS_SUCCESS);
    auto x = MakeComplex(VectorSize(p.n, p.incx), p.x, p.randomSeed);
    auto ap = MakeComplex(PackedSize(p.n), p.ap, p.randomSeed + 1);
    for (int j = 0; j < p.n; ++j) {
        ap[DiagonalIndex(j, p.n, p.uplo)].imag = 0;
    }
    RunValid(handle_, stream_, p.caseName, p.uplo, p.n, p.alpha, p.incx, x, ap, 1, p.mereThreshold, p.mareMultiplier);
}

TEST_F(ChprArch35Test, HandleAndQuickReturnPrecedence)
{
    float alpha = 0;
    EXPECT_EQ(aclblasChpr(nullptr, ACLBLAS_UPPER, 3, &alpha, nullptr, 1, nullptr), ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
    EXPECT_EQ(
        aclblasChpr(nullptr, static_cast<aclblasFillMode_t>(999), 0, nullptr, nullptr, INT_MIN, nullptr),
        ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
    EXPECT_EQ(
        aclblasChpr(nullptr, ACLBLAS_UPPER, -1, nullptr, nullptr, INT_MIN, nullptr), ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
    EXPECT_EQ(
        aclblasChpr(handle_, static_cast<aclblasFillMode_t>(999), 0, nullptr, nullptr, INT_MIN, nullptr),
        ACLBLAS_STATUS_SUCCESS);
    EXPECT_EQ(
        aclblasChpr(handle_, ACLBLAS_UPPER, -1, nullptr, nullptr, INT_MIN, nullptr), ACLBLAS_STATUS_INVALID_VALUE);
}

TEST_F(ChprArch35Test, GmFallbackAndWideStride)
{
    struct Case {
        int n;
        int incx;
        aclblasFillMode_t uplo;
        const char* id;
    };
    const Case cases[] = {
        {4097, 17, ACLBLAS_UPPER, "SUP_gm_upper_stride17"},  {4097, -17, ACLBLAS_LOWER, "SUP_gm_lower_strideN17"},
        {1, INT_MAX, ACLBLAS_UPPER, "SUP_n1_stride_intmax"}, {2, 65537, ACLBLAS_UPPER, "SUP_n2_stride65537"},
        {2, -65537, ACLBLAS_LOWER, "SUP_n2_strideN65537"},
    };
    const BlasFillMode fill("RANDOM_NORM_1_1");
    uint32_t seed = 20261029U;
    for (const auto& c : cases) {
        auto x = MakeComplex(VectorSize(c.n, c.incx), fill, seed++);
        auto ap = MakeComplex(PackedSize(c.n), fill, seed++);
        for (int j = 0; j < c.n; ++j) {
            ap[DiagonalIndex(j, c.n, c.uplo)].imag = 0;
        }
        RunValid(handle_, stream_, c.id, c.uplo, c.n, 0.75f, c.incx, x, ap);
    }
}

TEST_F(ChprArch35Test, ScalarNonfiniteAndZeroColumns)
{
    struct ScalarCase {
        float alpha;
        const char* name;
    };
    const ScalarCase scalars[] = {
        {std::numeric_limits<float>::quiet_NaN(), "nan"},
        {std::numeric_limits<float>::infinity(), "posinf"},
        {-std::numeric_limits<float>::infinity(), "neginf"},
    };
    constexpr int N = 5;
    const aclblasComplex logical[] = {{0.0f, 0.0f}, {1.0f, 2.0f}, {0.0f, 0.0f}, {-0.5f, 0.75f}, {0.0f, 0.0f}};
    const BlasFillMode fill("RANDOM_NORM_1_1");
    for (const auto& scalar : scalars) {
        for (const auto uplo : {ACLBLAS_UPPER, ACLBLAS_LOWER}) {
            const int incx = uplo == ACLBLAS_UPPER ? 3 : -3;
            auto x = MakeComplex(VectorSize(N, incx), fill, 20261030U);
            for (int j = 0; j < N; ++j) {
                const size_t physical = static_cast<size_t>(incx > 0 ? j : N - 1 - j) * 3U;
                x[physical] = logical[j];
            }
            auto ap = MakeComplex(PackedSize(N), fill, 20261031U);
            for (int j = 0; j < N; ++j) {
                ap[DiagonalIndex(j, N, uplo)].imag = 91.0f;
            }
            const std::string id =
                std::string("SUP_scalar_") + scalar.name + (uplo == ACLBLAS_UPPER ? "_upper" : "_lower");
            RunValid(handle_, stream_, id, uplo, N, scalar.alpha, incx, x, ap);
        }
    }
}

TEST_F(ChprArch35Test, SeededUniformNormalAndAsyncStreams)
{
    RunSeededDistributions(handle_, stream_);
    // Two independently bound streams enqueue multiple calls before either stream is synchronized.
    aclblasHandle_t handles[2] = {nullptr, nullptr};
    aclrtStream streams[2] = {nullptr, nullptr};
    std::vector<aclblasComplex> x(31, {1.25f, -0.5f}), initial(PackedSize(31), {0.25f, 0});
    DeviceBuffer dx(x), ap0(initial), ap1(initial);
    DeviceBuffer* outputs[2] = {&ap0, &ap1};
    float alpha = 0.75f;
    for (int s = 0; s < 2; ++s) {
        ASSERT_EQ(aclblasCreate(&handles[s]), ACLBLAS_STATUS_SUCCESS);
        CheckAcl(aclrtCreateStream(&streams[s]));
        ASSERT_EQ(aclblasSetStream(handles[s], streams[s]), ACLBLAS_STATUS_SUCCESS);
        for (int k = 0; k < 3; ++k) {
            EXPECT_EQ(
                aclblasChpr(handles[s], ACLBLAS_UPPER, 31, &alpha, dx.get(), 1, outputs[s]->get()),
                ACLBLAS_STATUS_SUCCESS);
        }
    }
    auto golden = initial;
    for (int k = 0; k < 3; ++k) {
        cblas_chpr(CblasColMajor, CblasUpper, 31, alpha, x.data(), 1, golden.data());
    }
    for (int s = 0; s < 2; ++s) {
        CheckAcl(aclrtSynchronizeStream(streams[s]));
        const auto out = outputs[s]->Read();
        EXPECT_TRUE(CompareComponent("SUP_stream_" + std::to_string(s), "real", out, golden, false, 0, 0));
        EXPECT_TRUE(CompareComponent("SUP_stream_" + std::to_string(s), "imag", out, golden, true, 0, 0));
        EXPECT_EQ(aclblasDestroy(handles[s]), ACLBLAS_STATUS_SUCCESS);
        CheckAcl(aclrtDestroyStream(streams[s]));
    }
}
