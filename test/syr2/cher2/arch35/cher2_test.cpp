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
#include <stdexcept>
#include <cstring>
#include <iomanip>
#include <limits>
#include <random>
#include <securec.h>
#include "blas_test.h"
#include "cher2_param.h"
#include "cher2_golden.h"
#include "cher2_npu_wrapper.h"

namespace {
using namespace cher2_test;
constexpr double TOL = 1.0 / 8192;

float FillValue(const std::string& fill, size_t index, std::mt19937& rng)
{
    if (fill == "VALUE_NORM_0") {
        return 0;
    }
    if (fill == "VALUE_NORM_INF") {
        return std::numeric_limits<float>::infinity();
    }
    if (fill == "VALUE_NORM_NAN") {
        return std::numeric_limits<float>::quiet_NaN();
    }
    if (fill == "RANDOM_ALTER") {
        return index % 2 ? -5.0f : 5.0f;
    }
    if (fill == "RANDOM_EXTREME") {
        return index % 2 ? -1.0e20f : 1.0e20f;
    }
    if (fill == "MIXED_SPECIAL") {
        const float inf = std::numeric_limits<float>::infinity();
        const float nan = std::numeric_limits<float>::quiet_NaN();
        const float values[] = {0.0f, -0.0f, 1.5f, -2.25f, inf, -inf, nan, 1.0e20f, -1.0e20f};
        return values[index % 9];
    }
    if (fill == "NORMAL") {
        return std::normal_distribution<float>(1.0f, 1.5f)(rng);
    }
    return std::uniform_real_distribution<float>(-5.0f, 5.0f)(rng);
}

bool SameNonFinite(float a, float b)
{
    if (std::isfinite(a) || std::isfinite(b)) {
        return false;
    }
    return std::isnan(a) || std::isnan(b) || std::signbit(a) == std::signbit(b);
}

struct ComponentStats {
    size_t finite = 0;
    size_t matched = 0;
    size_t special = 0;
    size_t specialMatched = 0;
    double maxAbs = 0;
    double sumRelative = 0;
    double maxRelative = 0;
    size_t relativeCount = 0;
    double mereThreshold = TOL;
    double mareMultiplier = 10;
    bool errorLimit = true;
    void Add(float actual, float golden)
    {
        if (!std::isfinite(actual) || !std::isfinite(golden)) {
            ++special;
            specialMatched += SameNonFinite(actual, golden);
            return;
        }
        ++finite;
        const double error = std::abs(static_cast<double>(actual) - golden);
        matched += error <= TOL + TOL * std::abs(static_cast<double>(golden));
        maxAbs = std::max(maxAbs, error);
        if (std::abs(golden) >= mereThreshold) {
            const double relative = error / (std::abs(static_cast<double>(golden)) + 1.0 / 16384);
            sumRelative += relative;
            maxRelative = std::max(maxRelative, relative);
            ++relativeCount;
        }
        // ULP measured on the finite side, avoiding an infinite ULP at FLT_MAX.
        const double ulp = std::abs(static_cast<double>(golden) - std::nextafter(golden, 0.0f));
        errorLimit &= error <= std::max(0.01, 32 * ulp);
    }
    void Verify(const std::string& label) const
    {
        const double mere = relativeCount ? sumRelative / relativeCount : 0;
        EXPECT_LT(mere, mereThreshold) << label;
        EXPECT_LT(maxRelative, mareMultiplier * mereThreshold) << label;
        EXPECT_TRUE(errorLimit) << label << " max_abs=" << maxAbs;
        EXPECT_GE(matched, 0.99 * finite) << label;
        EXPECT_EQ(special, specialMatched) << label;
        std::cout << "[ CHER2 ACC ] " << label << " finite=" << finite << " matched=" << matched
                  << " max_abs=" << maxAbs << " nonfinite=" << special << " nonfinite_matched=" << specialMatched
                  << " mere=" << mere << " mare=" << maxRelative << std::endl;
    }
};

void VerifyOutput(
    const Cher2Param& p, const std::vector<aclblasComplex>& actual, const std::vector<aclblasComplex>& golden,
    const std::vector<aclblasComplex>& original)
{
    ComponentStats real;
    ComponentStats imag;
    real.mereThreshold = p.mereThreshold > 0 ? p.mereThreshold : TOL;
    imag.mereThreshold = real.mereThreshold;
    real.mareMultiplier = p.mareMultiplier > 0 ? p.mareMultiplier : 10;
    imag.mareMultiplier = real.mareMultiplier;
    size_t changed = 0;
    size_t diagonalErrors = 0;
    for (int col = 0; col < p.n; ++col) {
        for (int row = 0; row < p.lda; ++row) {
            const size_t i = static_cast<size_t>(col) * p.lda + row;
            const bool referenced = row < p.n && (p.uplo == ACLBLAS_UPPER ? row <= col : row >= col);
            if (!referenced || (p.alpha.real == 0 && p.alpha.imag == 0)) {
                changed += std::memcmp(&actual[i], &original[i], sizeof(aclblasComplex)) != 0;
            } else {
                real.Add(actual[i].real, golden[i].real);
                imag.Add(actual[i].imag, golden[i].imag);
                if (row == col) {
                    diagonalErrors += actual[i].imag != 0.0f;
                }
            }
        }
    }
    EXPECT_EQ(changed, 0u) << "Unused triangle/padding/no-op changed";
    EXPECT_EQ(diagonalErrors, 0u);
    real.Verify(p.caseName + "/real");
    imag.Verify(p.caseName + "/imag");
}

bool ValidShape(const Cher2Param& p)
{
    return p.n > 0 && p.lda >= p.n && p.incx != 0 && p.incx != std::numeric_limits<int>::min() && p.incy != 0 &&
           p.incy != std::numeric_limits<int>::min();
}

std::vector<aclblasComplex> MakeVector(const std::string& fill, size_t size, std::mt19937& rng)
{
    std::vector<aclblasComplex> result(fill == "NULLPTR" ? 0 : size);
    for (size_t i = 0; i < result.size(); ++i) {
        result[i] = {FillValue(fill, 2 * i, rng), FillValue(fill, 2 * i + 1, rng)};
    }
    return result;
}

void FillMatrix(const Cher2Param& p, std::vector<aclblasComplex>& a, std::mt19937& rng, bool dirtyDiagonal)
{
    const uint32_t sentinel = 0x7fc13579;
    for (auto& value : a) {
        if (memcpy_s(&value.real, sizeof(value.real), &sentinel, sizeof(sentinel)) != EOK ||
            memcpy_s(&value.imag, sizeof(value.imag), &sentinel, sizeof(sentinel)) != EOK) {
            throw std::runtime_error("Matrix sentinel copy failed");
        }
    }
    if (!ValidShape(p) || a.empty()) {
        return;
    }
    for (int col = 0; col < p.n; ++col) {
        const int begin = p.uplo == ACLBLAS_UPPER ? 0 : col;
        const int end = p.uplo == ACLBLAS_UPPER ? col + 1 : p.n;
        for (int row = begin; row < end; ++row) {
            const size_t i = static_cast<size_t>(col) * p.lda + row;
            a[i] = {FillValue(p.aFill, 2 * i, rng), FillValue(p.aFill, 2 * i + 1, rng)};
            if (row == col) {
                a[i].imag = dirtyDiagonal ? std::numeric_limits<float>::quiet_NaN() : 0;
            }
        }
    }
}

void RunProfile(
    const Cher2Param& p, aclblasHandle_t handle, aclrtStream stream, const Cher2NpuWrapper& dx,
    const Cher2NpuWrapper& dy, const Cher2NpuWrapper& da)
{
    ASSERT_TRUE(p.caseName.rfind("TC_PF_", 0) == 0);
    const bool opMode = std::string(std::getenv("CHER2_PROFILE")) == "op";
    // msprof op performs its own five warmups for each collected launch.
    const int WARMUP = opMode ? 0 : 5;
    const char* sampleOverride = std::getenv("CHER2_PROFILE_SAMPLES");
    const int SAMPLES = opMode ? (sampleOverride ? std::stoi(sampleOverride) : 10) : 51;
    ASSERT_GE(SAMPLES, 10);
    ASSERT_LE(SAMPLES, 100);
    for (int iteration = 0; iteration < WARMUP + SAMPLES; ++iteration) {
        ASSERT_EQ(
            aclblasCher2(handle, p.uplo, p.n, &p.alpha, dx.Data(), p.incx, dy.Data(), p.incy, da.Data(), p.lda),
            ACLBLAS_STATUS_SUCCESS);
        CheckAcl(aclrtSynchronizeStream(stream));
    }
    std::cout << "[ CHER2 PERF ] " << p.caseName << " warmup=" << WARMUP << " samples=" << SAMPLES << std::endl;
}

void VerifyReadOnly(const std::vector<aclblasComplex>& original, const Cher2NpuWrapper& buffer)
{
    auto actual = original;
    buffer.Read(actual);
    if (!original.empty()) {
        EXPECT_EQ(std::memcmp(original.data(), actual.data(), original.size() * sizeof(aclblasComplex)), 0);
    }
}

void RunCase(
    const Cher2Param& p, aclblasHandle_t handle, aclrtStream stream, bool dirtyDiagonal = false,
    bool zeroColumn = false, size_t guard = 64)
{
    const bool validShape = ValidShape(p);
    // Negative cases need only non-null placeholders; no huge/negative allocations.
    const size_t xSize = validShape ? 1 + static_cast<size_t>(p.n - 1) * std::abs(static_cast<int64_t>(p.incx)) : 1;
    const size_t ySize = validShape ? 1 + static_cast<size_t>(p.n - 1) * std::abs(static_cast<int64_t>(p.incy)) : 1;
    const size_t aSize = validShape ? static_cast<size_t>(p.lda) * p.n : 1;
    std::mt19937 rng(p.randomSeed);
    auto x = MakeVector(p.xFill, xSize, rng);
    auto y = MakeVector(p.yFill, ySize, rng);
    std::vector<aclblasComplex> a(p.aFill == "NULLPTR" ? 0 : aSize);
    if (zeroColumn && !x.empty() && !y.empty()) {
        x[0] = {0, 0};
        y[0] = {0, 0};
    }
    FillMatrix(p, a, rng, dirtyDiagonal);
    const auto original = a;
    Cher2NpuWrapper dx(x, guard);
    Cher2NpuWrapper dy(y, guard);
    Cher2NpuWrapper da(a, guard);
    // Profiling replay modifies an in-place output. Run a dedicated path so
    // golden comparison always describes exactly one unprofiled update.
    if (std::getenv("CHER2_PROFILE") != nullptr) {
        RunProfile(p, handle, stream, dx, dy, da);
        return;
    }
    const auto result = aclblasCher2(
        handle, p.uplo, p.n, p.nullAlpha ? nullptr : &p.alpha, dx.Data(), p.incx, dy.Data(), p.incy, da.Data(), p.lda);
    ASSERT_EQ(result, p.expectResult) << p.caseName;
    CheckAcl(aclrtSynchronizeStream(stream));
    da.Read(a);
    if (result != ACLBLAS_STATUS_SUCCESS || p.n == 0 || a.empty()) {
        if (!a.empty()) {
            ASSERT_EQ(std::memcmp(a.data(), original.data(), a.size() * sizeof(aclblasComplex)), 0);
        }
        return;
    }
    auto golden = original;
    if (p.alpha.real != 0 || p.alpha.imag != 0) {
        Cher2Golden(p, x.data(), y.data(), golden.data());
    }
    VerifyOutput(p, a, golden, original);
    VerifyReadOnly(y, dy);
    VerifyReadOnly(x, dx);
}
void RunLargeStrideCases(aclblasHandle_t handle, aclrtStream stream)
{
    for (int n : {1, 65, 513}) {
        // n=1 needs one physical element even for the largest legal int strides.
        const int increments[] = {
            n == 1 ? std::numeric_limits<int>::min() + 1 : -1000, n == 1 ? std::numeric_limits<int>::max() : 1000};
        for (const char* uplo : {"ACLBLAS_UPPER", "ACLBLAS_LOWER"}) {
            for (int incx : increments) {
                for (int incy : increments) {
                    csv_map values{
                        {"case_name", "large_stride_" + std::to_string(n) + "_" + uplo + "_" + std::to_string(incx) +
                                          "_" + std::to_string(incy)},
                        {"uplo", uplo},
                        {"n", std::to_string(n)},
                        {"lda", std::to_string(n + 3)},
                        {"incx", std::to_string(incx)},
                        {"incy", std::to_string(incy)},
                        {"alpha_real", "1.25"},
                        {"alpha_imag", "-0.75"},
                        {"x_fill", "NORMAL"},
                        {"y_fill", "NORMAL"},
                        {"a_fill", "NORMAL"},
                        {"expect_result", "ACLBLAS_STATUS_SUCCESS"},
                        {"random_seed", "851337"}};
                    const Cher2Param p(values);
                    SCOPED_TRACE(p.caseName);
                    RunCase(p, handle, stream, true, false, 68);
                }
            }
        }
    }
}

std::vector<Cher2Param> MixedCases()
{
    std::vector<Cher2Param> result;
    std::mt19937 rng(20261028);
    const int sizes[] = {1, 3, 33, 65, 127, 513, 1025};
    for (const char* uplo : {"ACLBLAS_UPPER", "ACLBLAS_LOWER"}) {
        for (int incx : {-3, -2, -1, 1, 2, 3}) {
            for (int incy : {-3, -2, -1, 1, 2, 3}) {
                for (bool normal : {false, true}) {
                    const int id = static_cast<int>(result.size());
                    const int n = sizes[(id / 2) % 7];
                    std::uniform_real_distribution<float> uniform(-5, 5);
                    std::normal_distribution<float> gaussian(
                        uniform(rng), std::uniform_real_distribution<float>(0.1f, 2.0f)(rng));
                    const float ar = normal ? gaussian(rng) : uniform(rng);
                    const float ai = normal ? gaussian(rng) : uniform(rng);
                    const std::string fill = normal ? "NORMAL" : "RANDOM_NORM_5_5";
                    csv_map m{
                        {"case_name", "TC_MX_" + std::to_string(id)},
                        {"uplo", uplo},
                        {"n", std::to_string(n)},
                        {"lda", std::to_string(n + id % 3)},
                        {"incx", std::to_string(incx)},
                        {"incy", std::to_string(incy)},
                        {"alpha_real", std::to_string(ar)},
                        {"alpha_imag", std::to_string(ai)},
                        {"x_fill", fill},
                        {"y_fill", fill},
                        {"a_fill", fill},
                        {"random_seed", std::to_string(20261028 + id)}};
                    result.emplace_back(m);
                }
            }
        }
    }
    return result;
}

std::vector<Cher2Param> CheckedCsvCases()
{
    auto cases = GetCasesFromCsv<Cher2Param>(ReplaceFileExtension2Csv(__FILE__));
    if (cases.size() != 1200) {
        throw std::runtime_error("Cher2 requires all 1200 supplied CSV cases");
    }
    return cases;
}

} // namespace

class Cher2Test : public BlasTest<Cher2Param> {};
INSTANTIATE_TEST_SUITE_P(Cher2, Cher2Test, ::testing::ValuesIn(CheckedCsvCases()), PrintCaseInfoString<Cher2Param>);

INSTANTIATE_TEST_SUITE_P(Mixed, Cher2Test, ::testing::ValuesIn(MixedCases()), PrintCaseInfoString<Cher2Param>);

TEST_P(Cher2Test, CsvDriven)
{
    const auto& p = GetParam();
    RecordProperty("n", p.n);
    RecordProperty("uplo", std::to_string(p.uplo));
    RecordProperty("lda", p.lda);
    RecordProperty("incx", p.incx);
    RecordProperty("incy", p.incy);
    RecordProperty("alpha_real", std::to_string(p.alpha.real));
    RecordProperty("alpha_imag", std::to_string(p.alpha.imag));
    RecordProperty("x_fill", p.xFill);
    RecordProperty("y_fill", p.yFill);
    RecordProperty("a_fill", p.aFill);
    RecordProperty("seed", std::to_string(p.randomSeed));
    RunCase(p, handle_, stream_);
}

TEST_F(Cher2Test, NullHandle)
{
    EXPECT_EQ(
        aclblasCher2(nullptr, ACLBLAS_UPPER, 0, nullptr, nullptr, 0, nullptr, 0, nullptr, 0),
        ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
}

TEST_F(Cher2Test, NormalAndSpecialScalars)
{
    for (const auto& uplo : {"ACLBLAS_UPPER", "ACLBLAS_LOWER"}) {
        for (int stride : {-3, 1, 2}) {
            for (const auto& alpha : {"1", "-2.25", "inf", "nan", "0"}) {
                csv_map m{
                    {"case_name", std::string("extra_") + uplo + "_" + std::to_string(stride) + "_" + alpha},
                    {"uplo", uplo},
                    {"n", "65"},
                    {"lda", "73"},
                    {"incx", std::to_string(stride)},
                    {"alpha_real", alpha},
                    {"alpha_imag", std::string(alpha) == "0" ? "0" : "0.75"},
                    {"x_fill", "NORMAL"},
                    {"y_fill", "NORMAL"},
                    {"incy", "-2"},
                    {"a_fill", "NORMAL"},
                    {"random_seed", "91537"}};
                RunCase(Cher2Param(m), handle_, stream_, true, true, 68);
            }
        }
    }
    RunLargeStrideCases(handle_, stream_);
}

TEST_F(Cher2Test, MisalignedTileBoundaries)
{
    for (int n : {1, 63, 64, 65, 511, 512, 513, 1024, 2048, 4096}) {
        for (const char* uplo : {"ACLBLAS_UPPER", "ACLBLAS_LOWER"}) {
            csv_map m{
                {"case_name", "unaligned_" + std::to_string(n) + "_" + uplo},
                {"uplo", uplo},
                {"n", std::to_string(n)},
                {"lda", std::to_string(n + 1)},
                {"incx", "1"},
                {"incy", "1"},
                {"alpha_real", "1.25"},
                {"alpha_imag", "-0.75"},
                {"x_fill", "NORMAL"},
                {"y_fill", "NORMAL"},
                {"a_fill", "NORMAL"},
                {"random_seed", "38254"}};
            RunCase(Cher2Param(m), handle_, stream_, true, false, 68);
        }
    }
}

TEST_F(Cher2Test, MixedSpecialTilePaths)
{
    for (int n : {129, 1025}) {
        for (const char* uplo : {"ACLBLAS_UPPER", "ACLBLAS_LOWER"}) {
            for (const char* imag : {"0", "0.75"}) {
                for (const char* yFill : {"NORMAL", "MIXED_SPECIAL"}) {
                    for (size_t guard : {64, 68}) {
                        csv_map m{
                            {"case_name", std::string("mixed_special_") + std::to_string(n) + "_" + uplo + "_" + imag +
                                              "_" + yFill + "_" + std::to_string(guard)},
                            {"uplo", uplo},
                            {"n", std::to_string(n)},
                            {"lda", std::to_string(n + 4)},
                            {"incx", "1"},
                            {"incy", "1"},
                            {"alpha_real", "1"},
                            {"alpha_imag", imag},
                            {"x_fill", "MIXED_SPECIAL"},
                            {"y_fill", yFill},
                            {"a_fill", "NORMAL"},
                            {"random_seed", "12891"}};
                        RunCase(Cher2Param(m), handle_, stream_, true, true, guard);
                    }
                }
            }
        }
    }
}

TEST_F(Cher2Test, AivDispatchBoundaries)
{
    for (int n : {1023, 1024, 1025, 4095, 4096, 4097}) {
        for (const char* uplo : {"ACLBLAS_UPPER", "ACLBLAS_LOWER"}) {
            csv_map m{
                {"case_name", "aiv_boundary_" + std::to_string(n) + "_" + uplo},
                {"uplo", uplo},
                {"n", std::to_string(n)},
                {"lda", std::to_string(n + 3)},
                {"incx", "1"},
                {"incy", "1"},
                {"alpha_real", "-1.25"},
                {"alpha_imag", "0.75"},
                {"x_fill", "NORMAL"},
                {"y_fill", "NORMAL"},
                {"a_fill", "NORMAL"},
                {"random_seed", "95373"}};
            RunCase(Cher2Param(m), handle_, stream_, true, true);
        }
    }
}
