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
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>
#include "blas_test.h"
#include "csv_loader.h"
#include "fill.h"
#include "mixed_tolerance.h"
#include "srot_param.h"
#include "srot_golden.h"
#include "srot_npu_wrapper.h"
#include "verify.h"

namespace {
using namespace srot_arch22_test;
using SrotArch22Test = BlasTest<SrotParam>;

bool InvalidShape(const SrotParam& p)
{
    if (p.n < 0)
        return true;
    if (p.n == 0)
        return false;
    for (int inc : {p.incx, p.incy}) {
        if (inc == std::numeric_limits<int>::min())
            return true;
        if (static_cast<int64_t>(p.n - 1) * std::abs(static_cast<int64_t>(inc)) > std::numeric_limits<int32_t>::max())
            return true;
    }
    return false;
}

std::vector<SrotParam> LoadCases()
{
    const std::string path = ReplaceFileExtension2Csv(__FILE__);
    auto result = GetCasesFromCsv<SrotParam>(path);
    const auto slash = path.find_last_of('/');
    for (const char* name : {"srot_supplement.csv"}) {
        auto extra = GetCasesFromCsv<SrotParam>(path.substr(0, slash + 1) + name);
        result.insert(result.end(), extra.begin(), extra.end());
    }
    return result;
}

std::vector<float> MakeBuffer(const SrotParam& p, bool isX, bool dummy)
{
    const auto& fill = isX ? p.x : p.y;
    const int inc = isX ? p.incx : p.incy;
    const int64_t span =
        dummy || p.n <= 0 ? 1 : 1 + static_cast<int64_t>(p.n - 1) * std::abs(static_cast<int64_t>(inc));
    std::vector<float> output(static_cast<size_t>(span) + 2 * GUARD_ELEMENTS, GUARD_VALUE);
    if (fill.method == BlasFillMode::M_GAUSS) {
        // The shared GaussGenerator clamps samples to [-5, 5]. Task section 3.5
        // requires an untruncated normal distribution, with bounds on mean/sigma only.
        std::mt19937 rng(p.randomSeed + (isX ? 0 : 1));
        std::uniform_real_distribution<float> meanDistribution(-5.0f, 5.0f);
        std::uniform_real_distribution<float> sigmaDistribution(0.1f, 2.0f);
        const float mean = meanDistribution(rng);
        const float sigma = sigmaDistribution(rng);
        std::normal_distribution<float> normal(mean, sigma);
        for (int64_t i = 0; i < span; ++i) {
            output[GUARD_ELEMENTS + static_cast<size_t>(i)] = normal(rng);
        }
    } else if (fill.method != BlasFillMode::M_NULLPTR) {
        const auto data = makeBlasArray(span, fill, p.randomSeed + (isX ? 0 : 1));
        std::copy(data.begin(), data.end(), output.begin() + GUARD_ELEMENTS);
    }
    return output;
}

void SaveGolden(const std::string& name, const char* vectorName, const std::vector<float>& data)
{
    const char* dir = std::getenv("SROT_GOLDEN_DIR");
    if (dir == nullptr || *dir == '\0')
        return;
    const std::string path = std::string(dir) + "/" + name + "_" + vectorName + ".f32";
    std::ofstream file(path, std::ios::binary);
    ASSERT_TRUE(file.is_open()) << path;
    file.write(
        reinterpret_cast<const char*>(data.data() + GUARD_ELEMENTS),
        static_cast<std::streamsize>((data.size() - 2 * GUARD_ELEMENTS) * sizeof(float)));
    ASSERT_TRUE(file.good()) << path;
}

bool NonFiniteEquivalent(float actual, float golden)
{
    return (std::isnan(actual) && !std::isfinite(golden)) || (std::isnan(golden) && !std::isfinite(actual)) ||
           actual == golden;
}

void UpdateMaximumError(double error, float golden, double& maxError, double& maxErrorLimit)
{
    if (error > maxError) {
        maxError = error;
        // Task section 3.2 anchors the FP32 ULP limit at the maximum-error point.
        maxErrorLimit = std::max(1e-2, 32.0 * getUlpsAt(golden, 23, -126));
    }
}

template <typename Consume>
void VisitSelected(const std::vector<float>& actual, const std::vector<float>& golden, int n, int inc, Consume consume)
{
    const size_t step = static_cast<size_t>(std::abs(static_cast<int64_t>(inc)));
    // A zero stride revisits one physical element; verify its final value once.
    const int count = inc == 0 ? std::min(n, 1) : n;
    for (int i = 0; i < count; ++i) {
        const size_t index = GUARD_ELEMENTS + static_cast<size_t>(i) * step;
        consume(actual[index], golden[index]);
    }
}

size_t CountUntouchedErrors(const std::vector<float>& actual, const std::vector<float>& original, int n, int inc)
{
    const size_t span = actual.size() - 2 * GUARD_ELEMENTS;
    const size_t step = static_cast<size_t>(std::abs(static_cast<int64_t>(inc)));
    size_t errors = 0;
    for (size_t index = 0; index < actual.size(); ++index) {
        const bool inside = index >= GUARD_ELEMENTS && index < GUARD_ELEMENTS + span;
        const size_t offset = inside ? index - GUARD_ELEMENTS : 0;
        const bool selected =
            inside && n > 0 && (step == 0 ? offset == 0 : offset % step == 0 && offset / step < static_cast<size_t>(n));
        if (!selected && std::memcmp(&actual[index], &original[index], sizeof(float)) != 0)
            ++errors;
    }
    return errors;
}

bool VerifyFiniteValues(
    const std::string& name, const std::vector<float>& actual, const std::vector<float>& golden, int n, int inc,
    size_t finite, double maxErrorLimit)
{
    VerifyConfig cfg;
    applyMixedTolerance(cfg, ACL_FLOAT, golden.data(), golden.size());
    // Task section 3.2 defines this limit at the maximum-error point. Passing it
    // as the fixed limit avoids imposing a stricter per-element ULP condition.
    cfg.mixedMaxAbsErrorLimit = maxErrorLimit;
    const auto visit = [&](auto consume) {
        VisitSelected(actual, golden, n, inc, [&](float a, float g) {
            if (std::isfinite(a) && std::isfinite(g))
                consume(a, g);
        });
    };
    return Verifier::verifyEach(visit, finite, cfg, name);
}

void VerifyVector(
    const std::string& name, const char* vectorName, const std::vector<float>& actual, const std::vector<float>& golden,
    const std::vector<float>& original, int n, int inc)
{
    constexpr double tolerance = 1.0 / 8192.0;
    size_t finite = 0;
    size_t matched = 0;
    size_t nonfinite = 0;
    size_t nonfiniteMatched = 0;
    double maxError = 0.0;
    double maxErrorLimit = 1e-2;
    VisitSelected(actual, golden, n, inc, [&](float a, float g) {
        if (!std::isfinite(a) || !std::isfinite(g)) {
            ++nonfinite;
            if (NonFiniteEquivalent(a, g))
                ++nonfiniteMatched;
            return;
        }
        ++finite;
        const double error = std::abs(static_cast<double>(a) - g);
        UpdateMaximumError(error, g, maxError, maxErrorLimit);
        if (error <= tolerance + tolerance * std::abs(static_cast<double>(g)))
            ++matched;
    });
    EXPECT_TRUE(VerifyFiniteValues(name + "_" + vectorName, actual, golden, n, inc, finite, maxErrorLimit));
    const size_t untouchedErrors = CountUntouchedErrors(actual, original, n, inc);
    const double ratio = finite ? static_cast<double>(matched) / finite : 1.0;
    std::cout << "[ACCURACY] case=" << name << " vector=" << vectorName << " finite=" << finite
              << " matched=" << matched << " matched_ratio=" << std::setprecision(12) << ratio
              << " max_abs_error=" << maxError << " max_abs_error_limit=" << maxErrorLimit << " nonfinite=" << nonfinite
              << " nonfinite_matched=" << nonfiniteMatched << " untouched_errors=" << untouchedErrors << std::endl;
    EXPECT_LE(maxError, maxErrorLimit);
    EXPECT_EQ(nonfiniteMatched, nonfinite);
    EXPECT_EQ(untouchedErrors, 0U);
}

INSTANTIATE_TEST_SUITE_P(Srot, SrotArch22Test, ::testing::ValuesIn(LoadCases()), PrintCaseInfoString<SrotParam>);

TEST_P(SrotArch22Test, CsvDriven)
{
    const auto& p = GetParam();
    const bool invalid = InvalidShape(p);
    const bool dummy = invalid || p.expectResult != ACLBLAS_STATUS_SUCCESS;
    auto x = MakeBuffer(p, true, dummy);
    auto y = MakeBuffer(p, false, dummy);
    const auto beforeX = x;
    const auto beforeY = y;
    const bool performance = p.caseName.find("TC_PF_") == 0;
    const int iterations = performance ? 10 : 1;
    const auto ret = Execute(
        handle_, stream_, p.n, p.incx, p.incy, x, y, p.c, p.s, p.csPtrMode, p.x.method == BlasFillMode::M_NULLPTR,
        p.y.method == BlasFillMode::M_NULLPTR, false, false, iterations);
    std::cout << "[STATUS] case=" << p.caseName << " expected=" << static_cast<int>(p.expectResult)
              << " actual=" << static_cast<int>(ret)
              << " contract_conflict=" << (invalid && p.expectResult == ACLBLAS_STATUS_SUCCESS) << std::endl;
    EXPECT_EQ(ret, p.expectResult);
    if (ret != ACLBLAS_STATUS_SUCCESS || p.n == 0) {
        EXPECT_EQ(std::memcmp(x.data(), beforeX.data(), x.size() * sizeof(float)), 0);
        EXPECT_EQ(std::memcmp(y.data(), beforeY.data(), y.size() * sizeof(float)), 0);
        return;
    }
    // Invalid-success conflicts are never sent to Netlib with the dummy span.
    ASSERT_FALSE(dummy) << "Operator accepted an invalid shape; original CSV expectation is contradictory";
    if (performance) {
        // This entry collects launches; external msprof validation checks the performance threshold.
        std::cout << "[PROFILE_READY] case=" << p.caseName << " successful_launches=" << iterations
                  << " measurement=msprof_op_Task_Duration_us warmup=5" << std::endl;
        return;
    }
    auto goldenX = beforeX;
    auto goldenY = beforeY;
    ASSERT_EQ(
        aclblasSrot_cpu<SrotGolden::Arch22>(
            handle_, p.n, goldenX.data() + GUARD_ELEMENTS, p.incx, goldenY.data() + GUARD_ELEMENTS, p.incy, &p.c, &p.s),
        ACLBLAS_STATUS_SUCCESS);
    SaveGolden(p.caseName, "x", goldenX);
    SaveGolden(p.caseName, "y", goldenY);
    VerifyVector(p.caseName, "x", x, goldenX, beforeX, p.n, p.incx);
    VerifyVector(p.caseName, "y", y, goldenY, beforeY, p.n, p.incy);
}

TEST_F(SrotArch22Test, NullHandlePrecedence)
{
    EXPECT_EQ(aclblasSrot(nullptr, 0, nullptr, 0, nullptr, 0, nullptr, nullptr), ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
    EXPECT_EQ(aclblasSrot(nullptr, 1, nullptr, 1, nullptr, 1, nullptr, nullptr), ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
}

TEST_F(SrotArch22Test, ZeroLengthSkipsAllDataAndStrideValidation)
{
    EXPECT_EQ(
        aclblasSrot(handle_, 0, nullptr, 0, nullptr, std::numeric_limits<int>::min(), nullptr, nullptr),
        ACLBLAS_STATUS_SUCCESS);
}

TEST_F(SrotArch22Test, InvalidStrideAndLengthBoundaries)
{
    const int min = std::numeric_limits<int>::min();
    const int max = std::numeric_limits<int>::max();
    const int cases[][3] = {{-1, 1, 1},   {min, 1, 1}, {1, min, 1}, {1, 1, min}, {4, min, 1}, {4, 1, min}, {3, max, 1},
                            {3, 1, -max}, {-1, 0, 0},  {1, min, 0}, {1, 0, min}, {3, max, 0}, {3, 0, -max}};
    for (const auto& args : cases) {
        std::vector<float> x(2 * GUARD_ELEMENTS + 1, GUARD_VALUE);
        std::vector<float> y = x;
        EXPECT_EQ(
            Execute(handle_, stream_, args[0], args[1], args[2], x, y, 0.6f, 0.8f, "host", false, false),
            ACLBLAS_STATUS_INVALID_VALUE)
            << "n=" << args[0] << " incx=" << args[1] << " incy=" << args[2];
        EXPECT_TRUE(std::all_of(x.begin(), x.end(), [](float v) { return v == GUARD_VALUE; }));
        EXPECT_TRUE(std::all_of(y.begin(), y.end(), [](float v) { return v == GUARD_VALUE; }));
    }
}

void CheckZeroStrideSequence(aclblasHandle_t handle, aclrtStream stream, int n, int incx, int incy, const char* mode)
{
    const std::string name = "ZeroStrideSequentialUpdates_" + std::to_string(n) + "_" + std::to_string(incx) + "_" +
                             std::to_string(incy) + "_" + mode;
    SCOPED_TRACE(name);
    const SrotParam p(
        {{"case_name", name},
         {"n", std::to_string(n)},
         {"incx", std::to_string(incx)},
         {"incy", std::to_string(incy)},
         {"c", "0.6"},
         {"s", "-0.5"}});
    auto x = MakeBuffer(p, true, false);
    auto y = MakeBuffer(p, false, false);
    const auto beforeX = x;
    const auto beforeY = y;
    auto goldenX = x;
    auto goldenY = y;
    constexpr int iterations = 3;
    for (int i = 0; i < iterations; ++i) {
        ASSERT_EQ(
            aclblasSrot_cpu<SrotGolden::Arch22>(
                handle, n, goldenX.data() + GUARD_ELEMENTS, incx, goldenY.data() + GUARD_ELEMENTS, incy, &p.c, &p.s),
            ACLBLAS_STATUS_SUCCESS);
    }
    ASSERT_EQ(
        Execute(handle, stream, n, incx, incy, x, y, p.c, p.s, mode, false, false, false, false, iterations),
        ACLBLAS_STATUS_SUCCESS);
    VerifyVector(name, "x", x, goldenX, beforeX, n, incx);
    VerifyVector(name, "y", y, goldenY, beforeY, n, incy);
}

TEST_F(SrotArch22Test, ZeroStrideSequentialUpdates)
{
    const int strides[][2] = {{0, 2}, {3, 0}, {0, -2}, {-3, 0}, {0, 0}};
    for (const auto& pair : strides) {
        for (const char* mode : {"host", "device", "mixed", "mixed_reverse"}) {
            for (int n : {1, 9, 33, 255, 256, 257, 513}) {
                CheckZeroStrideSequence(handle_, stream_, n, pair[0], pair[1], mode);
            }
        }
    }
    std::vector<float> x(2 * GUARD_ELEMENTS + 1, GUARD_VALUE);
    std::vector<float> y = x;
    x[GUARD_ELEMENTS] = 2.0f;
    y[GUARD_ELEMENTS] = -3.0f;
    const auto beforeX = x;
    const auto beforeY = y;
    ASSERT_EQ(Execute(handle_, stream_, 3, 0, 0, x, y, 0.0f, 1.0f, "host", false, false), ACLBLAS_STATUS_SUCCESS);
    // Three serial rotations: (2, -3) -> (-3, -2) -> (-2, 3) -> (3, 2).
    EXPECT_FLOAT_EQ(x[GUARD_ELEMENTS], 3.0f);
    EXPECT_FLOAT_EQ(y[GUARD_ELEMENTS], 2.0f);
    EXPECT_EQ(CountUntouchedErrors(x, beforeX, 3, 0), 0U);
    EXPECT_EQ(CountUntouchedErrors(y, beforeY, 3, 0), 0U);
}

TEST_F(SrotArch22Test, RepeatIsBitwiseDeterministic)
{
    csv_map row = {{"case_name", "determinism"}, {"n", "257"}, {"incx", "-3"}, {"incy", "7"}};
    const SrotParam p(row);
    auto x = MakeBuffer(p, true, false);
    auto y = MakeBuffer(p, false, false);
    auto repeatX = x;
    auto repeatY = y;
    ASSERT_EQ(
        Execute(handle_, stream_, p.n, p.incx, p.incy, x, y, p.c, p.s, "mixed", false, false), ACLBLAS_STATUS_SUCCESS);
    ASSERT_EQ(
        Execute(handle_, stream_, p.n, p.incx, p.incy, repeatX, repeatY, p.c, p.s, "mixed", false, false),
        ACLBLAS_STATUS_SUCCESS);
    EXPECT_EQ(std::memcmp(x.data(), repeatX.data(), x.size() * sizeof(float)), 0);
    EXPECT_EQ(std::memcmp(y.data(), repeatY.data(), y.size() * sizeof(float)), 0);
}

TEST_F(SrotArch22Test, SingleElementAllowsIntMaxStrides)
{
    const int max = std::numeric_limits<int>::max();
    const int strides[][2] = {{max, -max}, {0, max}, {max, 0}, {0, -max}, {-max, 0}};
    for (const auto& pair : strides) {
        const std::string name =
            "SingleElementAllowsIntMaxStrides_" + std::to_string(pair[0]) + "_" + std::to_string(pair[1]);
        SCOPED_TRACE(name);
        std::vector<float> x(2 * GUARD_ELEMENTS + 1, GUARD_VALUE);
        std::vector<float> y = x;
        x[GUARD_ELEMENTS] = 2.0f;
        y[GUARD_ELEMENTS] = -3.0f;
        const auto beforeX = x;
        const auto beforeY = y;
        auto goldenX = x;
        auto goldenY = y;
        const float c = 0.6f;
        const float s = -0.5f;
        ASSERT_EQ(
            aclblasSrot_cpu<SrotGolden::Arch22>(
                handle_, 1, goldenX.data() + GUARD_ELEMENTS, pair[0], goldenY.data() + GUARD_ELEMENTS, pair[1], &c, &s),
            ACLBLAS_STATUS_SUCCESS);
        ASSERT_EQ(
            Execute(handle_, stream_, 1, pair[0], pair[1], x, y, c, s, "device", false, false), ACLBLAS_STATUS_SUCCESS);
        VerifyVector(name, "x", x, goldenX, beforeX, 1, pair[0]);
        VerifyVector(name, "y", y, goldenY, beforeY, 1, pair[1]);
    }
}

void WriteAsyncScalar(SrotDeviceBuffer& destination, const float* value, aclrtStream stream)
{
    CheckAcl(
        aclrtMemcpyAsync(destination.Data(), sizeof(float), value, sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE, stream),
        "async scalar update");
}

void CheckAsyncScalarUpdates(aclblasHandle_t handle, aclrtStream stream, const std::string& name, int incx, int incy)
{
    constexpr int n = 33;
    const SrotParam p(
        {{"case_name", name},
         {"n", std::to_string(n)},
         {"incx", std::to_string(incx)},
         {"incy", std::to_string(incy)}});
    auto x = MakeBuffer(p, true, false);
    auto y = MakeBuffer(p, false, false);
    const auto beforeX = x;
    const auto beforeY = y;
    auto goldenX = x;
    auto goldenY = y;
    SrotDeviceBuffer dx(x.size());
    SrotDeviceBuffer dy(y.size());
    SrotDeviceBuffer dc(1);
    SrotDeviceBuffer ds(1);
    dx.Write(x.data());
    dy.Write(y.data());
    void* raw = nullptr;
    CheckAcl(aclrtMallocHost(&raw, 4 * sizeof(float)), "aclrtMallocHost");
    std::unique_ptr<float, decltype(&aclrtFreeHost)> pinned(static_cast<float*>(raw), aclrtFreeHost);
    const float coefficients[] = {0.6f, -0.5f, 0.0f, 1.0f};
    std::copy(coefficients, coefficients + 4, pinned.get());
    for (int i = 0; i < 2; ++i) {
        WriteAsyncScalar(dc, pinned.get() + 2 * i, stream);
        WriteAsyncScalar(ds, pinned.get() + 2 * i + 1, stream);
        ASSERT_EQ(
            aclblasSrot(handle, n, dx.Data(GUARD_ELEMENTS), incx, dy.Data(GUARD_ELEMENTS), incy, dc.Data(), ds.Data()),
            ACLBLAS_STATUS_SUCCESS);
        ASSERT_EQ(
            aclblasSrot_cpu<SrotGolden::Arch22>(
                handle, n, goldenX.data() + GUARD_ELEMENTS, incx, goldenY.data() + GUARD_ELEMENTS, incy,
                coefficients + 2 * i, coefficients + 2 * i + 1),
            ACLBLAS_STATUS_SUCCESS);
    }
    CheckAcl(aclrtSynchronizeStream(stream), "async scalar stream completion");
    dx.Read(x.data());
    dy.Read(y.data());
    SaveGolden(name, "x", goldenX);
    SaveGolden(name, "y", goldenY);
    VerifyVector(name, "x", x, goldenX, beforeX, n, incx);
    VerifyVector(name, "y", y, goldenY, beforeY, n, incy);
}

TEST_F(SrotArch22Test, AsyncDeviceScalarUpdatesOnBoundStream)
{
    CheckAsyncScalarUpdates(handle_, stream_, "AsyncDeviceScalarUpdatesOnBoundStream", 1, 1);
}

TEST_F(SrotArch22Test, ZeroStrideAsyncDeviceScalarUpdatesOnBoundStream)
{
    for (const auto& pair : {std::make_pair(0, 0), std::make_pair(0, -3), std::make_pair(2, 0)}) {
        const std::string name =
            "ZeroStrideAsyncDeviceScalarUpdates_" + std::to_string(pair.first) + "_" + std::to_string(pair.second);
        SCOPED_TRACE(name);
        CheckAsyncScalarUpdates(handle_, stream_, name, pair.first, pair.second);
    }
}

TEST_F(SrotArch22Test, NullScalarPointers)
{
    for (bool nullC : {false, true}) {
        std::vector<float> x(2 * GUARD_ELEMENTS + 1, GUARD_VALUE);
        std::vector<float> y = x;
        EXPECT_EQ(
            Execute(handle_, stream_, 1, 1, 1, x, y, 0.6f, 0.8f, "host", false, false, nullC, !nullC),
            ACLBLAS_STATUS_INVALID_VALUE);
        EXPECT_TRUE(std::all_of(x.begin(), x.end(), [](float v) { return v == GUARD_VALUE; }));
        EXPECT_TRUE(std::all_of(y.begin(), y.end(), [](float v) { return v == GUARD_VALUE; }));
    }
}
} // namespace
