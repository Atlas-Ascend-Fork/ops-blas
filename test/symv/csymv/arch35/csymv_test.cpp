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
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <new>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "blas_test.h"
#include "csv_loader.h"
#include "csymv_golden.h"
#include "csymv_npu_wrapper.h"
#include "csymv_param.h"

class CsymvArch35Test : public ::testing::TestWithParam<CsymvParam> {
protected:
    static void SetUpTestSuite()
    {
        const aclError initRet = aclInit(nullptr);
        ASSERT_TRUE(initRet == ACL_SUCCESS || initRet == ACL_ERROR_REPEAT_INITIALIZE)
            << "aclInit failed with error: " << initRet;
        aclInitialized_ = true;
        ASSERT_EQ(aclrtSetDevice(TEST_DEVICE_ID), ACL_SUCCESS);
        deviceSet_ = true;
        ASSERT_EQ(aclblasCreate(&handle_), ACLBLAS_STATUS_SUCCESS);
        ASSERT_EQ(aclrtCreateStream(&stream_), ACL_SUCCESS);
        ASSERT_EQ(aclblasSetStream(handle_, stream_), ACLBLAS_STATUS_SUCCESS);
    }

    static void TearDownTestSuite()
    {
        if (stream_ != nullptr) {
            // issue #605: 收尾同步失败仅记录，不在清理路径抛出
            if (aclrtSynchronizeStream(stream_) != ACL_SUCCESS) {
                std::cerr << "csymv_test: teardown aclrtSynchronizeStream failed" << std::endl;
            }
        }
        if (handle_ != nullptr) {
            aclblasDestroy(handle_);
            handle_ = nullptr;
        }
        if (stream_ != nullptr) {
            aclrtDestroyStream(stream_);
            stream_ = nullptr;
        }
        if (deviceSet_) {
            aclrtResetDevice(TEST_DEVICE_ID);
            deviceSet_ = false;
        }
        if (aclInitialized_) {
            aclFinalize();
            aclInitialized_ = false;
        }
    }

    static aclblasHandle_t handle_;
    static aclrtStream stream_;
    static bool aclInitialized_;
    static bool deviceSet_;
};

aclblasHandle_t CsymvArch35Test::handle_ = nullptr;
aclrtStream CsymvArch35Test::stream_ = nullptr;
bool CsymvArch35Test::aclInitialized_ = false;
bool CsymvArch35Test::deviceSet_ = false;

namespace {

size_t StorageLength(int n, int increment)
{
    if (n <= 0) {
        return 0;
    }
    const int64_t absIncrement = increment >= 0 ? static_cast<int64_t>(increment) : -static_cast<int64_t>(increment);
    return 1U + static_cast<size_t>(n - 1) * static_cast<size_t>(absIncrement);
}

std::vector<aclblasComplex> MakeComplexStorage(
    size_t count, const BlasFillMode& fill, uint32_t seed, bool useNormalDistribution)
{
    if (count == 0 || fill.method == BlasFillMode::M_NULLPTR) {
        return {};
    }
    if (useNormalDistribution && fill.method == BlasFillMode::M_RANDOM && fill.pattern == BlasFillMode::P_NORM) {
        const float mean = -5.0f + 10.0f * static_cast<float>(seed % 10001U) / 10000.0f;
        const float sigma = 0.1f + 1.9f * static_cast<float>((seed / 17U) % 1901U) / 1900.0f;
        std::mt19937 realRng(seed == 0U ? 42U : seed);
        std::mt19937 imagRng(seed == 0U ? 1042U : seed + 1000U);
        std::normal_distribution<float> realDist(mean, sigma);
        std::normal_distribution<float> imagDist(mean, sigma);
        std::vector<aclblasComplex> data(count);
        for (auto& value : data) {
            value = {realDist(realRng), imagDist(imagRng)};
        }
        return data;
    }
    return makeBlasComplexMatrix(static_cast<int>(count), 1, static_cast<int>(count), fill, seed);
}

void PoisonUnusedTriangle(std::vector<aclblasComplex>& matrix, aclblasFillMode_t uplo, int n, int lda)
{
    if (n <= 0 || lda < n || matrix.size() < static_cast<size_t>(lda) * n) {
        return;
    }
    const float nan = std::numeric_limits<float>::quiet_NaN();
    for (int col = 0; col < n; ++col) {
        for (int row = 0; row < n; ++row) {
            const bool unused = uplo == ACLBLAS_UPPER ? row > col : row < col;
            if (unused) {
                matrix[static_cast<int64_t>(col) * lda + row] = {nan, nan};
            }
        }
    }
}

struct CsymvHostData {
    std::vector<aclblasComplex> A;
    std::vector<aclblasComplex> x;
    std::vector<aclblasComplex> y;
};

struct CsymvPerformanceBaseline {
    aclblasFillMode_t uplo;
    int n;
    int incx;
    int incy;
    double gpuMs;

    explicit CsymvPerformanceBaseline(const csv_map& row)
        : uplo(parseFillMode(ReadMap(row, "uplo"))),
          n(parseInt(ReadMap(row, "n"))),
          incx(parseInt(ReadMap(row, "incx", "1"))),
          incy(parseInt(ReadMap(row, "incy", "1"))),
          gpuMs(parseDouble(ReadMap(row, "gpu_ms")))
    {}
};

std::vector<CsymvParam> LoadCsymvCases(const std::string& csvPath)
{
    std::vector<CsymvParam> cases = GetCasesFromCsv<CsymvParam>(csvPath);
    const size_t separator = csvPath.find_last_of("/\\");
    const std::string baselinePath =
        (separator == std::string::npos ? std::string() : csvPath.substr(0, separator + 1U)) + "gpu_baseline.csv";
    const std::vector<CsymvPerformanceBaseline> baselines = GetCasesFromCsv<CsymvPerformanceBaseline>(baselinePath);
    size_t baselineIndex = 0;
    for (auto& param : cases) {
        if (!param.IsPerformanceCase()) {
            continue;
        }
        if (baselineIndex >= baselines.size()) {
            throw std::runtime_error("Missing GPU baseline for " + param.caseName);
        }
        const auto& baseline = baselines[baselineIndex++];
        if (param.uplo != baseline.uplo || param.n != baseline.n || param.incx != baseline.incx ||
            param.incy != baseline.incy || baseline.gpuMs <= 0.0) {
            throw std::runtime_error("GPU baseline does not match " + param.caseName);
        }
        param.gpuBaselineMs = baseline.gpuMs;
    }
    if (baselineIndex != baselines.size()) {
        throw std::runtime_error("GPU baseline contains unused rows");
    }
    return cases;
}

CsymvHostData MakeHostData(const CsymvParam& param)
{
    CsymvHostData data;
    const size_t aCount = param.n > 0 && param.lda > 0 ? static_cast<size_t>(param.lda) * param.n : 0;
    data.A = MakeComplexStorage(aCount, param.a, param.randomSeed, param.UseNormalDistribution());
    data.x = MakeComplexStorage(
        StorageLength(param.n, param.incx), param.x, param.randomSeed + 1U, param.UseNormalDistribution());
    data.y = MakeComplexStorage(
        StorageLength(param.n, param.incy), param.y, param.randomSeed + 2U, param.UseNormalDistribution());
    PoisonUnusedTriangle(data.A, param.uplo, param.n, param.lda);
    return data;
}

bool NonFiniteMatches(float actual, float golden)
{
    if (std::isnan(golden)) {
        return std::isnan(actual);
    }
    if (std::isinf(golden)) {
        return actual == golden;
    }
    return false;
}

bool VerifyComponent(
    const std::vector<aclblasComplex>& actual, const std::vector<aclblasComplex>& golden, bool imaginary,
    const std::string& caseName)
{
    constexpr double kRtol = 0.0009765625;
    constexpr double kAtol = 0.0000152587890625;
    constexpr double kRequiredRatio = 0.99;
    constexpr double kFixedMaxError = 0.01;
    size_t matched = 0;
    bool maxErrorPassed = true;
    double maxError = 0.0;

    for (size_t i = 0; i < actual.size(); ++i) {
        const float actualValue = imaginary ? actual[i].imag : actual[i].real;
        const float goldenValue = imaginary ? golden[i].imag : golden[i].real;
        if (!std::isfinite(actualValue) || !std::isfinite(goldenValue)) {
            if (NonFiniteMatches(actualValue, goldenValue)) {
                ++matched;
            } else {
                maxErrorPassed = false;
            }
            continue;
        }

        const double error = std::abs(static_cast<double>(actualValue) - goldenValue);
        maxError = std::max(maxError, error);
        const double tolerance = kAtol + kRtol * std::abs(static_cast<double>(goldenValue));
        if (error <= tolerance) {
            ++matched;
        }
        const float magnitude = std::abs(goldenValue);
        const float next = std::nextafter(magnitude, std::numeric_limits<float>::infinity());
        const double ulpLimit = 32.0 * static_cast<double>(next - magnitude);
        if (error > std::max(kFixedMaxError, ulpLimit)) {
            maxErrorPassed = false;
        }
    }

    const double ratio = actual.empty() ? 1.0 : static_cast<double>(matched) / actual.size();
    const bool passed = ratio >= kRequiredRatio && maxErrorPassed;
    std::cout << "[" << caseName << (imaginary ? "_imag" : "_real") << "] " << (passed ? "PASSED" : "FAILED")
              << " matched_ratio=" << ratio << " max_abs_error=" << maxError << std::endl;
    return passed;
}

std::vector<aclblasComplex> ExtractLogicalY(const std::vector<aclblasComplex>& storage, int n, int incy)
{
    std::vector<aclblasComplex> logical(static_cast<size_t>(std::max(0, n)));
    const int64_t absIncy = incy >= 0 ? static_cast<int64_t>(incy) : -static_cast<int64_t>(incy);
    for (int i = 0; i < n; ++i) {
        const int64_t index = incy >= 0 ? static_cast<int64_t>(i) * incy : static_cast<int64_t>(n - 1 - i) * absIncy;
        logical[static_cast<size_t>(i)] = storage[static_cast<size_t>(index)];
    }
    return logical;
}

bool RunCsymvBatch(
    aclblasHandle_t handle, aclrtStream stream, const CsymvParam& param, const CsymvDeviceBuffers& buffers,
    int iterations)
{
    for (int i = 0; i < iterations; ++i) {
        const aclblasStatus_t status = aclblasCsymv(
            handle, param.uplo, param.n, buffers.alphaArg, buffers.aArg, param.lda, buffers.xArg, param.incx,
            buffers.betaArg, buffers.yArg, param.incy);
        if (status != ACLBLAS_STATUS_SUCCESS) {
            return false;
        }
    }
    return aclrtSynchronizeStream(stream) == ACL_SUCCESS;
}

double RunPerformanceCase(aclblasHandle_t handle, aclrtStream stream, const CsymvParam& param)
{
    CsymvHostData data = MakeHostData(param);

    CsymvDeviceBuffers buffers;
    buffers.Prepare(
        param.n, &param.alpha, data.A.data(), param.lda, data.x.data(), param.incx, &param.beta, data.y.data(),
        param.incy, false);
    constexpr int kWarmup = 10;
    constexpr int kIterations = 60;
    if (!RunCsymvBatch(handle, stream, param, buffers, kWarmup)) {
        return -1.0;
    }

    const auto start = std::chrono::steady_clock::now();
    if (!RunCsymvBatch(handle, stream, param, buffers, kIterations)) {
        return -1.0;
    }
    const auto end = std::chrono::steady_clock::now();
    const double averageUs =
        std::chrono::duration<double, std::micro>(end - start).count() / static_cast<double>(kIterations);
    return averageUs;
}

} // namespace

TEST_F(CsymvArch35Test, NullHandle)
{
    const aclblasComplex alpha{1.0f, 0.0f};
    const aclblasComplex beta{0.0f, 0.0f};
    EXPECT_EQ(
        aclblasCsymv(nullptr, ACLBLAS_UPPER, 1, &alpha, nullptr, 1, nullptr, 1, &beta, nullptr, 1),
        ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
    EXPECT_EQ(
        aclblasCsymv(nullptr, ACLBLAS_UPPER, -1, &alpha, nullptr, 1, nullptr, 1, &beta, nullptr, 1),
        ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
}

TEST_F(CsymvArch35Test, HostScalarQuickReturnAllowsNullData)
{
    const aclblasComplex alpha{0.0f, 0.0f};
    const aclblasComplex beta{1.0f, 0.0f};
    EXPECT_EQ(
        aclblasCsymv(handle_, ACLBLAS_UPPER, 8, &alpha, nullptr, 8, nullptr, 1, &beta, nullptr, 1),
        ACLBLAS_STATUS_SUCCESS);
}

TEST_F(CsymvArch35Test, DeviceScalarQuickReturnAllowsNullData)
{
    const aclblasComplex alpha{0.0f, 0.0f};
    const aclblasComplex beta{1.0f, 0.0f};
    EXPECT_EQ(
        aclblasCsymv_npu(handle_, ACLBLAS_LOWER, 8, &alpha, nullptr, 8, nullptr, -1, &beta, nullptr, -1, true),
        ACLBLAS_STATUS_SUCCESS);
}

TEST_F(CsymvArch35Test, ScaleOnlyAllowsNullMatrixAndX)
{
    const aclblasComplex alpha{0.0f, 0.0f};
    const aclblasComplex beta{0.0f, 1.0f};
    std::vector<aclblasComplex> y(4, aclblasComplex{2.0f, -3.0f});
    ASSERT_EQ(
        aclblasCsymv_npu(handle_, ACLBLAS_UPPER, 4, &alpha, nullptr, 4, nullptr, 1, &beta, y.data(), 1, false),
        ACLBLAS_STATUS_SUCCESS);
    for (const auto& value : y) {
        EXPECT_FLOAT_EQ(value.real, 3.0f);
        EXPECT_FLOAT_EQ(value.imag, 2.0f);
    }
}

TEST_F(CsymvArch35Test, BetaOnePreservesInfiniteY)
{
    const aclblasComplex alpha{1.0f, 0.0f};
    const aclblasComplex beta{1.0f, 0.0f};
    const aclblasComplex A{1.0f, 0.0f};
    const aclblasComplex x{1.0f, 0.0f};
    const float infinity = std::numeric_limits<float>::infinity();
    aclblasComplex goldenY{infinity, 2.0f};
    aclblasComplex npuY = goldenY;

    ASSERT_EQ(
        aclblasCsymv_cpu(handle_, ACLBLAS_UPPER, 1, &alpha, &A, 1, &x, 1, &beta, &goldenY, 1), ACLBLAS_STATUS_SUCCESS);
    ASSERT_EQ(
        aclblasCsymv_npu(handle_, ACLBLAS_UPPER, 1, &alpha, &A, 1, &x, 1, &beta, &npuY, 1, false),
        ACLBLAS_STATUS_SUCCESS);

    EXPECT_EQ(goldenY.real, infinity);
    EXPECT_FLOAT_EQ(goldenY.imag, 2.0f);
    EXPECT_EQ(npuY.real, infinity);
    EXPECT_FLOAT_EQ(npuY.imag, 2.0f);
}

INSTANTIATE_TEST_SUITE_P(
    Csymv, CsymvArch35Test, ::testing::ValuesIn(LoadCsymvCases(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<CsymvParam>);

TEST_P(CsymvArch35Test, CsvDriven)
{
    const CsymvParam& param = GetParam();
    if (param.IsPerformanceCase()) {
        constexpr double kRequiredPerformanceRatio = 0.4;
        ASSERT_GT(param.gpuBaselineMs, 0.0);
        const double averageUs = RunPerformanceCase(handle_, stream_, param);
        ASSERT_GT(averageUs, 0.0);
        const double baselineUs = param.gpuBaselineMs * 1000.0;
        const double thresholdUs = baselineUs / kRequiredPerformanceRatio;
        const double ratio = baselineUs / averageUs;
        const bool passed = averageUs <= thresholdUs;
        std::cout << "[PERF] case=" << param.caseName << " uplo=" << (param.uplo == ACLBLAS_UPPER ? "UPPER" : "LOWER")
                  << " n=" << param.n << " avg_us=" << averageUs << " baseline_us=" << baselineUs << " ratio=" << ratio
                  << " threshold_us=" << thresholdUs << " verdict=" << (passed ? "PASS" : "FAIL") << std::endl;
        EXPECT_TRUE(passed);
        return;
    }

    CsymvHostData data;
    try {
        data = MakeHostData(param);
    } catch (const std::bad_alloc&) {
        GTEST_SKIP() << "Host allocation exceeds available memory";
    }

    const aclblasComplex* alpha = param.nullAlpha ? nullptr : &param.alpha;
    const aclblasComplex* beta = param.nullBeta ? nullptr : &param.beta;
    const aclblasComplex* A = data.A.empty() ? nullptr : data.A.data();
    const aclblasComplex* x = data.x.empty() ? nullptr : data.x.data();
    aclblasComplex* y = data.y.empty() ? nullptr : data.y.data();
    std::vector<aclblasComplex> goldenY = data.y;

    const aclblasStatus_t actualStatus = aclblasCsymv_npu(
        handle_, param.uplo, param.n, alpha, A, param.lda, x, param.incx, beta, y, param.incy,
        param.UseDeviceScalars());
    ASSERT_EQ(actualStatus, param.expectResult);
    if (actualStatus != ACLBLAS_STATUS_SUCCESS) {
        return;
    }

    const aclblasStatus_t goldenStatus = aclblasCsymv_cpu(
        handle_, param.uplo, param.n, alpha, A, param.lda, x, param.incx, beta,
        goldenY.empty() ? nullptr : goldenY.data(), param.incy);
    ASSERT_EQ(goldenStatus, ACLBLAS_STATUS_SUCCESS);
    if (param.n == 0) {
        return;
    }

    const std::vector<aclblasComplex> actualLogical = ExtractLogicalY(data.y, param.n, param.incy);
    const std::vector<aclblasComplex> goldenLogical = ExtractLogicalY(goldenY, param.n, param.incy);
    EXPECT_TRUE(VerifyComponent(actualLogical, goldenLogical, false, param.caseName));
    EXPECT_TRUE(VerifyComponent(actualLogical, goldenLogical, true, param.caseName));
}
