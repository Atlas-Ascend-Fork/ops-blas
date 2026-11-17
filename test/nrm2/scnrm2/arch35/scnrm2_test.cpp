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
#include <chrono>
#include <cmath>
#include <vector>
#include <cstdlib>
#include <limits>
#include <random>

#include "verify.h"
#include "blas_test.h"
#include "csv_loader.h"
#include "scnrm2_test_common.h"

class Scnrm2Arch35Test : public BlasTest<Scnrm2Param> {};

TEST_F(Scnrm2Arch35Test, NullHandle)
{
    float result = 0.0f;
    aclblasStatus_t ret = aclblasScnrm2_npu(nullptr, 5, nullptr, 1, &result);
    EXPECT_EQ(ret, ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
}

TEST_F(Scnrm2Arch35Test, SmallWorkspaceFallback)
{
    constexpr int n = 4096;
    std::vector<aclblasComplex> input(n, aclblasComplex{3.0f, 4.0f});
    // Enough for 64 custom-kernel partials, but smaller than the norm workspace.
    DeviceBuffer workspace(64 * 2 * sizeof(float));
    aclblasHandle_t rawHandle = nullptr;
    ASSERT_EQ(aclblasCreate(&rawHandle), ACLBLAS_STATUS_SUCCESS);
    std::unique_ptr<_aclblas_handle, decltype(&aclblasDestroy)> handle(rawHandle, aclblasDestroy);
    ASSERT_EQ(aclblasSetStream(handle.get(), stream_), ACLBLAS_STATUS_SUCCESS);
    ASSERT_EQ(aclblasSetWorkspace(handle.get(), workspace.ptr(), workspace.size()), ACLBLAS_STATUS_SUCCESS);

    float result = -1.0f;
    ASSERT_EQ(aclblasScnrm2_npu(handle.get(), n, input.data(), 1, &result), ACLBLAS_STATUS_SUCCESS);
    EXPECT_FLOAT_EQ(result, 320.0f);
}

TEST_F(Scnrm2Arch35Test, LargeUnalignedFallback)
{
    constexpr int n = 1048576;
    std::vector<aclblasComplex> input(n + 1, aclblasComplex{3.0f, 4.0f});
    DeviceBuffer workspace(64 * 2 * sizeof(float));
    aclblasHandle_t rawHandle = nullptr;
    ASSERT_EQ(aclblasCreate(&rawHandle), ACLBLAS_STATUS_SUCCESS);
    std::unique_ptr<_aclblas_handle, decltype(&aclblasDestroy)> handle(rawHandle, aclblasDestroy);
    ASSERT_EQ(aclblasSetStream(handle.get(), stream_), ACLBLAS_STATUS_SUCCESS);
    ASSERT_EQ(aclblasSetWorkspace(handle.get(), workspace.ptr(), workspace.size()), ACLBLAS_STATUS_SUCCESS);

    DeviceBuffer x(input.size() * sizeof(aclblasComplex));
    DeviceBuffer output(2 * sizeof(float));
    x.copyFromHost(input.data(), x.size());
    auto* xOffset = static_cast<const aclblasComplex*>(x.ptr()) + 1;
    auto* resultOffset = static_cast<float*>(output.ptr()) + 1;
    ASSERT_EQ(aclblasScnrm2(handle.get(), n, xOffset, 1, resultOffset), ACLBLAS_STATUS_SUCCESS);
    ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_SUCCESS);

    float result = 0.0f;
    ASSERT_EQ(aclrtMemcpy(&result, sizeof(float), resultOffset, sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST), ACL_SUCCESS);
    EXPECT_FLOAT_EQ(result, 5120.0f);
}

TEST_F(Scnrm2Arch35Test, InsufficientWorkspacePreservesHostResult)
{
    DeviceBuffer workspace(sizeof(float));
    aclblasHandle_t rawHandle = nullptr;
    ASSERT_EQ(aclblasCreate(&rawHandle), ACLBLAS_STATUS_SUCCESS);
    std::unique_ptr<_aclblas_handle, decltype(&aclblasDestroy)> handle(rawHandle, aclblasDestroy);
    ASSERT_EQ(aclblasSetStream(handle.get(), stream_), ACLBLAS_STATUS_SUCCESS);
    ASSERT_EQ(aclblasSetWorkspace(handle.get(), workspace.ptr(), workspace.size()), ACLBLAS_STATUS_SUCCESS);

    std::vector<aclblasComplex> input(7, aclblasComplex{3.0f, 4.0f});
    float result = -1.0f;
    EXPECT_EQ(aclblasScnrm2_npu(handle.get(), 4, input.data(), 2, &result), ACLBLAS_STATUS_ALLOC_FAILED);
    EXPECT_FLOAT_EQ(result, -1.0f);
}

TEST_F(Scnrm2Arch35Test, OffsetInputAndOutput)
{
    for (int n : {64, 4096}) {
        for (int incx : {1, -2}) {
            SCOPED_TRACE(::testing::Message() << "n=" << n << " incx=" << incx);
            size_t count = 1 + static_cast<size_t>(n - 1) * std::abs(incx);
            std::vector<aclblasComplex> input(count + 1, aclblasComplex{3.0f, 4.0f});
            DeviceBuffer x(input.size() * sizeof(aclblasComplex));
            DeviceBuffer output(2 * sizeof(float));
            x.copyFromHost(input.data(), x.size());
            auto* xOffset = static_cast<const aclblasComplex*>(x.ptr()) + 1;
            auto* resultOffset = static_cast<float*>(output.ptr()) + 1;
            ASSERT_EQ(aclblasScnrm2(handle_, n, xOffset, incx, resultOffset), ACLBLAS_STATUS_SUCCESS);
            ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_SUCCESS);
            float result = 0.0f;
            ASSERT_EQ(
                aclrtMemcpy(&result, sizeof(float), resultOffset, sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST),
                ACL_SUCCESS);
            EXPECT_FLOAT_EQ(result, 5.0f * std::sqrt(static_cast<float>(n)));
        }
    }
}

TEST_F(Scnrm2Arch35Test, TaskPerformanceSmoke)
{
    struct PerfCase {
        int n;
        double thresholdUs;
    };
    constexpr PerfCase cases[] = {
        {1048576, 32.70},
        {2097152, 35.91},
        {4194304, 42.05},
    };
    constexpr int warmup = 20;
    constexpr int iterations = 200;

    for (const auto& c : cases) {
        DeviceBuffer x(static_cast<size_t>(c.n) * sizeof(aclblasComplex));
        DeviceBuffer result(sizeof(float));
        ASSERT_EQ(aclrtMemset(x.ptr(), x.size(), 0, x.size()), ACL_SUCCESS);

        for (int i = 0; i < warmup; ++i) {
            ASSERT_EQ(
                aclblasScnrm2(
                    handle_, c.n, static_cast<const aclblasComplex*>(x.ptr()), 1, static_cast<float*>(result.ptr())),
                ACLBLAS_STATUS_SUCCESS);
        }
        ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_SUCCESS);

        aclrtEvent start = nullptr;
        aclrtEvent end = nullptr;
        ASSERT_EQ(aclrtCreateEvent(&start), ACL_SUCCESS);
        ASSERT_EQ(aclrtCreateEvent(&end), ACL_SUCCESS);
        ASSERT_EQ(aclrtRecordEvent(start, stream_), ACL_SUCCESS);
        for (int i = 0; i < iterations; ++i) {
            ASSERT_EQ(
                aclblasScnrm2(
                    handle_, c.n, static_cast<const aclblasComplex*>(x.ptr()), 1, static_cast<float*>(result.ptr())),
                ACLBLAS_STATUS_SUCCESS);
        }
        ASSERT_EQ(aclrtRecordEvent(end, stream_), ACL_SUCCESS);
        ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_SUCCESS);

        float elapsedMs = 0.0f;
        ASSERT_EQ(aclrtEventElapsedTime(&elapsedMs, start, end), ACL_SUCCESS);
        ASSERT_EQ(aclrtDestroyEvent(start), ACL_SUCCESS);
        ASSERT_EQ(aclrtDestroyEvent(end), ACL_SUCCESS);

        double elapsedUs = static_cast<double>(elapsedMs) * 1000.0 / iterations;
        std::cout << "[PERF][ascend950] aclblasScnrm2 n=" << c.n << " incx=1 average_us=" << elapsedUs
                  << " threshold_us=" << c.thresholdUs << std::endl;
        EXPECT_LE(elapsedUs, c.thresholdUs);
    }
}

INSTANTIATE_TEST_SUITE_P(
    Scnrm2, Scnrm2Arch35Test, ::testing::ValuesIn(GetCasesFromCsv<Scnrm2Param>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<Scnrm2Param>);

TEST_F(Scnrm2Arch35Test, NumericalBoundaries)
{
    const float values[] = {
        1.0e20f, 1.0e-20f, std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()};
    for (int n : {1, 64, 4096}) {
        for (int incx : {1, -2}) {
            for (float value : values) {
                SCOPED_TRACE(::testing::Message() << "n=" << n << " incx=" << incx << " value=" << value);
                std::vector<aclblasComplex> input(1 + (n - 1) * std::abs(incx), aclblasComplex{value, value});
                float result = 0.0f;
                float golden = 0.0f;
                ASSERT_EQ(aclblasScnrm2_npu(handle_, n, input.data(), incx, &result), ACLBLAS_STATUS_SUCCESS);
                ASSERT_EQ(aclblasScnrm2_cpu(handle_, n, input.data(), incx, &golden), ACLBLAS_STATUS_SUCCESS);
                if (std::isnan(golden)) {
                    EXPECT_TRUE(std::isnan(result));
                } else if (std::isinf(golden)) {
                    EXPECT_EQ(result, golden);
                } else {
                    EXPECT_TRUE(std::isfinite(result));
                    EXPECT_NEAR(static_cast<double>(result) / golden, 1.0, 0x1p-10);
                }
            }
        }
    }
}

TEST_F(Scnrm2Arch35Test, NormalDistributionPrecision)
{
    constexpr uint32_t seed = 20260918U;
    constexpr int sizes[] = {1, 64, 4096, 65536};
    constexpr int strides[] = {1, -2};

    auto makeNormalInput = [](int count) {
        std::mt19937 generator(seed);
        auto uniform = [&generator]() { return (static_cast<double>(generator()) + 0.5) / 4294967296.0; };

        std::vector<aclblasComplex> input(static_cast<size_t>(count));
        for (int i = 0; i < count; ++i) {
            double u1 = uniform();
            double u2 = uniform();
            double normal = std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2);
            input[static_cast<size_t>(i)] = aclblasComplex{static_cast<float>(normal), 0.0f};
            u1 = uniform();
            u2 = uniform();
            normal = std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2);
            input[static_cast<size_t>(i)].imag = static_cast<float>(normal);
        }
        return input;
    };

    for (int n : sizes) {
        for (int incx : strides) {
            SCOPED_TRACE(::testing::Message() << "n=" << n << " incx=" << incx << " seed=" << seed);
            const int physicalCount = 1 + (n - 1) * std::abs(incx);
            const std::vector<aclblasComplex> input = makeNormalInput(physicalCount);
            float result = 0.0f;
            ASSERT_EQ(aclblasScnrm2_npu(handle_, n, input.data(), incx, &result), ACLBLAS_STATUS_SUCCESS);

            float golden = 0.0f;
            ASSERT_EQ(aclblasScnrm2_cpu(handle_, n, input.data(), incx, &golden), ACLBLAS_STATUS_SUCCESS);
            VerifyScnrm2Result(result, golden, "NormalDistributionPrecision");
        }
    }
}

namespace {
struct MixedDistributionCase {
    uint32_t index;
    bool isNormal;
    int n;
    int incx;
    double muReal;
    double muImag;
    double sigmaReal;
    double sigmaImag;
};

constexpr uint32_t seed = 20260919U;
constexpr int uniformCaseCount = 500;
constexpr int normalCaseCount = 500;
constexpr double uniformLow = -5.0;
constexpr double uniformHigh = 5.0;
constexpr double normalMuLow = -5.0;
constexpr double normalMuHigh = 5.0;
constexpr double normalSigmaMin = 0.1;
constexpr double normalSigmaMax = 2.0;
constexpr int representativeSizes[] = {1, 17, 64, 257, 1024, 4096};
constexpr int representativeStrides[] = {1, -2, 3, -4};
constexpr double normalMus[] = {-5.0, -2.5, 0.0, 2.5, 5.0};
constexpr double normalSigmas[] = {0.1, 0.55, 1.0, 1.5, 2.0};

std::vector<MixedDistributionCase> MakeMixedCases()
{
    std::vector<MixedDistributionCase> cases;
    cases.reserve(static_cast<size_t>(uniformCaseCount + normalCaseCount));
    auto appendCase = [&](int sequenceIndex, int parameterIndex, bool isNormal) {
        const int n = representativeSizes[static_cast<size_t>(sequenceIndex) % std::size(representativeSizes)];
        const int incx = representativeStrides[static_cast<size_t>(sequenceIndex) % std::size(representativeStrides)];
        cases.push_back(MixedDistributionCase{
            static_cast<uint32_t>(sequenceIndex), isNormal, n, incx,
            isNormal ? normalMus[static_cast<size_t>(parameterIndex % 5)] : 0.0,
            isNormal ? normalMus[static_cast<size_t>((parameterIndex + 2) % 5)] : 0.0,
            isNormal ? normalSigmas[static_cast<size_t>((parameterIndex + 1) % 5)] : 0.0,
            isNormal ? normalSigmas[static_cast<size_t>((parameterIndex + 3) % 5)] : 0.0});
    };
    for (int i = 0; i < uniformCaseCount; ++i) {
        appendCase(i, i, false);
    }
    for (int i = 0; i < normalCaseCount; ++i) {
        appendCase(uniformCaseCount + i, i, true);
    }
    return cases;
}

double NextCanonicalUniform(std::mt19937& generator) { return (static_cast<double>(generator()) + 0.5) / 4294967296.0; }

double NextStandardNormal(std::mt19937& generator)
{
    const double u1 = NextCanonicalUniform(generator);
    const double u2 = NextCanonicalUniform(generator);
    return std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2);
}

std::vector<aclblasComplex> MakeMixedInput(const MixedDistributionCase& sample)
{
    std::mt19937 realGenerator(seed + 2U * sample.index);
    std::mt19937 imagGenerator(seed + 2U * sample.index + 1U);
    const int64_t physicalCount = Scnrm2BufferComplexCount(sample.n, sample.incx);
    std::vector<aclblasComplex> input(static_cast<size_t>(physicalCount));
    for (int64_t i = 0; i < physicalCount; ++i) {
        float real = 0.0f;
        float imag = 0.0f;
        if (sample.isNormal) {
            real = static_cast<float>(sample.muReal + sample.sigmaReal * NextStandardNormal(realGenerator));
            imag = static_cast<float>(sample.muImag + sample.sigmaImag * NextStandardNormal(imagGenerator));
        } else {
            real = static_cast<float>(uniformLow + (uniformHigh - uniformLow) * NextCanonicalUniform(realGenerator));
            imag = static_cast<float>(uniformLow + (uniformHigh - uniformLow) * NextCanonicalUniform(imagGenerator));
        }
        input[static_cast<size_t>(i)] = aclblasComplex{real, imag};
    }
    return input;
}

void CheckMixedParameters(const MixedDistributionCase& sample)
{
    if (sample.isNormal) {
        EXPECT_GE(sample.muReal, normalMuLow);
        EXPECT_LE(sample.muReal, normalMuHigh);
        EXPECT_GE(sample.muImag, normalMuLow);
        EXPECT_LE(sample.muImag, normalMuHigh);
        EXPECT_GE(sample.sigmaReal, normalSigmaMin);
        EXPECT_LE(sample.sigmaReal, normalSigmaMax);
        EXPECT_GE(sample.sigmaImag, normalSigmaMin);
        EXPECT_LE(sample.sigmaImag, normalSigmaMax);
    } else {
        EXPECT_DOUBLE_EQ(uniformLow, -5.0);
        EXPECT_DOUBLE_EQ(uniformHigh, 5.0);
    }
}

void CheckMixedInput(const MixedDistributionCase& sample, const std::vector<aclblasComplex>& input)
{
    for (const aclblasComplex& value : input) {
        if (!sample.isNormal) {
            EXPECT_GE(value.real, uniformLow);
            EXPECT_LE(value.real, uniformHigh);
            EXPECT_GE(value.imag, uniformLow);
            EXPECT_LE(value.imag, uniformHigh);
        }
    }
}

void RecordMixedCoverage(
    size_t generatedUniformCases, size_t generatedNormalCases, bool sawPositiveStride, bool sawNegativeStride,
    bool sawUnitSize, bool sawLargeSize, size_t totalCases)
{
    EXPECT_EQ(generatedUniformCases, static_cast<size_t>(uniformCaseCount));
    EXPECT_EQ(generatedNormalCases, static_cast<size_t>(normalCaseCount));
    EXPECT_TRUE(sawPositiveStride);
    EXPECT_TRUE(sawNegativeStride);
    EXPECT_TRUE(sawUnitSize);
    EXPECT_TRUE(sawLargeSize);
    ::testing::Test::RecordProperty("uniform_cases", std::to_string(generatedUniformCases));
    ::testing::Test::RecordProperty("normal_cases", std::to_string(generatedNormalCases));
    ::testing::Test::RecordProperty("total_cases", std::to_string(totalCases));
}
} // namespace

TEST_F(Scnrm2Arch35Test, MixedDistributionPrecision)
{
    const std::vector<MixedDistributionCase> cases = MakeMixedCases();
    EXPECT_EQ(cases.size(), static_cast<size_t>(uniformCaseCount + normalCaseCount));
    size_t generatedUniformCases = 0;
    size_t generatedNormalCases = 0;
    bool sawPositiveStride = false;
    bool sawNegativeStride = false;
    bool sawUnitSize = false;
    bool sawLargeSize = false;

    for (const MixedDistributionCase& sample : cases) {
        SCOPED_TRACE(
            ::testing::Message() << "index=" << sample.index
                                 << (sample.isNormal ? " distribution=normal" : " distribution=uniform")
                                 << " n=" << sample.n << " incx=" << sample.incx);
        CheckMixedParameters(sample);
        generatedNormalCases += sample.isNormal ? 1 : 0;
        generatedUniformCases += sample.isNormal ? 0 : 1;
        sawPositiveStride = sawPositiveStride || sample.incx > 0;
        sawNegativeStride = sawNegativeStride || sample.incx < 0;
        sawUnitSize = sawUnitSize || sample.n == 1;
        sawLargeSize = sawLargeSize || sample.n == 4096;

        const std::vector<aclblasComplex> input = MakeMixedInput(sample);
        CheckMixedInput(sample, input);

        float result = 0.0f;
        ASSERT_EQ(aclblasScnrm2_npu(handle_, sample.n, input.data(), sample.incx, &result), ACLBLAS_STATUS_SUCCESS);
        float golden = 0.0f;
        ASSERT_EQ(aclblasScnrm2_cpu(handle_, sample.n, input.data(), sample.incx, &golden), ACLBLAS_STATUS_SUCCESS);
        const std::string caseName = std::string("MixedDistributionPrecision.") +
                                     (sample.isNormal ? "normal_" : "uniform_") + std::to_string(sample.index);
        VerifyScnrm2Result(result, golden, caseName);
    }

    RecordMixedCoverage(
        generatedUniformCases, generatedNormalCases, sawPositiveStride, sawNegativeStride, sawUnitSize, sawLargeSize,
        cases.size());
}

static void TestErrorPath(const Scnrm2Param& p, aclblasHandle_t handle)
{
    std::vector<aclblasComplex> xHost = MakeComplexInput(p);
    const aclblasComplex* xPtr = (p.x.method == BlasFillMode::M_NULLPTR || xHost.empty()) ? nullptr : xHost.data();
    float result = 0.0f;
    float* resultPtr = p.resultIsNull ? nullptr : &result;

    aclblasStatus_t ret = aclblasScnrm2_npu(handle, p.n, xPtr, p.incx, resultPtr);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
}

static void TestNoOpPath(const Scnrm2Param& p, aclblasHandle_t handle)
{
    std::vector<aclblasComplex> xHost = MakeComplexInput(p);
    const aclblasComplex* xPtr = (p.x.method == BlasFillMode::M_NULLPTR || xHost.empty()) ? nullptr : xHost.data();
    float result = 123.0f;

    aclblasStatus_t ret = aclblasScnrm2_npu(handle, p.n, xPtr, p.incx, &result);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
    if (ret == ACLBLAS_STATUS_SUCCESS) {
        EXPECT_FLOAT_EQ(result, 0.0f) << "[" << p.caseName << "] early return should produce result=0.0f";
    }
}

static bool StrictPerfMode()
{
    const char* v = std::getenv("SCNRM2_STRICT_PERF");
    return v != nullptr && std::string(v) == "1";
}

static void TestStrictPerformance(const Scnrm2Param& p, aclblasHandle_t handle, aclrtStream stream)
{
    std::vector<aclblasComplex> xHost = MakeComplexInput(p);
    ASSERT_FALSE(xHost.empty());
    DeviceBuffer x(static_cast<size_t>(xHost.size()) * sizeof(aclblasComplex));
    DeviceBuffer result(sizeof(float));
    ASSERT_EQ(aclrtMemcpy(x.ptr(), x.size(), xHost.data(), x.size(), ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);
    constexpr int warmup = 20;
    constexpr int iterations = 100;
    for (int i = 0; i < warmup; ++i) {
        ASSERT_EQ(
            aclblasScnrm2(
                handle, p.n, static_cast<const aclblasComplex*>(x.ptr()), p.incx, static_cast<float*>(result.ptr())),
            ACLBLAS_STATUS_SUCCESS);
    }
    ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);
    aclrtEvent start = nullptr, end = nullptr;
    ASSERT_EQ(aclrtCreateEvent(&start), ACL_SUCCESS);
    ASSERT_EQ(aclrtCreateEvent(&end), ACL_SUCCESS);
    ASSERT_EQ(aclrtRecordEvent(start, stream), ACL_SUCCESS);
    for (int i = 0; i < iterations; ++i) {
        ASSERT_EQ(
            aclblasScnrm2(
                handle, p.n, static_cast<const aclblasComplex*>(x.ptr()), p.incx, static_cast<float*>(result.ptr())),
            ACLBLAS_STATUS_SUCCESS);
    }
    ASSERT_EQ(aclrtRecordEvent(end, stream), ACL_SUCCESS);
    ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);
    float elapsedMs = 0.0f;
    ASSERT_EQ(aclrtEventElapsedTime(&elapsedMs, start, end), ACL_SUCCESS);
    aclrtDestroyEvent(start);
    aclrtDestroyEvent(end);
    std::cout << "[STRICT_PERF] case=" << p.caseName << " n=" << p.n << " incx=" << p.incx << " warmup=" << warmup
              << " samples=" << iterations << " average_us=" << (static_cast<double>(elapsedMs) * 1000.0 / iterations)
              << std::endl;
}

TEST_P(Scnrm2Arch35Test, CsvDriven)
{
    const auto& p = GetParam();

    if (StrictPerfMode() && p.caseName.rfind("TC_PF_", 0) == 0) {
        TestStrictPerformance(p, Scnrm2Arch35Test::handle_, Scnrm2Arch35Test::stream_);
        return;
    }

    if (p.expectResult != ACLBLAS_STATUS_SUCCESS) {
        TestErrorPath(p, Scnrm2Arch35Test::handle_);
    } else if (p.n <= 0) {
        TestNoOpPath(p, Scnrm2Arch35Test::handle_);
    } else {
        TestScnrm2NormalPath(p, Scnrm2Arch35Test::handle_);
    }
}
