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
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <sys/resource.h>
#include <vector>

#include "blas_test.h"
#include "csv_loader.h"
#include "csyr2_golden.h"
#include "csyr2_npu_wrapper.h"
#include "csyr2_param.h"
#include "fill.h"
#include "verify.h"

class Csyr2Test : public BlasTest<Csyr2Param> {};

namespace {

bool IsZero(const aclblasComplex& value) { return value.real == 0.0f && value.imag == 0.0f; }

size_t VectorStorageSize(int n, int inc)
{
    if (n <= 0 || inc == 0) {
        return 0;
    }
    const int64_t stride = inc < 0 ? -static_cast<int64_t>(inc) : static_cast<int64_t>(inc);
    return static_cast<size_t>(1 + static_cast<int64_t>(n - 1) * stride);
}

std::vector<aclblasComplex> MakeComplexData(
    size_t count, const BlasFillMode& realFill, const BlasFillMode& imagFill, const Csyr2Param& p, uint32_t seed)
{
    if (count == 0 || realFill.method == BlasFillMode::M_NULLPTR || imagFill.method == BlasFillMode::M_NULLPTR) {
        return {};
    }
    std::vector<aclblasComplex> result(count);
    if (p.distribution == "NORMAL") {
        std::mt19937 realRng(seed);
        std::mt19937 imagRng(seed + 0x9e3779b9U);
        std::normal_distribution<float> realDist(p.distributionMean, p.distributionStddev);
        std::normal_distribution<float> imagDist(p.distributionMean, p.distributionStddev);
        for (auto& value : result) {
            value = {realDist(realRng), imagDist(imagRng)};
        }
        return result;
    }

    const std::vector<float> real = makeBlasArray(static_cast<int64_t>(count), realFill, seed);
    const std::vector<float> imag = makeBlasArray(static_cast<int64_t>(count), imagFill, seed + 0x9e3779b9U);
    for (size_t index = 0; index < count; ++index) {
        result[index] = {real[index], imag[index]};
    }
    return result;
}

struct Csyr2HostData {
    std::vector<aclblasComplex> x;
    std::vector<aclblasComplex> y;
    std::vector<aclblasComplex> a;
    std::vector<aclblasComplex> aOriginal;
};

Csyr2HostData PrepareHostData(const Csyr2Param& p)
{
    Csyr2HostData data;
    const size_t xCount = VectorStorageSize(p.n, p.incx);
    const size_t yCount = VectorStorageSize(p.n, p.incy);
    const size_t aCount = p.n > 0 && p.lda > 0 ? static_cast<size_t>(p.lda) * static_cast<size_t>(p.n) : 0;
    data.x = MakeComplexData(xCount, p.xRealFill, p.xImagFill, p, p.randomSeed + 1U);
    data.y = MakeComplexData(yCount, p.yRealFill, p.yImagFill, p, p.randomSeed + 2U);
    data.a = MakeComplexData(aCount, p.aRealFill, p.aImagFill, p, p.randomSeed + 3U);
    data.aOriginal = data.a;
    return data;
}

void VerifyUpdatedTriangle(
    const Csyr2Param& p, const std::vector<aclblasComplex>& actual, const std::vector<aclblasComplex>& original,
    const std::vector<aclblasComplex>& x, const std::vector<aclblasComplex>& y)
{
    csyr2_test::ComplexAccuracy stats;
    csyr2_test::AccuracyConfig config;
    config.mereThreshold = p.mereThreshold;
    config.mareMultiplier = p.mareMultiplier;
    std::vector<aclblasComplex> goldenColumn(static_cast<size_t>(p.lda));
    for (int col = 0; col < p.n; ++col) {
        std::copy_n(original.data() + static_cast<size_t>(col) * p.lda, p.lda, goldenColumn.data());
        csyr2_test::Csyr2GoldenColumn(
            p.uplo, p.n, p.alpha, x.data(), p.incx, y.data(), p.incy, col, goldenColumn.data());
        const int rowBegin = p.uplo == ACLBLAS_UPPER ? 0 : col;
        const int rowEnd = p.uplo == ACLBLAS_UPPER ? col + 1 : p.n;
        for (int row = rowBegin; row < rowEnd; ++row) {
            const size_t index = static_cast<size_t>(col) * static_cast<size_t>(p.lda) + static_cast<size_t>(row);
            csyr2_test::Accumulate(stats, actual[index], goldenColumn[static_cast<size_t>(row)], index, config);
        }
    }
    // Use the already tested streaming accumulator: do not materialize four
    // expanded triangle arrays (up to 256 MiB of vector capacity at n=4096).
    for (bool real : {true, false}) {
        const auto& component = real ? stats.real : stats.imag;
        std::printf(
            "[CSYR2_ACCURACY] case=%s component=%s total=%zu matchedRatio=%.9g "
            "maxAbsErr=%.9g MERE=%.9g MARE=%.9g cap_failures=%zu special_mismatch=%zu\n",
            p.caseName.c_str(), real ? "real" : "imag", component.total, component.matchedRatio(),
            component.maxAbsError, component.mere(), component.mare, component.capFailCount, component.specialMismatch);
        EXPECT_TRUE(csyr2_test::ComponentPass(component, config)) << p.caseName << (real ? " real" : " imag");
        EXPECT_TRUE(csyr2_test::MereMarePass(component, config)) << p.caseName << (real ? " real" : " imag");
    }
    const bool worstReal = stats.real.maxAbsError >= stats.imag.maxAbsError;
    const auto& worst = worstReal ? stats.real : stats.imag;
    std::printf(
        "[ACCURACY] case=%s max_error=%.9g ratio=%.9g worst_case=%s:index=%zu\n", p.caseName.c_str(), worst.maxAbsError,
        std::min(stats.real.matchedRatio(), stats.imag.matchedRatio()), worstReal ? "real" : "imag", worst.worstIndex);
}

void VerifyUnchangedStorage(
    const Csyr2Param& p, const std::vector<aclblasComplex>& actual, const std::vector<aclblasComplex>& original)
{
    for (int col = 0; col < p.n; ++col) {
        for (int row = 0; row < p.lda; ++row) {
            const bool selected = row < p.n && (p.uplo == ACLBLAS_UPPER ? row <= col : row >= col);
            if (selected) {
                continue;
            }
            const size_t index = static_cast<size_t>(col) * static_cast<size_t>(p.lda) + static_cast<size_t>(row);
            EXPECT_EQ(0, std::memcmp(&actual[index], &original[index], sizeof(aclblasComplex)))
                << "[" << p.caseName << "] modified non-uplo or padding element at col=" << col << ", row=" << row;
        }
    }
}

void VerifyManualN2(aclblasHandle_t handle, aclrtStream stream, aclblasFillMode_t uplo)
{
    const aclblasComplex alpha{0.5f, -0.25f};
    std::vector<aclblasComplex> x{{1.0f, 2.0f}, {3.0f, -1.0f}};
    std::vector<aclblasComplex> y{{2.0f, -1.0f}, {-1.0f, 0.5f}};
    std::vector<aclblasComplex> a{{1.0f, 0.0f}, {2.0f, 1.0f}, {-3.0f, 2.0f}, {4.0f, -2.0f}};
    std::vector<aclblasComplex> golden = a;
    csyr2_test::Csyr2Golden(uplo, 2, alpha, x.data(), 1, y.data(), 1, golden.data(), 2);
    ASSERT_EQ(
        ACLBLAS_STATUS_SUCCESS,
        aclblasCsyr2_npu(handle, stream, uplo, 2, &alpha, x.data(), 1, y.data(), 1, a.data(), 2));
    for (size_t index = 0; index < a.size(); ++index) {
        EXPECT_FLOAT_EQ(golden[index].real, a[index].real) << "real index=" << index;
        EXPECT_FLOAT_EQ(golden[index].imag, a[index].imag) << "imag index=" << index;
    }
}

void VerifyGuardedMatrix(
    const std::vector<aclblasComplex>& actual, const std::vector<aclblasComplex>& original,
    const std::vector<aclblasComplex>& golden, size_t guardCount, size_t matrixElements, int n, int lda,
    aclblasFillMode_t uplo)
{
    if (lda <= 0) {
        ADD_FAILURE() << "invalid lda=" << lda;
        return;
    }
    const auto expectGuard = [&](size_t index) {
        EXPECT_EQ(0, std::memcmp(&actual[index], &original[index], sizeof(aclblasComplex)))
            << "A guard index=" << index;
    };
    for (size_t index = 0; index < guardCount; ++index) {
        expectGuard(index);
    }
    int row = 0;
    int col = 0;
    for (size_t matrixIndex = 0; matrixIndex < matrixElements; ++matrixIndex) {
        const size_t actualIndex = guardCount + matrixIndex;
        const bool selected = row < n && (uplo == ACLBLAS_UPPER ? row <= col : row >= col);
        if (selected) {
            EXPECT_FLOAT_EQ(golden[matrixIndex].real, actual[actualIndex].real) << "A real index=" << matrixIndex;
            EXPECT_FLOAT_EQ(golden[matrixIndex].imag, actual[actualIndex].imag) << "A imag index=" << matrixIndex;
        } else {
            EXPECT_EQ(0, std::memcmp(&actual[actualIndex], &original[actualIndex], sizeof(aclblasComplex)))
                << "A untouched index=" << matrixIndex;
        }
        if (++row == lda) {
            row = 0;
            ++col;
        }
    }
    for (size_t index = guardCount + matrixElements; index < actual.size(); ++index) {
        expectGuard(index);
    }
}

aclError CopyComplexFromDevice(std::vector<aclblasComplex>& host, const void* device)
{
    const size_t bytes = host.size() * sizeof(aclblasComplex);
    return aclrtMemcpy(host.data(), bytes, device, bytes, ACL_MEMCPY_DEVICE_TO_HOST);
}

void VerifyGuardedNoncontiguousLayout(
    aclblasHandle_t handle, aclrtStream stream, aclblasFillMode_t uplo, int n = 7, int incx = -2, int incy = 3)
{
    const int lda = n + 4;
    constexpr size_t guardCount = 3;
    const aclblasComplex alpha{0.75f, -0.5f};
    const aclblasComplex canary{std::numeric_limits<float>::quiet_NaN(), -0.0f};
    const size_t xElements = VectorStorageSize(n, incx);
    const size_t yElements = VectorStorageSize(n, incy);
    const size_t aElements = static_cast<size_t>(lda) * n;
    std::vector<aclblasComplex> x(guardCount + xElements + guardCount, canary);
    std::vector<aclblasComplex> y(guardCount + yElements + guardCount, canary);
    std::vector<aclblasComplex> a(guardCount + aElements + guardCount, canary);
    for (size_t i = 0; i < xElements; ++i) {
        x[guardCount + i] = {static_cast<float>(i) - 4.0f, static_cast<float>(2 * i) - 5.0f};
    }
    for (size_t i = 0; i < yElements; ++i) {
        y[guardCount + i] = {1.5f - static_cast<float>(i), static_cast<float>(i) - 2.5f};
    }
    for (size_t i = 0; i < aElements; ++i) {
        a[guardCount + i] = {static_cast<float>(i) * 0.25f - 7.0f, 3.0f - static_cast<float>(i) * 0.125f};
    }
    const std::vector<aclblasComplex> xOriginal = x;
    const std::vector<aclblasComplex> yOriginal = y;
    const std::vector<aclblasComplex> aOriginal = a;
    std::vector<aclblasComplex> golden(a.begin() + guardCount, a.begin() + guardCount + aElements);
    csyr2_test::Csyr2Golden(
        uplo, n, alpha, x.data() + guardCount, incx, y.data() + guardCount, incy, golden.data(), lda);

    Csyr2DeviceBuffers buffers;
    ASSERT_EQ(ACLBLAS_STATUS_SUCCESS, Csyr2AllocAndCopy(x.data(), x.size() * sizeof(aclblasComplex), buffers.x));
    ASSERT_EQ(ACLBLAS_STATUS_SUCCESS, Csyr2AllocAndCopy(y.data(), y.size() * sizeof(aclblasComplex), buffers.y));
    ASSERT_EQ(ACLBLAS_STATUS_SUCCESS, Csyr2AllocAndCopy(a.data(), a.size() * sizeof(aclblasComplex), buffers.a));
    const auto* xDevice = static_cast<const aclblasComplex*>(buffers.x) + guardCount;
    const auto* yDevice = static_cast<const aclblasComplex*>(buffers.y) + guardCount;
    auto* aDevice = static_cast<aclblasComplex*>(buffers.a) + guardCount;
    ASSERT_EQ(
        ACLBLAS_STATUS_SUCCESS, aclblasCsyr2(handle, uplo, n, &alpha, xDevice, incx, yDevice, incy, aDevice, lda));
    ASSERT_EQ(ACL_SUCCESS, aclrtSynchronizeStream(stream));
    ASSERT_EQ(ACL_SUCCESS, CopyComplexFromDevice(x, buffers.x));
    ASSERT_EQ(ACL_SUCCESS, CopyComplexFromDevice(y, buffers.y));
    ASSERT_EQ(ACL_SUCCESS, CopyComplexFromDevice(a, buffers.a));

    EXPECT_EQ(0, std::memcmp(x.data(), xOriginal.data(), x.size() * sizeof(aclblasComplex)));
    EXPECT_EQ(0, std::memcmp(y.data(), yOriginal.data(), y.size() * sizeof(aclblasComplex)));
    VerifyGuardedMatrix(a, aOriginal, golden, guardCount, aElements, n, lda, uplo);
}

constexpr int WARMUP_COUNT = 10;
constexpr int REPEAT_COUNT = 100;
constexpr int SAMPLE_COUNT = 60;

struct ScopedEvents {
    aclrtEvent start = nullptr;
    aclrtEvent end = nullptr;

    ~ScopedEvents()
    {
        if (start != nullptr) {
            aclrtDestroyEvent(start);
        }
        if (end != nullptr) {
            aclrtDestroyEvent(end);
        }
    }

    bool Create(int n)
    {
        if (aclrtCreateEvent(&start) == ACL_SUCCESS && aclrtCreateEvent(&end) == ACL_SUCCESS) {
            return true;
        }
        ADD_FAILURE() << "event creation failed for n=" << n;
        return false;
    }
};

bool RunWarmup(
    aclblasHandle_t handle, aclrtStream stream, const Csyr2Param& p, const aclblasComplex* xDevice,
    const aclblasComplex* yDevice, aclblasComplex* aDevice)
{
    for (int i = 0; i < WARMUP_COUNT; ++i) {
        if (aclblasCsyr2(handle, p.uplo, p.n, &p.alpha, xDevice, p.incx, yDevice, p.incy, aDevice, p.lda) !=
            ACLBLAS_STATUS_SUCCESS) {
            ADD_FAILURE() << "warmup launch failed for n=" << p.n;
            return false;
        }
    }
    if (aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
        ADD_FAILURE() << "warmup synchronization failed for n=" << p.n;
        return false;
    }
    return true;
}

bool RunTimedBatch(
    aclblasHandle_t handle, const Csyr2Param& p, const aclblasComplex* xDevice, const aclblasComplex* yDevice,
    aclblasComplex* aDevice)
{
    for (int repeat = 0; repeat < REPEAT_COUNT; ++repeat) {
        if (aclblasCsyr2(handle, p.uplo, p.n, &p.alpha, xDevice, p.incx, yDevice, p.incy, aDevice, p.lda) !=
            ACLBLAS_STATUS_SUCCESS) {
            ADD_FAILURE() << "timed launch failed for n=" << p.n;
            return false;
        }
    }
    return true;
}

double ThreadCpuUs(const rusage& usage)
{
    return 1e6 * (usage.ru_utime.tv_sec + usage.ru_stime.tv_sec) + usage.ru_utime.tv_usec + usage.ru_stime.tv_usec;
}

bool MeasureSample(
    aclblasHandle_t handle, aclrtStream stream, const Csyr2Param& p, const std::vector<aclblasComplex>& original,
    const aclblasComplex* xDevice, const aclblasComplex* yDevice, aclblasComplex* aDevice, const ScopedEvents& events,
    int sample, float& sampleUs)
{
    const size_t matrixBytes = original.size() * sizeof(aclblasComplex);
    if (aclrtMemcpy(aDevice, matrixBytes, original.data(), matrixBytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS ||
        aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
        ADD_FAILURE() << "sample reset failed for n=" << p.n;
        return false;
    }
    struct rusage beforeUsage {};
    struct rusage afterUsage {};
    const bool beforeUsageOk = getrusage(RUSAGE_THREAD, &beforeUsage) == 0;
    const auto wallStart = std::chrono::steady_clock::now();
    if (aclrtRecordEvent(events.start, stream) != ACL_SUCCESS) {
        ADD_FAILURE() << "start event record failed for n=" << p.n;
        return false;
    }
    if (!RunTimedBatch(handle, p, xDevice, yDevice, aDevice)) {
        // issue #605: 失败分支的同步失败同样要可见
        if (aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
            ADD_FAILURE() << "sample reset sync failed for n=" << p.n;
        }
        return false;
    }
    if (aclrtRecordEvent(events.end, stream) != ACL_SUCCESS || aclrtSynchronizeEvent(events.end) != ACL_SUCCESS) {
        ADD_FAILURE() << "end event record/synchronization failed for n=" << p.n;
        return false;
    }
    float elapsedMs = 0.0f;
    if (aclrtEventElapsedTime(&elapsedMs, events.start, events.end) != ACL_SUCCESS) {
        ADD_FAILURE() << "event elapsed-time query failed for n=" << p.n;
        return false;
    }
    const double wallUs =
        std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - wallStart).count();
    const bool usageOk = getrusage(RUSAGE_THREAD, &afterUsage) == 0 && beforeUsageOk;
    sampleUs = elapsedMs * 1000.0f / REPEAT_COUNT;
    std::printf(
        "CSYR2_SAMPLE case=%s n=%d sample=%d repeats=%d mean_us=%.6f "
        "batch_wall_us=%.3f thread_cpu_us=%.3f voluntary_switches=%ld involuntary_switches=%ld\n",
        p.caseName.c_str(), p.n, sample, REPEAT_COUNT, sampleUs, wallUs,
        usageOk ? ThreadCpuUs(afterUsage) - ThreadCpuUs(beforeUsage) : -1.0,
        usageOk ? afterUsage.ru_nvcsw - beforeUsage.ru_nvcsw : -1L,
        usageOk ? afterUsage.ru_nivcsw - beforeUsage.ru_nivcsw : -1L);
    return true;
}

float AverageSamples(const std::array<float, SAMPLE_COUNT>& samples)
{
    double totalUs = 0.0;
    for (float sample : samples) {
        if (!std::isfinite(sample) || sample <= 0.0f) {
            return std::numeric_limits<float>::quiet_NaN();
        }
        totalUs += sample;
    }
    return static_cast<float>(totalUs / SAMPLE_COUNT);
}

float MeasureMeanDeviceUs(aclblasHandle_t handle, aclrtStream stream, const Csyr2Param& p, const Csyr2HostData& data)
{
    const auto& x = data.x;
    const auto& y = data.y;
    const auto& a = data.aOriginal;
    Csyr2DeviceBuffers buffers;
    if (Csyr2AllocAndCopy(x.data(), x.size() * sizeof(aclblasComplex), buffers.x) != ACLBLAS_STATUS_SUCCESS ||
        Csyr2AllocAndCopy(y.data(), y.size() * sizeof(aclblasComplex), buffers.y) != ACLBLAS_STATUS_SUCCESS ||
        Csyr2AllocAndCopy(a.data(), a.size() * sizeof(aclblasComplex), buffers.a) != ACLBLAS_STATUS_SUCCESS) {
        ADD_FAILURE() << "device allocation/copy failed for n=" << p.n;
        return std::numeric_limits<float>::quiet_NaN();
    }
    const auto* xDevice = static_cast<const aclblasComplex*>(buffers.x);
    const auto* yDevice = static_cast<const aclblasComplex*>(buffers.y);
    auto* aDevice = static_cast<aclblasComplex*>(buffers.a);
    if (!RunWarmup(handle, stream, p, xDevice, yDevice, aDevice)) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    ScopedEvents events;
    if (!events.Create(p.n)) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    std::array<float, SAMPLE_COUNT> samples;
    samples.fill(std::numeric_limits<float>::quiet_NaN());
    for (int sample = 0; sample < SAMPLE_COUNT; ++sample) {
        float& sampleUs = samples[static_cast<size_t>(sample)];
        const bool measured = MeasureSample(handle, stream, p, a, xDevice, yDevice, aDevice, events, sample, sampleUs);
        if (!measured) {
            return std::numeric_limits<float>::quiet_NaN();
        }
    }
    return AverageSamples(samples);
}

struct ScopedBlasStream {
    aclblasHandle_t handle = nullptr;
    aclrtStream stream = nullptr;

    ~ScopedBlasStream()
    {
        if (stream != nullptr) {
            // issue #605: 收尾同步失败仅记录，不在析构路径抛出
            if (aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
                std::cerr << "csyr2_test: scoped stream sync failed" << std::endl;
            }
        }
        if (handle != nullptr) {
            aclblasDestroy(handle);
        }
        if (stream != nullptr) {
            aclrtDestroyStream(stream);
        }
    }
};

} // namespace

TEST_F(Csyr2Test, NullHandle)
{
    const aclblasComplex alpha{1.0f, 0.0f};
    aclblasComplex value{0.0f, 0.0f};
    EXPECT_EQ(
        ACLBLAS_STATUS_HANDLE_IS_NULLPTR,
        aclblasCsyr2_npu(nullptr, nullptr, ACLBLAS_UPPER, 1, &alpha, &value, 1, &value, 1, &value, 1));
}

TEST_F(Csyr2Test, ManualN2UpperAndLowerOnBoundStream)
{
    VerifyManualN2(handle_, stream_, ACLBLAS_UPPER);
    VerifyManualN2(handle_, stream_, ACLBLAS_LOWER);
}

TEST_F(Csyr2Test, GuardedNoncontiguousLayoutPreservesInputsPaddingAndBounds)
{
    VerifyGuardedNoncontiguousLayout(handle_, stream_, ACLBLAS_UPPER);
    VerifyGuardedNoncontiguousLayout(handle_, stream_, ACLBLAS_LOWER);
}

TEST_F(Csyr2Test, ExtremeIncrementsForSingleElementAreNotNarrowed)
{
    const aclblasComplex alpha{1.0f, -0.25f};
    std::vector<aclblasComplex> x{{2.0f, -3.0f}};
    std::vector<aclblasComplex> y{{-1.0f, 4.0f}};
    std::vector<aclblasComplex> a{{0.5f, 1.5f}};
    const std::vector<aclblasComplex> original = a;
    std::vector<aclblasComplex> golden = a;
    csyr2_test::Csyr2Golden(
        ACLBLAS_UPPER, 1, alpha, x.data(), std::numeric_limits<int>::min(), y.data(), std::numeric_limits<int>::max(),
        golden.data(), 1);
    ASSERT_EQ(
        ACLBLAS_STATUS_SUCCESS,
        aclblasCsyr2_npu(
            handle_, stream_, ACLBLAS_UPPER, 1, &alpha, x.data(), std::numeric_limits<int>::min(), y.data(),
            std::numeric_limits<int>::max(), a.data(), 1));
    EXPECT_FLOAT_EQ(golden[0].real, a[0].real);
    EXPECT_FLOAT_EQ(golden[0].imag, a[0].imag);
    EXPECT_NE(0, std::memcmp(a.data(), original.data(), sizeof(aclblasComplex)));
}

TEST_F(Csyr2Test, ContiguousVectorTailsPreserveGuardsAndTriangle)
{
    for (int n : {31, 32, 33, 39, 40, 63, 64, 65, 127, 129, 257}) {
        SCOPED_TRACE(n);
        VerifyGuardedNoncontiguousLayout(handle_, stream_, ACLBLAS_UPPER, n, 1, 1);
        VerifyGuardedNoncontiguousLayout(handle_, stream_, ACLBLAS_LOWER, n, 1, 1);
    }
}

TEST_F(Csyr2Test, LargeIndexHostValidationDoesNotLaunch)
{
    const aclblasComplex alpha{1.0f, 0.0f};
    aclblasComplex value{0.0f, 0.0f};
    // This exercises the largest legal structural values without allocating
    // impossible vectors.  Null x must be rejected before tiling/launch.
    EXPECT_EQ(
        ACLBLAS_STATUS_INVALID_VALUE,
        aclblasCsyr2(
            handle_, ACLBLAS_LOWER, std::numeric_limits<int>::max(), &alpha, nullptr, std::numeric_limits<int>::min(),
            &value, std::numeric_limits<int>::max(), &value, std::numeric_limits<int>::max()));
}

TEST_F(Csyr2Test, GenericPathBeyondVectorCapacity)
{
    for (const auto uplo : {ACLBLAS_UPPER, ACLBLAS_LOWER}) {
        Csyr2Param p(csv_map{{"case_name", "GenericPathBeyondVectorCapacity"}});
        p.n = 4097;
        p.lda = 4100;
        p.uplo = uplo;
        p.alpha = {0.5f, -0.25f};
        auto data = PrepareHostData(p);
        ASSERT_EQ(
            ACLBLAS_STATUS_SUCCESS, aclblasCsyr2_npu(
                                        handle_, stream_, p.uplo, p.n, &p.alpha, data.x.data(), p.incx, data.y.data(),
                                        p.incy, data.a.data(), p.lda));
        VerifyUpdatedTriangle(p, data.a, data.aOriginal, data.x, data.y);
        VerifyUnchangedStorage(p, data.a, data.aOriginal);
    }
}

TEST_F(Csyr2Test, IndependentStreamsAndConsecutiveInPlaceUpdates)
{
    ScopedBlasStream second;
    ASSERT_EQ(ACL_SUCCESS, aclrtCreateStream(&second.stream));
    ASSERT_EQ(ACLBLAS_STATUS_SUCCESS, aclblasCreate(&second.handle));
    ASSERT_EQ(ACLBLAS_STATUS_SUCCESS, aclblasSetStream(second.handle, second.stream));

    constexpr int n = 33;
    constexpr int lda = 37;
    const std::vector<aclblasComplex> x(n, {1.0f, -0.5f});
    const std::vector<aclblasComplex> y(n, {-0.25f, 2.0f});
    const std::vector<aclblasComplex> initial(n * lda, {0.5f, 1.0f});
    const size_t vectorBytes = x.size() * sizeof(aclblasComplex);
    const size_t matrixBytes = initial.size() * sizeof(aclblasComplex);
    Csyr2DeviceBuffers firstBuffers;
    Csyr2DeviceBuffers secondBuffers;
    for (auto* buffers : {&firstBuffers, &secondBuffers}) {
        ASSERT_EQ(ACLBLAS_STATUS_SUCCESS, Csyr2AllocAndCopy(x.data(), vectorBytes, buffers->x));
        ASSERT_EQ(ACLBLAS_STATUS_SUCCESS, Csyr2AllocAndCopy(y.data(), vectorBytes, buffers->y));
        ASSERT_EQ(ACLBLAS_STATUS_SUCCESS, Csyr2AllocAndCopy(initial.data(), matrixBytes, buffers->a));
    }
    aclblasComplex alpha{0.5f, 0.25f};
    auto expectedFirst = initial;
    auto expectedSecond = initial;
    for (int iteration = 0; iteration < 3; ++iteration) {
        EXPECT_EQ(
            ACLBLAS_STATUS_SUCCESS,
            aclblasCsyr2(
                handle_, ACLBLAS_UPPER, n, &alpha, static_cast<const aclblasComplex*>(firstBuffers.x), 1,
                static_cast<const aclblasComplex*>(firstBuffers.y), 1, static_cast<aclblasComplex*>(firstBuffers.a),
                lda));
        csyr2_test::Csyr2Golden(ACLBLAS_UPPER, n, alpha, x.data(), 1, y.data(), 1, expectedFirst.data(), lda);
        alpha.imag = -alpha.imag;
        EXPECT_EQ(
            ACLBLAS_STATUS_SUCCESS,
            aclblasCsyr2(
                second.handle, ACLBLAS_LOWER, n, &alpha, static_cast<const aclblasComplex*>(secondBuffers.x), 1,
                static_cast<const aclblasComplex*>(secondBuffers.y), 1, static_cast<aclblasComplex*>(secondBuffers.a),
                lda));
        csyr2_test::Csyr2Golden(ACLBLAS_LOWER, n, alpha, x.data(), 1, y.data(), 1, expectedSecond.data(), lda);
        alpha.real = -alpha.real;
    }
    // No synchronization between the interleaved public API calls. Reusing
    // and changing host alpha also checks that each launch captured its value.
    ASSERT_EQ(ACL_SUCCESS, aclrtSynchronizeStream(second.stream));
    ASSERT_EQ(ACL_SUCCESS, aclrtSynchronizeStream(stream_));
    auto actual = initial;
    for (bool first : {true, false}) {
        const auto& expected = first ? expectedFirst : expectedSecond;
        const auto& buffers = first ? firstBuffers : secondBuffers;
        ASSERT_EQ(ACL_SUCCESS, CopyComplexFromDevice(actual, buffers.a));
        EXPECT_EQ(0, std::memcmp(actual.data(), expected.data(), matrixBytes));
    }
}

TEST_F(Csyr2Test, InfinityPropagationMatchesCublasInBothLayouts)
{
    constexpr int n = 2;
    constexpr int lda = 5;
    const float inf = std::numeric_limits<float>::infinity();
    for (bool contiguous : {true, false}) {
        for (const auto uplo : {ACLBLAS_UPPER, ACLBLAS_LOWER}) {
            Csyr2Param p(csv_map{{"case_name", "InfinityPropagationMatchesCublasInBothLayouts"}});
            p.n = n;
            p.lda = lda;
            p.uplo = uplo;
            p.alpha = {0.5f, -0.25f};
            p.incx = contiguous ? 1 : -2;
            p.incy = contiguous ? 1 : 3;
            std::vector<aclblasComplex> x(VectorStorageSize(n, p.incx), {99.0f, 99.0f});
            std::vector<aclblasComplex> y(VectorStorageSize(n, p.incy), {99.0f, 99.0f});
            x[csyr2_test::LogicalOffset(0, n, p.incx)] = {inf, 1.0f};
            x[csyr2_test::LogicalOffset(1, n, p.incx)] = {1.0f, 2.0f};
            y[csyr2_test::LogicalOffset(0, n, p.incy)] = {2.0f, 1.0f};
            y[csyr2_test::LogicalOffset(1, n, p.incy)] = {3.0f, 4.0f};
            std::vector<aclblasComplex> a(n * lda, {0.5f, 0.75f});
            const auto original = a;
            ASSERT_EQ(
                ACLBLAS_STATUS_SUCCESS,
                aclblasCsyr2_npu(
                    handle_, stream_, uplo, n, &p.alpha, x.data(), p.incx, y.data(), p.incy, a.data(), lda));
            // Independent observed cuBLAS 12.6.4 classifications. Checking
            // these directly prevents a shared Golden/kernel order mistake.
            if (uplo == ACLBLAS_UPPER) {
                EXPECT_TRUE(std::isnan(a[lda].imag));
            } else {
                EXPECT_EQ(inf, a[1].imag);
            }
            VerifyUpdatedTriangle(p, a, original, x, y);
            VerifyUnchangedStorage(p, a, original);
        }
    }
}

TEST_F(Csyr2Test, TC_PF_DeviceEventTimingFixedThresholds)
{
    struct PerfCase {
        const char* name;
        aclblasFillMode_t uplo;
        int n;
        float maxUs;
    };
    constexpr std::array<PerfCase, 4> cases{{
        {"TC_PF_1001", ACLBLAS_UPPER, 512, 11.0f},
        {"TC_PF_1002", ACLBLAS_LOWER, 1024, 12.17f},
        {"TC_PF_1003", ACLBLAS_UPPER, 2048, 24.89f},
        {"TC_PF_1004", ACLBLAS_LOWER, 4096, 131.97f},
    }};
    for (const PerfCase& perfCase : cases) {
        Csyr2Param p(csv_map{{"case_name", perfCase.name}});
        p.uplo = perfCase.uplo;
        p.n = perfCase.n;
        p.lda = perfCase.n;
        Csyr2HostData data;
        data.x.assign(static_cast<size_t>(p.n), {1.25f, -0.75f});
        data.y.assign(static_cast<size_t>(p.n), {-0.5f, 2.0f});
        data.aOriginal.assign(static_cast<size_t>(p.n) * p.n, {0.25f, -1.0f});
        const float meanUs = MeasureMeanDeviceUs(handle_, stream_, p, data);
        std::printf("%s device_event_mean_us=%.4f threshold_us=%.4f\n", perfCase.name, meanUs, perfCase.maxUs);
        EXPECT_TRUE(std::isfinite(meanUs)) << perfCase.name << " event timing failed";
        EXPECT_LE(meanUs, perfCase.maxUs) << perfCase.name << " exceeds task threshold";
    }
}

INSTANTIATE_TEST_SUITE_P(
    Csyr2, Csyr2Test, ::testing::ValuesIn(GetCasesFromCsv<Csyr2Param>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<Csyr2Param>);

TEST_P(Csyr2Test, CsvDriven)
{
    const Csyr2Param& p = GetParam();
    Csyr2HostData data = PrepareHostData(p);
    const aclblasComplex* alpha = p.alphaNull ? nullptr : &p.alpha;
    const aclblasComplex* x = data.x.empty() ? nullptr : data.x.data();
    const aclblasComplex* y = data.y.empty() ? nullptr : data.y.data();
    aclblasComplex* a = data.a.empty() ? nullptr : data.a.data();
    const aclblasHandle_t handle = p.handleNull ? nullptr : handle_;

    const aclblasStatus_t status =
        aclblasCsyr2_npu(handle, stream_, p.uplo, p.n, alpha, x, p.incx, y, p.incy, a, p.lda);
    EXPECT_EQ(p.expectResult, status);
    if (status != ACLBLAS_STATUS_SUCCESS || p.n <= 0 || alpha == nullptr || a == nullptr) {
        return;
    }
    if (IsZero(p.alpha)) {
        EXPECT_EQ(0, std::memcmp(data.a.data(), data.aOriginal.data(), data.a.size() * sizeof(aclblasComplex)));
        return;
    }

    VerifyUpdatedTriangle(p, data.a, data.aOriginal, data.x, data.y);
    VerifyUnchangedStorage(p, data.a, data.aOriginal);
    if (p.caseName.rfind("TC_PF_", 0) == 0 && !HasFailure()) {
        const float meanUs = MeasureMeanDeviceUs(handle_, stream_, p, data);
        ASSERT_TRUE(std::isfinite(meanUs));
        struct rusage usage {};
        ASSERT_EQ(0, getrusage(RUSAGE_SELF, &usage));
        std::printf(
            "[PERF] case=%s average_us=%.6f warmup=10 samples=60 repeats=100 "
            "timer=device_event peak_memory_bytes=%llu memory_scope=process_high_watermark\n",
            p.caseName.c_str(), meanUs, static_cast<unsigned long long>(usage.ru_maxrss) * 1024ULL);
    }
}
