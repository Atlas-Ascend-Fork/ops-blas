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
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <mutex>
#include <new>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unistd.h>
#include <vector>

#include "mspti.h"

#include "blas_test.h"
#include "csv_loader.h"
#include "csymv_golden.h"
#include "csymv_npu_wrapper.h"
#include "csymv_param.h"

class CsymvTest : public BlasTest<CsymvParam> {};

namespace {

aclblasComplex AccumulateCsymvDouble(const CsymvCpuArguments& args, int row)
{
    if (CsymvComplexIsZero(*args.alpha)) {
        return {0.0f, 0.0f};
    }
    double real = 0;
    double imag = 0;
    for (int col = 0; col < args.n; ++col) {
        const aclblasComplex aValue = args.a[CsymvMatrixIndex(args, row, col)];
        const aclblasComplex xValue = args.x[CsymvVectorIndex(col, args.n, args.incx)];
        // Products retain FP32 overflow/NaN behavior; only the long sum is widened.
        const float productReal = aValue.real * xValue.real - aValue.imag * xValue.imag;
        const float productImag = aValue.real * xValue.imag + aValue.imag * xValue.real;
        real += productReal;
        imag += productImag;
    }
    return {static_cast<float>(real), static_cast<float>(imag)};
}

aclblasStatus_t CsymvCpuDouble(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n, const aclblasComplex* alpha, const aclblasComplex* A,
    int lda, const aclblasComplex* x, int incx, const aclblasComplex* beta, aclblasComplex* y, int incy)
{
    if (n < 0) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (handle == nullptr) {
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    const CsymvCpuArguments args{n, lda, incx, incy, uplo, alpha, A, x, beta, y};
    const aclblasStatus_t status = ValidateCsymvCpuArguments(args);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }
    if (CsymvComplexIsZero(*alpha) && CsymvComplexIsOne(*beta)) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    for (int row = 0; row < n; ++row) {
        CsymvCpuStoreResult(args, row, AccumulateCsymvDouble(args, row));
    }
    return ACLBLAS_STATUS_SUCCESS;
}

struct KernelActivitySample {
    uint64_t start;
    uint64_t end;
    uint32_t device;
    uint32_t stream;
    uint64_t correlation;
    std::string type;
    std::string name;
};

struct KernelActivityState {
    std::mutex mutex;
    std::vector<KernelActivitySample> samples;
    std::atomic<bool> failed{false};
    msptiSubscriberHandle subscriber = nullptr;
};

KernelActivityState kernelActivity;

void RequestKernelBuffer(uint8_t** buffer, size_t* size, size_t* maxRecords)
{
    constexpr size_t bufferBytes = 2U * 1024U * 1024U;
    *buffer = static_cast<uint8_t*>(std::malloc(bufferBytes));
    *size = *buffer == nullptr ? 0 : bufferBytes;
    *maxRecords = 0;
    if (*buffer == nullptr) {
        kernelActivity.failed = true;
    }
}

void CompleteKernelBuffer(uint8_t* buffer, size_t, size_t validSize)
{
    if (validSize == 0) {
        std::free(buffer);
        return;
    }
    try {
        std::lock_guard<std::mutex> guard(kernelActivity.mutex);
        msptiActivity* record = nullptr;
        msptiResult status = msptiActivityGetNextRecord(buffer, validSize, &record);
        while (status == MSPTI_SUCCESS && record != nullptr) {
            if (record->kind == MSPTI_ACTIVITY_KIND_KERNEL) {
                const auto* sample = reinterpret_cast<const msptiActivityKernel*>(record);
                kernelActivity.samples.push_back(
                    {sample->start, sample->end, sample->ds.deviceId, sample->ds.streamId, sample->correlationId,
                     sample->type == nullptr ? "" : sample->type, sample->name == nullptr ? "" : sample->name});
            }
            status = msptiActivityGetNextRecord(buffer, validSize, &record);
        }
        if (status != MSPTI_ERROR_MAX_LIMIT_REACHED) {
            kernelActivity.failed = true;
        }
    } catch (...) {
        // No exception may cross the C callback boundary.
        kernelActivity.failed = true;
    }
    std::free(buffer);
}

bool InitializeKernelActivity()
{
    {
        std::lock_guard<std::mutex> guard(kernelActivity.mutex);
        kernelActivity.samples.clear();
        kernelActivity.failed = false;
    }
    const msptiResult subscribed = msptiSubscribe(&kernelActivity.subscriber, nullptr, nullptr);
    if (subscribed != MSPTI_SUCCESS) {
        std::cerr << "msptiSubscribe failed: " << subscribed << std::endl;
        return false;
    }
    const msptiResult registered = msptiActivityRegisterCallbacks(RequestKernelBuffer, CompleteKernelBuffer);
    if (registered != MSPTI_SUCCESS) {
        std::cerr << "msptiActivityRegisterCallbacks failed: " << registered << std::endl;
        msptiUnsubscribe(kernelActivity.subscriber);
        kernelActivity.subscriber = nullptr;
        return false;
    }
    return true;
}

bool FinalizeKernelActivity()
{
    if (kernelActivity.subscriber == nullptr) {
        return true;
    }
    const msptiResult status = msptiUnsubscribe(kernelActivity.subscriber);
    kernelActivity.subscriber = nullptr;
    return status == MSPTI_SUCCESS;
}

bool SaveKernelActivities(
    const std::string& directory, const std::vector<KernelActivitySample>& samples, bool failed, double& totalUs)
{
    if (failed || samples.size() != 100U) {
        std::cerr << "Incomplete kernel activities: " << samples.size() << std::endl;
        return false;
    }
    std::set<uint64_t> correlations;
    const auto& first = samples.front();
    long double totalNs = 0;
    for (const auto& sample : samples) {
        if (sample.start == 0 || sample.end <= sample.start || sample.device != first.device ||
            sample.stream != first.stream || !correlations.insert(sample.correlation).second) {
            std::cerr << "Invalid kernel activity timestamps or identity" << std::endl;
            return false;
        }
        totalNs += sample.end - sample.start;
    }
    const std::string path = directory + "/kernel_activity.csv";
    std::ofstream output(path);
    output << "kind,start_ns,end_ns,device_id,stream_id,correlation_id,type,name\n";
    for (const auto& sample : samples) {
        output << MSPTI_ACTIVITY_KIND_KERNEL << ',' << sample.start << ',' << sample.end << ',' << sample.device << ','
               << sample.stream << ',' << sample.correlation << ',' << std::quoted(sample.type, '"', '"') << ','
               << std::quoted(sample.name, '"', '"') << '\n';
    }
    output.close();
    if (!output) {
        return false;
    }
    totalUs = static_cast<double>(totalNs / 1000.0L);
    std::cout << "[PROFILE] path=" << path << " samples=" << samples.size() << std::endl;
    return true;
}

int MeasureNpuKernelBatch(
    aclrtStream stream, const std::function<bool()>& reset, const std::function<bool()>& launch, bool externalProfiler,
    double& totalUs)
{
    totalUs = 0.0;
    for (int i = 0; i < 5; ++i) {
        if (!reset() || !launch() || aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
            return 1;
        }
    }
    if (!reset() || aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
        return 1;
    }
    char directory[] = "/tmp/csymv-kernel-XXXXXX";
    if (!externalProfiler) {
        if (mkdtemp(directory) == nullptr) {
            return 1;
        }
        if (!InitializeKernelActivity()) {
            return 1;
        }
        // Subscribe only to typed kernel activities, not memory-copy or API traces.
        const msptiResult enabled = msptiActivityEnable(MSPTI_ACTIVITY_KIND_KERNEL);
        if (enabled != MSPTI_SUCCESS) {
            std::cerr << "msptiActivityEnable failed: " << enabled << std::endl;
            FinalizeKernelActivity();
            msptiActivityFlushAll(1);
            return 1;
        }
    }
    bool valid = true;
    for (int i = 0; i < 100; ++i) {
        if (!launch()) {
            valid = false;
            break;
        }
    }
    valid = (aclrtSynchronizeStream(stream) == ACL_SUCCESS) && valid;
    if (externalProfiler) {
        return valid ? 0 : 1;
    }
    // Unsubscribe joins the device readers; FlushAll then delivers their final buffers.
    const bool stopped = FinalizeKernelActivity();
    const msptiResult flushed = msptiActivityFlushAll(1);
    if (!valid || !stopped || flushed != MSPTI_SUCCESS) {
        std::cerr << "Kernel activity collection failed: flush=" << flushed << " stopped=" << stopped << std::endl;
        return 1;
    }
    std::lock_guard<std::mutex> guard(kernelActivity.mutex);
    return SaveKernelActivities(directory, kernelActivity.samples, kernelActivity.failed, totalUs) ? 0 : 1;
}

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
    constexpr double kRtol = 0.0001220703125;
    constexpr double kAtol = 0.0001220703125;
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

struct PerformanceReference {
    std::string id;
    aclblasFillMode_t uplo;
    int n;
    int incx;
    int incy;
    double milliseconds;

    explicit PerformanceReference(const csv_map& row)
        : id(row.at("id")),
          uplo(parseFillMode("ACLBLAS_" + row.at("uplo"))),
          n(std::stoi(row.at("n"))),
          incx(std::stoi(row.at("incx"))),
          incy(std::stoi(row.at("incy"))),
          milliseconds(std::stod(row.at("gpu_ms")))
    {
        if (row.at("uplo") != "UPPER" && row.at("uplo") != "LOWER") {
            throw std::runtime_error("Invalid baseline uplo: " + id);
        }
    }
};

double DocumentedPerformanceLimitUs(const std::string& name)
{
    if (name == "TC_PF_1001") {
        return 15.45;
    }
    if (name == "TC_PF_1002") {
        return 46.14;
    }
    if (name == "TC_PF_1003") {
        return 38.57;
    }
    if (name == "TC_PF_1004") {
        return 146.0;
    }
    if (name == "TC_PF_1005") {
        return 99.55;
    }
    return std::numeric_limits<double>::infinity();
}

std::map<std::string, PerformanceReference> LoadPerformanceReferences()
{
    const std::string source(__FILE__);
    const auto references =
        GetCasesFromCsv<PerformanceReference>(source.substr(0, source.find_last_of("/\\") + 1U) + "gpu_baseline.csv");
    std::map<std::string, PerformanceReference> result;
    for (const auto& reference : references) {
        const size_t separator = reference.id.find_last_of('-');
        const std::string number = reference.id.substr(separator + 1U);
        if (number.size() != 3U || number.find_first_not_of("0123456789") != std::string::npos) {
            throw std::runtime_error("Invalid Csymv baseline ID: " + reference.id);
        }
        const int index = std::stoi(number);
        const std::string prefix = index < 5 ? "csymv-base-" : "csymv-perf-";
        if (index >= 200 || reference.id != prefix + number || !std::isfinite(reference.milliseconds) ||
            reference.milliseconds <= 0.0) {
            throw std::runtime_error("Invalid Csymv baseline: " + reference.id);
        }
        if (!result.emplace("TC_PF_" + std::to_string(1001 + index), reference).second) {
            throw std::runtime_error("Duplicate Csymv baseline: " + reference.id);
        }
    }
    if (references.size() != 200U || result.size() != 200U) {
        throw std::runtime_error("Expected all 200 Csymv GPU baselines");
    }
    return result;
}

void AttachPerformanceReference(CsymvParam& param, const PerformanceReference& reference)
{
    const auto actual = std::tie(
        param.n, param.uplo, param.lda, param.incx, param.incy, param.alpha.real, param.alpha.imag, param.beta.real,
        param.beta.imag, param.nullAlpha, param.nullBeta);
    const auto expected = std::make_tuple(
        reference.n, reference.uplo, reference.n, reference.incx, reference.incy, 1.0f, 0.0f, 0.0f, 0.0f, false, false);
    if (actual != expected) {
        throw std::runtime_error("Performance baseline parameters differ: " + param.caseName);
    }
    param.gpuBaselineMs = reference.milliseconds;
}

std::vector<CsymvParam> LoadCsymvCases()
{
    auto cases = GetCasesFromCsv<CsymvParam>(ReplaceFileExtension2Csv(__FILE__));
    const auto references = LoadPerformanceReferences();
    std::set<std::string> names;
    size_t performanceCount = 0;
    for (auto& param : cases) {
        if (!names.insert(param.caseName).second) {
            throw std::runtime_error("Duplicate Csymv case: " + param.caseName);
        }
        if (!param.IsPerformanceCase()) {
            continue;
        }
        const auto reference = references.find(param.caseName);
        if (reference == references.end()) {
            throw std::runtime_error("Missing Csymv GPU baseline: " + param.caseName);
        }
        AttachPerformanceReference(param, reference->second);
        ++performanceCount;
    }
    if (cases.size() != 1244U || performanceCount != 200U) {
        throw std::runtime_error("Expected 1044 accuracy and 200 performance cases");
    }
    return cases;
}

double RunPerformanceCase(
    aclblasHandle_t handle, aclrtStream stream, const CsymvParam& param, bool externalProfiler, double& totalUs)
{
    CsymvHostData data = MakeHostData(param);
    CsymvDeviceBuffers buffers;
    buffers.Prepare(
        param.n, &param.alpha, data.A.data(), param.lda, data.x.data(), param.incx, &param.beta, data.y.data(),
        param.incy, false);
    const auto reset = [&]() {
        // Restore the original y before each batch, before profiling.
        return aclrtMemcpy(buffers.yArg, buffers.yBytes, data.y.data(), buffers.yBytes, ACL_MEMCPY_HOST_TO_DEVICE) ==
               ACL_SUCCESS;
    };
    const auto launch = [&]() {
        const aclblasStatus_t status = aclblasCsymv(
            handle, param.uplo, param.n, buffers.alphaArg, buffers.aArg, param.lda, buffers.xArg, param.incx,
            buffers.betaArg, buffers.yArg, param.incy);
        return status == ACLBLAS_STATUS_SUCCESS;
    };
    auto golden = data.y;
    const aclblasStatus_t goldenStatus = CsymvCpuDouble(
        handle, param.uplo, param.n, &param.alpha, data.A.data(), param.lda, data.x.data(), param.incx, &param.beta,
        golden.data(), param.incy);
    if (goldenStatus != ACLBLAS_STATUS_SUCCESS || !launch() || aclrtSynchronizeStream(stream) != ACL_SUCCESS)
        return -1.0;
    const auto goldenLogical = ExtractLogicalY(golden, param.n, param.incy);
    const auto verify = [&]() {
        auto actual = data.y;
        buffers.CopyYToHost(actual.data());
        const auto logical = ExtractLogicalY(actual, param.n, param.incy);
        const bool realPassed = VerifyComponent(logical, goldenLogical, false, param.caseName);
        const bool imagPassed = VerifyComponent(logical, goldenLogical, true, param.caseName);
        return realPassed && imagPassed;
    };
    if (!verify())
        return -1.0;
    // beta=0 makes each repeated call independent of old y.
    if (param.beta.real != 0.0f || param.beta.imag != 0.0f)
        return -1.0;
    const int status = MeasureNpuKernelBatch(stream, reset, launch, externalProfiler, totalUs);
    if (status != 0 || !verify())
        return -1.0;
    return totalUs / 100.0;
}

} // namespace

TEST_F(CsymvTest, NullHandle)
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

TEST_F(CsymvTest, HostScalarQuickReturnAllowsNullData)
{
    const aclblasComplex alpha{0.0f, 0.0f};
    const aclblasComplex beta{1.0f, 0.0f};
    EXPECT_EQ(
        aclblasCsymv(handle_, ACLBLAS_UPPER, 8, &alpha, nullptr, 8, nullptr, 1, &beta, nullptr, 1),
        ACLBLAS_STATUS_SUCCESS);
}

TEST_F(CsymvTest, DeviceScalarQuickReturnAllowsNullData)
{
    const aclblasComplex alpha{0.0f, 0.0f};
    const aclblasComplex beta{1.0f, 0.0f};
    EXPECT_EQ(
        aclblasCsymv_npu(handle_, ACLBLAS_LOWER, 8, &alpha, nullptr, 8, nullptr, -1, &beta, nullptr, -1, true),
        ACLBLAS_STATUS_SUCCESS);
}

TEST_F(CsymvTest, ScaleOnlyAllowsNullMatrixAndX)
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

TEST_F(CsymvTest, BetaOnePreservesInfiniteY)
{
    const aclblasComplex alpha{1.0f, 0.0f};
    const aclblasComplex beta{1.0f, 0.0f};
    const aclblasComplex A{1.0f, 0.0f};
    const aclblasComplex x{1.0f, 0.0f};
    const float infinity = std::numeric_limits<float>::infinity();
    aclblasComplex goldenY{infinity, 2.0f};
    aclblasComplex npuY = goldenY;

    ASSERT_EQ(
        CsymvCpuDouble(handle_, ACLBLAS_UPPER, 1, &alpha, &A, 1, &x, 1, &beta, &goldenY, 1), ACLBLAS_STATUS_SUCCESS);
    ASSERT_EQ(
        aclblasCsymv_npu(handle_, ACLBLAS_UPPER, 1, &alpha, &A, 1, &x, 1, &beta, &npuY, 1, false),
        ACLBLAS_STATUS_SUCCESS);

    EXPECT_EQ(goldenY.real, infinity);
    EXPECT_FLOAT_EQ(goldenY.imag, 2.0f);
    EXPECT_EQ(npuY.real, infinity);
    EXPECT_FLOAT_EQ(npuY.imag, 2.0f);
}

TEST_F(CsymvTest, SymmetricNotHermitian)
{
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const aclblasComplex alpha{1, 0};
    const aclblasComplex beta{0, 0};
    const aclblasComplex matrix[] = {{1, 2}, {nan, nan}, {3, 4}, {5, 6}};
    const aclblasComplex x[] = {{1, 0}, {0, 1}};
    aclblasComplex y[] = {{nan, nan}, {nan, nan}};
    ASSERT_EQ(
        aclblasCsymv_npu(handle_, ACLBLAS_UPPER, 2, &alpha, matrix, 2, x, 1, &beta, y, 1, false),
        ACLBLAS_STATUS_SUCCESS);
    EXPECT_FLOAT_EQ(y[0].real, -3);
    EXPECT_FLOAT_EQ(y[0].imag, 5);
    EXPECT_FLOAT_EQ(y[1].real, -3);
    EXPECT_FLOAT_EQ(y[1].imag, 9);
}

namespace {
CsymvHostData MakeTriangleData(aclblasFillMode_t uplo, int n, int lda, int incy)
{
    const float nan = std::numeric_limits<float>::quiet_NaN();
    CsymvHostData data;
    data.A.assign(static_cast<size_t>(lda) * n, {nan, nan});
    data.x.resize(n);
    data.y.assign(StorageLength(n, incy), {123, -456});
    for (int c = 0; c < n; ++c) {
        data.x[c] = {0.25f, -0.5f};
        data.y[CsymvVectorIndex(c, n, incy)] = {0.5f, 0.25f};
        for (int r = 0; r < n; ++r) {
            if ((uplo == ACLBLAS_UPPER && r <= c) || (uplo == ACLBLAS_LOWER && r >= c)) {
                data.A[static_cast<size_t>(c) * lda + r] = {static_cast<float>((r + 3 * c) % 7 - 3) * 0.125f, 0.25f};
            }
        }
    }
    return data;
}

} // namespace

TEST_F(CsymvTest, RuntimePaddingPreservesLogicalOutputs)
{
    const aclblasComplex alpha{0.75f, -0.5f};
    const aclblasComplex beta{0.25f, 0.125f};
    const aclblasComplex guard{123.0f, -456.0f};
    for (int n : {16, 32, 64, 65, 127, 129, 506}) {
        for (auto uplo : {ACLBLAS_UPPER, ACLBLAS_LOWER}) {
            SCOPED_TRACE(::testing::Message() << "n=" << n << " uplo=" << uplo);
            const int lda = n + 3;
            auto data = MakeTriangleData(uplo, n, lda, 1);
            std::vector<aclblasComplex> output(n + 2, guard);
            std::copy(data.y.begin(), data.y.end(), output.begin() + 1);
            auto golden = data.y;
            DeviceBuffer a(data.A.size() * sizeof(aclblasComplex));
            DeviceBuffer x(data.x.size() * sizeof(aclblasComplex));
            DeviceBuffer y(output.size() * sizeof(aclblasComplex));
            a.copyFromHost(data.A.data(), a.size());
            x.copyFromHost(data.x.data(), x.size());
            y.copyFromHost(output.data(), y.size());
            // Two queued calls exercise workspace reuse and nonzero beta without a host sync.
            for (int call = 0; call < 2; ++call) {
                EXPECT_EQ(
                    CsymvCpuDouble(
                        handle_, uplo, n, &alpha, data.A.data(), lda, data.x.data(), 1, &beta, golden.data(), 1),
                    ACLBLAS_STATUS_SUCCESS);
                EXPECT_EQ(
                    aclblasCsymv(
                        handle_, uplo, n, &alpha, static_cast<const aclblasComplex*>(a.ptr()), lda,
                        static_cast<const aclblasComplex*>(x.ptr()), 1, &beta,
                        static_cast<aclblasComplex*>(y.ptr()) + 1, 1),
                    ACLBLAS_STATUS_SUCCESS);
            }
            ASSERT_EQ(aclrtSynchronizeStream(stream_), ACL_SUCCESS);
            y.copyToHost(output.data(), y.size());
            EXPECT_EQ(0, std::memcmp(&output.front(), &guard, sizeof(guard)));
            EXPECT_EQ(0, std::memcmp(&output.back(), &guard, sizeof(guard)));
            const std::vector<aclblasComplex> actual(output.begin() + 1, output.end() - 1);
            EXPECT_TRUE(VerifyComponent(actual, golden, false, "runtime_padding"));
            EXPECT_TRUE(VerifyComponent(actual, golden, true, "runtime_padding"));
        }
    }
}

TEST_F(CsymvTest, RuntimePaddingPreservesOwnerReduction)
{
    const aclblasComplex one{1.0f, 0.0f};
    const aclblasComplex zero{0.0f, 0.0f};
    const float huge = std::numeric_limits<float>::max();
    for (int n : {506, 639}) {
        for (auto uplo : {ACLBLAS_UPPER, ACLBLAS_LOWER}) {
            std::vector<aclblasComplex> a(static_cast<size_t>(n) * n, zero);
            std::vector<aclblasComplex> x(n, one);
            std::vector<aclblasComplex> y(n, zero);
            // A triangular FP32 partial sum would overflow before cancellation.
            for (int col : {16, 17, 144, 145}) {
                const size_t index = uplo == ACLBLAS_UPPER ? static_cast<size_t>(col) * n : col;
                a[index] = {col < 32 ? huge : -huge, 0.0f};
            }
            ASSERT_EQ(
                aclblasCsymv_npu(handle_, uplo, n, &one, a.data(), n, x.data(), 1, &zero, y.data(), 1, false),
                ACLBLAS_STATUS_SUCCESS);
            EXPECT_FLOAT_EQ(y[0].real, 0.0f);
            EXPECT_FLOAT_EQ(y[0].imag, 0.0f);
        }
    }
}

TEST_F(CsymvTest, SmallWorkspaceFallsBack)
{
    constexpr int n = 4096;
    const aclblasComplex one{1, 0};
    const aclblasComplex zero{0, 0};
    DeviceBuffer workspace(sizeof(float));
    aclblasHandle_t handle = nullptr;
    ASSERT_EQ(aclblasCreate(&handle), ACLBLAS_STATUS_SUCCESS);
    EXPECT_EQ(aclblasSetStream(handle, stream_), ACLBLAS_STATUS_SUCCESS);
    EXPECT_EQ(aclblasSetWorkspace(handle, workspace.ptr(), sizeof(float)), ACLBLAS_STATUS_SUCCESS);
    std::vector<aclblasComplex> a(static_cast<size_t>(n) * n, {1, 0});
    std::vector<aclblasComplex> x(n, one);
    std::vector<aclblasComplex> y(
        n, {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN()});
    EXPECT_EQ(
        aclblasCsymv_npu(handle, ACLBLAS_LOWER, n, &one, a.data(), n, x.data(), 1, &zero, y.data(), 1, false),
        ACLBLAS_STATUS_SUCCESS);
    for (const auto& value : y) {
        EXPECT_FLOAT_EQ(value.real, n);
        EXPECT_FLOAT_EQ(value.imag, 0);
    }
    EXPECT_EQ(aclblasDestroy(handle), ACLBLAS_STATUS_SUCCESS);
}

INSTANTIATE_TEST_SUITE_P(Csymv, CsymvTest, ::testing::ValuesIn(LoadCsymvCases()), PrintCaseInfoString<CsymvParam>);

class CsymvMemoryCheckTest : public BlasTest<int> {};

TEST_P(CsymvMemoryCheckTest, RouteBoundary)
{
    const int n = GetParam();
    const aclblasComplex one{1, 0};
    const aclblasComplex zero{0, 0};
    std::vector<aclblasComplex> a(static_cast<size_t>(n) * n, one);
    const std::vector<aclblasComplex> x(n, one);
    for (aclblasFillMode_t uplo : {ACLBLAS_UPPER, ACLBLAS_LOWER}) {
        SCOPED_TRACE(::testing::Message() << "n=" << n << " uplo=" << uplo);
        std::fill(a.begin(), a.end(), one);
        PoisonUnusedTriangle(a, uplo, n, n);
        std::vector<aclblasComplex> y(n, zero);
        ASSERT_EQ(
            aclblasCsymv_npu(handle_, uplo, n, &one, a.data(), n, x.data(), 1, &zero, y.data(), 1, true),
            ACLBLAS_STATUS_SUCCESS);
        for (const auto& value : y) {
            EXPECT_FLOAT_EQ(value.real, n);
            EXPECT_FLOAT_EQ(value.imag, 0);
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    Csymv, CsymvMemoryCheckTest, ::testing::Values(16, 32, 64, 697, 2048, 4096, 8192),
    [](const ::testing::TestParamInfo<int>& info) { return "N" + std::to_string(info.param); });

class CsymvPhaseWindowTest : public BlasTest<int> {
protected:
    aclblasHandle_t workspaceHandle_ = nullptr;

    void SetUp() override
    {
        ASSERT_EQ(aclblasCreate(&workspaceHandle_), ACLBLAS_STATUS_SUCCESS);
        ASSERT_EQ(aclblasSetStream(workspaceHandle_, stream_), ACLBLAS_STATUS_SUCCESS);
    }

    void TearDown() override
    {
        if (workspaceHandle_ != nullptr) {
            EXPECT_EQ(aclblasDestroy(workspaceHandle_), ACLBLAS_STATUS_SUCCESS);
        }
    }
};

namespace {

void CheckPhaseWindowOutput(
    aclblasHandle_t handle, aclrtStream stream, int n, const std::vector<aclblasComplex>& a,
    const std::vector<aclblasComplex>& x)
{
    const aclblasComplex one{1, 0};
    const aclblasComplex zero{0, 0};
    const aclblasComplex guard{123, -456};
    std::vector<aclblasComplex> golden(n, zero);
    ASSERT_EQ(
        CsymvCpuDouble(handle, ACLBLAS_LOWER, n, &one, a.data(), n, x.data(), 1, &zero, golden.data(), 1),
        ACLBLAS_STATUS_SUCCESS);
    std::vector<aclblasComplex> output(n + 8, guard);
    auto deviceA = CsymvCopyToDevice(a.data(), a.size() * sizeof(aclblasComplex));
    auto deviceX = CsymvCopyToDevice(x.data(), x.size() * sizeof(aclblasComplex));
    auto deviceY = CsymvCopyToDevice(output.data(), output.size() * sizeof(aclblasComplex));
    auto* y = static_cast<aclblasComplex*>(deviceY->ptr()) + 4;
    const uintptr_t pointers = reinterpret_cast<uintptr_t>(deviceA->ptr()) |
                               reinterpret_cast<uintptr_t>(deviceX->ptr()) | reinterpret_cast<uintptr_t>(y);
    ASSERT_EQ(pointers & 31U, 0U);
    ASSERT_EQ(
        aclblasCsymv(
            handle, ACLBLAS_LOWER, n, &one, static_cast<const aclblasComplex*>(deviceA->ptr()), n,
            static_cast<const aclblasComplex*>(deviceX->ptr()), 1, &zero, y, 1),
        ACLBLAS_STATUS_SUCCESS);
    ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);
    deviceY->copyToHost(output.data(), deviceY->size());
    for (int i = 0; i < 4; ++i) {
        EXPECT_EQ(0, std::memcmp(&output[i], &guard, sizeof(guard)));
        EXPECT_EQ(0, std::memcmp(&output[n + 4 + i], &guard, sizeof(guard)));
    }
    const std::vector<aclblasComplex> actual(output.begin() + 4, output.end() - 4);
    EXPECT_TRUE(VerifyComponent(actual, golden, false, "phase_window"));
    EXPECT_TRUE(VerifyComponent(actual, golden, true, "phase_window"));
}

size_t PhaseWindowWorkspaceBytes(int n)
{
    const size_t blocks = (static_cast<size_t>(n) + 63U) / 64U;
    return 1536U * blocks * (blocks + 1U) / 2U;
}

} // namespace

TEST_P(CsymvPhaseWindowTest, WorkspaceThreshold)
{
    const int n = GetParam();
    const size_t required = PhaseWindowWorkspaceBytes(n);
    DeviceBuffer workspace(required + 64U);
    ASSERT_EQ(reinterpret_cast<uintptr_t>(workspace.ptr()) & 31U, 0U);
    std::vector<uint8_t> before(workspace.size(), 0xa5U);
    std::vector<uint8_t> after(workspace.size());
    const std::vector<aclblasComplex> a(static_cast<size_t>(n) * n, {1, 0});
    const std::vector<aclblasComplex> x(n, {1, 0});
    int64_t cores = 0;
    ASSERT_EQ(
        aclrtGetDeviceInfo(
            static_cast<uint32_t>(blas_test_detail::RuntimeDeviceId()), ACL_DEV_ATTR_VECTOR_CORE_NUM, &cores),
        ACL_SUCCESS);
    const bool enoughCores = cores >= static_cast<int64_t>(required / 1536U);
    for (size_t bytes : {required - 1U, required, required + 32U}) {
        SCOPED_TRACE(::testing::Message() << "n=" << n << " workspace_bytes=" << bytes);
        workspace.copyFromHost(before.data(), before.size());
        ASSERT_EQ(aclblasSetWorkspace(workspaceHandle_, workspace.ptr(), bytes), ACLBLAS_STATUS_SUCCESS);
        ASSERT_NO_FATAL_FAILURE(CheckPhaseWindowOutput(workspaceHandle_, stream_, n, a, x));
        workspace.copyToHost(after.data(), after.size());
        const bool changed = !std::equal(before.begin(), before.begin() + required, after.begin());
        // Below n=512 only phase-window uses workspace, so writes prove dispatch.
        EXPECT_EQ(changed, enoughCores && bytes >= required);
        EXPECT_TRUE(std::equal(before.begin() + bytes, before.end(), after.begin() + bytes));
    }
    if (!enoughCores) {
        GTEST_SKIP() << "Resource fallback checked; phase entry needs " << required / 1536U << " AIV cores";
    }
}

TEST_P(CsymvPhaseWindowTest, NonfiniteAndOwnerReplay)
{
    const int n = GetParam();
    const aclblasComplex zero{0, 0};
    const float huge = std::numeric_limits<float>::max();
    const size_t required = PhaseWindowWorkspaceBytes(n);
    int64_t cores = 0;
    ASSERT_EQ(
        aclrtGetDeviceInfo(
            static_cast<uint32_t>(blas_test_detail::RuntimeDeviceId()), ACL_DEV_ATTR_VECTOR_CORE_NUM, &cores),
        ACL_SUCCESS);
    if (cores < static_cast<int64_t>(required / 1536U)) {
        GTEST_SKIP() << "Phase replay needs " << required / 1536U << " AIV cores";
    }
    DeviceBuffer workspace(required);
    ASSERT_EQ(reinterpret_cast<uintptr_t>(workspace.ptr()) & 31U, 0U);
    ASSERT_EQ(aclblasSetWorkspace(workspaceHandle_, workspace.ptr(), required), ACLBLAS_STATUS_SUCCESS);
    const std::vector<uint8_t> before(required, 0xa5U);
    std::vector<uint8_t> after(required);
    const std::vector<aclblasComplex> x(n, {1, 0});
    const auto check = [&](const std::vector<aclblasComplex>& a) {
        workspace.copyFromHost(before.data(), before.size());
        ASSERT_NO_FATAL_FAILURE(CheckPhaseWindowOutput(workspaceHandle_, stream_, n, a, x));
        workspace.copyToHost(after.data(), after.size());
        EXPECT_NE(before, after) << "Nonfinite regression must execute the phase entry";
    };
    for (float value : {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity(), huge}) {
        SCOPED_TRACE(::testing::Message() << "n=" << n << " diagonal=" << value);
        std::vector<aclblasComplex> a(static_cast<size_t>(n) * n, zero);
        a.back() = {value, 0};
        PoisonUnusedTriangle(a, ACLBLAS_LOWER, n, n);
        ASSERT_NO_FATAL_FAILURE(check(a));
    }
    std::vector<aclblasComplex> a(static_cast<size_t>(n) * n, zero);
    // Both a full finalizer group and the short last group must replay finite cancellation.
    for (int row : {0, n - 1}) {
        for (int col : {16, 17, 144, 145}) {
            const size_t index = static_cast<size_t>(std::min(row, col)) * n + std::max(row, col);
            const float value = col < 32 ? huge : -huge;
            a[index] = {value, value};
        }
    }
    PoisonUnusedTriangle(a, ACLBLAS_LOWER, n, n);
    ASSERT_NO_FATAL_FAILURE(check(a));
}

// Reuse CSV shape IDs so incremental and smoke filters also select these checks.
INSTANTIATE_TEST_SUITE_P(
    Csymv, CsymvPhaseWindowTest, ::testing::Values(310, 350, 414, 478), [](const ::testing::TestParamInfo<int>& info) {
        return "TC_PHASE" + std::to_string((info.param + 63) / 64) + "_001";
    });

void CheckPerformanceCase(aclblasHandle_t handle, aclrtStream stream, const CsymvParam& param)
{
    // External batch profiling owns collection; do not subscribe to MSPTI as well.
    const char* preload = std::getenv("LD_PRELOAD");
    const char* external = std::getenv("CSYMV_EXTERNAL_PROFILER");
    const bool externalProfiler = (external != nullptr && std::strcmp(external, "1") == 0) ||
                                  (preload != nullptr && std::strstr(preload, "libmsopprof_injection.so") != nullptr);
    double totalUs = 0.0;
    const double averageUs = RunPerformanceCase(handle, stream, param, externalProfiler, totalUs);
    if (averageUs < 0.0 || !std::isfinite(averageUs)) {
        ADD_FAILURE() << "Csymv performance execution or collection failed";
        return;
    }
    std::cout << "[NUMERIC] case=" << param.caseName << " n=" << param.n
              << " uplo=" << (param.uplo == ACLBLAS_UPPER ? "UPPER" : "LOWER") << " verdict=PASS" << std::endl;
    if (externalProfiler) {
        std::cout << "[PROFILE] case=" << param.caseName
                  << " collector=external numeric_launches=1 warmup=5 repeats=100 skip=6" << std::endl;
        return;
    }
    if (averageUs == 0.0) {
        ADD_FAILURE() << "Invalid NPU kernel timing: " << averageUs;
        return;
    }
    constexpr double requiredRatio = 0.8;
    const double baselineUs = param.gpuBaselineMs * 1000.0;
    ASSERT_GT(baselineUs, 0.0);
    ASSERT_TRUE(std::isfinite(baselineUs));
    const double thresholdUs = std::min(DocumentedPerformanceLimitUs(param.caseName), baselineUs / requiredRatio);
    const double ratio = baselineUs / averageUs;
    const bool passed = ratio >= requiredRatio && averageUs <= thresholdUs;
    std::cout << std::setprecision(std::numeric_limits<double>::max_digits10) << "[PERF] case=" << param.caseName
              << " uplo=" << (param.uplo == ACLBLAS_UPPER ? "UPPER" : "LOWER") << " n=" << param.n
              << " metric=kernel_sum warmup=5 repeats=100 batches=1 total_us=" << totalUs << " avg_us=" << averageUs;
    std::cout << " baseline_us=" << baselineUs << " threshold_us=" << thresholdUs << " ratio=" << ratio
              << " required_ratio=" << requiredRatio << " verdict=" << (passed ? "PASS" : "FAIL") << std::endl;
    if (std::isfinite(DocumentedPerformanceLimitUs(param.caseName))) {
        EXPECT_GE(ratio, requiredRatio);
        EXPECT_LE(averageUs, thresholdUs);
    }
}

TEST_P(CsymvTest, CsvDriven)
{
    const CsymvParam& param = GetParam();
    if (param.IsPerformanceCase()) {
        CheckPerformanceCase(handle_, stream_, param);
        return;
    }

    CsymvHostData data;
    try {
        data = MakeHostData(param);
    } catch (const std::bad_alloc&) {
        FAIL() << "Host allocation exceeds available memory";
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

    const aclblasStatus_t goldenStatus = CsymvCpuDouble(
        handle_, param.uplo, param.n, alpha, A, param.lda, x, param.incx, beta,
        goldenY.empty() ? nullptr : goldenY.data(), param.incy);
    ASSERT_EQ(goldenStatus, ACLBLAS_STATUS_SUCCESS);
    if (param.n == 0 || data.y.empty()) {
        return;
    }

    const size_t step = static_cast<size_t>(std::abs(static_cast<int64_t>(param.incy)));
    for (size_t i = 0; i < data.y.size(); ++i) {
        if (i % step != 0) {
            EXPECT_EQ(0, std::memcmp(&data.y[i], &goldenY[i], sizeof(aclblasComplex))) << "padding index=" << i;
        }
    }

    const std::vector<aclblasComplex> actualLogical = ExtractLogicalY(data.y, param.n, param.incy);
    const std::vector<aclblasComplex> goldenLogical = ExtractLogicalY(goldenY, param.n, param.incy);
    EXPECT_TRUE(VerifyComponent(actualLogical, goldenLogical, false, param.caseName));
    EXPECT_TRUE(VerifyComponent(actualLogical, goldenLogical, true, param.caseName));
}
