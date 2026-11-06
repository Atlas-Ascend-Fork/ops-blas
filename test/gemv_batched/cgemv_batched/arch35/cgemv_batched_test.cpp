/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file cgemv_batched_test.cpp
 * \brief CSV-driven GTest for aclblasCgemvBatched on arch35 (Ascend 950PR).
 */

#include <cstring>
#include <vector>
#include <climits>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>
#include <algorithm>
#include <string>
#include "verify.h"
#include "blas_test.h"
#include "csv_loader.h"
#include "cgemv_batched_param.h"
#include "cgemv_batched_golden.h"
#include "cgemv_batched_npu_wrapper.h"

class CgemvBatchedTest : public BlasTest<CgemvBatchedParam> {};

namespace cgemv_batched_test {

// ---------------------------------------------------------------------------
// Performance measurement: aclrtEvent-based kernel timing (5 warmup + 10 samples)
// Equivalent to msprof op task duration when msprof profiling channel is unavailable.
// ---------------------------------------------------------------------------

std::string GetGpuBaselinePath()
{
    std::string file = __FILE__;
    size_t pos = file.find_last_of('/');
    if (pos == std::string::npos)
        return "gpu_baseline.csv";
    return file.substr(0, pos + 1) + "gpu_baseline.csv";
}

std::vector<double> LoadGpuBaselineUs()
{
    std::vector<double> baseline;
    std::ifstream ifs(GetGpuBaselinePath());
    if (!ifs)
        return baseline;
    std::string line;
    std::getline(ifs, line); // skip header
    while (std::getline(ifs, line)) {
        std::stringstream ss(line);
        std::string field;
        for (int i = 0; i < 8; ++i)
            std::getline(ss, field, ','); // skip to gpu_ms
        std::getline(ss, field, ',');
        if (!field.empty())
            baseline.push_back(std::stod(field) * 1000.0); // ms -> us
    }
    return baseline;
}

struct PerfResult {
    double avgUs = 0.0;
    double minUs = 0.0;
    double maxUs = 0.0;
    bool valid = false;
};

// Owns the per-batch device buffers and the pointer arrays handed to the API.
// DeviceBuffer is move-only and cannot be default-constructed (size 0 is an
// invalid allocation), so the pointer arrays are held by unique_ptr and built
// once the batch count is known. Everything is released when the struct dies.
struct BatchedDeviceInputs {
    std::vector<DeviceBuffer> aBufs;
    std::vector<DeviceBuffer> xBufs;
    std::vector<DeviceBuffer> yBufs;
    std::unique_ptr<DeviceBuffer> aPtrArr;
    std::unique_ptr<DeviceBuffer> xPtrArr;
    std::unique_ptr<DeviceBuffer> yPtrArr;
};

// Per-batch upload of A/x/y plus construction of the device-side pointer arrays.
// Reuses the batched-upload helpers shared with the functional NPU wrapper so
// the timed launch path stays identical to the path under test.
BatchedDeviceInputs UploadBatchedInputs(
    const aclblasComplex* a, const aclblasComplex* x, aclblasComplex* y, int m, int n, int lda, int incx, int incy,
    aclblasOperation_t trans, int batchCount)
{
    using namespace cgemv_batched_npu_wrapper;

    BatchedDeviceInputs in;
    const CgemvBatchedStrides s = ComputeStrides(trans, m, n, lda, incx, incy);
    const size_t perBatchABytes = s.aStride * sizeof(aclblasComplex);
    const size_t perBatchXBytes = s.xStride * sizeof(aclblasComplex);
    const size_t perBatchYBytes = s.yStride * sizeof(aclblasComplex);

    std::vector<const aclblasComplex*> hAPtrs(batchCount);
    std::vector<const aclblasComplex*> hXPtrs(batchCount);
    std::vector<aclblasComplex*> hYPtrs(batchCount);
    UploadReadOnlyBatches(a, s.aStride, perBatchABytes, batchCount, in.aBufs, hAPtrs);
    UploadReadOnlyBatches(x, s.xStride, perBatchXBytes, batchCount, in.xBufs, hXPtrs);
    UploadWritableBatches(y, s.yStride, perBatchYBytes, batchCount, in.yBufs, hYPtrs);

    in.aPtrArr.reset(new DeviceBuffer(BuildReadOnlyPtrArray(hAPtrs, batchCount)));
    in.xPtrArr.reset(new DeviceBuffer(BuildReadOnlyPtrArray(hXPtrs, batchCount)));
    in.yPtrArr.reset(new DeviceBuffer(BuildWritablePtrArray(hYPtrs, batchCount)));
    return in;
}

// Launch the operator once with the prepared device-side pointer arrays.
aclblasStatus_t LaunchBatched(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, const aclblasComplex* alpha, int lda,
    const aclblasComplex* beta, int incx, int incy, int batchCount, const DeviceBuffer* dAPtrArr,
    const DeviceBuffer* dXPtrArr, const DeviceBuffer* dYPtrArr)
{
    return aclblasCgemvBatched(
        handle, trans, m, n, alpha, reinterpret_cast<const aclblasComplex* const*>(dAPtrArr->ptr()), lda,
        reinterpret_cast<const aclblasComplex* const*>(dXPtrArr->ptr()), incx, beta,
        reinterpret_cast<aclblasComplex* const*>(dYPtrArr->ptr()), incy, batchCount);
}

// Warm up the operator so the timed samples run on a hot pipeline.
void RunWarmup(
    aclblasHandle_t handle, aclrtStream stream, aclblasOperation_t trans, int m, int n, const aclblasComplex* alpha,
    int lda, const aclblasComplex* beta, int incx, int incy, int batchCount, const DeviceBuffer* dAPtrArr,
    const DeviceBuffer* dXPtrArr, const DeviceBuffer* dYPtrArr, int warmupCount)
{
    for (int i = 0; i < warmupCount; ++i) {
        LaunchBatched(handle, trans, m, n, alpha, lda, beta, incx, incy, batchCount, dAPtrArr, dXPtrArr, dYPtrArr);
        aclrtSynchronizeStream(stream);
    }
}

// Task-book semantics: after warmup, the operator is executed sampleCount times
// back-to-back and the interval is divided by the number of valid samples.
// msprof op reports pure on-device task duration, which excludes host-side
// launch latency; a per-sample record+sync would fold that latency into
// every short kernel and systematically inflate sub-50us cases.
double TimeBatchedLaunches(
    aclblasHandle_t handle, aclrtStream stream, aclblasOperation_t trans, int m, int n, const aclblasComplex* alpha,
    int lda, const aclblasComplex* beta, int incx, int incy, int batchCount, const DeviceBuffer* dAPtrArr,
    const DeviceBuffer* dXPtrArr, const DeviceBuffer* dYPtrArr, int sampleCount, int& validSamples)
{
    aclrtEvent startEvent = nullptr;
    aclrtEvent endEvent = nullptr;
    if (aclrtCreateEvent(&startEvent) != ACL_SUCCESS || aclrtCreateEvent(&endEvent) != ACL_SUCCESS) {
        return -1.0;
    }
    if (aclrtRecordEvent(startEvent, stream) != ACL_SUCCESS) {
        aclrtDestroyEvent(startEvent);
        aclrtDestroyEvent(endEvent);
        return -1.0;
    }

    validSamples = 0;
    for (int i = 0; i < sampleCount; ++i) {
        aclblasStatus_t ret =
            LaunchBatched(handle, trans, m, n, alpha, lda, beta, incx, incy, batchCount, dAPtrArr, dXPtrArr, dYPtrArr);
        if (ret != ACLBLAS_STATUS_SUCCESS)
            break;
        ++validSamples;
    }

    float batchMs = 0.0f;
    bool timed = (validSamples > 0) && (aclrtRecordEvent(endEvent, stream) == ACL_SUCCESS) &&
                 (aclrtSynchronizeEvent(endEvent) == ACL_SUCCESS) &&
                 (aclrtEventElapsedTime(&batchMs, startEvent, endEvent) == ACL_SUCCESS);
    aclrtDestroyEvent(startEvent);
    aclrtDestroyEvent(endEvent);
    return timed ? (static_cast<double>(batchMs) * 1000.0 / static_cast<double>(validSamples)) : -1.0;
}

PerfResult MeasureCgemvBatchedPerf(
    aclblasHandle_t handle, aclrtStream stream, aclblasOperation_t trans, int m, int n, const aclblasComplex* alpha,
    const aclblasComplex* a, int lda, const aclblasComplex* x, int incx, const aclblasComplex* beta, aclblasComplex* y,
    int incy, int batchCount, int warmupCount, int sampleCount)
{
    PerfResult result;
    BatchedDeviceInputs in = UploadBatchedInputs(a, x, y, m, n, lda, incx, incy, trans, batchCount);

    RunWarmup(
        handle, stream, trans, m, n, alpha, lda, beta, incx, incy, batchCount, in.aPtrArr.get(), in.xPtrArr.get(),
        in.yPtrArr.get(), warmupCount);

    int validSamples = 0;
    double avgUs = TimeBatchedLaunches(
        handle, stream, trans, m, n, alpha, lda, beta, incx, incy, batchCount, in.aPtrArr.get(), in.xPtrArr.get(),
        in.yPtrArr.get(), sampleCount, validSamples);
    if (avgUs < 0.0)
        return result;

    result.avgUs = avgUs;
    result.minUs = avgUs;
    result.maxUs = avgUs;
    result.valid = true;
    return result;
}

// Global accumulator for performance summary.
struct PerfRecord {
    std::string caseName;
    double npuAvgUs;
    double npuMinUs;
    double npuMaxUs;
    double gpuUs;
    double speedup;
    bool pass;
};
std::vector<PerfRecord> g_perfRecords;
const int PERF_WARMUP = 5;
const int PERF_SAMPLES = 10;

struct PerfSummaryPrinter {
    ~PerfSummaryPrinter()
    {
        if (g_perfRecords.empty()) {
            return;
        }
        int pass = 0;
        int slower = 0;
        double totalSpeedup = 0.0;
        for (const auto& r : g_perfRecords) {
            if (r.pass) {
                pass++;
            } else {
                slower++;
            }
            totalSpeedup += r.speedup;
        }
        std::cout << "\n[perfreg] ==================== SUMMARY ====================" << std::endl;
        std::cout << "[perfreg] total=" << g_perfRecords.size() << " PASS=" << pass << " SLOWER=" << slower
                  << std::endl;
        std::cout << "[perfreg] mean speedup(gpu/npu)=" << std::fixed << std::setprecision(3)
                  << totalSpeedup / g_perfRecords.size() << std::endl;
        std::cout << "[perfreg] ====================================================\n" << std::endl;
    }
};
PerfSummaryPrinter g_perfSummaryPrinter;

// Extract the numeric suffix of "TC_PF_<n>" without sscanf (unsafe function).
// Returns -1 when the name does not follow the expected shape.
int ParsePerfCaseIndex(const std::string& caseName)
{
    static const std::string kPrefix = "TC_PF_";
    if (caseName.compare(0, kPrefix.size(), kPrefix) != 0) {
        return -1;
    }
    int idx = -1;
    try {
        size_t used = 0;
        idx = std::stoi(caseName.substr(kPrefix.size()), &used);
        // Reject trailing garbage such as "TC_PF_12x".
        if (used != caseName.size() - kPrefix.size()) {
            return -1;
        }
    } catch (const std::exception&) {
        return -1;
    }
    return idx;
}

// Record the measurement and evaluate the 0.4 speedup gate.
void ReportPerfCase(const CgemvBatchedParam& p, const PerfResult& pr, const std::vector<double>& gpuBaseline)
{
    int idx = ParsePerfCaseIndex(p.caseName);
    if (idx > 0) {
        idx -= 1001;
    }
    double gpuUs = 0.0;
    if (idx >= 0 && idx < static_cast<int>(gpuBaseline.size())) {
        gpuUs = gpuBaseline[idx];
    }
    double speedup = (gpuUs > 0.0) ? (gpuUs / pr.avgUs) : 0.0;
    bool pass = (speedup >= 0.4);

    g_perfRecords.push_back({p.caseName, pr.avgUs, pr.minUs, pr.maxUs, gpuUs, speedup, pass});
    std::cout << "[perfreg] " << p.caseName << " npu_avg=" << std::fixed << std::setprecision(1) << pr.avgUs << "us"
              << " min=" << pr.minUs << " max=" << pr.maxUs << " gpu=" << gpuUs << "us"
              << " speedup=" << std::setprecision(3) << speedup << (pass ? " PASS" : " SLOWER") << std::endl;
    EXPECT_GE(speedup, 0.4) << "Performance gate failed: " << p.caseName << " npu_avg=" << pr.avgUs
                            << "us gpu=" << gpuUs << "us speedup=" << speedup;
}

// Run the timed variant of a performance case (TC_PF_*).
void RunPerfCase(
    aclblasHandle_t handle, aclrtStream stream, const CgemvBatchedParam& p, const aclblasComplex& alpha,
    const aclblasComplex& beta, const std::vector<aclblasComplex>& aHost, const std::vector<aclblasComplex>& xHost,
    std::vector<aclblasComplex>& yHost)
{
    static const auto gpuBaseline = LoadGpuBaselineUs();
    PerfResult pr = MeasureCgemvBatchedPerf(
        handle, stream, p.trans, p.m, p.n, &alpha, aHost.data(), p.lda, xHost.data(), p.incx, &beta, yHost.data(),
        p.incy, p.batchCount, PERF_WARMUP, PERF_SAMPLES);
    ASSERT_TRUE(pr.valid) << "Performance measurement failed for " << p.caseName;
    ReportPerfCase(p, pr, gpuBaseline);
}

// Run an error-path case: only the returned status code matters.
// yPtr is nullptr when the case models a null y array so the host-side
// validation path is exercised.
void RunErrorCase(
    aclblasHandle_t handle, aclrtStream stream, const CgemvBatchedParam& p, const aclblasComplex* alphaPtr,
    const aclblasComplex* betaPtr, const aclblasComplex* aPtr, const aclblasComplex* xPtr, aclblasComplex* yPtr)
{
    aclblasStatus_t ret = aclblasCgemvBatched_npu(
        handle, stream, p.trans, p.m, p.n, alphaPtr, aPtr, p.lda, xPtr, p.incx, betaPtr, yPtr, p.incy, p.batchCount);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
}

// De-stride a strided complex buffer into separate real/imag float vectors so
// the mixed-tolerance verifier can run on each part independently.
void SplitComplexForVerify(
    const std::vector<aclblasComplex>& src, int incy, int yCount, size_t yStride, int batchCount,
    std::vector<float>& outReal, std::vector<float>& outImag)
{
    const int absIncy = std::abs(incy);
    const size_t totalOut = static_cast<size_t>(batchCount) * yCount;
    outReal.resize(totalOut);
    outImag.resize(totalOut);
    for (int b = 0; b < batchCount; ++b) {
        for (int i = 0; i < yCount; ++i) {
            int yIdx = (incy > 0) ? (i * incy) : ((yCount - 1 - i) * absIncy);
            size_t idx = static_cast<size_t>(b) * yCount + i;
            outReal[idx] = src[static_cast<size_t>(b) * yStride + yIdx].real;
            outImag[idx] = src[static_cast<size_t>(b) * yStride + yIdx].imag;
        }
    }
}

// Run a normal case: NPU vs CPU golden with FP32 mixed tolerance.
void RunNormalCase(
    aclblasHandle_t handle, aclrtStream stream, const CgemvBatchedParam& p, const aclblasComplex* alphaPtr,
    const aclblasComplex* betaPtr, const std::vector<aclblasComplex>& aHost, const std::vector<aclblasComplex>& xHost,
    const std::vector<aclblasComplex>& yHost, int yCount, size_t yStride)
{
    std::vector<aclblasComplex> npuY = yHost;
    aclblasStatus_t ret = aclblasCgemvBatched_npu(
        handle, stream, p.trans, p.m, p.n, alphaPtr, aHost.data(), p.lda, xHost.data(), p.incx, betaPtr, npuY.data(),
        p.incy, p.batchCount);
    ASSERT_EQ(ret, ACLBLAS_STATUS_SUCCESS);

    std::vector<aclblasComplex> goldenY = yHost;
    aclblasCgemvBatched_cpu(
        handle, p.trans, p.m, p.n, alphaPtr, aHost.data(), p.lda, xHost.data(), p.incx, betaPtr, goldenY.data(), p.incy,
        p.batchCount);

    if (yCount == 0 || p.batchCount == 0)
        return;

    std::vector<float> npuReal;
    std::vector<float> npuImag;
    std::vector<float> goldenReal;
    std::vector<float> goldenImag;
    SplitComplexForVerify(npuY, p.incy, yCount, yStride, p.batchCount, npuReal, npuImag);
    SplitComplexForVerify(goldenY, p.incy, yCount, yStride, p.batchCount, goldenReal, goldenImag);

    const size_t totalOut = npuReal.size();
    VerifyConfig cfg;
    applyMixedTolerance(cfg, ACL_FLOAT, goldenReal.data(), goldenReal.size());
    EXPECT_TRUE(Verifier::verifyVector(npuReal.data(), goldenReal.data(), totalOut, 1, cfg, p.caseName + "_real"));
    EXPECT_TRUE(Verifier::verifyVector(npuImag.data(), goldenImag.data(), totalOut, 1, cfg, p.caseName + "_imag"));
}

// Generate a complex array from a fill mode: real and imag parts are sampled
// independently from the same distribution.
std::vector<aclblasComplex> makeComplexArray(int64_t size, const BlasFillMode& fill, uint32_t seed)
{
    if (fill.method == BlasFillMode::M_NULLPTR || size <= 0) {
        return {};
    }
    auto realPart = makeBlasArray(size, fill, seed);
    auto imagPart = makeBlasArray(size, fill, seed + 1000000u);
    std::vector<aclblasComplex> out(static_cast<size_t>(size));
    for (size_t i = 0; i < static_cast<size_t>(size); ++i) {
        out[i] = aclblasComplex{realPart[i], imagPart[i]};
    }
    return out;
}

// Generate a strided complex vector (supports negative stride).
std::vector<aclblasComplex> makeComplexStrided(int count, int inc, const BlasFillMode& fill, uint32_t seed)
{
    int absInc = (inc == INT_MIN) ? 1 : std::abs(inc);
    size_t stride = static_cast<size_t>((count - 1) * absInc + 1);
    auto realPart = makeBlasArray(static_cast<int64_t>(stride), fill, seed);
    auto imagPart = makeBlasArray(static_cast<int64_t>(stride), fill, seed + 1000000u);
    std::vector<aclblasComplex> out(stride);
    for (size_t i = 0; i < stride; ++i) {
        out[i] = aclblasComplex{realPart[i], imagPart[i]};
    }
    return out;
}

// Allocate the x/y host buffers for every batch according to the fill mode.
void FillBatchVectors(
    const CgemvBatchedParam& p, int xCount, int yCount, int xStride, int yStride, std::vector<aclblasComplex>& xHost,
    std::vector<aclblasComplex>& yHost)
{
    for (int b = 0; b < p.batchCount; ++b) {
        if (!p.xIsNull) {
            auto xBatch = makeComplexStrided(xCount, p.incx, p.xFill, p.randomSeed + 1 + b);
            for (size_t i = 0; i < static_cast<size_t>(xStride); ++i)
                xHost[b * xStride + i] = xBatch[i];
        }
        if (!p.yIsNull) {
            auto yBatch = makeComplexStrided(yCount, p.incy, p.yFill, p.randomSeed + 2 + b);
            for (size_t i = 0; i < static_cast<size_t>(yStride); ++i)
                yHost[b * yStride + i] = yBatch[i];
        }
    }
}

} // namespace cgemv_batched_test

// Load-time composition guard for the CSV case set. A truncated or edited CSV
// would otherwise silently shrink the precision / performance coverage while CI
// still reports green, so the expected case counts and name uniqueness are
// asserted here, at instantiation time.
std::vector<CgemvBatchedParam> LoadCgemvBatchedCasesWithAssert(const std::string& csvPath)
{
    std::vector<CgemvBatchedParam> cases = GetCasesFromCsv<CgemvBatchedParam>(csvPath);

    const size_t expectedTotal = 1212; // 1012 precision/negative + 200 TC_PF
    const size_t expectedPerf = 200;
    const size_t expectedNonPerf = expectedTotal - expectedPerf;

    size_t perfCount = 0;
    size_t nonPerfCount = 0;
    std::map<std::string, int> nameCount;
    for (const auto& c : cases) {
        if (c.caseName.compare(0, 5, "TC_PF") == 0) {
            perfCount++;
        } else {
            nonPerfCount++;
        }
        nameCount[c.caseName]++;
    }

    std::vector<std::string> duplicated;
    for (const auto& kv : nameCount) {
        if (kv.second > 1) {
            duplicated.push_back(kv.first);
        }
    }
    std::sort(duplicated.begin(), duplicated.end());

    EXPECT_EQ(cases.size(), expectedTotal) << "CSV case count mismatch: " << csvPath;
    EXPECT_EQ(perfCount, expectedPerf) << "TC_PF case count mismatch: " << csvPath;
    EXPECT_EQ(nonPerfCount, expectedNonPerf) << "non-TC_PF case count mismatch: " << csvPath;
    if (!duplicated.empty()) {
        std::ostringstream oss;
        for (size_t i = 0; i < duplicated.size(); ++i) {
            if (i != 0) {
                oss << ", ";
            }
            oss << duplicated[i];
        }
        ADD_FAILURE() << "duplicate case names in " << csvPath << ": " << oss.str();
    }

    return cases;
}

TEST_F(CgemvBatchedTest, NullHandle)
{
    const aclblasComplex alpha{1.0f, 0.0f};
    const aclblasComplex beta{0.0f, 0.0f};
    const std::vector<aclblasComplex> af(64, {1.0f, 0.0f});
    const std::vector<aclblasComplex> xf(8, {1.0f, 0.0f});
    std::vector<aclblasComplex> yf(8, {0.0f, 0.0f});
    aclblasStatus_t ret = aclblasCgemvBatched_npu(
        nullptr, nullptr, ACLBLAS_OP_N, 8, 8, &alpha, af.data(), 8, xf.data(), 1, &beta, yf.data(), 1, 1);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_HANDLE_IS_NULLPTR));
}

// issue #848: 超限形状在 host 跨度校验阶段拒绝（INVALID_VALUE，不进入
// kernel）。直接调用公开 API（RunErrorCase 路径的缓冲随维度分配，超限形状
// 无法经 CSV 承载）；校验先于任何指针解引用与 kernel 下发，用小哑指针即可。
TEST_F(CgemvBatchedTest, OversizeShapeSpanRejected)
{
    const aclblasComplex alpha{1.0f, 0.0f};
    const aclblasComplex beta{0.0f, 0.0f};
    aclblasComplex dummy{};
    const aclblasComplex* const arrays[1] = {&dummy};
    auto* yArrays = const_cast<aclblasComplex* const*>(arrays);

    // A 元素跨度 (n-1)*lda+m ≈ 4.3e9（×2 float 寻址后超 INT32_MAX）
    aclblasStatus_t ret = aclblasCgemvBatched(
        handle_, ACLBLAS_OP_T, 65536, 65536, &alpha, arrays, 65536, arrays, 2, &beta, yArrays, 2, 1);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE));

    // x 跨度 1+(dotDim-1)*|incx| 超界（trans=N 时 dotDim=n）
    ret = aclblasCgemvBatched(handle_, ACLBLAS_OP_N, 4, 4, &alpha, arrays, 4, arrays, 800000000, &beta, yArrays, 1, 2);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE));

    // y 跨度 1+(outDim-1)*|incy| 超界（trans=N 时 outDim=m）
    ret = aclblasCgemvBatched(handle_, ACLBLAS_OP_N, 4, 4, &alpha, arrays, 4, arrays, 1, &beta, yArrays, 800000000, 2);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE));

    // 总输出 batchCount*outSize 超 INT32_MAX
    ret = aclblasCgemvBatched(handle_, ACLBLAS_OP_N, 4, 4, &alpha, arrays, 4, arrays, 1, &beta, yArrays, 1, 600000000);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(ACLBLAS_STATUS_INVALID_VALUE));
}

INSTANTIATE_TEST_SUITE_P(
    CgemvBatched, CgemvBatchedTest,
    ::testing::ValuesIn(LoadCgemvBatchedCasesWithAssert(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<CgemvBatchedParam>);

TEST_P(CgemvBatchedTest, CsvDriven)
{
    using namespace cgemv_batched_test;

    const auto& p = GetParam();

    // Use safe dimensions for data generation (negative dims would overflow).
    int safeM = std::max(1, std::abs(p.m));
    int safeN = std::max(1, std::abs(p.n));
    int safeBatchCount = std::max(1, p.batchCount);
    int safeLda = std::max(safeM, p.lda);

    const bool isTransN = (p.trans == ACLBLAS_OP_N);
    const int xCount = isTransN ? safeN : safeM;
    const int yCount = isTransN ? safeM : safeN;
    // Safe abs: std::abs(INT_MIN) overflows; clamp to 1 for negative-test allocation.
    auto safeAbsInc = [](int inc) -> int { return (inc == INT_MIN) ? 1 : std::abs(inc); };
    const size_t xStride = static_cast<size_t>((xCount - 1) * safeAbsInc(p.incx) + 1);
    const size_t yStride = static_cast<size_t>((yCount - 1) * safeAbsInc(p.incy) + 1);

    // Generate complex data using safe dimensions.
    auto aHost = makeComplexArray(static_cast<int64_t>(safeBatchCount) * safeLda * safeN, p.aFill, p.randomSeed);

    std::vector<aclblasComplex> xHost(safeBatchCount * xStride);
    std::vector<aclblasComplex> yHost(safeBatchCount * yStride);
    FillBatchVectors(p, xCount, yCount, static_cast<int>(xStride), static_cast<int>(yStride), xHost, yHost);

    aclblasComplex alpha{p.alphaReal, p.alphaImag};
    aclblasComplex beta{p.betaReal, p.betaImag};
    const aclblasComplex* alphaPtr = p.alphaIsNull ? nullptr : &alpha;
    const aclblasComplex* betaPtr = p.betaIsNull ? nullptr : &beta;
    const aclblasComplex* aPtr = p.aIsNull ? nullptr : aHost.data();
    const aclblasComplex* xPtr = p.xIsNull ? nullptr : xHost.data();
    aclblasComplex* yErrPtr = p.yIsNull ? nullptr : yHost.data();

    // Performance test cases: 5 warmup + 10 samples with aclrtEvent timing.
    if (p.caseName.compare(0, 5, "TC_PF") == 0 && p.expectResult == ACLBLAS_STATUS_SUCCESS) {
        RunPerfCase(handle_, stream_, p, alpha, beta, aHost, xHost, yHost);
        return;
    }

    // Error / early-return cases: only check the status code.
    if (p.expectResult != ACLBLAS_STATUS_SUCCESS) {
        RunErrorCase(handle_, stream_, p, alphaPtr, betaPtr, aPtr, xPtr, yErrPtr);
        return;
    }

    // Normal case: run NPU and CPU golden, then compare.
    RunNormalCase(handle_, stream_, p, alphaPtr, betaPtr, aHost, xHost, yHost, yCount, yStride);
}
