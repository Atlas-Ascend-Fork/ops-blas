/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cblas.h>
#include <atomic>
#include <thread>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <limits>
#include <memory>
#include <random>
#include "blas_test.h"
#include "aclblas_handle_internal.h"
#include "ctrsmbatched_param.h"

namespace {
using Complex = aclblasComplex;
void CheckAcl(aclError status)
{
    if (status != ACL_SUCCESS)
        throw std::runtime_error("ACL error " + std::to_string(status));
}

struct DeviceBuffer {
    void* ptr = nullptr;
    size_t bytes;
    explicit DeviceBuffer(size_t n) : bytes(n) { CheckAcl(aclrtMalloc(&ptr, n, ACL_MEM_MALLOC_NORMAL_ONLY)); }
    ~DeviceBuffer()
    {
        if (ptr)
            aclrtFree(ptr);
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    void Put(const void* data) { CheckAcl(aclrtMemcpy(ptr, bytes, data, bytes, ACL_MEMCPY_HOST_TO_DEVICE)); }
    void Get(void* data) { CheckAcl(aclrtMemcpy(data, bytes, ptr, bytes, ACL_MEMCPY_DEVICE_TO_HOST)); }
};

struct Event {
    aclrtEvent event = nullptr;
    Event() { CheckAcl(aclrtCreateEventExWithFlag(&event, ACL_EVENT_TIME_LINE)); }
    ~Event()
    {
        if (event)
            aclrtDestroyEvent(event);
    }
};

std::vector<Complex> MakeComplex(size_t size, const std::string& fill, uint32_t seed)
{
    // Explicit supplementary distributions. Original attachment fill modes are unchanged.
    if (fill == "TASK_UNIFORM_5_5" || fill == "TASK_NORMAL") {
        std::mt19937 realEngine(seed), imagEngine(seed + 1);
        std::uniform_real_distribution<float> uniform(-5.0f, 5.0f), sigma(0.1f, 2.0f);
        std::vector<Complex> result(size);
        if (fill == "TASK_UNIFORM_5_5") {
            for (auto& value : result)
                value = {uniform(realEngine), uniform(imagEngine)};
        } else {
            const float realMean = uniform(realEngine), realSigma = sigma(realEngine);
            const float imagMean = uniform(imagEngine), imagSigma = sigma(imagEngine);
            std::normal_distribution<float> realNormal(realMean, realSigma), imagNormal(imagMean, imagSigma);
            for (auto& value : result)
                value = {realNormal(realEngine), imagNormal(imagEngine)};
            std::cout << "CTRSM_NORMAL_INPUT," << seed << ',' << realMean << ',' << realSigma << ',' << imagMean << ','
                      << imagSigma << '\n';
        }
        return result;
    }
    auto real = makeBlasArray(size, fill, seed);
    auto imag = makeBlasArray(size, fill, seed + 1);
    std::vector<Complex> result(size);
    for (size_t i = 0; i < size; ++i)
        result[i] = {real[i], imag[i]};
    return result;
}

struct Metrics {
    size_t count = 0, matched = 0, nonfinite = 0, goldenNonfinite = 0, actualNonfinite = 0;
    double maxError = 0;
    bool maxLimit = true, specialMatch = true;
    void Add(float actual, float golden)
    {
        ++count;
        if (!std::isfinite(golden) || !std::isfinite(actual)) {
            ++nonfinite;
            goldenNonfinite += !std::isfinite(golden);
            actualNonfinite += !std::isfinite(actual);
            const bool equal = (std::isnan(golden) && std::isnan(actual)) || actual == golden;
            specialMatch &= equal;
            matched += equal;
            return;
        }
        const double error = std::abs(static_cast<double>(actual) - golden);
        maxError = std::max(maxError, error);
        matched += error <= std::ldexp(1.0, -16) + std::ldexp(1.0, -10) * std::abs(golden);
        const float magnitude = std::abs(golden);
        float ulp = std::nextafter(magnitude, std::numeric_limits<float>::infinity()) - magnitude;
        if (!std::isfinite(ulp))
            ulp = magnitude - std::nextafter(magnitude, 0.0f);
        maxLimit &= error <= std::max(1e-2, 32.0 * ulp);
    }
    double Ratio() const { return count ? static_cast<double>(matched) / count : 1.0; }
};

std::vector<CtrsmBatchedParam> LoadCases()
{
    const char* csv = std::getenv("CTRSM_BATCHED_CSV");
    return GetCasesFromCsv<CtrsmBatchedParam>(csv ? csv : ReplaceFileExtension2Csv(__FILE__));
}

std::vector<csv_map> LoadGpuBaselines()
{
    const char* overridePath = std::getenv("CTRSM_BATCHED_GPU_BASELINE");
    const std::string source = __FILE__;
    const std::string path =
        overridePath ? overridePath : source.substr(0, source.find_last_of("/\\") + 1) + "gpu_baseline.csv";
    auto data = GetCasesFromCsv<csv_map>(path);
    if (data.size() != 200)
        throw std::runtime_error("GPU baseline must contain all 200 attachment rows");
    for (size_t i = 0; i < data.size(); ++i) {
        std::ostringstream id;
        id << "ctrsmBatched-" << (i < 5 ? "base-" : "perf-") << std::setw(3) << std::setfill('0') << i;
        if (ReadMap(data[i], "id") != id.str())
            throw std::runtime_error("GPU baseline ID/order mismatch at row " + std::to_string(i));
    }
    return data;
}

void CheckGpuMetadata(const csv_map& row, const CtrsmBatchedParam& p)
{
    if (std::stoi(row.at("m")) != p.m || std::stoi(row.at("n")) != p.n ||
        std::stoi(row.at("batchCount")) != p.batchCount ||
        row.at("side") != (p.side == ACLBLAS_SIDE_LEFT ? "LEFT" : "RIGHT") ||
        row.at("uplo") != (p.uplo == ACLBLAS_UPPER ? "UPPER" : "LOWER") ||
        row.at("trans") != (p.trans == ACLBLAS_OP_N ? "N" : (p.trans == ACLBLAS_OP_T ? "T" : "C")) ||
        row.at("diag") != (p.diag == ACLBLAS_UNIT ? "UNIT" : "NON_UNIT"))
        throw std::runtime_error("GPU baseline metadata mismatch for " + p.caseName);
}

double GpuBaselineUs(const CtrsmBatchedParam& p)
{
    static const auto rows = LoadGpuBaselines();
    const int index = std::stoi(p.caseName.substr(6)) - 1001;
    if (index < 0 || index >= static_cast<int>(rows.size()))
        throw std::runtime_error("No attachment GPU baseline for " + p.caseName);
    const auto& row = rows[index];
    CheckGpuMetadata(row, p);
    size_t used = 0;
    const double gpuUs = std::stod(row.at("gpu_ms"), &used) * 1000.0;
    if (used != row.at("gpu_ms").size() || !std::isfinite(gpuUs) || gpuUs <= 0)
        throw std::runtime_error("Invalid GPU baseline for " + p.caseName);
    return gpuUs;
}
aclblasStatus_t InvokeCase(
    const CtrsmBatchedParam& p, aclblasHandle_t handle, const Complex* const* a, Complex* const* b)
{
    return aclblasCtrsmBatched(
        p.description == "handle_null" ? nullptr : handle, p.side, p.uplo, p.trans, p.diag, p.m, p.n,
        p.nullAlpha ? nullptr : &p.alpha, p.aFill == "NULLPTR" ? nullptr : a, p.lda, p.bFill == "NULLPTR" ? nullptr : b,
        p.ldb, p.batchCount);
}

struct CaseData {
    const CtrsmBatchedParam& p;
    aclblasHandle_t handle;
    aclrtStream stream;
    bool zero, extremes, mixedUnit;
    int k;
    size_t aSize, bSize;
    Complex poison;
    std::vector<std::vector<Complex>> aHost, original, references;
    std::vector<std::unique_ptr<DeviceBuffer>> aDevice, bDevice;
    std::vector<const Complex*> aPtrs;
    std::vector<Complex*> bPtrs;
    DeviceBuffer ap, bp;
    CaseData(const CtrsmBatchedParam& param, aclblasHandle_t h, aclrtStream st)
        : p(param),
          handle(h),
          stream(st),
          zero(p.alpha.real == 0 && p.alpha.imag == 0),
          extremes(p.aFill == "DIAGONAL_EXTREMES"),
          mixedUnit(p.aFill == "MIXED_UNIT"),
          k(p.side == ACLBLAS_SIDE_LEFT ? p.m : p.n),
          aSize(static_cast<size_t>(p.lda) * k),
          bSize(static_cast<size_t>(p.ldb) * p.n),
          poison{std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN()},
          aHost(p.batchCount),
          original(p.batchCount),
          references(p.batchCount),
          aPtrs(p.batchCount, nullptr),
          bPtrs(p.batchCount),
          ap(sizeof(Complex*) * p.batchCount),
          bp(sizeof(Complex*) * p.batchCount)
    {}

    aclblasStatus_t Run()
    {
        return InvokeCase(p, handle, static_cast<const Complex* const*>(ap.ptr), static_cast<Complex* const*>(bp.ptr));
    }
    void PrepareA(int batch)
    {
        auto& a = aHost[batch];
        const bool bounded = p.aFill == "CONTRACT_BOUNDED", noAccess = p.aFill == "NOACCESS_UNIT";
        a = extremes ?
                std::vector<Complex>(aSize, {0, 0}) :
                MakeComplex(
                    aSize, mixedUnit || bounded || noAccess ? "RANDOM_NORM_5_5" : p.aFill, p.randomSeed + batch * 4);
        if (bounded || (mixedUnit && batch % 2 == 0))
            for (auto& value : a) {
                value.real /= 4 * k;
                value.imag /= 4 * k;
            }
        PoisonA(a);
        if (noAccess) {
            ASSERT_EQ(k, 1);
            ASSERT_EQ(p.diag, ACLBLAS_UNIT);
            aPtrs[batch] = reinterpret_cast<const Complex*>(uintptr_t{1});
            return;
        }
        aDevice.emplace_back(new DeviceBuffer(aSize * sizeof(Complex)));
        aDevice.back()->Put(a.data());
        aPtrs[batch] = static_cast<const Complex*>(aDevice.back()->ptr);
    }

    void PoisonA(std::vector<Complex>& a)
    {
        for (int col = 0; col < k; ++col) {
            for (int row = 0; row < p.lda; ++row) {
                const size_t off = row + static_cast<size_t>(col) * p.lda;
                if (row >= k || (p.uplo == ACLBLAS_UPPER ? row > col : row < col) ||
                    (p.diag == ACLBLAS_UNIT && row == col))
                    a[off] = poison;
                else if (row == col) {
                    if (extremes) {
                        const float scale = row % 2 == 0 ? 2e-38f : 2e38f;
                        a[off] = {scale, scale};
                    } else {
                        const float boost = std::max(5, k);
                        a[off].real += std::copysign(boost, a[off].real);
                        a[off].imag += std::copysign(boost, a[off].imag);
                    }
                }
            }
        }
    }

    void PrepareB(int batch)
    {
        original[batch] = MakeComplex(bSize, p.bFill, p.randomSeed + batch * 4 + 2);
        if (extremes) {
            ASSERT_EQ(p.diag, ACLBLAS_NON_UNIT);
            ASSERT_EQ(p.alpha.real, 1.0f);
            ASSERT_EQ(p.alpha.imag, 0.0f);
            for (int col = 0; col < p.n; ++col)
                for (int row = 0; row < p.m; ++row) {
                    const int index = p.side == ACLBLAS_SIDE_LEFT ? row : col;
                    original[batch][row + static_cast<size_t>(col) * p.ldb] = {index % 2 == 0 ? 2e-38f : 2e38f, 0};
                }
        }
        bDevice.emplace_back(new DeviceBuffer(bSize * sizeof(Complex)));
        bDevice.back()->Put(original[batch].data());
        bPtrs[batch] = static_cast<Complex*>(bDevice.back()->ptr);
    }

    void Prepare()
    {
        if (mixedUnit)
            ASSERT_EQ(p.diag, ACLBLAS_UNIT);
        for (int batch = 0; batch < p.batchCount; ++batch) {
            if (!zero)
                PrepareA(batch);
            PrepareB(batch);
            if (::testing::Test::HasFatalFailure())
                return;
        }
        ap.Put(aPtrs.data());
        bp.Put(bPtrs.data());
    }

    void Reference(int batch)
    {
        auto& golden = references[batch];
        golden = original[batch];
        if (zero || extremes) {
            for (int col = 0; col < p.n; ++col)
                for (int row = 0; row < p.m; ++row)
                    golden[row + static_cast<size_t>(col) * p.ldb] =
                        zero ? Complex{0, 0} : Complex{0.5f, p.trans == ACLBLAS_OP_C ? 0.5f : -0.5f};
        } else {
            cblas_ctrsm(
                CblasColMajor, p.side == ACLBLAS_SIDE_LEFT ? CblasLeft : CblasRight,
                p.uplo == ACLBLAS_UPPER ? CblasUpper : CblasLower,
                p.trans == ACLBLAS_OP_N ? CblasNoTrans :
                p.trans == ACLBLAS_OP_T ? CblasTrans :
                                          CblasConjTrans,
                p.diag == ACLBLAS_UNIT ? CblasUnit : CblasNonUnit, p.m, p.n, &p.alpha, aHost[batch].data(), p.lda,
                golden.data(), p.ldb);
        }
    }

    void BuildReferences()
    {
        if (p.batchCount > 1 && !zero && !extremes) {
            std::atomic<int> next{0};
            std::vector<std::thread> workers;
            for (int i = 0; i < std::min(8, p.batchCount); ++i)
                workers.emplace_back([&]() {
                    for (int batch = next.fetch_add(1); batch < p.batchCount; batch = next.fetch_add(1))
                        Reference(batch);
                });
            for (auto& worker : workers)
                worker.join();
        } else {
            for (int batch = 0; batch < p.batchCount; ++batch)
                Reference(batch);
        }
    }

    void MeasureComponents(
        int batch, const std::vector<Complex>& actual, const std::vector<Complex>& golden, Metrics& real, Metrics& imag)
    {
        for (int col = 0; col < p.n; ++col) {
            for (int row = 0; row < p.ldb; ++row) {
                const size_t off = row + static_cast<size_t>(col) * p.ldb;
                if (row < p.m) {
                    real.Add(actual[off].real, golden[off].real);
                    imag.Add(actual[off].imag, golden[off].imag);
                } else
                    EXPECT_EQ(std::memcmp(&actual[off], &original[batch][off], sizeof(Complex)), 0)
                        << "B padding changed";
            }
        }
    }

    void ReportAccuracy(int batch, const Metrics& real, const Metrics& imag)
    {
        EXPECT_GE(real.Ratio(), 0.99);
        EXPECT_GE(imag.Ratio(), 0.99);
        EXPECT_TRUE(real.maxLimit && imag.maxLimit && real.specialMatch && imag.specialMatch);
        // The task specifies comparison against Netlib, including special values.
        // Record reference overflow separately; it is not a finite-valued solve.
        // Unexpected actual overflow still fails specialMatch above.
        if (p.bFill != "VALUE_NORM_INF" && p.bFill != "VALUE_NORM_NAN" &&
            real.goldenNonfinite + imag.goldenNonfinite != 0)
            std::cout << "CTRSM_REFERENCE_OVERFLOW," << p.caseName << ',' << batch << ','
                      << real.goldenNonfinite + imag.goldenNonfinite << '\n';
        std::cout << "CTRSM_ACCURACY," << p.caseName << ',' << batch << ',' << std::setprecision(10) << real.Ratio()
                  << ',' << imag.Ratio() << ',' << real.maxError << ',' << imag.maxError << ',' << real.nonfinite << ','
                  << imag.nonfinite << ',' << real.goldenNonfinite << ',' << imag.goldenNonfinite << ','
                  << real.actualNonfinite << ',' << imag.actualNonfinite << '\n';
    }

    void DumpMismatches(int batch, const std::vector<Complex>& actual, const std::vector<Complex>& golden)
    {
        if (std::getenv("CTRSM_BATCHED_DUMP")) {
            size_t printed = 0;
            for (int col = 0; col < p.n && printed < 8; ++col) {
                for (int row = 0; row < p.m && printed < 8; ++row) {
                    const size_t off = row + static_cast<size_t>(col) * p.ldb;
                    Metrics rm, im;
                    rm.Add(actual[off].real, golden[off].real);
                    im.Add(actual[off].imag, golden[off].imag);
                    if (rm.Ratio() < 1 || im.Ratio() < 1 || !rm.maxLimit || !im.maxLimit) {
                        std::cout << "CTRSM_MISMATCH," << p.caseName << ',' << batch << ',' << row << ',' << col << ','
                                  << actual[off].real << ',' << actual[off].imag << ',' << golden[off].real << ','
                                  << golden[off].imag << '\n';
                        ++printed;
                    }
                }
            }
        }
    }

    void CheckBatch(int batch)
    {
        const auto& golden = references[batch];
        std::vector<Complex> actual(bSize);
        bDevice[batch]->Get(actual.data());
        if (!zero && p.aFill != "NOACCESS_UNIT") {
            std::vector<Complex> after(aSize);
            aDevice[batch]->Get(after.data());
            EXPECT_EQ(std::memcmp(after.data(), aHost[batch].data(), aSize * sizeof(Complex)), 0) << "A changed";
        }
        Metrics real, imag;
        MeasureComponents(batch, actual, golden, real, imag);
        ReportAccuracy(batch, real, imag);
        DumpMismatches(batch, actual, golden);
    }

    void CheckAccuracy()
    {
        for (int batch = 0; batch < p.batchCount; ++batch)
            CheckBatch(batch);
        std::vector<const Complex*> afterA(p.batchCount);
        std::vector<Complex*> afterB(p.batchCount);
        ap.Get(afterA.data());
        bp.Get(afterB.data());
        EXPECT_EQ(afterA, aPtrs) << "A pointer array changed";
        EXPECT_EQ(afterB, bPtrs) << "B must be overwritten at the original batch addresses";
    }

    void Benchmark()
    {
        Event start, stop;
        double totalUs = 0;
        // Sample-count overrides are for diagnostics; acceptance uses the default 60 samples.
        const char* sampleOverride = std::getenv("CTRSM_BATCHED_BENCH_SAMPLES");
        const int samples = sampleOverride ? std::stoi(sampleOverride) : 60;
        ASSERT_GE(samples, 1);
        ASSERT_LE(samples, 60);
        // Restoring B before every solve prevents repeated in-place solves from changing the workload.
        for (int sample = -5; sample < samples; ++sample) {
            for (int batch = 0; batch < p.batchCount; ++batch)
                bDevice[batch]->Put(original[batch].data());
            CheckAcl(aclrtSynchronizeStream(stream));
            CheckAcl(aclrtRecordEvent(start.event, stream));
            ASSERT_EQ(Run(), ACLBLAS_STATUS_SUCCESS);
            CheckAcl(aclrtRecordEvent(stop.event, stream));
            CheckAcl(aclrtSynchronizeStream(stream));
            float ms = 0;
            CheckAcl(aclrtEventElapsedTime(&ms, start.event, stop.event));
            if (sample >= 0)
                totalUs += ms * 1000.0;
        }
        const double averageUs = totalUs / samples;
        // Every attachment performance case is gated; missing baselines fail closed.
        const double gpuUs = GpuBaselineUs(p);
        EXPECT_TRUE(std::isfinite(averageUs) && averageUs > 0);
        EXPECT_GT(gpuUs, 0.4 * averageUs) << "GPU/NPU must be strictly greater than 0.4 for " << p.caseName;
        const size_t deviceBytes = (zero ? 0 : aSize * sizeof(Complex) * p.batchCount) +
                                   bSize * sizeof(Complex) * p.batchCount + 2 * sizeof(Complex*) * p.batchCount;
        std::cout << "CTRSM_PERF," << p.caseName << ',' << averageUs << ',' << samples << ',' << deviceBytes << ','
                  << GetEffectiveWorkspaceSize(handle) << '\n';
    }
};

} // namespace

class CtrsmBatchedArch35Test : public BlasTest<CtrsmBatchedParam> {
protected:
    std::unique_ptr<DeviceBuffer> suppliedWorkspace;
    void SetUp() override
    {
        // Optional independent run to exercise user-owned workspace and fallback.
        if (const char* bytes = std::getenv("CTRSM_BATCHED_WORKSPACE_BYTES")) {
            suppliedWorkspace.reset(new DeviceBuffer(std::stoull(bytes)));
            if (suppliedWorkspace->bytes == 16) {
                const uint32_t canary[4] = {0xa5a5a5a5, 0xa5a5a5a5, 0xa5a5a5a5, 0xa5a5a5a5};
                suppliedWorkspace->Put(canary);
            }
            ASSERT_EQ(
                aclblasSetWorkspace(handle_, suppliedWorkspace->ptr, suppliedWorkspace->bytes), ACLBLAS_STATUS_SUCCESS);
        }
    }
    void TearDown() override
    {
        if (suppliedWorkspace) {
            EXPECT_EQ(GetEffectiveWorkspace(handle_), suppliedWorkspace->ptr);
            EXPECT_EQ(GetEffectiveWorkspaceSize(handle_), suppliedWorkspace->bytes);
            EXPECT_EQ(aclrtSynchronizeStream(stream_), ACL_SUCCESS);
            if (suppliedWorkspace->bytes == 16) {
                uint32_t after[4]{};
                suppliedWorkspace->Get(after);
                for (const auto value : after)
                    EXPECT_EQ(value, 0xa5a5a5a5U) << "Insufficient user workspace must remain untouched";
            }
            // The public API restores library workspace when setting the stream.
            EXPECT_EQ(aclblasSetStream(handle_, stream_), ACLBLAS_STATUS_SUCCESS);
            suppliedWorkspace.reset();
        }
    }
};
INSTANTIATE_TEST_SUITE_P(
    CSV, CtrsmBatchedArch35Test, ::testing::ValuesIn(LoadCases()), PrintCaseInfoString<CtrsmBatchedParam>);

TEST_P(CtrsmBatchedArch35Test, Solve)
{
    const auto& p = GetParam();
    // Invalid metadata/no-op calls do not allocate or access device buffers.
    if (p.expectResult != ACLBLAS_STATUS_SUCCESS || p.m == 0 || p.n == 0) {
        Complex dummy{};
        const Complex* a[] = {&dummy};
        Complex* b[] = {&dummy};
        EXPECT_EQ(InvokeCase(p, handle_, a, b), p.expectResult);
        return;
    }
    CaseData data(p, handle_, stream_);
    data.Prepare();
    ASSERT_FALSE(HasFatalFailure());
    ASSERT_EQ(data.Run(), ACLBLAS_STATUS_SUCCESS);
    CheckAcl(aclrtSynchronizeStream(stream_));
    data.BuildReferences();
    data.CheckAccuracy();
    if (p.caseName.rfind("TC_PF_", 0) == 0 && !HasFailure())
        data.Benchmark();
}
