/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <chrono>
#include <cstdint>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

#include "csv_loader.h"
#include "triangular_fixture.h"

namespace ctpttr_arch35_test {

using namespace complex_triangular_test;

namespace {

constexpr int BASELINE_WARMUP_ITERATIONS = 10;
constexpr int BASELINE_TIMED_ITERATIONS = 100;
constexpr double BASELINE_THRESHOLD_FACTOR = 2.5;

// One row of gpu_baseline.csv: the raw H100 single-call time for one shape,
// keyed by id (ctpttr-base-000..003 task-book shapes, ctpttr-perf-004..199).
struct CtpttrPerfBaseline {
    explicit CtpttrPerfBaseline(const csv_map& row)
        : id(ReadMap(row, "id")),
          n(std::stoi(ReadMap(row, "n"))),
          lda(std::stoi(ReadMap(row, "lda"))),
          uploText(ReadMap(row, "uplo")),
          gpuMs(std::stod(ReadMap(row, "gpu_ms")))
    {}

    aclblasFillMode_t Uplo() const { return uploText == "ACLBLAS_UPPER" ? ACLBLAS_UPPER : ACLBLAS_LOWER; }

    std::string id;
    int n = 0;
    int lda = 0;
    std::string uploText;
    double gpuMs = 0.0;
};

std::string GpuBaselinePath()
{
    const std::string source = __FILE__;
    return source.substr(0, source.find_last_of("/\\") + 1U) + "gpu_baseline.csv";
}

// Warms up, then times timedIterations calls between two stream synchronisations
// and reports the total microseconds across them (same host wall-clock style as
// the ctbmv arch35 CSV performance path). The caller divides by the compile-time
// iteration count, allocation and device copies stay outside the interval.
template <typename LaunchFn>
bool MeasureTotalUs(
    aclrtStream stream, int warmupIterations, int timedIterations, const LaunchFn& launch, double& totalUs)
{
    for (int i = 0; i < warmupIterations; ++i) {
        if (launch(i) != ACLBLAS_STATUS_SUCCESS) {
            return false;
        }
    }
    if (aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
        return false;
    }
    const auto begin = std::chrono::steady_clock::now();
    for (int i = 0; i < timedIterations; ++i) {
        if (launch(warmupIterations + i) != ACLBLAS_STATUS_SUCCESS) {
            return false;
        }
    }
    if (aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
        return false;
    }
    const auto end = std::chrono::steady_clock::now();
    totalUs = std::chrono::duration<double, std::micro>(end - begin).count();
    return true;
}

} // namespace

class CtpttrTest : public TriangularFixture {};

// Preserve the suite name used by profiling and existing GoogleTest filters.
using CtpttrArch35Test = CtpttrTest;

inline std::vector<TriangularCase> Cases()
{
    const std::string source = __FILE__;
    return LoadCases(source.substr(0, source.rfind('.')) + ".csv");
}

INSTANTIATE_TEST_SUITE_P(
    CtpttrArch35, CtpttrArch35Test, ::testing::ValuesIn(Cases()),
    [](const ::testing::TestParamInfo<TriangularCase>& info) { return info.param.caseName; });

TEST_P(CtpttrArch35Test, CsvDriven) { this->Run(GetParam()); }

TEST_F(CtpttrArch35Test, NullHandlePriority)
{
    EXPECT_EQ(
        Invoke(nullptr, static_cast<aclblasFillMode_t>(255), -1, nullptr, nullptr, 0),
        ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
}

TEST_F(CtpttrArch35Test, EmptyValidationOrder)
{
    EXPECT_EQ(Invoke(this->handle_, ACLBLAS_UPPER, 0, nullptr, nullptr, 0), ACLBLAS_STATUS_INVALID_VALUE);
    EXPECT_EQ(Invoke(this->handle_, ACLBLAS_LOWER, 0, nullptr, nullptr, 1), ACLBLAS_STATUS_SUCCESS);
    EXPECT_EQ(
        Invoke(this->handle_, static_cast<aclblasFillMode_t>(255), 0, nullptr, nullptr, 1),
        ACLBLAS_STATUS_INVALID_VALUE);
}

TEST_F(CtpttrArch35Test, PayloadsAndPackedOffsets) { this->RunPayloads(); }

TEST_F(CtpttrArch35Test, OutputInitialValues)
{
    for (const char* output : {"VALUE_NORM_0", "INDEX", "RANDOM_EXTREME"}) {
        for (const auto uplo : {ACLBLAS_UPPER, ACLBLAS_LOWER}) {
            auto testCase = ManualCase(5, 11, uplo);
            testCase.aFill = output;
            this->Run(testCase);
        }
    }
}

TEST_F(CtpttrArch35Test, MissingCsvColumnsRejected)
{
    const CaseRow emptyRow;
    EXPECT_THROW(TriangularCase{emptyRow}, std::runtime_error);
}

TEST_F(CtpttrArch35Test, EmptyPreservesBuffers) { this->RunNoOpProtection(); }

TEST_F(CtpttrArch35Test, OverlappingPointersAreCallerError) { this->RunOverlap(); }

TEST_F(CtpttrArch35Test, ComparatorRejectsCorruption) { this->RunComparatorChecks(); }

// Threshold assertion against gpu_baseline.csv: the baseline file previously had
// no consumer, so the performance gate was not enforced anywhere. Aligned with
// the ctbmv arch35 CSV performance path: wall-clock total over timed calls and
// a 2.5x budget over the H100 reference, asserted on the ctpttr-base-* task-book
// shapes only; small-n rows are dominated by per-call host launch overhead and
// keep truthful diagnostics until budgets are validated per size class on the
// target device.
TEST_F(CtpttrArch35Test, PerformanceBaselineCases)
{
    if (std::getenv("TRIANGULAR_VERIFY_PERF") == nullptr) {
        GTEST_SKIP() << "set TRIANGULAR_VERIFY_PERF=1 to run the gpu_baseline threshold cases";
    }
    std::vector<CtpttrPerfBaseline> baselines;
    try {
        baselines = GetCasesFromCsv<CtpttrPerfBaseline>(GpuBaselinePath());
    } catch (const std::exception& error) {
        FAIL() << "cannot load gpu_baseline.csv: " << error.what();
    }
    ASSERT_EQ(baselines.size(), 200U);

    for (const auto& baseline : baselines) {
        SCOPED_TRACE(baseline.id);
        const uint64_t packed = static_cast<uint64_t>(baseline.n) * (baseline.n + 1U) / 2U;
        const uint64_t matrix = static_cast<uint64_t>(baseline.n) * static_cast<uint64_t>(baseline.lda);
        DeviceBuffer inputDevice((packed + GUARD_ELEMENTS) * sizeof(ComplexBits));
        DeviceBuffer outputDevice((matrix + GUARD_ELEMENTS) * sizeof(ComplexBits));
        const auto launch = [&](int) {
            return Invoke(
                this->handle_, baseline.Uplo(), baseline.n, inputDevice.ptr(), outputDevice.ptr(), baseline.lda);
        };
        double totalUs = 0.0;
        ASSERT_TRUE(
            MeasureTotalUs(this->stream_, BASELINE_WARMUP_ITERATIONS, BASELINE_TIMED_ITERATIONS, launch, totalUs));
        const double averageUs = totalUs / BASELINE_TIMED_ITERATIONS;
        const double gpuUs = baseline.gpuMs * 1000.0;
        const bool perfPass = averageUs <= gpuUs * BASELINE_THRESHOLD_FACTOR;
        std::cout << "[PERF_CSV][ctpttr] id=" << baseline.id << " n=" << baseline.n << " uplo=" << baseline.uploText
                  << " lda=" << baseline.lda << " gpu_ms=" << baseline.gpuMs << " gpu_us=" << gpuUs
                  << " average_us=" << averageUs << " execution=" << (perfPass ? "PASS" : "FAIL") << std::endl;
        if (baseline.id.rfind("ctpttr-base-", 0) == 0) {
            EXPECT_LE(averageUs, gpuUs * BASELINE_THRESHOLD_FACTOR);
        }
    }
}

} // namespace ctpttr_arch35_test
