/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Kernel-level performance benchmark for aclblasCsrot (arch35).
// Device buffers are allocated and initialized ONCE per case; the timed region
// covers only the 100 operator dispatches (device events on the bound stream),
// so host-side allocation / H2D copy / result handling is excluded — this is the
// direct implementation of the task book's "warmup >= 5, valid samples > 50,
// average" requirement and is comparable with the pure-kernel GPU baseline.
// The three typical shapes of task book §3.3 carry hard threshold assertions
// (GPU baseline / 0.4).

#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "blas_test.h"
#include "cann_ops_blas.h"
#include "csrot_param.h"
#include "csv_loader.h"

namespace {

constexpr int CSROT_PERF_WARMUP = 20;   // task book requires >= 5
constexpr int CSROT_PERF_REPEATS = 100; // task book requires > 50
constexpr size_t CSROT_ACCEPTANCE_CASES = 200;
constexpr float CSROT_BENCH_COS = 0.6f;
constexpr float CSROT_BENCH_SIN = 0.8f;

struct CsrotBenchmarkCase {
    std::string caseName;
    int n;
};

struct CsrotBenchmarkResources {
    void* x = nullptr;
    void* y = nullptr;

    ~CsrotBenchmarkResources()
    {
        if (x != nullptr) {
            aclrtFree(x);
        }
        if (y != nullptr) {
            aclrtFree(y);
        }
    }
};

struct CsrotBenchmarkEvents {
    aclrtEvent start = nullptr;
    aclrtEvent stop = nullptr;

    ~CsrotBenchmarkEvents()
    {
        if (start != nullptr) {
            aclrtDestroyEvent(start);
        }
        if (stop != nullptr) {
            aclrtDestroyEvent(stop);
        }
    }
};

template <typename Launch>
bool MeasureAverageUs(aclrtStream stream, Launch&& launch, float& averageUs)
{
    for (int i = 0; i < CSROT_PERF_WARMUP; ++i) {
        if (!launch()) {
            return false;
        }
    }
    if (aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
        return false;
    }

    CsrotBenchmarkEvents events;
    if (aclrtCreateEvent(&events.start) != ACL_SUCCESS || aclrtCreateEvent(&events.stop) != ACL_SUCCESS ||
        aclrtRecordEvent(events.start, stream) != ACL_SUCCESS) {
        return false;
    }
    for (int i = 0; i < CSROT_PERF_REPEATS; ++i) {
        if (!launch()) {
            return false;
        }
    }
    if (aclrtRecordEvent(events.stop, stream) != ACL_SUCCESS || aclrtSynchronizeEvent(events.stop) != ACL_SUCCESS) {
        return false;
    }

    float elapsedMs = 0.0F;
    if (aclrtEventElapsedTime(&elapsedMs, events.start, events.stop) != ACL_SUCCESS) {
        return false;
    }
    averageUs = elapsedMs * 1000.0F / static_cast<float>(CSROT_PERF_REPEATS);
    return averageUs > 0.0F;
}

std::vector<CsrotBenchmarkCase> LoadPerformanceCases()
{
    std::string csvPath(__FILE__);
    constexpr const char* sourceName = "csrot_benchmark.cpp";
    size_t sourcePos = csvPath.rfind(sourceName);
    if (sourcePos == std::string::npos) {
        throw std::runtime_error("Cannot derive csrot_test.csv from benchmark source path");
    }
    csvPath.replace(sourcePos, std::char_traits<char>::length(sourceName), "csrot_test.csv");

    std::vector<CsrotBenchmarkCase> performanceCases;
    for (const auto& param : GetCasesFromCsv<CsrotParam>(csvPath)) {
        if (param.caseName.rfind("TC_PF_", 0) != 0) {
            continue;
        }
        if (param.n <= 0 || param.incx != 1 || param.incy != 1) {
            throw std::runtime_error("Performance cases must use positive n and unit strides: " + param.caseName);
        }
        performanceCases.push_back({param.caseName, param.n});
    }
    return performanceCases;
}

bool MatchesCaseFilter(const std::string& caseName)
{
    const char* value = std::getenv("CSROT_BENCHMARK_CASE");
    return value == nullptr || *value == '\0' || caseName == value;
}

// Task book §3.3 typical shapes: GPU baseline / 0.4 (acceptance ratio >= 0.4).
float AcceptanceThresholdUs(int n)
{
    if (n == 1048576) {
        return 21.69F;
    }
    if (n == 2097152) {
        return 43.19F;
    }
    if (n == 4194304) {
        return 119.38F;
    }
    return 0.0F;
}

void PrintPerformance(const std::string& caseName, int n, size_t vectorBytes, float averageUs, float thresholdUs)
{
    std::cout << "[CSROT_PERF] case=" << caseName << " n=" << n << " vector_bytes=" << vectorBytes
              << " traffic_bytes=" << 4 * vectorBytes << " warmup=" << CSROT_PERF_WARMUP
              << " repeats=" << CSROT_PERF_REPEATS << " avg_us=" << std::fixed << std::setprecision(3) << averageUs;
    if (thresholdUs > 0.0F) {
        std::cout << " threshold_us=" << thresholdUs;
    }
    std::cout << std::endl;
}

} // namespace

class CsrotBenchmark : public BlasTest<CsrotParam> {};

TEST_F(CsrotBenchmark, AcceptanceContinuousShapes)
{
    const auto performanceCases = LoadPerformanceCases();
    ASSERT_EQ(performanceCases.size(), CSROT_ACCEPTANCE_CASES);
    size_t selectedCases = 0;
    for (const auto& benchmarkCase : performanceCases) {
        if (!MatchesCaseFilter(benchmarkCase.caseName)) {
            continue;
        }
        ++selectedCases;
        int n = benchmarkCase.n;
        CsrotBenchmarkResources resources;
        size_t bytes = static_cast<size_t>(n) * sizeof(aclblasComplex);
        ASSERT_EQ(aclrtMalloc(&resources.x, bytes, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
        ASSERT_EQ(aclrtMalloc(&resources.y, bytes, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
        ASSERT_EQ(aclrtMemset(resources.x, bytes, 0x5a, bytes), ACL_SUCCESS);
        ASSERT_EQ(aclrtMemset(resources.y, bytes, 0xa5, bytes), ACL_SUCCESS);

        float averageUs = 0.0F;
        ASSERT_TRUE(MeasureAverageUs(
            CsrotBenchmark::stream_,
            [&]() {
                return aclblasCsrot(
                           CsrotBenchmark::handle_, n, static_cast<aclblasComplex*>(resources.x), 1,
                           static_cast<aclblasComplex*>(resources.y), 1, &CSROT_BENCH_COS,
                           &CSROT_BENCH_SIN) == ACLBLAS_STATUS_SUCCESS;
            },
            averageUs));
        float thresholdUs = AcceptanceThresholdUs(n);
        PrintPerformance(benchmarkCase.caseName, n, bytes, averageUs, thresholdUs);
        if (thresholdUs > 0.0F) {
            EXPECT_LE(averageUs, thresholdUs) << benchmarkCase.caseName;
        }
    }
    ASSERT_GT(selectedCases, 0U) << "CSROT_BENCHMARK_CASE did not match a performance case";
}
