/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

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
#include "cswap_param.h"
#include "csv_loader.h"

namespace {

constexpr int CSWAP_PERF_WARMUP = 20;
constexpr int CSWAP_PERF_REPEATS = 100;
constexpr size_t CSWAP_ACCEPTANCE_CASES = 200;

struct CswapBenchmarkCase {
    std::string caseName;
    int n;
};

struct CswapBenchmarkResources {
    void* x = nullptr;
    void* y = nullptr;

    ~CswapBenchmarkResources()
    {
        if (x != nullptr) {
            aclrtFree(x);
        }
        if (y != nullptr) {
            aclrtFree(y);
        }
    }
};

struct CswapBenchmarkEvents {
    aclrtEvent start = nullptr;
    aclrtEvent stop = nullptr;

    ~CswapBenchmarkEvents()
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
    for (int i = 0; i < CSWAP_PERF_WARMUP; ++i) {
        if (!launch()) {
            return false;
        }
    }
    if (aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
        return false;
    }

    CswapBenchmarkEvents events;
    if (aclrtCreateEvent(&events.start) != ACL_SUCCESS || aclrtCreateEvent(&events.stop) != ACL_SUCCESS ||
        aclrtRecordEvent(events.start, stream) != ACL_SUCCESS) {
        return false;
    }
    for (int i = 0; i < CSWAP_PERF_REPEATS; ++i) {
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
    averageUs = elapsedMs * 1000.0F / static_cast<float>(CSWAP_PERF_REPEATS);
    return averageUs > 0.0F;
}

std::vector<CswapBenchmarkCase> LoadPerformanceCases()
{
    std::string csvPath(__FILE__);
    constexpr const char* sourceName = "cswap_benchmark.cpp";
    size_t sourcePos = csvPath.rfind(sourceName);
    if (sourcePos == std::string::npos) {
        throw std::runtime_error("Cannot derive cswap_test.csv from benchmark source path");
    }
    csvPath.replace(sourcePos, std::char_traits<char>::length(sourceName), "cswap_test.csv");

    std::vector<CswapBenchmarkCase> performanceCases;
    for (const auto& param : GetCasesFromCsv<CswapParam>(csvPath)) {
        if (param.caseName.rfind("TC_PF_", 0) != 0) {
            continue;
        }
        if (param.n <= 0 || param.incx != 1 || param.incy != 1 || param.xAlignOffset != 0 || param.yAlignOffset != 0) {
            throw std::runtime_error(
                "Performance cases must use positive n, unit strides, and zero offsets: " + param.caseName);
        }
        performanceCases.push_back({param.caseName, param.n});
    }
    return performanceCases;
}

bool MatchesCaseFilter(const std::string& caseName)
{
    const char* value = std::getenv("CSWAP_BENCHMARK_CASE");
    return value == nullptr || *value == '\0' || caseName == value;
}

float AcceptanceThresholdUs(int n)
{
    if (n == 1048576) {
        return 23.11F;
    }
    if (n == 2097152) {
        return 63.82F;
    }
    if (n == 4194304) {
        return 166.11F;
    }
    return 0.0F;
}

void PrintPerformance(const std::string& caseName, int n, size_t vectorBytes, float averageUs, float thresholdUs)
{
    std::cout << "[CSWAP_PERF] case=" << caseName << " n=" << n << " vector_bytes=" << vectorBytes
              << " traffic_bytes=" << 4 * vectorBytes << " warmup=" << CSWAP_PERF_WARMUP
              << " repeats=" << CSWAP_PERF_REPEATS << " avg_us=" << std::fixed << std::setprecision(3) << averageUs;
    if (thresholdUs > 0.0F) {
        std::cout << " threshold_us=" << thresholdUs;
    }
    std::cout << std::endl;
}

} // namespace

class CswapBenchmark : public BlasTest<CswapParam> {};

TEST_F(CswapBenchmark, AcceptanceContinuousShapes)
{
    const auto performanceCases = LoadPerformanceCases();
    ASSERT_EQ(performanceCases.size(), CSWAP_ACCEPTANCE_CASES);
    size_t selectedCases = 0;
    for (const auto& benchmarkCase : performanceCases) {
        if (!MatchesCaseFilter(benchmarkCase.caseName)) {
            continue;
        }
        ++selectedCases;
        int n = benchmarkCase.n;
        CswapBenchmarkResources resources;
        size_t bytes = static_cast<size_t>(n) * sizeof(aclblasComplex);
        ASSERT_EQ(aclrtMalloc(&resources.x, bytes, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
        ASSERT_EQ(aclrtMalloc(&resources.y, bytes, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
        ASSERT_EQ(aclrtMemset(resources.x, bytes, 0x5a, bytes), ACL_SUCCESS);
        ASSERT_EQ(aclrtMemset(resources.y, bytes, 0xa5, bytes), ACL_SUCCESS);

        float averageUs = 0.0F;
        ASSERT_TRUE(MeasureAverageUs(
            CswapBenchmark::stream_,
            [&]() {
                return aclblasCswap(
                           CswapBenchmark::handle_, n, static_cast<aclblasComplex*>(resources.x), 1,
                           static_cast<aclblasComplex*>(resources.y), 1) == ACLBLAS_STATUS_SUCCESS;
            },
            averageUs));
        float thresholdUs = AcceptanceThresholdUs(n);
        PrintPerformance(benchmarkCase.caseName, n, bytes, averageUs, thresholdUs);
        if (thresholdUs > 0.0F) {
            EXPECT_LE(averageUs, thresholdUs) << benchmarkCase.caseName;
        }
    }
    ASSERT_GT(selectedCases, 0U) << "CSWAP_BENCHMARK_CASE did not match a performance case";
}
