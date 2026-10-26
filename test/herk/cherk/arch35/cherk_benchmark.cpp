/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <charconv>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "blas_test.h"
#include "cherk_npu_wrapper.h"
#include "cherk_param.h"
#include "csv_loader.h"
#include "fill.h"

namespace {

constexpr int kDefaultWarmup = 1;
constexpr int kDefaultRepeats = 3;
constexpr size_t kAcceptanceCases = 200;

int ParsePositiveEnv(const char* name, int fallback, bool allowZero = false)
{
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0')
        return fallback;
    int parsed = fallback;
    const char* end = value + std::strlen(value);
    const auto result = std::from_chars(value, end, parsed);
    const int minimum = allowZero ? 0 : 1;
    return result.ec == std::errc() && result.ptr == end && parsed >= minimum ? parsed : fallback;
}

std::vector<CherkParam> LoadPerformanceCases()
{
    std::string csvPath(__BASE_FILE__);
    constexpr const char* sourceName = "cherk_benchmark.cpp";
    const size_t sourcePos = csvPath.rfind(sourceName);
    if (sourcePos == std::string::npos)
        throw std::runtime_error("Cannot derive cherk_test.csv from benchmark source path");
    csvPath.replace(sourcePos, std::char_traits<char>::length(sourceName), "cherk_test.csv");
    // arch22 与 arch35 均维护 smoke/full CSV 对：benchmark 恒读 full（含 TC_PF_ 性能验收用例）。
    if (csvPath.find("arch22") != std::string::npos || csvPath.find("arch35") != std::string::npos) {
        csvPath.insert(csvPath.size() - 4U, "_full");
    }

    std::vector<CherkParam> performanceCases;
    for (const auto& param : GetCasesFromCsv<CherkParam>(csvPath)) {
        if (param.caseName.rfind("TC_PF_", 0U) != 0U)
            continue;
        if (param.expectResult != ACLBLAS_STATUS_SUCCESS || param.n <= 0 || param.k <= 0 || param.nullAlpha ||
            param.nullBeta || param.nullA || param.nullC) {
            throw std::runtime_error("Invalid performance case: " + param.caseName);
        }
        performanceCases.push_back(param);
    }
    return performanceCases;
}

bool MatchesCaseFilter(const std::string& caseName)
{
    const char* value = std::getenv("CHERK_BENCHMARK_CASE");
    return value == nullptr || *value == '\0' || caseName == value;
}

} // namespace

class CherkBenchmark : public BlasTest<CherkParam> {};

TEST_F(CherkBenchmark, PerformanceCases)
{
    const auto performanceCases = LoadPerformanceCases();
    ASSERT_EQ(performanceCases.size(), kAcceptanceCases);
    const int warmup = ParsePositiveEnv("CHERK_PERF_WARMUP", kDefaultWarmup, true);
    const int repeats = ParsePositiveEnv("CHERK_PERF_ITERATIONS", kDefaultRepeats);
    size_t selectedCases = 0;

    for (const auto& p : performanceCases) {
        if (!MatchesCaseFilter(p.caseName))
            continue;
        ++selectedCases;

        const int aCols = p.trans == ACLBLAS_OP_N ? p.k : p.n;
        std::vector<aclblasComplex> aHost = makeBlasComplexMatrix(p.lda, aCols, p.lda, p.aFill, p.randomSeed);
        std::vector<aclblasComplex> cHost = makeBlasComplexMatrix(p.n, p.n, p.ldc, p.cFill, p.randomSeed + 1U);
        float averageUs = 0.0F;
        ASSERT_EQ(
            aclblasCherk_npu_benchmark(
                CherkBenchmark::handle_, CherkBenchmark::stream_, p.uplo, p.trans, p.n, p.k, &p.alpha, aHost.data(),
                p.lda, &p.beta, cHost.data(), p.ldc, warmup, repeats, &averageUs),
            ACLBLAS_STATUS_SUCCESS)
            << p.caseName;
        std::cout << "[CHERK_PERF] case=" << p.caseName << " metric=e2e_api"
                  << " n=" << p.n << " k=" << p.k << " warmup=" << warmup << " repeats=" << repeats
                  << " avg_us=" << std::fixed << std::setprecision(3) << averageUs << std::endl;
    }

    ASSERT_GT(selectedCases, 0U) << "CHERK_BENCHMARK_CASE did not match a performance case";
}
