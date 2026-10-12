/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "blas_test.h"
#include "cann_ops_blas.h"
#include "cgbmv_param.h"
#include "csv_loader.h"

namespace {

constexpr long CGBMV_BENCH_MAX_COUNT = 1000000L; // upper bound accepted for warmup / repeat counts

// Defaults satisfy the acceptance rule (warmup first, then > 50 timed samples); both are overridable
// through CGBMV_BENCH_WARMUP (>= 0) / CGBMV_BENCH_REPEATS (> 0), e.g. to shorten runs under the simulator.
constexpr int CGBMV_PERF_WARMUP = 10;
constexpr int CGBMV_PERF_REPEATS = 100;

int EnvInt(const char* name, int def, int minValue)
{
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0') {
        return def;
    }
    char* end = nullptr;
    long parsed = std::strtol(value, &end, 10);
    if (end == value || *end != '\0' || parsed < minValue || parsed > CGBMV_BENCH_MAX_COUNT) {
        return def;
    }
    return static_cast<int>(parsed);
}

struct DeviceBuffer {
    void* ptr = nullptr;
    ~DeviceBuffer()
    {
        if (ptr != nullptr) {
            aclrtFree(ptr);
        }
    }
};

struct BenchEvents {
    aclrtEvent start = nullptr;
    aclrtEvent stop = nullptr;
    ~BenchEvents()
    {
        if (start != nullptr) {
            aclrtDestroyEvent(start);
        }
        if (stop != nullptr) {
            aclrtDestroyEvent(stop);
        }
    }
};

bool UploadRandom(DeviceBuffer& buf, size_t complexCount, uint32_t seed)
{
    std::vector<aclblasComplex> host(complexCount);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : host) {
        v = {dist(rng), dist(rng)};
    }
    size_t bytes = complexCount * sizeof(aclblasComplex);
    return aclrtMalloc(&buf.ptr, bytes, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS &&
           aclrtMemcpy(buf.ptr, bytes, host.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS;
}

template <typename Launch>
bool MeasureAverageUs(aclrtStream stream, Launch&& launch, int warmup, int repeats, float& averageUs)
{
    for (int i = 0; i < warmup; ++i) {
        if (!launch()) {
            return false;
        }
    }
    BenchEvents ev;
    if (aclrtSynchronizeStream(stream) != ACL_SUCCESS || aclrtCreateEvent(&ev.start) != ACL_SUCCESS ||
        aclrtCreateEvent(&ev.stop) != ACL_SUCCESS || aclrtRecordEvent(ev.start, stream) != ACL_SUCCESS) {
        return false;
    }
    for (int i = 0; i < repeats; ++i) {
        if (!launch()) {
            return false;
        }
    }
    float elapsedMs = 0.0f;
    if (aclrtRecordEvent(ev.stop, stream) != ACL_SUCCESS || aclrtSynchronizeEvent(ev.stop) != ACL_SUCCESS ||
        aclrtEventElapsedTime(&elapsedMs, ev.start, ev.stop) != ACL_SUCCESS) {
        return false;
    }
    averageUs = elapsedMs * 1000.0f / static_cast<float>(repeats);
    return true;
}

std::vector<CgbmvParam> LoadPerformanceCases()
{
    std::string csvPath(__FILE__);
    const std::string sourceName = "cgbmv_benchmark.cpp";
    size_t pos = csvPath.rfind(sourceName);
    if (pos == std::string::npos) {
        throw std::runtime_error("cannot derive cgbmv_test.csv from the benchmark source path");
    }
    csvPath.replace(pos, sourceName.size(), "cgbmv_test.csv");
    const char* filter = std::getenv("CGBMV_BENCHMARK_CASE");
    std::vector<CgbmvParam> cases;
    for (const auto& p : GetCasesFromCsv<CgbmvParam>(csvPath)) {
        if (p.caseName.rfind("TC_PF_", 0) != 0) {
            continue;
        }
        if (filter != nullptr && *filter != '\0' && p.caseName.find(filter) == std::string::npos) {
            continue;
        }
        cases.push_back(p);
    }
    return cases;
}

const char* TransName(aclblasOperation_t trans)
{
    return trans == ACLBLAS_OP_N ? "N" : (trans == ACLBLAS_OP_T ? "T" : "C");
}

} // namespace

class CgbmvBenchmark : public BlasTest<CgbmvParam> {};

TEST_F(CgbmvBenchmark, PerformanceCases)
{
    const int warmup = EnvInt("CGBMV_BENCH_WARMUP", CGBMV_PERF_WARMUP, 0);
    const int repeats = EnvInt("CGBMV_BENCH_REPEATS", CGBMV_PERF_REPEATS, 1);
    const auto cases = LoadPerformanceCases();
    ASSERT_FALSE(cases.empty()) << "no TC_PF_ case selected";
    for (const auto& p : cases) {
        bool isTransN = (p.trans == ACLBLAS_OP_N);
        int xCount = isTransN ? p.n : p.m;
        int yCount = isTransN ? p.m : p.n;
        DeviceBuffer a;
        DeviceBuffer x;
        DeviceBuffer y;
        ASSERT_TRUE(UploadRandom(a, static_cast<size_t>(p.lda) * p.n, p.randomSeed));
        ASSERT_TRUE(UploadRandom(x, static_cast<size_t>(xCount), p.randomSeed + 1U));
        ASSERT_TRUE(UploadRandom(y, static_cast<size_t>(yCount), p.randomSeed + 2U));
        aclblasComplex alpha = {p.alphaReal, p.alphaImag};
        aclblasComplex beta = {p.betaReal, p.betaImag};
        float avgUs = 0.0f;
        ASSERT_TRUE(MeasureAverageUs(
            CgbmvBenchmark::stream_,
            [&]() {
                const aclblasStatus_t st = aclblasCgbmv(
                    CgbmvBenchmark::handle_, p.trans, p.m, p.n, p.kl, p.ku, &alpha,
                    static_cast<const aclblasComplex*>(a.ptr), p.lda, static_cast<const aclblasComplex*>(x.ptr), p.incx,
                    &beta, static_cast<aclblasComplex*>(y.ptr), p.incy);
                return st == ACLBLAS_STATUS_SUCCESS;
            },
            warmup, repeats, avgUs))
            << p.caseName;
        std::cout << "[CGBMV_PERF] case=" << p.caseName << " trans=" << TransName(p.trans) << " m=" << p.m
                  << " n=" << p.n << " kl=" << p.kl << " ku=" << p.ku << " warmup=" << warmup << " repeats=" << repeats
                  << " avg_us=" << std::fixed << std::setprecision(3) << avgUs << std::endl;
    }
}
