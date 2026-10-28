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
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "ctbsv_bench_cases.h"

namespace {

constexpr int WARMUP = 20;
constexpr int REPEATS = 100;
constexpr int TOTAL_RUNS = WARMUP + REPEATS;
constexpr double RATIO_THRESHOLD = 0.8;
constexpr int CORE_CASE_COUNT = 5;
constexpr int MIN_DIAGONAL_BOOST = 5;     // Task test guide: boost = max(5, n).
constexpr double INITIAL_MIN_RATIO = 1e9; // Reporting sentinel, not the acceptance threshold.
constexpr int CONSOLE_DECIMAL_PLACES = 3;
constexpr int CSV_DECIMAL_PLACES = 6;

// Keep the deterministic 32-bit LCG and its [-5, 5) value mapping unchanged.
constexpr uint32_t RANDOM_MULTIPLIER = 1664525U;
constexpr uint32_t RANDOM_INCREMENT = 1013904223U;
constexpr uint32_t RANDOM_BUCKET_COUNT = 10000U;
constexpr float RANDOM_VALUE_SCALE = 1000.0f;
constexpr float RANDOM_VALUE_OFFSET = 5.0f;
constexpr uint32_t MATRIX_RANDOM_SEED = 42U;
constexpr uint32_t RHS_RANDOM_SEED = 43U;

struct Case {
    const char* name;
    aclblasFillMode_t uplo;
    aclblasOperation_t trans;
    aclblasDiagType_t diag;
    int n;
    int k;
    int lda;
    double gpuUs;
};

struct BenchOpts {
    int fromIdx = 0;
    int toIdx = PERF_CASE_COUNT;
    const char* outPath = nullptr;
};

struct BenchStats {
    int pass = 0;
    int fail = 0;
    double minRatio = INITIAL_MIN_RATIO;
    std::string minCase;
};

const char* UploStr(aclblasFillMode_t v) { return v == ACLBLAS_UPPER ? "UPPER" : "LOWER"; }

const char* TransStr(aclblasOperation_t v)
{
    if (v == ACLBLAS_OP_N) {
        return "N";
    }
    if (v == ACLBLAS_OP_T) {
        return "T";
    }
    return "C";
}

const char* DiagStr(aclblasDiagType_t v) { return v == ACLBLAS_UNIT ? "UNIT" : "NON_UNIT"; }

void FillRandom(std::vector<aclblasComplex>& data, uint32_t seed)
{
    uint32_t s = seed;
    for (auto& v : data) {
        s = s * RANDOM_MULTIPLIER + RANDOM_INCREMENT;
        v.real = static_cast<float>(static_cast<int32_t>(s % RANDOM_BUCKET_COUNT)) / RANDOM_VALUE_SCALE -
                 RANDOM_VALUE_OFFSET;
        s = s * RANDOM_MULTIPLIER + RANDOM_INCREMENT;
        v.imag = static_cast<float>(static_cast<int32_t>(s % RANDOM_BUCKET_COUNT)) / RANDOM_VALUE_SCALE -
                 RANDOM_VALUE_OFFSET;
    }
}

void StrengthenDiagonal(std::vector<aclblasComplex>& a, int n, int k, int lda, bool isUpper)
{
    const int effK = std::min(k, std::max(n - 1, 0));
    const int diagRow = isUpper ? effK : 0;
    const float boost = static_cast<float>(std::max(MIN_DIAGONAL_BOOST, n));
    for (int j = 0; j < n; ++j) {
        aclblasComplex& d = a[diagRow + j * lda];
        d.real += (d.real >= 0.0f) ? boost : -boost;
    }
}

void FreePair(aclblasComplex* aDev, aclblasComplex* xDev)
{
    if (aDev != nullptr) {
        aclrtFree(aDev);
    }
    if (xDev != nullptr) {
        aclrtFree(xDev);
    }
}

bool AllocAndCopy(
    const std::vector<aclblasComplex>& aHost, const std::vector<aclblasComplex>& xHost, aclblasComplex*& aDev,
    aclblasComplex*& xDev)
{
    const size_t aBytes = aHost.size() * sizeof(aclblasComplex);
    const size_t xBytes = xHost.size() * sizeof(aclblasComplex);
    if (aclrtMalloc(reinterpret_cast<void**>(&aDev), aBytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS ||
        aclrtMalloc(reinterpret_cast<void**>(&xDev), xBytes * TOTAL_RUNS, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        FreePair(aDev, xDev);
        aDev = nullptr;
        xDev = nullptr;
        return false;
    }
    aclrtMemcpy(aDev, aBytes, aHost.data(), aBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    for (int i = 0; i < TOTAL_RUNS; ++i) {
        aclrtMemcpy(
            xDev + static_cast<size_t>(i) * xHost.size(), xBytes, xHost.data(), xBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    }
    return true;
}

bool TimedRepeats(
    aclblasHandle_t handle, aclrtStream stream, const Case& c, aclblasComplex* aDev, aclblasComplex* xDev, float& avgUs)
{
    auto launch = [&](int run) {
        aclblasComplex* runX = xDev + static_cast<size_t>(run) * static_cast<size_t>(c.n);
        return aclblasCtbsv(handle, c.uplo, c.trans, c.diag, c.n, c.k, aDev, c.lda, runX, 1) == ACLBLAS_STATUS_SUCCESS;
    };
    for (int i = 0; i < WARMUP; ++i) {
        if (!launch(i)) {
            return false;
        }
    }
    if (aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
        return false;
    }

    aclrtEvent start = nullptr;
    aclrtEvent stop = nullptr;
    aclrtCreateEvent(&start);
    aclrtCreateEvent(&stop);
    aclrtRecordEvent(start, stream);
    for (int i = 0; i < REPEATS; ++i) {
        if (!launch(WARMUP + i)) {
            aclrtDestroyEvent(start);
            aclrtDestroyEvent(stop);
            return false;
        }
    }
    aclrtRecordEvent(stop, stream);
    if (aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
        aclrtDestroyEvent(start);
        aclrtDestroyEvent(stop);
        return false;
    }
    float totalMs = 0.0f;
    aclrtEventElapsedTime(&totalMs, start, stop);
    avgUs = totalMs * 1000.0f / static_cast<float>(REPEATS);
    aclrtDestroyEvent(start);
    aclrtDestroyEvent(stop);
    return true;
}

bool MeasureUs(aclblasHandle_t handle, aclrtStream stream, const Case& c, float& avgUs)
{
    const int n = c.n;
    const int lda = c.lda;
    std::vector<aclblasComplex> aHost(static_cast<size_t>(lda) * static_cast<size_t>(std::max(n, 1)));
    std::vector<aclblasComplex> xHost(static_cast<size_t>(std::max(n, 1)));
    FillRandom(aHost, MATRIX_RANDOM_SEED);
    FillRandom(xHost, RHS_RANDOM_SEED);
    if (n > 0 && c.diag == ACLBLAS_NON_UNIT) {
        StrengthenDiagonal(aHost, n, c.k, lda, c.uplo == ACLBLAS_UPPER);
    }

    aclblasComplex* aDev = nullptr;
    aclblasComplex* xDev = nullptr;
    if (!AllocAndCopy(aHost, xHost, aDev, xDev)) {
        return false;
    }
    const bool ok = TimedRepeats(handle, stream, c, aDev, xDev, avgUs);
    FreePair(aDev, xDev);
    return ok;
}

void ParseArgs(int argc, char** argv, BenchOpts& opts)
{
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--five") {
            opts.fromIdx = 0;
            opts.toIdx = std::min(CORE_CASE_COUNT, PERF_CASE_COUNT);
            continue;
        }
        if (a == "--from" && i + 1 < argc) {
            opts.fromIdx = std::atoi(argv[++i]);
            continue;
        }
        if (a == "--to" && i + 1 < argc) {
            opts.toIdx = std::atoi(argv[++i]);
            continue;
        }
        if (a == "--csv" && i + 1 < argc) {
            opts.outPath = argv[++i];
            continue;
        }
        opts.outPath = argv[i];
    }
    if (opts.fromIdx < 0) {
        opts.fromIdx = 0;
    }
    if (opts.toIdx > PERF_CASE_COUNT) {
        opts.toIdx = PERF_CASE_COUNT;
    }
    if (opts.fromIdx > opts.toIdx) {
        opts.fromIdx = opts.toIdx;
    }
}

void EmitRow(std::ostream& os, const Case& c, const char* extra)
{
    os << c.name << "," << c.n << "," << c.k << "," << UploStr(c.uplo) << "," << TransStr(c.trans) << ","
       << DiagStr(c.diag) << extra;
}

void RecordCase(const Case& c, double npuUs, BenchStats& stats, std::ofstream& csv)
{
    const double gpuUs = c.gpuUs;
    const double ratio = npuUs > 0.0 ? (gpuUs / npuUs) : 0.0;
    const double targetUs = gpuUs / RATIO_THRESHOLD;
    const bool ok = ratio >= RATIO_THRESHOLD;
    if (ok) {
        ++stats.pass;
    } else {
        ++stats.fail;
    }
    if (ratio < stats.minRatio) {
        stats.minRatio = ratio;
        stats.minCase = c.name;
    }
    std::ostringstream extra;
    extra << std::fixed << std::setprecision(CONSOLE_DECIMAL_PLACES) << "," << npuUs << "," << gpuUs << "," << ratio
          << "," << targetUs << "," << (ok ? "PASS" : "FAIL") << "\n";
    EmitRow(std::cout, c, extra.str().c_str());
    if (csv.is_open()) {
        csv << c.name << "," << c.n << "," << c.k << "," << UploStr(c.uplo) << "," << TransStr(c.trans) << ","
            << DiagStr(c.diag) << "," << c.lda << "," << npuUs << "," << gpuUs << "," << ratio << "," << targetUs << ","
            << (ok ? "PASS" : "FAIL") << "\n";
    }
}

int RunCases(aclblasHandle_t handle, aclrtStream stream, const BenchOpts& opts)
{
    std::ofstream csv;
    if (opts.outPath != nullptr) {
        csv.open(opts.outPath);
        csv << "case_name,n,k,uplo,trans,diag,lda,npu_us,gpu_us,ratio,target_us,verdict\n";
        csv << std::fixed << std::setprecision(CSV_DECIMAL_PLACES);
    }
    std::cout << std::fixed << std::setprecision(CONSOLE_DECIMAL_PLACES);
    std::cout << "case,n,k,uplo,trans,diag,npu_us,gpu_us,ratio,target_us,verdict\n";

    BenchStats stats;
    for (int i = opts.fromIdx; i < opts.toIdx; ++i) {
        const PerfCase& pc = PERF_CASES[i];
        Case c{pc.name, pc.uplo, pc.trans, pc.diag, pc.n, pc.k, pc.lda, pc.gpuUs};
        float avgUs = 0.0f;
        if (!MeasureUs(handle, stream, c, avgUs)) {
            ++stats.fail;
            EmitRow(std::cout, c, ",MEASURE_FAILED\n");
            continue;
        }
        RecordCase(c, static_cast<double>(avgUs), stats, csv);
    }
    std::cout << "summary,pass=" << stats.pass << ",fail=" << stats.fail << ",min_ratio=" << stats.minRatio
              << ",min_case=" << stats.minCase << ",threshold=" << RATIO_THRESHOLD << "\n";
    return stats.fail;
}

} // namespace

int main(int argc, char** argv)
{
    BenchOpts opts;
    ParseArgs(argc, argv, opts);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclblasHandle_t handle = nullptr;
    aclblasCreate(&handle);
    aclrtStream stream = nullptr;
    aclrtCreateStream(&stream);
    aclblasSetStream(handle, stream);

    const int fail = RunCases(handle, stream, opts);

    aclrtDestroyStream(stream);
    aclblasDestroy(handle);
    aclrtResetDevice(0);
    aclFinalize();
    return fail == 0 ? 0 : 1;
}
