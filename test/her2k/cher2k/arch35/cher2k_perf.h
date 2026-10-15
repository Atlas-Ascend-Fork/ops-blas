/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "log/log.h"
#include "cann_ops_blas.h"

#include "cher2k_npu_wrapper.h"

// ═══════════════════════════════════════════════════════════════════════════════
// cher2k performance-capture framework (test plan §6.5)
//   Timing metric : aclrtEvent timing (authoritative), warmup 5 + 60 valid samples averaged;
//                   the GTest end-to-end time (host prep/golden/compare included) is only a
//                   reference upper bound.
//   Bottleneck    : derive the compute/load/store/scalar four-dimension split from the shape
//                   (byte and FLOPs model) and print the ratio of each dimension's theoretical
//                   time to the measured time (bound ratio).
//   Multi-round   : env var CHER2K_PERF_ROUNDS (default 1) enables multi-round capture; each
//                   round warms up and samples independently, recording median and dispersion
//                   (min/max/CV).
// ═══════════════════════════════════════════════════════════════════════════════

namespace cher2k_perf {

constexpr int kWarmupIters = 5;  // task spec / plan §6.5: 5 warmup iterations (+ 1 sync)
constexpr int kSampleIters = 60; // 60 valid samples (> 50)
constexpr int kDefaultRounds = 1;

// 950PR reference bandwidth/compute (theoretical model for bottleneck breakdown, not acceptance thresholds)
constexpr double kHbmBandwidthGbs = 1300.0; // ~1.3 TB/s
constexpr double kCubeTflops = 100.0;       // FP32 Cube peak (conservative)
constexpr double kAivTflops = 20.0;         // AIV SIMD peak (conservative)

struct PerfSample {
    std::string caseName;
    int n = 0;
    int k = 0;
    double eventAvgUs = 0.0; // aclrtEvent average per-launch time (authoritative metric)
    double eventMedUs = 0.0; // multi-round median
    double eventMinUs = 0.0;
    double eventMaxUs = 0.0;
    double cvPercent = 0.0; // dispersion: std/mean (%)
    int rounds = 1;
    int samples = kSampleIters;
    // Bottleneck breakdown (§6.5): theoretical time share of each dimension
    double flops = 0.0;       // total floating-point operations
    double bytesIn = 0.0;     // bytes read for A/B + old C
    double bytesOut = 0.0;    // bytes written for the C uplo triangle
    double scalarBytes = 0.0; // alpha/beta staging
    double computeUs = 0.0;   // theoretical compute time
    double loadUs = 0.0;      // theoretical load time
    double storeUs = 0.0;     // theoretical store time
    double scalarUs = 0.0;    // theoretical scalar time
    double boundRatio = 0.0;  // max(dimension theory)/measured -> <1 means slower than the tightest dimension
    std::string boundDim;     // compute / load / store / scalar
};

inline double Median(std::vector<double> v)
{
    if (v.empty()) {
        return 0.0;
    }
    std::sort(v.begin(), v.end());
    size_t m = v.size() / 2;
    return (v.size() % 2 == 0) ? (v[m - 1] + v[m]) / 2.0 : v[m];
}

struct EventPair {
    aclrtEvent start = nullptr;
    aclrtEvent stop = nullptr;
    ~EventPair()
    {
        if (start)
            aclrtDestroyEvent(start);
        if (stop)
            aclrtDestroyEvent(stop);
    }
};

// Single-round capture: after kWarmupIters warmup launches, aclrtEvent records the total
// time of kSampleIters launches; returns the average per-launch time (us).
template <typename Launch>
inline bool MeasureRound(aclrtStream stream, Launch&& launch, double& avgUs)
{
    for (int i = 0; i < kWarmupIters; ++i) {
        if (!launch()) {
            return false;
        }
    }
    if (aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
        return false;
    }

    EventPair ev;
    if (aclrtCreateEvent(&ev.start) != ACL_SUCCESS || aclrtCreateEvent(&ev.stop) != ACL_SUCCESS ||
        aclrtRecordEvent(ev.start, stream) != ACL_SUCCESS) {
        return false;
    }
    for (int i = 0; i < kSampleIters; ++i) {
        if (!launch()) {
            return false;
        }
    }
    if (aclrtRecordEvent(ev.stop, stream) != ACL_SUCCESS || aclrtSynchronizeEvent(ev.stop) != ACL_SUCCESS) {
        return false;
    }
    float elapsedMs = 0.0F;
    if (aclrtEventElapsedTime(&elapsedMs, ev.start, ev.stop) != ACL_SUCCESS) {
        return false;
    }
    avgUs = static_cast<double>(elapsedMs) * 1000.0 / static_cast<double>(kSampleIters);
    return avgUs > 0.0;
}

// Multi-round capture: each round warms up and samples independently; aggregates avg/median/min/max/CV.
template <typename Launch>
inline bool Measure(aclrtStream stream, int rounds, Launch&& launch, PerfSample& s)
{
    std::vector<double> perRoundUs;
    perRoundUs.reserve(static_cast<size_t>(rounds));
    for (int r = 0; r < rounds; ++r) {
        double us = 0.0;
        if (!MeasureRound(stream, launch, us)) {
            return false;
        }
        perRoundUs.push_back(us);
    }
    s.rounds = rounds;
    s.eventAvgUs = 0.0;
    for (double v : perRoundUs) {
        s.eventAvgUs += v;
    }
    s.eventAvgUs /= static_cast<double>(perRoundUs.size());
    s.eventMedUs = Median(perRoundUs);
    s.eventMinUs = *std::min_element(perRoundUs.begin(), perRoundUs.end());
    s.eventMaxUs = *std::max_element(perRoundUs.begin(), perRoundUs.end());
    double var = 0.0;
    for (double v : perRoundUs) {
        var += (v - s.eventAvgUs) * (v - s.eventAvgUs);
    }
    var /= static_cast<double>(perRoundUs.size());
    s.cvPercent = (s.eventAvgUs > 0.0) ? std::sqrt(var) / s.eventAvgUs * 100.0 : 0.0;
    return true;
}

// Bottleneck breakdown (§6.5): model the cher2k three-phase data flow
//   Phase0 load : 2*n*k*8B (A/B complex read) + 4*n*k*4B (Ar/Ai/Br/Bi write) ~ 12B/elem in the model
//   Phase1 compute: 4*(2*n*n*k) FLOP (4 real GEMMs)
//   Phase2 store: t1..t4 read 4*n*n*4B*2 (transposed double read) + C read n*n*8B + C uplo write n(n+1)*8B
inline void AnalyzeBottleneck(PerfSample& s)
{
    const double nn = static_cast<double>(s.n) * static_cast<double>(s.n);
    const double nk = static_cast<double>(s.n) * static_cast<double>(s.k);
    s.flops = 4.0 * (2.0 * nn * static_cast<double>(s.k));                          // 4 real GEMMs
    s.bytesIn = 2.0 * nk * 8.0 + 4.0 * nk * 4.0 + nn * 8.0;                         // A/B + interleave out + old C
    s.bytesOut = 4.0 * nn * 4.0 * 2.0 + static_cast<double>(s.n) * (s.n + 1) * 8.0; // t reread + C store
    s.scalarBytes = 3.0 * sizeof(float);

    s.computeUs = s.flops / (kCubeTflops * 1e12) * 1e6;
    s.loadUs = s.bytesIn / (kHbmBandwidthGbs * 1e9) * 1e6;
    s.storeUs = s.bytesOut / (kHbmBandwidthGbs * 1e9) * 1e6;
    s.scalarUs = s.scalarBytes / (kHbmBandwidthGbs * 1e9) * 1e6 + 5.0; // +5us launch/staging constant term

    double best = std::max(std::max(s.computeUs, s.loadUs), std::max(s.storeUs, s.scalarUs));
    s.boundRatio = (s.eventAvgUs > 0.0) ? best / s.eventAvgUs : 0.0;
    if (best == s.computeUs) {
        s.boundDim = "compute";
    } else if (best == s.loadUs) {
        s.boundDim = "load";
    } else if (best == s.storeUs) {
        s.boundDim = "store";
    } else {
        s.boundDim = "scalar";
    }
}

struct Cher2kPerfRecorder {
public:
    static Cher2kPerfRecorder& instance()
    {
        static Cher2kPerfRecorder rec;
        return rec;
    }

    void add(const PerfSample& s)
    {
        samples_.push_back(s);
        AppendRow(s);
        ReportOne(s);
    }

    // Per-case immediate summary (printed while gtest is alive, not relying on atexit/static destruction order)
    static void ReportOne(const PerfSample& s)
    {
        std::cout << "[Cher2kPerf] " << s.caseName << " n=" << s.n << " k=" << s.k << " warmup=" << kWarmupIters
                  << " samples=" << kSampleIters << " rounds=" << s.rounds << std::fixed << std::setprecision(2)
                  << " avg_us=" << s.eventAvgUs << " med_us=" << s.eventMedUs << " min_us=" << s.eventMinUs
                  << " max_us=" << s.eventMaxUs << std::setprecision(3) << " cv%=" << s.cvPercent
                  << " bound=" << s.boundDim << " ratio=" << s.boundRatio << std::setprecision(2)
                  << " (comp_us=" << s.computeUs << " load_us=" << s.loadUs << " store_us=" << s.storeUs << ")"
                  << " | GTest end-to-end is a reference upper bound only, aclrtEvent is authoritative" << std::endl;
    }

    void report()
    {
        if (samples_.empty() || reported_) {
            return;
        }
        reported_ = true;
        std::cout << "\n[Cher2kPerf] aclrtEvent metric: warmup=" << kWarmupIters << " samples=" << kSampleIters
                  << " (GTest end-to-end time is a reference upper bound only)\n";
        std::cout << "[Cher2kPerF] " << std::left << std::setw(18) << "case" << std::right << std::setw(6) << "n"
                  << std::setw(6) << "k" << std::setw(12) << "avg_us" << std::setw(12) << "med_us" << std::setw(10)
                  << "min_us" << std::setw(10) << "max_us" << std::setw(8) << "cv%" << std::setw(6) << "rnd"
                  << std::setw(10) << "bound" << std::setw(9) << "ratio" << std::setw(11) << "comp_us" << std::setw(10)
                  << "load_us" << std::setw(10) << "store_us"
                  << "\n";
        for (const auto& s : samples_) {
            std::cout << "[Cher2kPerf] " << std::left << std::setw(18) << s.caseName << std::right << std::setw(6)
                      << s.n << std::setw(6) << s.k << std::fixed << std::setprecision(2) << std::setw(12)
                      << s.eventAvgUs << std::setw(12) << s.eventMedUs << std::setw(10) << s.eventMinUs << std::setw(10)
                      << s.eventMaxUs << std::setprecision(3) << std::setw(8) << s.cvPercent << std::setw(6) << s.rounds
                      << std::setw(10) << s.boundDim << std::setprecision(3) << std::setw(9) << s.boundRatio
                      << std::setprecision(2) << std::setw(11) << s.computeUs << std::setw(10) << s.loadUs
                      << std::setw(10) << s.storeUs << "\n";
        }
        std::cout << "[Cher2kPerf] bound=bottleneck dimension (the largest of compute/load/store/scalar "
                     "theoretical time), ratio=tightest-dimension theoretical time/measured time (<=1 means "
                     "the measurement has not reached the theoretical bound)\n";
        std::cout << "[Cher2kPerf] perf csv -> " << CsvPath() << " (" << samples_.size() << " cases)\n";
    }

private:
    // Row-by-row append (open-append-close), avoiding file writes during static destruction.
    static void AppendRow(const PerfSample& s)
    {
        const std::string p = CsvPath();
        bool needHeader = true;
        {
            std::ifstream probe(p);
            if (probe.good() && probe.peek() != std::char_traits<char>::eof()) {
                needHeader = false;
            }
        }
        std::ofstream ofs(p, std::ios::app);
        if (!ofs.is_open()) {
            return;
        }
        if (needHeader) {
            ofs << "case_name,n,k,event_avg_us,event_median_us,event_min_us,event_max_us,cv_percent,"
                   "rounds,samples,flops,bytes_in,bytes_out,scalar_bytes,compute_us,load_us,store_us,"
                   "scalar_us,bound_dim,bound_ratio\n";
        }
        ofs << s.caseName << "," << s.n << "," << s.k << "," << std::scientific << std::setprecision(6) << s.eventAvgUs
            << "," << s.eventMedUs << "," << s.eventMinUs << "," << s.eventMaxUs << "," << s.cvPercent << ","
            << s.rounds << "," << s.samples << "," << s.flops << "," << s.bytesIn << "," << s.bytesOut << ","
            << s.scalarBytes << "," << s.computeUs << "," << s.loadUs << "," << s.storeUs << "," << s.scalarUs << ","
            << s.boundDim << "," << s.boundRatio << "\n";
    }

    static std::string CsvPath()
    {
        // Write to the binary's run directory (cwd) to avoid polluting the source tree; CHER2K_RESULT_DIR overrides.
        const char* dir = std::getenv("CHER2K_RESULT_DIR");
        if (dir != nullptr && *dir != '\0') {
            return std::string(dir) + "/cher2k_perf_results.csv";
        }
        return "cher2k_perf_results.csv";
    }

    std::vector<PerfSample> samples_;
    bool reported_ = false;
};

// Results are written row-by-row via append (AppendRow, immediately per case).
// The summary print does not flush during static destruction/atexit -- under atexit ordering the
// CANN-side cleanup may already have corrupted the iostream state (observed as large amounts of
// garbled output); instead Cher2kRunPerformanceCase prints a per-case summary (ReportOne) right
// after each capture, keeping output within the gtest lifetime.
static const bool g_perfUnused = false;

} // namespace cher2k_perf

// ─────────────────────────────────────────────────────────────────────────────
// Capture entry point for a single TC_PF case (called from the CsvDriven branch in cher2k_test.cpp).
// The device buffers stay alive across the whole capture window (host data is prepared only once).
// ─────────────────────────────────────────────────────────────────────────────
// [codecheck #7] Upload stage (outside the timing window): allocate the five device buffers; if
// any fails, free the already-allocated prefix and return false.
struct Cher2kPerfBuffers {
    void* dAlpha = nullptr;
    void* dA = nullptr;
    void* dB = nullptr;
    void* dBeta = nullptr;
    void* dC = nullptr;
};

inline bool Cher2kAllocPerfBuffers(
    const char* caseName, size_t aBytes, size_t bBytes, size_t cBytes, Cher2kPerfBuffers& bufs)
{
    bool ok = aclrtMalloc(&bufs.dAlpha, sizeof(aclblasComplex), ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS &&
              aclrtMalloc(&bufs.dA, aBytes, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS &&
              aclrtMalloc(&bufs.dB, bBytes, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS &&
              aclrtMalloc(&bufs.dBeta, sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS &&
              aclrtMalloc(&bufs.dC, cBytes, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS;
    if (!ok) {
        std::cout << "[Cher2kPerf] skip " << caseName << ": device alloc failed" << std::endl;
        Cher2kFreeDeviceBuffers(bufs.dAlpha, bufs.dA, bufs.dB, bufs.dBeta, bufs.dC);
        return false;
    }
    return true;
}

// [codecheck #7] Upload stage (outside the timing window): five H2D copies (alpha/A/B/beta/C); if
// any fails, free all allocated segments and return false. Falls back to the original aPtr when C's
// cPtr is null.
template <typename Param, typename HostData>
inline bool Cher2kUploadPerfBuffers(
    const char* caseName, const Param& p, const HostData& d, size_t aBytes, size_t bBytes, size_t cBytes,
    Cher2kPerfBuffers& bufs)
{
    aclblasStatus_t uploadSt = ACLBLAS_STATUS_SUCCESS;
    do {
        if (aclrtMemcpy(
                bufs.dAlpha, sizeof(aclblasComplex), &d.alpha, sizeof(aclblasComplex), ACL_MEMCPY_HOST_TO_DEVICE) !=
            ACL_SUCCESS) {
            uploadSt = ACLBLAS_STATUS_INTERNAL_ERROR;
            break;
        }
        if (aclrtMemcpy(bufs.dA, aBytes, d.aPtr, aBytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
            uploadSt = ACLBLAS_STATUS_INTERNAL_ERROR;
            break;
        }
        if (aclrtMemcpy(bufs.dB, bBytes, d.bPtr, bBytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
            uploadSt = ACLBLAS_STATUS_INTERNAL_ERROR;
            break;
        }
        if (aclrtMemcpy(bufs.dBeta, sizeof(float), &d.beta, sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
            uploadSt = ACLBLAS_STATUS_INTERNAL_ERROR;
            break;
        }
        if (aclrtMemcpy(bufs.dC, cBytes, d.cPtr ? d.cPtr : d.aPtr, cBytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
            uploadSt = ACLBLAS_STATUS_INTERNAL_ERROR;
            break;
        }
    } while (false);

    if (uploadSt != ACLBLAS_STATUS_SUCCESS) {
        std::cout << "[Cher2kPerf] skip " << caseName << ": H2D upload failed" << std::endl;
        Cher2kFreeDeviceBuffers(bufs.dAlpha, bufs.dA, bufs.dB, bufs.dBeta, bufs.dC);
        return false;
    }
    return true;
}

// [codecheck #7] Record stage: bottleneck breakdown on the successful sampling side + CSV append.
inline void Cher2kRecordPerfSample(cher2k_perf::PerfSample& s)
{
    cher2k_perf::AnalyzeBottleneck(s);
    cher2k_perf::Cher2kPerfRecorder::instance().add(s);
}

template <typename Param, typename HostData>
inline void Cher2kRunPerformanceCase(const Param& p, const HostData& d, aclblasHandle_t handle, aclrtStream stream)
{
    using namespace cher2k_perf;

    if (p.n <= 0 || d.aPtr == nullptr || d.bPtr == nullptr) {
        std::cout << "[Cher2kPerf] skip " << p.caseName << ": no compute path (n=" << p.n << ")" << std::endl;
        return;
    }

    const int abCols = (p.trans == ACLBLAS_OP_N) ? p.k : p.n;
    const size_t aBytes = static_cast<size_t>(p.lda) * static_cast<size_t>(abCols) * sizeof(aclblasComplex);
    const size_t bBytes = static_cast<size_t>(p.ldb) * static_cast<size_t>(abCols) * sizeof(aclblasComplex);
    const size_t cBytes = static_cast<size_t>(p.ldc) * static_cast<size_t>(p.n) * sizeof(aclblasComplex);
    if (aBytes == 0 || bBytes == 0 || cBytes == 0) {
        std::cout << "[Cher2kPerf] skip " << p.caseName << ": zero-size operand (k=" << p.k << ")" << std::endl;
        return;
    }

    // [codecheck #7] The upload stages were moved wholesale into Cher2kAllocPerfBuffers/
    // Cher2kUploadPerfBuffers (both outside the timing window); the buffer lifetime still spans
    // the whole warmup+sampling window.
    Cher2kPerfBuffers bufs;
    if (!Cher2kAllocPerfBuffers(p.caseName.c_str(), aBytes, bBytes, cBytes, bufs)) {
        return;
    }
    if (!Cher2kUploadPerfBuffers(p.caseName.c_str(), p, d, aBytes, bBytes, cBytes, bufs)) {
        return;
    }

    // Capture uses the stream bound to the handle by BlasTest (same stream as the measured
    // operator enqueue, so aclrtEvent can capture the operator execution time).
    aclblasHandle h = handle;
    PerfSample s;
    s.caseName = p.caseName;
    s.n = p.n;
    s.k = p.k;

    const char* roundsEnv = std::getenv("CHER2K_PERF_ROUNDS");
    int rounds = roundsEnv ? std::max(1, std::atoi(roundsEnv)) : kDefaultRounds;

    // [codecheck #7] The sampling loop (between aclrtEvent start/stop) must stay in the main
    // function: the timing metric stays transparent and is not moved around by helper splitting.
    bool measured = Measure(
        stream, rounds,
        [&]() -> bool {
            aclblasStatus_t st = aclblasCher2k(
                h, p.uplo, p.trans, p.n, p.k, static_cast<const aclblasComplex*>(bufs.dAlpha),
                static_cast<const aclblasComplex*>(bufs.dA), p.lda, static_cast<const aclblasComplex*>(bufs.dB), p.ldb,
                static_cast<const float*>(bufs.dBeta), static_cast<aclblasComplex*>(bufs.dC), p.ldc);
            if (st != ACLBLAS_STATUS_SUCCESS) {
                OP_LOGE("Cher2kRunPerformanceCase", "aclblasCher2k returned st=%d", static_cast<int>(st));
            }
            return st == ACLBLAS_STATUS_SUCCESS;
        },
        s);
    if (!measured) {
        std::cout << "[Cher2kPerf] measure failed for " << p.caseName << std::endl;
    } else {
        Cher2kRecordPerfSample(s);
    }

    Cher2kFreeDeviceBuffers(bufs.dAlpha, bufs.dA, bufs.dB, bufs.dBeta, bufs.dC);
}
