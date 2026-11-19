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
#include <chrono>
#include <cstdint>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "acl/acl.h"
#include "cann_ops_blas.h"

namespace {

constexpr int PERF_WARMUP = 5;
constexpr int PERF_ITERATIONS = 100;
// arch35 性能程序的 GPU/NPU 门限：gpu_ms * 1000 / npu_us >= 0.4。
constexpr double PERF_REQUIRED_RATIO = 0.4;

struct PerformanceCase {
    const char* caseName;
    int m;
    int n;
    double targetUs; // 任务书 910B3 单次调用上限（us）
};

// 任务书 §3.3 的四个典型方阵；incx=incy=1，lda=max(1,m)=m。
const PerformanceCase kPerformanceCases[] = {
    {"doc_512_512", 512, 512, 5.7},
    {"doc_1024_1024", 1024, 1024, 10.1},
    {"doc_2048_2048", 2048, 2048, 55.71},
    {"doc_4096_4096", 4096, 4096, 219.35},
};

struct DeviceResources {
    aclrtStream stream = nullptr;
    aclblasHandle_t handle = nullptr;
    aclrtEvent start = nullptr;
    aclrtEvent end = nullptr;
    void* x = nullptr;
    void* y = nullptr;
    void* a = nullptr;

    ~DeviceResources()
    {
        if (start != nullptr) {
            aclrtDestroyEvent(start);
        }
        if (end != nullptr) {
            aclrtDestroyEvent(end);
        }
        if (a != nullptr) {
            aclrtFree(a);
        }
        if (y != nullptr) {
            aclrtFree(y);
        }
        if (x != nullptr) {
            aclrtFree(x);
        }
        if (handle != nullptr) {
            aclblasDestroy(handle);
        }
        if (stream != nullptr) {
            aclrtDestroyStream(stream);
        }
    }
};

bool CheckAcl(aclError status, const char* operation)
{
    if (status == ACL_SUCCESS) {
        return true;
    }
    std::cerr << operation << " failed: " << status << std::endl;
    return false;
}

bool CheckBlas(aclblasStatus_t status, const char* operation)
{
    if (status == ACLBLAS_STATUS_SUCCESS) {
        return true;
    }
    std::cerr << operation << " failed: " << static_cast<int>(status) << std::endl;
    return false;
}

// gpu_baseline.csv 的 (m, n) -> gpu_ms。文件缺失或对应行缺失时视为"无参考"，
// 只打印 ratio=NA，不参与退出码判定。
std::map<std::pair<int, int>, double> LoadGpuBaseline(const std::string& path)
{
    std::map<std::pair<int, int>, double> baseline;
    std::ifstream input(path);
    if (!input.is_open()) {
        std::cerr << "[PERF] WARN: cannot open " << path << ", GPU/NPU ratio disabled" << std::endl;
        return baseline;
    }
    std::string line;
    std::getline(input, line); // header: id,m,n,incx,incy,gpu_ms
    while (std::getline(input, line)) {
        std::istringstream row(line);
        std::vector<std::string> fields;
        std::string field;
        while (std::getline(row, field, ',')) {
            fields.push_back(field);
        }
        if (fields.size() < 6) {
            continue;
        }
        try {
            const int m = std::stoi(fields[1]);
            const int n = std::stoi(fields[2]);
            const double gpuMs = std::stod(fields[5]);
            if (gpuMs > 0.0) {
                baseline.emplace(std::make_pair(m, n), gpuMs);
            }
        } catch (const std::exception&) {
            continue;
        }
    }
    std::cerr << "[PERF] loaded gpu_baseline entries=" << baseline.size() << " from " << path << std::endl;
    return baseline;
}

bool InitializePerformanceResources(DeviceResources& resources)
{
    int maxM = 0;
    int maxN = 0;
    uint64_t maxAElements = 0;
    for (const PerformanceCase& item : kPerformanceCases) {
        maxM = std::max(maxM, item.m);
        maxN = std::max(maxN, item.n);
        maxAElements = std::max(maxAElements, static_cast<uint64_t>(item.m) * static_cast<uint64_t>(item.n));
    }

    if (!CheckAcl(aclrtCreateStream(&resources.stream), "aclrtCreateStream") ||
        !CheckBlas(aclblasCreate(&resources.handle), "aclblasCreate") ||
        !CheckBlas(aclblasSetStream(resources.handle, resources.stream), "aclblasSetStream")) {
        return false;
    }

    const size_t xBytes = static_cast<size_t>(maxM) * sizeof(aclblasComplex);
    const size_t yBytes = static_cast<size_t>(maxN) * sizeof(aclblasComplex);
    const size_t aBytes = static_cast<size_t>(maxAElements) * sizeof(aclblasComplex);
    if (!CheckAcl(aclrtMalloc(&resources.x, xBytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc(x)") ||
        !CheckAcl(aclrtMalloc(&resources.y, yBytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc(y)") ||
        !CheckAcl(aclrtMalloc(&resources.a, aBytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc(A)") ||
        !CheckAcl(aclrtCreateEvent(&resources.start), "aclrtCreateEvent(start)") ||
        !CheckAcl(aclrtCreateEvent(&resources.end), "aclrtCreateEvent(end)")) {
        return false;
    }

    std::vector<aclblasComplex> xHost(static_cast<size_t>(maxM));
    std::vector<aclblasComplex> yHost(static_cast<size_t>(maxN));
    for (size_t i = 0; i < xHost.size(); ++i) {
        xHost[i] = aclblasComplex{0.125f + static_cast<float>(i % 7U) * 0.03125f, -0.25f};
    }
    for (size_t i = 0; i < yHost.size(); ++i) {
        yHost[i] = aclblasComplex{-0.375f, 0.0625f + static_cast<float>(i % 5U) * 0.03125f};
    }
    if (!CheckAcl(aclrtMemcpy(resources.x, xBytes, xHost.data(), xBytes, ACL_MEMCPY_HOST_TO_DEVICE), "copy x") ||
        !CheckAcl(aclrtMemcpy(resources.y, yBytes, yHost.data(), yBytes, ACL_MEMCPY_HOST_TO_DEVICE), "copy y")) {
        return false;
    }
    return true;
}

bool LaunchPerformanceCase(const PerformanceCase& item, const DeviceResources& resources, const aclblasComplex& alpha)
{
    return CheckBlas(
        aclblasCgeru(
            resources.handle, item.m, item.n, &alpha, static_cast<const aclblasComplex*>(resources.x), 1,
            static_cast<const aclblasComplex*>(resources.y), 1, static_cast<aclblasComplex*>(resources.a), item.m),
        "aclblasCgeru");
}

bool MeasurePerformanceCase(
    const PerformanceCase& item, const DeviceResources& resources, const aclblasComplex& alpha, double& npuUs,
    double& wallUs)
{
    const size_t aBytes = static_cast<size_t>(item.m) * static_cast<size_t>(item.n) * sizeof(aclblasComplex);
    if (!CheckAcl(aclrtMemset(resources.a, aBytes, 0, aBytes), "clear A")) {
        return false;
    }
    for (int i = 0; i < PERF_WARMUP; ++i) {
        if (!LaunchPerformanceCase(item, resources, alpha)) {
            return false;
        }
    }
    if (!CheckAcl(aclrtSynchronizeStream(resources.stream), "warmup synchronize") ||
        !CheckAcl(aclrtRecordEvent(resources.start, resources.stream), "record start")) {
        return false;
    }

    const auto wallStart = std::chrono::steady_clock::now();
    for (int i = 0; i < PERF_ITERATIONS; ++i) {
        if (!LaunchPerformanceCase(item, resources, alpha)) {
            return false;
        }
    }
    if (!CheckAcl(aclrtRecordEvent(resources.end, resources.stream), "record end") ||
        !CheckAcl(aclrtSynchronizeEvent(resources.end), "synchronize end")) {
        return false;
    }
    const auto wallEnd = std::chrono::steady_clock::now();

    float elapsedMs = 0.0f;
    if (!CheckAcl(aclrtEventElapsedTime(&elapsedMs, resources.start, resources.end), "event elapsed")) {
        return false;
    }
    npuUs = static_cast<double>(elapsedMs) * 1000.0 / PERF_ITERATIONS;
    wallUs = std::chrono::duration<double, std::micro>(wallEnd - wallStart).count() / PERF_ITERATIONS;
    return true;
}

int RunPerformance(const std::map<std::pair<int, int>, double>& gpuBaseline)
{
    DeviceResources resources;
    if (!InitializePerformanceResources(resources)) {
        return 2;
    }

    const aclblasComplex alpha{0.5f, 0.25f};
    std::cout << "case,m,n,incx,incy,gpu_us,npu_us,gpu_npu_ratio,target_us,verdict,gate0.4,wall_us" << std::endl;
    int targetFailures = 0;
    int gateFailures = 0;
    for (const PerformanceCase& item : kPerformanceCases) {
        double npuUs = 0.0;
        double wallUs = 0.0;
        if (!MeasurePerformanceCase(item, resources, alpha, npuUs, wallUs)) {
            return 2;
        }

        const bool targetPassed = npuUs <= item.targetUs;
        targetFailures += targetPassed ? 0 : 1;

        std::string gpuText = "NA";
        std::string ratioText = "NA";
        std::string gateText = "NO_REF";
        const auto it = gpuBaseline.find(std::make_pair(item.m, item.n));
        if (it != gpuBaseline.end()) {
            const double gpuUs = it->second * 1000.0;
            const double ratio = gpuUs / npuUs;
            const bool gatePassed = ratio >= PERF_REQUIRED_RATIO;
            gateFailures += gatePassed ? 0 : 1;
            std::ostringstream gpuStream;
            gpuStream << std::fixed << std::setprecision(4) << gpuUs;
            gpuText = gpuStream.str();
            std::ostringstream ratioStream;
            ratioStream << std::fixed << std::setprecision(4) << ratio;
            ratioText = ratioStream.str();
            gateText = gatePassed ? "PASS" : "FAIL";
        }

        std::cout << item.caseName << ',' << item.m << ',' << item.n << ",1,1," << gpuText << ',' << std::fixed
                  << std::setprecision(4) << npuUs << ',' << ratioText << ',' << std::setprecision(2) << item.targetUs
                  << ',' << (targetPassed ? "PASS" : "FAIL") << ',' << gateText << ',' << std::setprecision(4)
                  << wallUs << std::endl;
    }

    const int total = static_cast<int>(sizeof(kPerformanceCases) / sizeof(kPerformanceCases[0]));
    std::cerr << "SUMMARY total=" << total << " target_pass=" << (total - targetFailures)
              << " target_fail=" << targetFailures << " gate0.4_fail=" << gateFailures << " warmup=" << PERF_WARMUP
              << " iterations=" << PERF_ITERATIONS << std::endl;
    return targetFailures == 0 ? 0 : 1;
}

std::string DefaultBaselinePath()
{
    // 先看当前目录（CTest 会把 CSV 拷到可执行文件目录），再回退到源码树相对路径。
    const std::vector<std::string> candidates = {
        "gpu_baseline.csv", "../../../../test/ger/cgeru/arch22/gpu_baseline.csv",
        "test/ger/cgeru/arch22/gpu_baseline.csv"};
    for (const std::string& candidate : candidates) {
        std::ifstream probe(candidate);
        if (probe.is_open()) {
            return candidate;
        }
    }
    return "gpu_baseline.csv";
}

} // namespace

int main(int argc, char** argv)
{
    std::string baselinePath;
    if (argc == 1) {
        baselinePath = DefaultBaselinePath();
    } else if (argc == 2) {
        baselinePath = argv[1];
    } else {
        std::cerr << "usage: cgeru_perf [gpu_baseline.csv]" << std::endl;
        return 2;
    }

    const std::map<std::pair<int, int>, double> gpuBaseline = LoadGpuBaseline(baselinePath);
    if (!CheckAcl(aclInit(nullptr), "aclInit") || !CheckAcl(aclrtSetDevice(0), "aclrtSetDevice")) {
        return 2;
    }
    const int result = RunPerformance(gpuBaseline);
    aclrtResetDevice(0);
    aclFinalize();
    if (result == 2) {
        std::cerr << "cgeru_perf: runtime error, no valid measurements" << std::endl;
    }
    return result;
}
