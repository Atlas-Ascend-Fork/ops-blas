/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file cgeru_perf_common.h
 * \brief Shared plumbing for the per-arch cgeru performance tools (issue #677:
 *        the arch22/arch35 copies previously duplicated ~226 lines; the two
 *        executables now differ only in their PERF_WARMUP / PERF_REQUIRED_RATIO
 *        tuning constants and the launch wrapper below).
 */

#pragma once

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "cann_ops_blas.h"

namespace cgeru_perf {

struct PerformanceCase {
    std::string caseName;
    int m = 0;
    int n = 0;
    int incx = 1;
    int incy = 1;
    double gpuMs = 0.0;
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
        if (x != nullptr) {
            aclrtFree(x);
        }
        if (y != nullptr) {
            aclrtFree(y);
        }
        if (a != nullptr) {
            aclrtFree(a);
        }
        if (handle != nullptr) {
            aclblasDestroy(handle);
        }
        if (stream != nullptr) {
            aclrtDestroyStream(stream);
        }
    }
};

inline bool CheckAcl(aclError status, const char* operation)
{
    if (status == ACL_SUCCESS) {
        return true;
    }
    std::cerr << operation << " failed: " << status << std::endl;
    return false;
}

inline bool CheckBlas(aclblasStatus_t status, const char* operation)
{
    if (status == ACLBLAS_STATUS_SUCCESS) {
        return true;
    }
    std::cerr << operation << " failed: " << static_cast<int>(status) << std::endl;
    return false;
}

inline uint64_t PhysicalLength(int logicalLength, int increment)
{
    const uint64_t stride =
        increment < 0 ? static_cast<uint64_t>(-static_cast<int64_t>(increment)) : static_cast<uint64_t>(increment);
    return 1U + static_cast<uint64_t>(logicalLength - 1) * stride;
}

inline std::vector<PerformanceCase> LoadCases(const char* path)
{
    std::ifstream input(path);
    std::vector<PerformanceCase> cases;
    std::string line;
    std::getline(input, line);
    size_t lineNo = 1;
    while (std::getline(input, line)) {
        lineNo++;
        std::istringstream row(line);
        std::vector<std::string> fields;
        std::string field;
        while (std::getline(row, field, ',')) {
            fields.push_back(field);
        }
        if (fields.size() != 6 || fields[5].empty()) {
            continue;
        }
        try {
            cases.push_back(PerformanceCase{
                fields[0], std::stoi(fields[1]), std::stoi(fields[2]), std::stoi(fields[3]), std::stoi(fields[4]),
                std::stod(fields[5])});
        } catch (const std::exception& e) {
            // issue #677: 畸形行（空值/非数字）报行号与原因后跳过，工具不终止
            std::cerr << "[cgeru_perf] " << path << ":" << lineNo << " malformed row skipped: " << e.what() << " | "
                      << line << std::endl;
        }
    }
    return cases;
}

inline bool InitializePerformanceResources(const std::vector<PerformanceCase>& cases, DeviceResources& resources)
{
    uint64_t maxXElements = 0;
    uint64_t maxYElements = 0;
    uint64_t maxAElements = 0;
    for (const PerformanceCase& item : cases) {
        maxXElements = std::max(maxXElements, PhysicalLength(item.m, item.incx));
        maxYElements = std::max(maxYElements, PhysicalLength(item.n, item.incy));
        maxAElements = std::max(maxAElements, static_cast<uint64_t>(item.m) * item.n);
    }

    if (!CheckAcl(aclrtCreateStream(&resources.stream), "aclrtCreateStream") ||
        !CheckBlas(aclblasCreate(&resources.handle), "aclblasCreate") ||
        !CheckBlas(aclblasSetStream(resources.handle, resources.stream), "aclblasSetStream")) {
        return false;
    }

    const size_t xBytes = static_cast<size_t>(maxXElements) * sizeof(aclblasComplex);
    const size_t yBytes = static_cast<size_t>(maxYElements) * sizeof(aclblasComplex);
    const size_t aBytes = static_cast<size_t>(maxAElements) * sizeof(aclblasComplex);
    if (!CheckAcl(aclrtMalloc(&resources.x, xBytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc(x)") ||
        !CheckAcl(aclrtMalloc(&resources.y, yBytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc(y)") ||
        !CheckAcl(aclrtMalloc(&resources.a, aBytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc(A)")) {
        return false;
    }

    std::vector<aclblasComplex> xHost(static_cast<size_t>(maxXElements));
    std::vector<aclblasComplex> yHost(static_cast<size_t>(maxYElements));
    for (size_t i = 0; i < xHost.size(); ++i) {
        xHost[i] = aclblasComplex{0.125f + static_cast<float>(i % 7U) * 0.03125f, -0.25f};
    }
    for (size_t i = 0; i < yHost.size(); ++i) {
        yHost[i] = aclblasComplex{-0.375f, 0.0625f + static_cast<float>(i % 5U) * 0.03125f};
    }
    if (!CheckAcl(aclrtMemcpy(resources.x, xBytes, xHost.data(), xBytes, ACL_MEMCPY_HOST_TO_DEVICE), "copy x") ||
        !CheckAcl(aclrtMemcpy(resources.y, yBytes, yHost.data(), yBytes, ACL_MEMCPY_HOST_TO_DEVICE), "copy y") ||
        !CheckAcl(aclrtCreateEvent(&resources.start), "create start event") ||
        !CheckAcl(aclrtCreateEvent(&resources.end), "create end event")) {
        return false;
    }
    return true;
}

inline int RunPerformance(
    const std::vector<PerformanceCase>& cases, int warmup, int iterations, double requiredRatio,
    bool (*launchCase)(const PerformanceCase&, const DeviceResources&, const aclblasComplex&, const char*))
{
    // issue #677 G.EXP.22 整改：除数守卫（iterations/requiredRatio 由调用方调优常量
    // 传入，防御性校验避免除零；npuUs 同理依赖 iterations 非零）
    if (iterations <= 0 || requiredRatio <= 0.0) {
        std::cerr << "[cgeru_perf] invalid tuning constants: iterations=" << iterations
                  << " requiredRatio=" << requiredRatio << std::endl;
        return 2;
    }

    DeviceResources resources;
    if (!InitializePerformanceResources(cases, resources)) {
        return 2;
    }

    const aclblasComplex alpha{0.5f, 0.25f};
    std::cout << "id,m,n,gpu_us,npu_us,ratio,threshold_us,verdict,wall_us" << std::endl;
    int failures = 0;
    for (const PerformanceCase& item : cases) {
        double npuUs = 0.0;
        double wallUs = 0.0;
        const size_t currentABytes = static_cast<size_t>(item.m) * item.n * sizeof(aclblasComplex);
        bool ok = CheckAcl(aclrtMemset(resources.a, currentABytes, 0, currentABytes), "clear A");
        for (int i = 0; ok && i < warmup; ++i) {
            ok = launchCase(item, resources, alpha, "warmup launch");
        }
        ok = ok && CheckAcl(aclrtSynchronizeStream(resources.stream), "warmup sync") &&
             CheckAcl(aclrtRecordEvent(resources.start, resources.stream), "record start");
        const auto wallStart = std::chrono::steady_clock::now();
        for (int i = 0; ok && i < iterations; ++i) {
            ok = launchCase(item, resources, alpha, "measured launch");
        }
        ok = ok && CheckAcl(aclrtRecordEvent(resources.end, resources.stream), "record end") &&
             CheckAcl(aclrtSynchronizeEvent(resources.end), "sync end");
        const auto wallEnd = std::chrono::steady_clock::now();
        float elapsedMs = 0.0f;
        ok = ok && CheckAcl(aclrtEventElapsedTime(&elapsedMs, resources.start, resources.end), "event elapsed");
        if (!ok) {
            return 2;
        }
        npuUs = elapsedMs * 1000.0 / iterations;
        wallUs = std::chrono::duration<double, std::micro>(wallEnd - wallStart).count() / iterations;

        const double gpuUs = item.gpuMs * 1000.0;
        const double ratio = npuUs > 0.0 ? gpuUs / npuUs : 0.0;
        const double thresholdUs = gpuUs / requiredRatio;
        const bool passed = ratio >= requiredRatio;
        failures += passed ? 0 : 1;
        std::cout << item.caseName << ',' << item.m << ',' << item.n << ',' << std::fixed << std::setprecision(4)
                  << gpuUs << ',' << npuUs << ',' << ratio << ',' << thresholdUs << ',' << (passed ? "PASS" : "FAIL")
                  << ',' << wallUs << std::endl;
    }

    std::cerr << "SUMMARY total=" << cases.size() << " pass=" << (cases.size() - failures) << " fail=" << failures
              << std::endl;
    return failures == 0 ? 0 : 1;
}

// arch 工具入口公共骨架：命令行解析、初始化、执行、清理（消除 arch22/arch35 的 main 重复）
inline int CgeruPerfMain(
    int argc, char** argv, int warmup, int iterations, double requiredRatio,
    bool (*launchCase)(const PerformanceCase&, const DeviceResources&, const aclblasComplex&, const char*))
{
    const char* baselinePath = argc == 2 ? argv[1] : "gpu_baseline.csv";
    if (argc > 2) {
        std::cerr << "usage: cgeru_perf [gpu_baseline.csv]" << std::endl;
        return 2;
    }
    const std::vector<PerformanceCase> cases = LoadCases(baselinePath);
    if (cases.empty()) {
        std::cerr << "no GPU baseline cases loaded from " << baselinePath << std::endl;
        return 2;
    }
    if (!CheckAcl(aclInit(nullptr), "aclInit") || !CheckAcl(aclrtSetDevice(0), "aclrtSetDevice")) {
        return 2;
    }
    const int result = RunPerformance(cases, warmup, iterations, requiredRatio, launchCase);
    aclrtResetDevice(0);
    aclFinalize();
    return result;
}

// cgeru 缩放下发 wrapper（aclblasCgeru 为 arch 无关 API，两份工具共用）
inline bool LaunchCgeruCase(
    const PerformanceCase& item, const DeviceResources& resources, const aclblasComplex& alpha, const char* operation)
{
    return CheckBlas(
        aclblasCgeru(
            resources.handle, item.m, item.n, &alpha, static_cast<const aclblasComplex*>(resources.x), item.incx,
            static_cast<const aclblasComplex*>(resources.y), item.incy, static_cast<aclblasComplex*>(resources.a),
            item.m),
        operation);
}

} // namespace cgeru_perf
