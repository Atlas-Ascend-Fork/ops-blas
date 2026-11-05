/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file ssbmv_perf.cpp
 * \brief aclblasSsbmv arch22 的性能门禁：事件计时的暖平均，对照任务书 §3.3 的 GPU 基线。
 *
 * 口径：同一流上 warmup 10 次后连续下发 64 次，用一对 event 卡住整段再除以次数。窗口含
 * 每次下发的 host 入队开销，是暖平均的可复现上界（msprof 的 kernel Task Duration 不含这部分，
 * 小 shape 上两者差距明显）。基线 ssbmv_perf_baseline.csv 由任务包的 gpu_baseline.csv 与
 * TC_PF 用例逐条配对生成，限值 = gpu_ms * 1000 / 0.8（任务书 §3.3「不高于标杆耗时」）。
 *
 * §3.3 的 4 组标杆（TC_PF_1001~1004）是硬门禁；其余 196 条只打印测量值。4 组标杆的实测余量
 * 在 2 倍以上，host 入队开销不影响判定；小 shape 的判定以 msprof 口径为准（见测试 README §4）。
 */

#include <climits>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "blas_test.h"
#include "cann_ops_blas.h"
#include "csv_loader.h"
#include "ssbmv_param.h"

namespace {

constexpr int SSBMV_PERF_WARMUP = 10;
constexpr int SSBMV_PERF_REPEATS = 64;
constexpr size_t SSBMV_PERF_CASES = 200;
constexpr double SSBMV_PERF_THRESHOLD = 0.8; /* 任务书 §3.3：gpu_ms / 0.8 = 标杆耗时 */

/*! 任务书 §3.3 的 4 组标杆。按用例名选定，因为基线里 n=4096/k=64/UPPER 有两条同参用例
 *   （TC_PF_1004 与 TC_PF_1079），按形状匹配会多门禁一条。 */
bool IsTaskBookCase(const std::string& caseName)
{
    return caseName == "TC_PF_1001" || caseName == "TC_PF_1002" || caseName == "TC_PF_1003" || caseName == "TC_PF_1004";
}

struct SsbmvPerfCase {
    std::string caseName;
    aclblasFillMode_t uplo = ACLBLAS_UPPER;
    int n = 0;
    int k = 0;
    int lda = 0;
    int incx = 1;
    int incy = 1;
    double gpuMs = 0.0;

    explicit SsbmvPerfCase(const csv_map& map)
    {
        caseName = ReadMap(map, "case_name");
        uplo = ParseUplo(map);
        n = parseInt(ReadMap(map, "n", "0"));
        k = parseInt(ReadMap(map, "k", "0"));
        /* k == INT_MAX 时 k + 1 在 int 域溢出，默认 lda 直接钳到 INT_MAX（与 ssbmv_param.h 同口径）。 */
        lda = parseInt(ReadMap(map, "lda", std::to_string(k >= INT_MAX ? INT_MAX : k + 1)));
        incx = parseInt(ReadMap(map, "incx", "1"));
        incy = parseInt(ReadMap(map, "incy", "1"));
        gpuMs = parseDouble(ReadMap(map, "gpu_ms", "0"));
    }
};

struct SsbmvPerfBuffers {
    void* a = nullptr;
    void* x = nullptr;
    void* y = nullptr;

    ~SsbmvPerfBuffers()
    {
        if (a != nullptr) {
            aclrtFree(a);
        }
        if (x != nullptr) {
            aclrtFree(x);
        }
        if (y != nullptr) {
            aclrtFree(y);
        }
    }
};

/*! \brief |stride| 的 64 位写法：int 取负在 INT_MIN 上是 UB。 */
uint64_t AbsStride(int stride)
{
    const int64_t value = stride;
    return static_cast<uint64_t>(value > 0 ? value : -value);
}

std::string PerfBaselinePath()
{
    std::string path(__FILE__);
    constexpr const char* sourceName = "ssbmv_perf.cpp";
    const size_t pos = path.rfind(sourceName);
    if (pos == std::string::npos) {
        throw std::runtime_error("Cannot derive ssbmv_perf_baseline.csv from __FILE__");
    }
    path.replace(pos, std::char_traits<char>::length(sourceName), "ssbmv_perf_baseline.csv");
    return path;
}

bool MatchesCaseFilter(const std::string& caseName)
{
    const char* value = std::getenv("SSBMV_PERF_CASE");
    return value == nullptr || *value == '\0' || caseName == value;
}

template <typename Launch>
bool MeasureAverageUs(aclrtStream stream, Launch&& launch, float& averageUs)
{
    for (int i = 0; i < SSBMV_PERF_WARMUP; ++i) {
        if (!launch()) {
            return false;
        }
    }
    if (aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
        return false;
    }

    aclrtEvent start = nullptr;
    aclrtEvent stop = nullptr;
    if (aclrtCreateEvent(&start) != ACL_SUCCESS || aclrtCreateEvent(&stop) != ACL_SUCCESS ||
        aclrtRecordEvent(start, stream) != ACL_SUCCESS) {
        if (start != nullptr) {
            aclrtDestroyEvent(start);
        }
        if (stop != nullptr) {
            aclrtDestroyEvent(stop);
        }
        return false;
    }
    for (int i = 0; i < SSBMV_PERF_REPEATS; ++i) {
        if (!launch()) {
            aclrtDestroyEvent(start);
            aclrtDestroyEvent(stop);
            return false;
        }
    }
    const bool recorded =
        (aclrtRecordEvent(stop, stream) == ACL_SUCCESS) && (aclrtSynchronizeEvent(stop) == ACL_SUCCESS);
    float elapsedMs = 0.0F;
    const bool timed = recorded && (aclrtEventElapsedTime(&elapsedMs, start, stop) == ACL_SUCCESS);
    aclrtDestroyEvent(start);
    aclrtDestroyEvent(stop);
    if (!timed) {
        return false;
    }
    averageUs = elapsedMs * 1000.0F / static_cast<float>(SSBMV_PERF_REPEATS);
    return averageUs > 0.0F;
}

void PrintPerformance(const SsbmvPerfCase& testCase, float averageUs, double limitUs)
{
    const double ratio = (averageUs > 0.0F) ? (limitUs / averageUs) : 0.0;
    std::cout << "[SSBMV_PERF] case=" << testCase.caseName << " n=" << testCase.n << " k=" << testCase.k
              << " avg_us=" << std::fixed << std::setprecision(3) << averageUs << " limit_us=" << std::setprecision(3)
              << limitUs << " ratio=" << ratio;
    if (IsTaskBookCase(testCase.caseName)) {
        std::cout << " verdict=" << ((averageUs <= limitUs) ? "PASS" : "FAIL");
    } else {
        std::cout << " verdict=INFO";
    }
    std::cout << std::endl;
}

/*! \brief 为一个用例分配设备缓冲并测暖平均；分配、清零或下发失败时返回 false。 */
bool MeasureCase(const SsbmvPerfCase& testCase, aclblasHandle_t handle, aclrtStream stream, float& averageUs)
{
    SsbmvPerfBuffers buffers;
    const size_t aBytes = static_cast<size_t>(testCase.lda) * static_cast<size_t>(testCase.n) * sizeof(float);
    const uint64_t xCount = (static_cast<uint64_t>(testCase.n) - 1U) * AbsStride(testCase.incx) + 1U;
    const uint64_t yCount = (static_cast<uint64_t>(testCase.n) - 1U) * AbsStride(testCase.incy) + 1U;
    const size_t xBytes = static_cast<size_t>(xCount) * sizeof(float);
    const size_t yBytes = static_cast<size_t>(yCount) * sizeof(float);
    if (aclrtMalloc(&buffers.a, aBytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS ||
        aclrtMalloc(&buffers.x, xBytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS ||
        aclrtMalloc(&buffers.y, yBytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS ||
        aclrtMemset(buffers.a, aBytes, 0, aBytes) != ACL_SUCCESS ||
        aclrtMemset(buffers.x, xBytes, 0, xBytes) != ACL_SUCCESS ||
        aclrtMemset(buffers.y, yBytes, 0, yBytes) != ACL_SUCCESS) {
        return false;
    }

    float alpha = 1.0f;
    float beta = 1.0f;
    return MeasureAverageUs(
        stream,
        [&]() {
            const aclblasStatus_t status = aclblasSsbmv(
                handle, testCase.uplo, testCase.n, testCase.k, &alpha, static_cast<const float*>(buffers.a),
                testCase.lda, static_cast<const float*>(buffers.x), testCase.incx, &beta,
                static_cast<float*>(buffers.y), testCase.incy);
            if (status != ACLBLAS_STATUS_SUCCESS) {
                std::cerr << "[SSBMV_PERF] " << testCase.caseName << " status=" << static_cast<int>(status)
                          << std::endl;
            }
            return status == ACLBLAS_STATUS_SUCCESS;
        },
        averageUs);
}

} // namespace

class SsbmvPerf : public BlasTest<SsbmvParam> {};

/*!
 * \brief 对基线里的每一条 TC_PF 用例测暖平均，并对任务书 §3.3 的 4 组标杆断言达标。
 *
 * 只测一条：SSBMV_PERF_CASE=TC_PF_1001 ./ssbmv_perf。
 */
TEST_F(SsbmvPerf, TaskBookAcceptance)
{
    const auto cases = GetCasesFromCsv<SsbmvPerfCase>(PerfBaselinePath());
    ASSERT_EQ(cases.size(), SSBMV_PERF_CASES) << "perf baseline must cover the 200 TC_PF cases";

    size_t measured = 0;
    size_t gated = 0;
    size_t gateFailed = 0;
    for (const auto& testCase : cases) {
        if (!MatchesCaseFilter(testCase.caseName)) {
            continue;
        }
        ++measured;
        ASSERT_GT(testCase.n, 0) << testCase.caseName;
        ASSERT_GT(testCase.gpuMs, 0.0) << testCase.caseName;

        float averageUs = 0.0F;
        ASSERT_TRUE(MeasureCase(testCase, SsbmvPerf::handle_, SsbmvPerf::stream_, averageUs)) << testCase.caseName;

        const double limitUs = testCase.gpuMs * 1000.0 / SSBMV_PERF_THRESHOLD;
        PrintPerformance(testCase, averageUs, limitUs);
        if (IsTaskBookCase(testCase.caseName)) {
            ++gated;
            if (averageUs > limitUs) {
                ++gateFailed;
                ADD_FAILURE() << testCase.caseName << ": avg_us=" << averageUs << " > limit_us=" << limitUs;
            }
        }
    }

    ASSERT_GT(measured, 0U) << "SSBMV_PERF_CASE did not match a baseline case";
    if (std::getenv("SSBMV_PERF_CASE") == nullptr) {
        ASSERT_EQ(measured, SSBMV_PERF_CASES);
        ASSERT_EQ(gated, 4U) << "the four task-book §3.3 cases must all be measured";
    }
    EXPECT_EQ(gateFailed, 0U);
}
