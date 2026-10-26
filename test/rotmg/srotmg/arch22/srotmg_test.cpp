/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <chrono>
#include <cstdlib>
#include <vector>

#include "verify.h"
#include "blas_test.h"
#include "csv_loader.h"
#include "srotmg_param.h"
#include "srotmg_golden.h"
#include "srotmg_npu_wrapper.h"

// arch22 test param: adds the perf-loop count carried by the "iters" CSV column
// (accuracy cases keep the default 1 and ignore it), plus "null" markers for the
// null-pointer negative cases (TC_ED_*): a literal "null" cell injects nullptr.
struct SrotmgArch22Param : public SrotmgParam {
    int iters = 1;
    bool d1Null = false;
    bool d2Null = false;
    bool x1Null = false;
    bool y1Null = false;
    bool paramNull = false;

    SrotmgArch22Param(const csv_map& map) : SrotmgParam(map)
    {
        iters = parseInt(ReadMap(map, "iters", "1"));
        if (iters < 1) {
            iters = 1;
        }
        d1Null = ReadMap(map, "d1", "") == "null";
        d2Null = ReadMap(map, "d2", "") == "null";
        x1Null = ReadMap(map, "x1", "") == "null";
        y1Null = ReadMap(map, "y1", "") == "null";
        paramNull = ReadMap(map, "param", "") == "null";
    }
};

class SrotmgArch22Test : public BlasTest<SrotmgArch22Param> {};

// Compare the eight output scalars (d1/d2/x1 + param[0..4]) against golden.
static void SrotmgVerifyOutputs(
    float d1, float d2, float x1, const float* param, float d1G, float d2G, float x1G, const float* paramG,
    const VerifyConfig& cfg, const std::string& caseName)
{
    EXPECT_TRUE(Verifier::verifyScalar(d1, d1G, cfg, caseName + "_d1"));
    EXPECT_TRUE(Verifier::verifyScalar(d2, d2G, cfg, caseName + "_d2"));
    EXPECT_TRUE(Verifier::verifyScalar(x1, x1G, cfg, caseName + "_x1"));
    EXPECT_TRUE(Verifier::verifyScalar(param[0], paramG[0], cfg, caseName + "_param_sflag"));
    EXPECT_TRUE(Verifier::verifyScalar(param[1], paramG[1], cfg, caseName + "_param_h11"));
    EXPECT_TRUE(Verifier::verifyScalar(param[2], paramG[2], cfg, caseName + "_param_h21"));
    EXPECT_TRUE(Verifier::verifyScalar(param[3], paramG[3], cfg, caseName + "_param_h12"));
    EXPECT_TRUE(Verifier::verifyScalar(param[4], paramG[4], cfg, caseName + "_param_h22"));
}

static bool IsPerfCase(const std::string& name) { return name.rfind("TC_PF_", 0) == 0; }

// Null handle error path test (not in CSV)
TEST_F(SrotmgArch22Test, NullHandle)
{
    float d1 = 1.0f;
    float d2 = 2.0f;
    float x1 = 3.0f;
    float y1 = 4.0f;
    float param[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    aclblasStatus_t ret = aclblasSrotmg_npu(nullptr, &d1, &d2, &x1, &y1, param);
    EXPECT_EQ(ret, ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
}

INSTANTIATE_TEST_SUITE_P(
    Srotmg, SrotmgArch22Test,
    ::testing::ValuesIn(GetCasesFromCsv<SrotmgArch22Param>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<SrotmgArch22Param>);

// Timed median of ROUNDS independent rounds of continuous same-stream calls.
// Long batch runs occasionally hit transient SoC clock/power states that
// inflate a single round by ~1us; the median filters that out.
template <typename CallT>
static double SrotmgTimedMedian(CallT& call, int samples)
{
    const int safeSamples = (samples > 0) ? samples : 1;
    constexpr int ROUNDS = 3;
    double roundAvgs[ROUNDS];
    for (int r = 0; r < ROUNDS; r++) {
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < samples; i++) {
            if (call() != ACLBLAS_STATUS_SUCCESS) {
                return -1.0;
            }
        }
        aclrtSynchronizeDevice();
        auto t1 = std::chrono::steady_clock::now();
        roundAvgs[r] = std::chrono::duration<double, std::micro>(t1 - t0).count() / safeSamples;
    }
    double a = roundAvgs[0];
    double b = roundAvgs[1];
    double c = roundAvgs[2];
    return (a > b) ? ((b > c) ? b : ((a > c) ? c : a)) : ((a > c) ? a : ((b > c) ? c : b));
}

// ==========================================================================
// Perf path (TC_PF_*): fixed-cost / batch-throughput measurement.
// The five scalars live in one packed device buffer so each iteration resets
// d1/d2/x1/y1 with a single 16-byte async H2D copy on the same stream, then
// calls aclblasSrotmg on device pointers. Average per-call latency is taken
// over >50 timed iterations after warmup, per task requirements.
// ==========================================================================
static void SrotmgPerfDevice(aclblasHandle_t handle, const SrotmgArch22Param& p, double* avgUsOut)
{
    constexpr int WARMUP = 50;
    constexpr int MIN_SAMPLES = 1000;

    float init[4] = {p.d1, p.d2, p.x1, p.y1};
    float* devBuf = nullptr; // [d1, d2, x1, y1, param[0..4]]
    ASSERT_EQ(
        aclrtMalloc(reinterpret_cast<void**>(&devBuf), 9 * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    ASSERT_EQ(aclrtMemcpy(devBuf, 4 * sizeof(float), init, 4 * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);

    auto call = [&]() { return aclblasSrotmg(handle, devBuf + 0, devBuf + 1, devBuf + 2, devBuf + 3, devBuf + 4); };
    auto resetInput = [&]() {
        // issue #613: 重置拷贝失败会使正确性检查基于错误输入，须显式失败
        ASSERT_EQ(
            aclrtMemcpy(devBuf, 4 * sizeof(float), init, 4 * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);
    };

    // golden (CPU cblas) on the same inputs
    float d1G = p.d1;
    float d2G = p.d2;
    float x1G = p.x1;
    float y1G = p.y1;
    float paramG[5] = {};
    ASSERT_EQ(aclblasSrotmg_cpu(handle, &d1G, &d2G, &x1G, &y1G, paramG), ACLBLAS_STATUS_SUCCESS);

    for (int i = 0; i < WARMUP; i++) {
        resetInput();
        ASSERT_EQ(call(), ACLBLAS_STATUS_SUCCESS);
    }
    ASSERT_EQ(aclrtSynchronizeDevice(), ACL_SUCCESS);

    // Timed loop: continuous same-stream calls only. Previous outputs feed the
    // next iteration in-place (QR-iteration usage model), so the fixed launch
    // overhead is what is measured — no per-call reset memcpy inside the loop.
    // Every case takes at least MIN_SAMPLES timed calls: small-iters cases
    // (5/10/20) otherwise average in the first-launch warm-up spike and the
    // trailing sync cost, skewing the per-call latency high.
    // Each case is timed over ROUNDS independent rounds and the median round
    // average is reported: long batch runs occasionally hit transient SoC
    // clock/power states that inflate a single round by ~1us.
    const int samples = p.iters > MIN_SAMPLES ? p.iters : MIN_SAMPLES;
    double avgUs = SrotmgTimedMedian(call, samples);
    // issue #613: 计时失败哨兵值 -1.0 必须使用例 FAIL，不得静默记为性能数据
    ASSERT_GE(avgUs, 0.0) << "srotmg perf timing failed (sentinel -1.0)";
    if (avgUsOut != nullptr) {
        *avgUsOut = avgUs;
    }

    // correctness check on a fresh single call from the original inputs
    resetInput();
    ASSERT_EQ(call(), ACLBLAS_STATUS_SUCCESS);
    ASSERT_EQ(aclrtSynchronizeDevice(), ACL_SUCCESS);

    // correctness of the last iteration (device path result)
    float out[9] = {};
    ASSERT_EQ(aclrtMemcpy(out, 9 * sizeof(float), devBuf, 9 * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST), ACL_SUCCESS);

    printf("[PERF-DEV] %s iters=%d avg=%.3f us\n", p.caseName.c_str(), samples, avgUs);

    VerifyConfig cfg;
    cfg.mode = PrecisionMode::EXACT;
    EXPECT_TRUE(Verifier::verifyScalar(out[0], d1G, cfg, p.caseName + "_perf_d1"));
    EXPECT_TRUE(Verifier::verifyScalar(out[1], d2G, cfg, p.caseName + "_perf_d2"));
    EXPECT_TRUE(Verifier::verifyScalar(out[2], x1G, cfg, p.caseName + "_perf_x1"));
    EXPECT_TRUE(Verifier::verifyScalar(out[4 + 0], paramG[0], cfg, p.caseName + "_perf_sflag"));
    EXPECT_TRUE(Verifier::verifyScalar(out[5], paramG[1], cfg, p.caseName + "_perf_param_h11"));
    EXPECT_TRUE(Verifier::verifyScalar(out[6], paramG[2], cfg, p.caseName + "_perf_param_h21"));
    EXPECT_TRUE(Verifier::verifyScalar(out[7], paramG[3], cfg, p.caseName + "_perf_param_h12"));
    EXPECT_TRUE(Verifier::verifyScalar(out[8], paramG[4], cfg, p.caseName + "_perf_param_h22"));

    aclrtFree(devBuf);
}

// CSV-driven: all-device path (via NPU wrapper)
TEST_P(SrotmgArch22Test, CsvDrivenDevice)
{
    const auto& p = GetParam();

    if (IsPerfCase(p.caseName)) {
        double avgUs = 0.0;
        SrotmgPerfDevice(handle_, p, &avgUs);
        RecordProperty("avg_us", avgUs);
        return;
    }

    float d1Npu = p.d1;
    float d2Npu = p.d2;
    float x1Npu = p.x1;
    float y1Val = p.y1;
    float paramNpu[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    // Golden
    float d1Golden = p.d1;
    float d2Golden = p.d2;
    float x1Golden = p.x1;
    float y1Golden = p.y1;
    float paramGolden[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    EXPECT_EQ(
        aclblasSrotmg_cpu(handle_, &d1Golden, &d2Golden, &x1Golden, &y1Golden, paramGolden), ACLBLAS_STATUS_SUCCESS);

    aclblasStatus_t ret = aclblasSrotmg_npu(
        handle_, p.d1Null ? nullptr : &d1Npu, p.d2Null ? nullptr : &d2Npu, p.x1Null ? nullptr : &x1Npu,
        p.y1Null ? nullptr : &y1Val, p.paramNull ? nullptr : paramNpu);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
    if (p.expectResult != ACLBLAS_STATUS_SUCCESS) {
        return;
    }

    VerifyConfig cfg;
    applyMixedTolerance(cfg, ACL_FLOAT, d1Golden);

    EXPECT_TRUE(Verifier::verifyScalar(d1Npu, d1Golden, cfg, p.caseName + "_d1"));
    EXPECT_TRUE(Verifier::verifyScalar(d2Npu, d2Golden, cfg, p.caseName + "_d2"));
    EXPECT_TRUE(Verifier::verifyScalar(x1Npu, x1Golden, cfg, p.caseName + "_x1"));
    EXPECT_TRUE(Verifier::verifyScalar(paramNpu[0], paramGolden[0], cfg, p.caseName + "_param_sflag"));
    EXPECT_TRUE(Verifier::verifyScalar(paramNpu[1], paramGolden[1], cfg, p.caseName + "_param_h11"));
    EXPECT_TRUE(Verifier::verifyScalar(paramNpu[2], paramGolden[2], cfg, p.caseName + "_param_h21"));
    EXPECT_TRUE(Verifier::verifyScalar(paramNpu[3], paramGolden[3], cfg, p.caseName + "_param_h12"));
    EXPECT_TRUE(Verifier::verifyScalar(paramNpu[4], paramGolden[4], cfg, p.caseName + "_param_h22"));
}

// CSV-driven: all-host path (aclrtMallocHost pinned memory, direct call)
// Allocate the five pinned host scalars for the CSV-driven host path.
static void SrotmgAllocHostBuffers(float** d1, float** d2, float** x1, float** y1, float** param)
{
    ASSERT_EQ(aclrtMallocHost(reinterpret_cast<void**>(d1), sizeof(float)), ACL_SUCCESS);
    ASSERT_EQ(aclrtMallocHost(reinterpret_cast<void**>(d2), sizeof(float)), ACL_SUCCESS);
    ASSERT_EQ(aclrtMallocHost(reinterpret_cast<void**>(x1), sizeof(float)), ACL_SUCCESS);
    ASSERT_EQ(aclrtMallocHost(reinterpret_cast<void**>(y1), sizeof(float)), ACL_SUCCESS);
    ASSERT_EQ(aclrtMallocHost(reinterpret_cast<void**>(param), 5 * sizeof(float)), ACL_SUCCESS);
}

// Release the five pinned host scalars.
static void SrotmgFreeHostBuffers(float* d1, float* d2, float* x1, float* y1, float* param)
{
    if (d1) {
        aclrtFreeHost(d1);
    }
    if (d2) {
        aclrtFreeHost(d2);
    }
    if (x1) {
        aclrtFreeHost(x1);
    }
    if (y1) {
        aclrtFreeHost(y1);
    }
    if (param) {
        aclrtFreeHost(param);
    }
}

// CSV-driven host path: accuracy case body (pinned host memory, direct call).
static void SrotmgHostAccuracyCase(aclblasHandle_t handle, const SrotmgArch22Param& p)
{
    float* d1 = nullptr;
    float* d2 = nullptr;
    float* x1 = nullptr;
    float* y1 = nullptr;
    float* param = nullptr;
    SrotmgAllocHostBuffers(&d1, &d2, &x1, &y1, &param);

    *d1 = p.d1;
    *d2 = p.d2;
    *x1 = p.x1;
    *y1 = p.y1;

    // Golden
    float d1G = p.d1;
    float d2G = p.d2;
    float x1G = p.x1;
    float y1G = p.y1;
    float paramG[5] = {};
    ASSERT_EQ(aclblasSrotmg_cpu(handle, &d1G, &d2G, &x1G, &y1G, paramG), ACLBLAS_STATUS_SUCCESS);

    aclblasStatus_t ret = aclblasSrotmg(
        handle, p.d1Null ? nullptr : d1, p.d2Null ? nullptr : d2, p.x1Null ? nullptr : x1, p.y1Null ? nullptr : y1,
        p.paramNull ? nullptr : param);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));

    if (ret == ACLBLAS_STATUS_SUCCESS) {
        VerifyConfig cfg;
        if (p.mereThreshold > 0.0) {
            cfg.mode = PrecisionMode::MERE_MARE;
            cfg.mereThreshold = p.mereThreshold;
            cfg.mareMultiplier = p.mareMultiplier;
        } else {
            cfg.mode = PrecisionMode::EXACT;
        }
        SrotmgVerifyOutputs(*d1, *d2, *x1, param, d1G, d2G, x1G, paramG, cfg, p.caseName);
    }

    SrotmgFreeHostBuffers(d1, d2, x1, y1, param);
}

// CSV-driven host path: perf case body (plain stack scalars, direct call).
static void SrotmgHostPerfCase(aclblasHandle_t handle, const SrotmgArch22Param& p)
{
    constexpr int WARMUP = 20;
    constexpr int SAMPLES = 1000;
    float init[3] = {p.d1, p.d2, p.x1};
    float d1 = init[0];
    float d2 = init[1];
    float x1 = init[2];
    float y1 = p.y1;
    float param[5] = {};
    for (int i = 0; i < WARMUP; i++) {
        d1 = init[0];
        d2 = init[1];
        x1 = init[2];
        aclblasSrotmg(handle, &d1, &d2, &x1, &y1, param);
    }
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < SAMPLES; i++) {
        d1 = init[0];
        d2 = init[1];
        x1 = init[2];
        aclblasSrotmg(handle, &d1, &d2, &x1, &y1, param);
    }
    auto t1 = std::chrono::steady_clock::now();
    double avgUs = std::chrono::duration<double, std::micro>(t1 - t0).count() / SAMPLES;
    printf("[PERF-HOST] %s iters=%d avg=%.3f us\n", p.caseName.c_str(), SAMPLES, avgUs);
}

TEST_P(SrotmgArch22Test, CsvDrivenHost)
{
    const auto& p = GetParam();
    if (IsPerfCase(p.caseName)) {
        SrotmgHostPerfCase(handle_, p);
        return;
    }
    SrotmgHostAccuracyCase(handle_, p);
}
