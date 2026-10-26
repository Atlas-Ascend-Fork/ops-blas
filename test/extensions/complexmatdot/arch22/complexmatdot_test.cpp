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
 * \file complexmatdot_test.cpp
 * \brief Test for complex matrix dot product
 */

#include <cstdint>
#include <iostream>
#include <vector>
#include <algorithm>
#include <iterator>
#include <cmath>
#include "acl/acl.h"
#include "cann_ops_blas.h"

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

// ── Test data ────────────────────────────────────────────────────────────────

constexpr uint32_t M = 4;
constexpr uint32_t N = 4;
constexpr uint32_t COMPLEX_SIZE = M * N * 2;

struct ComplexMatDotTestData {
    std::vector<float> matx;
    std::vector<float> maty;
    std::vector<float> result;
};

static ComplexMatDotTestData BuildComplexMatDotInput()
{
    ComplexMatDotTestData data;
    data.matx.resize(COMPLEX_SIZE);
    data.maty.resize(COMPLEX_SIZE);
    data.result.assign(COMPLEX_SIZE, 0.0f);

    for (uint32_t i = 0; i < M * N; i++) {
        data.matx[i * 2] = 1.0f;
        data.matx[i * 2 + 1] = 2.0f;
        data.maty[i * 2] = 3.0f;
        data.maty[i * 2 + 1] = 4.0f;
    }
    return data;
}

static std::vector<float> BuildComplexMatDotGolden(const ComplexMatDotTestData& data)
{
    std::vector<float> golden(COMPLEX_SIZE);
    for (uint32_t i = 0; i < M * N; i++) {
        golden[i * 2] = data.matx[i * 2] * data.maty[i * 2] - data.matx[i * 2 + 1] * data.maty[i * 2 + 1];
        golden[i * 2 + 1] = data.matx[i * 2] * data.maty[i * 2 + 1] + data.matx[i * 2 + 1] * data.maty[i * 2];
    }
    return golden;
}

// ── Test runtime context with RAII cleanup ───────────────────────────────────

struct ComplexMatDotTestContext {
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    aclblasHandle handle = nullptr;
    aclblasComplex* matxDevice = nullptr;
    aclblasComplex* matyDevice = nullptr;
    aclblasComplex* resultDevice = nullptr;
};

static void CleanupTestRuntime(ComplexMatDotTestContext& ctx)
{
    if (ctx.resultDevice != nullptr) {
        aclrtFree(ctx.resultDevice);
        ctx.resultDevice = nullptr;
    }
    if (ctx.matyDevice != nullptr) {
        aclrtFree(ctx.matyDevice);
        ctx.matyDevice = nullptr;
    }
    if (ctx.matxDevice != nullptr) {
        aclrtFree(ctx.matxDevice);
        ctx.matxDevice = nullptr;
    }
    if (ctx.handle != nullptr) {
        aclblasDestroy(ctx.handle);
        ctx.handle = nullptr;
    }
    if (ctx.stream != nullptr) {
        aclrtDestroyStream(ctx.stream);
        ctx.stream = nullptr;
    }
    aclrtResetDevice(ctx.deviceId);
    aclFinalize();
}

static int32_t InitTestRuntime(ComplexMatDotTestContext& ctx)
{
    aclInit(nullptr);
    aclrtSetDevice(ctx.deviceId);
    aclrtCreateStream(&ctx.stream);
    aclblasCreate(&ctx.handle);
    aclblasSetStream(ctx.handle, ctx.stream);
    return 0;
}

// ── Run operator on NPU ──────────────────────────────────────────────────────

static int32_t RunComplexMatDot(ComplexMatDotTestContext& ctx, const ComplexMatDotTestData& data)
{
    size_t dataSize = COMPLEX_SIZE * sizeof(float);

    aclError aclRet = aclrtMalloc((void**)&ctx.matxDevice, dataSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclRet != ACL_SUCCESS) {
        LOG_PRINT("aclrtMalloc matxDevice failed. ERROR: %d\n", aclRet);
        return static_cast<int32_t>(aclRet);
    }
    aclRet = aclrtMalloc((void**)&ctx.matyDevice, dataSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclRet != ACL_SUCCESS) {
        LOG_PRINT("aclrtMalloc matyDevice failed. ERROR: %d\n", aclRet);
        return static_cast<int32_t>(aclRet);
    }
    aclRet = aclrtMalloc((void**)&ctx.resultDevice, dataSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclRet != ACL_SUCCESS) {
        LOG_PRINT("aclrtMalloc resultDevice failed. ERROR: %d\n", aclRet);
        return static_cast<int32_t>(aclRet);
    }

    aclRet = aclrtMemcpy(ctx.matxDevice, dataSize, data.matx.data(), dataSize, ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_SUCCESS) {
        LOG_PRINT("aclrtMemcpy matxDevice failed. ERROR: %d\n", aclRet);
        return static_cast<int32_t>(aclRet);
    }
    aclRet = aclrtMemcpy(ctx.matyDevice, dataSize, data.maty.data(), dataSize, ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_SUCCESS) {
        LOG_PRINT("aclrtMemcpy matyDevice failed. ERROR: %d\n", aclRet);
        return static_cast<int32_t>(aclRet);
    }
    aclRet = aclrtMemcpy(ctx.resultDevice, dataSize, data.result.data(), dataSize, ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_SUCCESS) {
        LOG_PRINT("aclrtMemcpy resultDevice failed. ERROR: %d\n", aclRet);
        return static_cast<int32_t>(aclRet);
    }

    auto ret = aclblasComplexMatDot(ctx.handle, M, N, ctx.matxDevice, ctx.matyDevice, ctx.resultDevice);
    if (ret != ACLBLAS_STATUS_SUCCESS) {
        LOG_PRINT("aclblasComplexMatDot failed. ERROR: %d\n", ret);
        return static_cast<int32_t>(ret);
    }

    aclRet = aclrtSynchronizeStream(ctx.stream);
    if (aclRet != ACL_SUCCESS) {
        LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet);
        return static_cast<int32_t>(aclRet);
    }

    return 0;
}

// ── Verify ───────────────────────────────────────────────────────────────────

static uint32_t VerifyResult(const std::vector<float>& output, const std::vector<float>& golden)
{
    if (output.size() != golden.size()) {
        std::cout << "[Failed] Size mismatch: output=" << output.size() << " golden=" << golden.size() << std::endl;
        return 1;
    }

    auto printTensor = [](const std::vector<float>& tensor, const char* name) {
        constexpr size_t maxPrintSize = 20;
        std::cout << name << ": ";
        std::copy(
            tensor.begin(), tensor.begin() + std::min(tensor.size(), maxPrintSize),
            std::ostream_iterator<float>(std::cout, " "));
        if (tensor.size() > maxPrintSize) {
            std::cout << "...";
        }
        std::cout << std::endl;
    };
    printTensor(output, "Output");
    printTensor(golden, "Golden");

    constexpr float epsilon = 1e-5f;
    size_t errorCount = 0;
    for (size_t i = 0; i < output.size(); i++) {
        float relError = std::abs(output[i] - golden[i]) / (std::abs(golden[i]) + 1e-10f);
        if (relError > epsilon) {
            errorCount++;
        }
    }

    if (errorCount == 0) {
        std::cout << "[Success] Case accuracy is verification passed." << std::endl;
        return 0;
    }
    std::cout << "[Failed] Case accuracy is verification failed! Error count: " << errorCount << std::endl;
    return 1;
}

// issue #581: handle 为空与非法 m/n 须返回对应错误码，而非解引用崩溃或按回绕后的形状计算
static int32_t TestComplexMatDotNegative()
{
    std::cout << "Testing complexmatdot negative cases:" << std::endl;
    int failed = 0;
    aclblasComplex dummy{};

    int32_t deviceId = 0;
    aclInit(nullptr);
    aclrtSetDevice(deviceId);
    aclblasHandle_t handle = nullptr;
    if (aclblasCreate(&handle) != ACLBLAS_STATUS_SUCCESS) {
        std::cout << "[Failed] aclblasCreate failed" << std::endl;
        return 1;
    }

    // handle 为空：期望 HANDLE_IS_NULLPTR
    auto ret = aclblasComplexMatDot(nullptr, 8, 8, &dummy, &dummy, &dummy);
    if (ret != ACLBLAS_STATUS_HANDLE_IS_NULLPTR) {
        std::cout << "[Failed] null handle: expected " << ACLBLAS_STATUS_HANDLE_IS_NULLPTR << " got " << ret
                  << std::endl;
        ++failed;
    }
    // 负维度 / 零维度
    ret = aclblasComplexMatDot(handle, -1, 8, &dummy, &dummy, &dummy);
    if (ret != ACLBLAS_STATUS_INVALID_VALUE) {
        std::cout << "[Failed] negative m: expected " << ACLBLAS_STATUS_INVALID_VALUE << " got " << ret << std::endl;
        ++failed;
    }
    ret = aclblasComplexMatDot(handle, 8, 0, &dummy, &dummy, &dummy);
    if (ret != ACLBLAS_STATUS_INVALID_VALUE) {
        std::cout << "[Failed] zero n: expected " << ACLBLAS_STATUS_INVALID_VALUE << " got " << ret << std::endl;
        ++failed;
    }
    // m*n 溢出 uint32（65536 * 65536 = 2^32）
    ret = aclblasComplexMatDot(handle, 65536, 65536, &dummy, &dummy, &dummy);
    if (ret != ACLBLAS_STATUS_INVALID_VALUE) {
        std::cout << "[Failed] m*n overflow: expected " << ACLBLAS_STATUS_INVALID_VALUE << " got " << ret << std::endl;
        ++failed;
    }
    aclblasDestroy(handle);
    aclrtResetDevice(deviceId);
    aclFinalize();

    if (failed == 0) {
        std::cout << "[Success] complexmatdot negative cases passed." << std::endl;
    }
    return failed == 0 ? 0 : 1;
}

// ── Main — orchestration only ────────────────────────────────────────────────

int32_t main()
{
    int32_t negativeStatus = TestComplexMatDotNegative();
    if (negativeStatus != 0) {
        return negativeStatus;
    }

    ComplexMatDotTestData data = BuildComplexMatDotInput();

    ComplexMatDotTestContext ctx;
    InitTestRuntime(ctx);

    int32_t runStatus = RunComplexMatDot(ctx, data);
    if (runStatus != 0) {
        CleanupTestRuntime(ctx);
        return runStatus;
    }

    size_t dataSize = COMPLEX_SIZE * sizeof(float);
    aclError aclRet = aclrtMemcpy(data.result.data(), dataSize, ctx.resultDevice, dataSize, ACL_MEMCPY_DEVICE_TO_HOST);
    if (aclRet != ACL_SUCCESS) {
        LOG_PRINT("aclrtMemcpy result failed. ERROR: %d\n", aclRet);
        CleanupTestRuntime(ctx);
        return static_cast<int32_t>(aclRet);
    }

    CleanupTestRuntime(ctx);

    std::vector<float> golden = BuildComplexMatDotGolden(data);
    return VerifyResult(data.result, golden);
}
