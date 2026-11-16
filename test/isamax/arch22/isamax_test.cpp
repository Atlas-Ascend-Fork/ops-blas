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
 * \file isamax_test.cpp
 * \brief Test for aclblasIsamax operator (arch22)
 */

#include <cstdint>
#include <iostream>
#include <vector>
#include <algorithm>
#include <iterator>
#include <cmath>
#include "acl/acl.h"
#include "cann_ops_blas.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

uint32_t VerifyResult(int32_t output, int32_t golden, const char* test_name)
{
    std::cout << "\n========== " << test_name << " ==========" << std::endl;
    std::cout << "Output: " << output << std::endl;
    std::cout << "Golden: " << golden << std::endl;

    if (output == golden) {
        std::cout << "[Success] " << test_name << " verification passed." << std::endl;
        return 0;
    }
    std::cout << "[Failed] " << test_name << " verification failed!" << std::endl;
    return 1;
}

int TestBasicRealFloat(aclblasHandle_t handle, aclrtStream stream)
{
    std::cout << "\n========== Testing Real Float ==========" << std::endl;
    constexpr uint32_t totalLength = 128;
    std::vector<float> x(totalLength);

    for (uint32_t i = 0; i < totalLength; i++) {
        x[i] = static_cast<float>(i) * 0.1f;
    }
    x[50] = 100.0f;

    int incx = 1;
    int32_t result = 0;

    float* xDevice = nullptr;
    int32_t* resultDevice = nullptr;
    size_t inputByteSize = totalLength * sizeof(float);
    size_t outputByteSize = sizeof(int32_t);

    aclError aclRet = aclrtMalloc((void**)&xDevice, inputByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc xDevice failed. ERROR: %d\n", aclRet); return 1);
    aclRet = aclrtMalloc((void**)&resultDevice, outputByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc resultDevice failed. ERROR: %d\n", aclRet); return 1);
    aclRet = aclrtMemcpy(xDevice, inputByteSize, x.data(), inputByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy xDevice failed. ERROR: %d\n", aclRet); return 1);

    aclblasStatus_t statusRet = aclblasIsamax(handle, totalLength, xDevice, incx, resultDevice);
    CHECK_RET(statusRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasIsamax failed. ERROR: %d\n", statusRet); return 1);

    aclRet = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet); return 1);
    aclRet = aclrtMemcpy(&result, outputByteSize, resultDevice, outputByteSize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy result failed. ERROR: %d\n", aclRet); return 1);

    aclrtFree(xDevice);
    aclrtFree(resultDevice);

    return VerifyResult(result, 51, "Real Float Test");
}

int TestQuickReturn(aclblasHandle_t handle, aclrtStream stream, int n, int incx, const char* testName)
{
    std::cout << "\n========== Testing " << testName << " ==========" << std::endl;
    float input = 1.0f;
    int32_t sentinel = -1;
    int32_t result = -1;
    float* xDevice = nullptr;
    int32_t* resultDevice = nullptr;
    size_t inputByteSize = sizeof(float);
    size_t outputByteSize = sizeof(int32_t);

    aclError aclRet = aclrtMalloc((void**)&xDevice, inputByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc xDevice failed. ERROR: %d\n", aclRet); return 1);
    aclRet = aclrtMalloc((void**)&resultDevice, outputByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclRet != ACL_SUCCESS) {
        LOG_PRINT("aclrtMalloc resultDevice failed. ERROR: %d\n", aclRet);
        aclrtFree(xDevice);
        return 1;
    }
    aclRet = aclrtMemcpy(xDevice, inputByteSize, &input, inputByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet == ACL_SUCCESS) {
        aclRet = aclrtMemcpy(resultDevice, outputByteSize, &sentinel, outputByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    }
    if (aclRet != ACL_SUCCESS) {
        LOG_PRINT("aclrtMemcpy initialization failed. ERROR: %d\n", aclRet);
        aclrtFree(xDevice);
        aclrtFree(resultDevice);
        return 1;
    }

    auto statusRet = aclblasIsamax(handle, n, xDevice, incx, resultDevice);
    if (statusRet == ACLBLAS_STATUS_SUCCESS) {
        aclRet = aclrtSynchronizeStream(stream);
        if (aclRet == ACL_SUCCESS) {
            aclRet = aclrtMemcpy(&result, outputByteSize, resultDevice, outputByteSize, ACL_MEMCPY_DEVICE_TO_HOST);
        }
    }
    aclrtFree(xDevice);
    aclrtFree(resultDevice);
    bool pass = (statusRet == ACLBLAS_STATUS_SUCCESS && result == 0);
    std::cout << "Status: " << statusRet << ", Result: " << result << std::endl;
    if (pass) {
        std::cout << "[Success] " << testName << " passed." << std::endl;
        return 0;
    }
    std::cout << "[Failed] " << testName << " failed (API status: " << statusRet
              << ", sync/copy status: " << aclRet << ")." << std::endl;
    return 1;
}

int TestNullHandle()
{
    std::cout << "\n========== Testing Null Handle ==========" << std::endl;
    int32_t result = 0;
    float dummyX = 1.0f;
    auto statusRet = aclblasIsamax(nullptr, 1, &dummyX, 1, &result);
    bool pass = (statusRet == ACLBLAS_STATUS_NOT_INITIALIZED);
    std::cout << "Status: " << statusRet << std::endl;
    if (pass) {
        std::cout << "[Success] Null handle check passed." << std::endl;
        return 0;
    }
    std::cout << "[Failed] Null handle check failed!" << std::endl;
    return 1;
}

int32_t main(int32_t argc, char* argv[])
{
    int32_t deviceId = 0;

    aclInit(nullptr);
    aclrtSetDevice(deviceId);

    aclblasHandle_t handle = nullptr;
    auto ret = aclblasCreate(&handle);
    CHECK_RET(ret == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasCreate failed. ERROR: %d\n", ret); return ret);

    aclrtStream stream = nullptr;
    aclError aclRet = aclrtCreateStream(&stream);
    CHECK_RET(
        aclRet == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", aclRet); aclblasDestroy(handle);
        return aclRet);
    ret = aclblasSetStream(handle, stream);
    CHECK_RET(ret == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasSetStream failed. ERROR: %d\n", ret); return ret);

    int testRet = 0;
    testRet |= TestNullHandle();
    testRet |= TestQuickReturn(handle, stream, 0, 1, "n=0 Quick Return");
    testRet |= TestQuickReturn(handle, stream, 1, 0, "incx=0 Quick Return");
    testRet |= TestBasicRealFloat(handle, stream);

    aclblasDestroy(handle);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    if (testRet == 0) {
        std::cout << "\n========================================" << std::endl;
        std::cout << "All tests passed successfully!" << std::endl;
        std::cout << "========================================" << std::endl;
    } else {
        std::cout << "\n========================================" << std::endl;
        std::cout << "Some tests failed!" << std::endl;
        std::cout << "========================================" << std::endl;
    }

    return testRet;
}
