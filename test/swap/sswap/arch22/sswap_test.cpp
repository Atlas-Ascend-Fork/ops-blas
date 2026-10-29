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
 * \file sswap_test.cpp
 * \brief
 */

#include <cstdint>
#include <iostream>
#include <vector>
#include <algorithm>
#include <iterator>
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

uint32_t VerifyResult(
    std::vector<float>& outputX, std::vector<float>& outputY, std::vector<float>& goldenX, std::vector<float>& goldenY)
{
    auto printTensor = [](std::vector<float>& tensor, const char* name) {
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
    printTensor(outputX, "OutputX");
    printTensor(goldenX, "GoldenX");
    printTensor(outputY, "OutputY");
    printTensor(goldenY, "GoldenY");
    constexpr float EPSILON = 1e-5f;
    for (size_t i = 0; i < outputX.size(); i++) {
        if (std::abs(outputX[i] - goldenX[i]) > EPSILON) {
            std::cout << "[Failed] X accuracy is verification failed!" << std::endl;
            return 1;
        }
    }
    for (size_t i = 0; i < outputY.size(); i++) {
        if (std::abs(outputY[i] - goldenY[i]) > EPSILON) {
            std::cout << "[Failed] Y accuracy is verification failed!" << std::endl;
            return 1;
        }
    }
    std::cout << "[Success] Case accuracy is verification passed." << std::endl;
    return 0;
}

// negative checks (issue #635): null handle, invalid n, null x/y
static int32_t RunSswapNegativeChecks(aclblasHandle_t handle)
{
    float sx = 1.0f;
    float sy = 2.0f;
    int32_t negativeResult = 0;
    if (aclblasSswap(nullptr, 8, &sx, 1, &sy, 1) != ACLBLAS_STATUS_HANDLE_IS_NULLPTR) {
        std::cout << "[Failed] sswap null handle not rejected" << std::endl;
        negativeResult = 1;
    }
    if (aclblasSswap(handle, -1, &sx, 1, &sy, 1) != ACLBLAS_STATUS_INVALID_VALUE) {
        std::cout << "[Failed] sswap negative n not rejected" << std::endl;
        negativeResult = 1;
    }
    if (aclblasSswap(handle, 0, &sx, 1, &sy, 1) != ACLBLAS_STATUS_SUCCESS) {
        std::cout << "[Failed] sswap zero n not fast-returned" << std::endl;
        negativeResult = 1;
    }
    if (aclblasSswap(handle, 8, nullptr, 1, &sy, 1) != ACLBLAS_STATUS_INVALID_VALUE ||
        aclblasSswap(handle, 8, &sx, 1, nullptr, 1) != ACLBLAS_STATUS_INVALID_VALUE) {
        std::cout << "[Failed] sswap null x/y not rejected" << std::endl;
        negativeResult = 1;
    }
    if (negativeResult == 0) {
        std::cout << "[Success] sswap negative checks passed." << std::endl;
    }
    return negativeResult;
}

// Basic unit-stride functional case: swap two full vectors and verify.
static int32_t RunSswapBasicCase(
    aclblasHandle_t handle, aclrtStream stream, uint32_t totalLength, std::vector<float>& x, std::vector<float>& y)
{
    float* xDevice = nullptr;
    float* yDevice = nullptr;
    size_t totalByteSize = totalLength * sizeof(float);
    aclError aclRet = aclrtMalloc((void**)&xDevice, totalByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc xDevice failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMalloc((void**)&yDevice, totalByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc yDevice failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMemcpy(xDevice, totalByteSize, x.data(), totalByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy xDevice failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMemcpy(yDevice, totalByteSize, y.data(), totalByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy yDevice failed. ERROR: %d\n", aclRet); return aclRet);

    const aclblasStatus_t ret = aclblasSswap(handle, static_cast<int>(totalLength), xDevice, 1, yDevice, 1);
    CHECK_RET(ret == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasSswap failed. ERROR: %d\n", ret); return ret);

    aclRet = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMemcpy(x.data(), totalByteSize, xDevice, totalByteSize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy x failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMemcpy(y.data(), totalByteSize, yDevice, totalByteSize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy y failed. ERROR: %d\n", aclRet); return aclRet);

    aclrtFree(xDevice);
    aclrtFree(yDevice);

    const float valueX = 1.5f;
    const float valueY = 2.5f;
    std::vector<float> goldenX(totalLength, valueY);
    std::vector<float> goldenY(totalLength, valueX);
    return VerifyResult(x, y, goldenX, goldenY);
}

int32_t main(int32_t argc, char* argv[])
{
    int32_t deviceId = 0;

    constexpr uint32_t totalLength = 8 * 2048;
    constexpr float valueX = 1.5f;
    constexpr float valueY = 2.5f;
    std::vector<float> x(totalLength, valueX);
    std::vector<float> y(totalLength, valueY);

    aclInit(nullptr);
    aclrtSetDevice(deviceId);

    aclblasHandle_t handle = nullptr;
    auto ret = aclblasCreate(&handle);
    CHECK_RET(ret == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasCreate failed. ERROR: %d\n", ret); return ret);

    aclrtStream stream = nullptr;
    aclrtCreateStream(&stream);
    ret = aclblasSetStream(handle, stream);
    CHECK_RET(ret == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasSetStream failed. ERROR: %d\n", ret); return ret);

    const int32_t negativeResult = RunSswapNegativeChecks(handle);
    const int32_t basicResult = RunSswapBasicCase(handle, stream, totalLength, x, y);

    aclblasDestroy(handle);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    if (basicResult != 0) {
        return basicResult;
    }
    return negativeResult;
}