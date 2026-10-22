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
 * \file sasum_test.cpp
 * \brief
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

uint32_t VerifyResult(float output, float golden)
{
    std::cout << "Output: " << output << std::endl;
    std::cout << "Golden: " << golden << std::endl;

    const float epsilon = 1e-4;
    if (std::abs(output - golden) < epsilon) {
        std::cout << "[Success] Case accuracy is verification passed." << std::endl;
        return 0;
    } else {
        std::cout << "[Failed] Case accuracy is verification failed!" << std::endl;
        return 1;
    }
}

int32_t RunStridedCase(aclblasHandle_t handle, aclrtStream stream, float* xDevice, float* resultDevice,
                      float& stridedResult)
{
    constexpr size_t outputByteSize = sizeof(float);
    const float stridedInput[] = {1.0f, 99.0f, 2.0f, 99.0f, 3.0f};
    constexpr size_t stridedInputBytes = sizeof(stridedInput);
    aclError aclRet = aclrtMemcpy(xDevice, stridedInputBytes, stridedInput, stridedInputBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy strided input failed. ERROR: %d\n", aclRet); return aclRet);

    constexpr int stridedLength = 3;
    constexpr int stridedIncrement = 2;
    auto ret = aclblasSasum(handle, stridedLength, xDevice, stridedIncrement, resultDevice);
    CHECK_RET(ret == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasSasum strided call failed. ERROR: %d\n", ret);
              return ret);
    aclRet = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream strided call failed. ERROR: %d\n", aclRet);
              return aclRet);
    aclRet = aclrtMemcpy(&stridedResult, outputByteSize, resultDevice, outputByteSize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy strided result failed. ERROR: %d\n", aclRet);
              return aclRet);

    return 0;
}

int32_t main(int32_t argc, char* argv[])
{
    int32_t deviceId = 0;

    constexpr uint32_t totalLength = 128;
    constexpr float valueX = 1.2f;
    constexpr int incx = 1;

    std::vector<float> x(totalLength, valueX);
    float result = 0.0f;
    float stridedResult = 0.0f;

    aclInit(nullptr);
    aclrtSetDevice(deviceId);

    aclblasHandle_t handle = nullptr;
    auto ret = aclblasCreate(&handle);
    CHECK_RET(ret == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasCreate failed. ERROR: %d\n", ret); return ret);

    aclrtStream stream = nullptr;
    aclrtCreateStream(&stream);
    ret = aclblasSetStream(handle, stream);
    CHECK_RET(ret == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasSetStream failed. ERROR: %d\n", ret); return ret);

    float* xDevice = nullptr;
    float* resultDevice = nullptr;

    size_t totalByteSize = totalLength * sizeof(float);
    size_t outputByteSize = sizeof(float);
    aclError aclRet = aclrtMalloc((void**)&xDevice, totalByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc xDevice failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMalloc((void**)&resultDevice, outputByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc resultDevice failed. ERROR: %d\n", aclRet); return aclRet);

    aclRet = aclrtMemcpy(xDevice, totalByteSize, x.data(), totalByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy xDevice failed. ERROR: %d\n", aclRet); return aclRet);

    ret = aclblasSasum(handle, totalLength, xDevice, incx, resultDevice);
    CHECK_RET(ret == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasSasum failed. ERROR: %d\n", ret); return ret);

    aclRet = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMemcpy(&result, outputByteSize, resultDevice, outputByteSize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy result failed. ERROR: %d\n", aclRet); return aclRet);

    int32_t stridedStatus = RunStridedCase(handle, stream, xDevice, resultDevice, stridedResult);
    CHECK_RET(stridedStatus == 0, return stridedStatus);

    aclrtFree(xDevice);
    aclrtFree(resultDevice);

    aclblasDestroy(handle);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    float golden = totalLength * std::abs(valueX);
    uint32_t status = VerifyResult(result, golden);
    status |= VerifyResult(stridedResult, 6.0f);
    return status;
}
