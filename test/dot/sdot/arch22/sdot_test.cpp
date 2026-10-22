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
 * \file sdot_test.cpp
 * \brief Test for real vector dot product
 */

#include <cstdint>
#include <iostream>
#include <vector>
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

inline uint32_t VerifyResult(float output, float golden)
{
    std::cout << "Output: " << output << std::endl;
    std::cout << "Golden: " << golden << std::endl;

    if (!std::isfinite(output) || !std::isfinite(golden)) {
        std::cout << "[Failed] Case accuracy is verification failed!" << std::endl;
        return 1;
    }

    if (golden == 0.0f && output == 0.0f) {
        std::cout << "[Success] Case accuracy is verification passed." << std::endl;
        return 0;
    }

    float diff = std::abs(output - golden);
    float maxVal = std::max(std::abs(output), std::abs(golden));
    if (maxVal > 0 && diff / maxVal > 1e-5) {
        std::cout << "[Failed] Case accuracy is verification failed!" << std::endl;
        return 1;
    }

    std::cout << "[Success] Case accuracy is verification passed." << std::endl;
    return 0;
}

int32_t main(int32_t argc, char* argv[])
{
    int32_t deviceId = 0;

    constexpr uint32_t n = 8 * 1024;
    constexpr int incx = 1;
    constexpr int incy = 1;

    std::vector<float> x(n);
    std::vector<float> y(n);
    float result = 0.0f;

    for (uint32_t i = 0; i < n; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    float golden = 0.0f;
    for (uint32_t i = 0; i < n; i++) {
        golden += x[i] * y[i];
    }

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
    float* yDevice = nullptr;
    float* resultDevice = nullptr;

    size_t totalByteSize = n * sizeof(float);
    aclError aclRet = aclrtMalloc((void**)&xDevice, totalByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc xDevice failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMalloc((void**)&yDevice, totalByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc yDevice failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMalloc((void**)&resultDevice, sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc resultDevice failed. ERROR: %d\n", aclRet); return aclRet);

    aclRet = aclrtMemcpy(xDevice, totalByteSize, x.data(), totalByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy xDevice failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMemcpy(yDevice, totalByteSize, y.data(), totalByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy yDevice failed. ERROR: %d\n", aclRet); return aclRet);

    ret = aclblasSdot(handle, n, xDevice, incx, yDevice, incy, resultDevice);
    CHECK_RET(ret == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasSdot failed. ERROR: %d\n", ret); return ret);

    aclRet = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMemcpy(&result, sizeof(float), resultDevice, sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy result failed. ERROR: %d\n", aclRet); return aclRet);

    aclrtFree(xDevice);
    aclrtFree(yDevice);
    aclrtFree(resultDevice);

    aclblasDestroy(handle);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    std::cout << "Testing sdot:" << std::endl;
    return VerifyResult(result, golden);
}
