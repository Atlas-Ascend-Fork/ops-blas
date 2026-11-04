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
 * \file scopy_test.cpp
 * \brief
 */

#include <cstdint>
#include <iostream>
#include <vector>
#include <algorithm>
#include <iterator>
#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "complex.h"

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

uint32_t VerifyResult(std::vector<float>& output, std::vector<float>& golden)
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
    printTensor(output, "Output");
    printTensor(golden, "Golden");
    if (std::equal(output.begin(), output.end(), golden.begin())) {
        std::cout << "[Success] Case accuracy is verification passed." << std::endl;
        return 0;
    } else {
        std::cout << "[Failed] Case accuracy is verification failed!" << std::endl;
        return 1;
    }
    return 0;
}

uint32_t VerifyResultComplex(std::vector<aclblasComplex>& output, std::vector<aclblasComplex>& golden)
{
    auto printTensor = [](std::vector<aclblasComplex>& tensor, const char* name) {
        constexpr size_t maxPrintSize = 10;
        std::cout << name << ": ";
        for (size_t i = 0; i < std::min(tensor.size(), maxPrintSize); i++) {
            std::cout << "(" << tensor[i].real << "," << tensor[i].imag << ") ";
        }
        if (tensor.size() > maxPrintSize) {
            std::cout << "...";
        }
        std::cout << std::endl;
    };
    printTensor(output, "Output");
    printTensor(golden, "Golden");
    if (std::equal(output.begin(), output.end(), golden.begin())) {
        std::cout << "[Success] Case accuracy is verification passed." << std::endl;
        return 0;
    } else {
        std::cout << "[Failed] Case accuracy is verification failed!" << std::endl;
        return 1;
    }
    return 0;
}

// issue #809: scopy 测试独立成函数
static uint32_t RunScopyContiguousTest()
{
    constexpr uint32_t totalLength = 8 * 2048;
    constexpr float valueX = 1.2f;
    constexpr float valueY = 2.3f;
    std::vector<float> xHost(totalLength, valueX);
    std::vector<float> yHost(totalLength, valueY);
    int64_t incx = 1;
    int64_t incy = 1;

    size_t totalByteSize = totalLength * sizeof(float);

    aclInit(nullptr);
    aclrtSetDevice(0);

    aclblasHandle_t handle = nullptr;
    aclblasStatus_t ret = aclblasCreate(&handle);
    CHECK_RET(ret == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasCreate failed. ERROR: %d\n", ret); return 1);

    aclrtStream stream = nullptr;
    aclError aclRet = aclrtCreateStream(&stream);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", aclRet); return aclRet);

    ret = aclblasSetStream(handle, stream);
    CHECK_RET(ret == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasSetStream failed.\n"); return 1);

    uint8_t* xDevice = nullptr;
    uint8_t* yDevice = nullptr;

    aclRet = aclrtMalloc((void**)&xDevice, totalByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc xDevice failed.\n"); return 1);
    aclRet = aclrtMalloc((void**)&yDevice, totalByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc yDevice failed.\n"); return 1);

    aclRet = aclrtMemcpy(xDevice, totalByteSize, xHost.data(), totalByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy xDevice failed.\n"); return 1);
    aclRet = aclrtMemcpy(yDevice, totalByteSize, yHost.data(), totalByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy yDevice failed.\n"); return 1);

    // issue #779: 空 handle 必须返回 HANDLE_IS_NULLPTR，不得静默取空流继续执行
    ret = aclblasScopy_legacy(nullptr, xDevice, yDevice, totalLength, incx, incy);
    CHECK_RET(ret == ACLBLAS_STATUS_HANDLE_IS_NULLPTR,
              LOG_PRINT("aclblasScopy_legacy null handle expected HANDLE_IS_NULLPTR, got %d\n", ret);
              return 1);

    ret = aclblasScopy_legacy(handle, xDevice, yDevice, totalLength, incx, incy);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclblasScopy_legacy failed.\n"); return 1);

    aclRet = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed.\n"); return 1);

    aclRet = aclrtMemcpy(yHost.data(), totalByteSize, yDevice, totalByteSize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy yHost failed.\n"); return 1);

    aclrtFree(xDevice);
    aclrtFree(yDevice);
    aclblasDestroy(handle);
    aclrtDestroyStream(stream);

    std::vector<float> golden(totalLength, valueX);
    return VerifyResult(yHost, golden);
}

// issue #809: ccopy 连续测试独立成函数
static uint32_t RunCcopyContiguousTest(int32_t deviceId)
{
    constexpr uint32_t complexLength = 4 * 1024;
    aclblasComplex valueComplexX{1.5f, 2.5f};
    aclblasComplex valueComplexY{3.0f, 4.0f};
    std::vector<aclblasComplex> cxHost(complexLength, valueComplexX);
    std::vector<aclblasComplex> cyHost(complexLength, valueComplexY);
    size_t complexByteSize = complexLength * sizeof(aclblasComplex);

    aclrtSetDevice(deviceId);

    aclblasHandle_t handle2 = nullptr;
    aclblasStatus_t ret = aclblasCreate(&handle2);
    CHECK_RET(ret == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasCreate failed.\n"); return 1);

    aclrtStream stream2 = nullptr;
    aclError aclRet = aclrtCreateStream(&stream2);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed.\n"); return 1);

    ret = aclblasSetStream(handle2, stream2);
    CHECK_RET(ret == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasSetStream failed.\n"); return 1);

    aclblasComplex* cxDevice = nullptr;
    aclblasComplex* cyDevice = nullptr;
    aclRet = aclrtMalloc((void**)&cxDevice, complexByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc cxDevice failed.\n"); return 1);
    aclRet = aclrtMalloc((void**)&cyDevice, complexByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc cyDevice failed.\n"); return 1);

    aclRet = aclrtMemcpy(cxDevice, complexByteSize, cxHost.data(), complexByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy cxDevice failed.\n"); return 1);
    aclRet = aclrtMemcpy(cyDevice, complexByteSize, cyHost.data(), complexByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy cyDevice failed.\n"); return 1);

    int64_t incx = 1;
    int64_t incy = 1;
    // issue #779: 空 handle 必须返回 HANDLE_IS_NULLPTR，不得静默取空流继续执行
    ret = aclblasCcopy(nullptr, complexLength, cxDevice, incx, cyDevice, incy);
    CHECK_RET(ret == ACLBLAS_STATUS_HANDLE_IS_NULLPTR,
              LOG_PRINT("aclblasCcopy null handle expected HANDLE_IS_NULLPTR, got %d\n", ret);
              return 1);

    ret = aclblasCcopy(handle2, complexLength, cxDevice, incx, cyDevice, incy);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclblasCcopy failed.\n"); return 1);

    aclRet = aclrtSynchronizeStream(stream2);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed.\n"); return 1);

    aclRet = aclrtMemcpy(cyHost.data(), complexByteSize, cyDevice, complexByteSize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy cyHost failed.\n"); return 1);

    aclrtFree(cxDevice);
    aclrtFree(cyDevice);
    aclblasDestroy(handle2);
    aclrtDestroyStream(stream2);

    std::vector<aclblasComplex> goldenComplex(complexLength, valueComplexX);
    return VerifyResultComplex(cyHost, goldenComplex);
}

int32_t main(int32_t argc, char* argv[])
{
    int32_t deviceId = 0;
    aclInit(nullptr);
    aclrtSetDevice(deviceId);

    uint32_t scopyResult = RunScopyContiguousTest();
    uint32_t ccopyResult = RunCcopyContiguousTest(deviceId);

    aclrtResetDevice(deviceId);
    aclFinalize();

    return (scopyResult + ccopyResult);
}
