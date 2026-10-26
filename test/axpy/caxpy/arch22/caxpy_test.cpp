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
 * \file caxpy_test.cpp
 * \brief Test for aclblasCaxpy interface
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

uint32_t VerifyCaxpyResult(std::vector<aclblasComplex>& output, std::vector<aclblasComplex>& golden)
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
    constexpr float EPSILON = 1e-3f;
    for (size_t i = 0; i < output.size(); i++) {
        float diff = blasComplexAbs(output[i] - golden[i]);
        if (diff > EPSILON) {
            std::cout << "[Failed] Caxpy Index " << i << ": output=(" << output[i].real << "," << output[i].imag
                      << ") golden=(" << golden[i].real << "," << golden[i].imag << ") diff=" << diff << std::endl;
            return 1;
        }
    }
    std::cout << "[Success] Caxpy accuracy verification passed." << std::endl;
    return 0;
}

int32_t TestCaxpy(aclblasHandle handle, aclrtStream stream)
{
    constexpr uint32_t totalLength = 8 * 2048;
    constexpr aclblasComplex valueX{1.0f, 0.5f};
    constexpr aclblasComplex valueY{2.0f, 1.0f};
    constexpr aclblasComplex alpha{2.0f, 1.0f};
    std::vector<aclblasComplex> x(totalLength, valueX);
    std::vector<aclblasComplex> y(totalLength, valueY);
    int incx = 1;
    int incy = 1;

    aclblasComplex* xDevice = nullptr;
    aclblasComplex* yDevice = nullptr;
    size_t xByteSize = totalLength * sizeof(aclblasComplex);
    size_t yByteSize = totalLength * sizeof(aclblasComplex);
    aclError aclRet = aclrtMalloc((void**)&xDevice, xByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc xDevice failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMalloc((void**)&yDevice, yByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc yDevice failed. ERROR: %d\n", aclRet); aclrtFree(xDevice);
              return aclRet);
    aclRet = aclrtMemcpy(xDevice, xByteSize, x.data(), xByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy xDevice failed. ERROR: %d\n", aclRet); aclrtFree(xDevice);
              aclrtFree(yDevice); return aclRet);
    aclRet = aclrtMemcpy(yDevice, yByteSize, y.data(), yByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy yDevice failed. ERROR: %d\n", aclRet); aclrtFree(xDevice);
              aclrtFree(yDevice); return aclRet);

    std::cout << "========== Testing aclblasCaxpy ==========" << std::endl;
    std::cout << "Formula: y = alpha * x + y" << std::endl;
    std::cout << "alpha = (" << alpha.real << ", " << alpha.imag << ")" << std::endl;
    std::cout << "x = (" << valueX.real << ", " << valueX.imag << ") * " << totalLength << std::endl;
    std::cout << "y = (" << valueY.real << ", " << valueY.imag << ") * " << totalLength << std::endl;

    auto ret = aclblasCaxpy(handle, totalLength, &alpha, xDevice, incx, yDevice, incy);
    CHECK_RET(ret == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasCaxpy failed. ERROR: %d\n", ret); aclrtFree(xDevice);
              aclrtFree(yDevice); return ret);

    aclRet = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet);
              aclrtFree(xDevice); aclrtFree(yDevice); return aclRet);
    aclRet = aclrtMemcpy(y.data(), yByteSize, yDevice, yByteSize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy y failed. ERROR: %d\n", aclRet); aclrtFree(xDevice);
              aclrtFree(yDevice); return aclRet);
    aclrtFree(xDevice);
    aclrtFree(yDevice);

    std::vector<aclblasComplex> golden(totalLength);
    for (size_t i = 0; i < totalLength; i++) {
        golden[i] = alpha * x[i] + valueY;
    }

    return VerifyCaxpyResult(y, golden);
}

// issue #578: handle 为空时期望返回 HANDLE_IS_NULLPTR，而非解引用崩溃
int32_t TestCaxpyNegative()
{
    std::cout << "========== Testing aclblasCaxpy negative case ==========" << std::endl;
    constexpr aclblasComplex alpha{1.0f, 0.0f};
    aclblasComplex dummy{};
    auto ret = aclblasCaxpy(nullptr, 8, &alpha, &dummy, 1, &dummy, 1);
    if (ret != ACLBLAS_STATUS_HANDLE_IS_NULLPTR) {
        std::cout << "[FAIL] null handle: expected " << ACLBLAS_STATUS_HANDLE_IS_NULLPTR << " got " << ret << std::endl;
        return 1;
    }
    std::cout << "[PASS] Caxpy negative case passed." << std::endl;
    return 0;
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
    aclrtCreateStream(&stream);
    ret = aclblasSetStream(handle, stream);
    CHECK_RET(ret == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasSetStream failed. ERROR: %d\n", ret); return ret);

    int32_t result = 0;

    result = TestCaxpy(handle, stream);
    if (result != 0) {
        std::cout << "[FAIL] Caxpy test failed" << std::endl;
        aclblasDestroy(handle);
        aclrtDestroyStream(stream);
        aclrtResetDevice(deviceId);
        aclFinalize();
        return result;
    }
    std::cout << "[PASS] Caxpy test passed" << std::endl;

    result = TestCaxpyNegative();
    if (result != 0) {
        std::cout << "[FAIL] Caxpy negative test failed" << std::endl;
        aclblasDestroy(handle);
        aclrtDestroyStream(stream);
        aclrtResetDevice(deviceId);
        aclFinalize();
        return result;
    }

    aclblasDestroy(handle);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    std::cout << "========================================" << std::endl;
    std::cout << "Test Summary:" << std::endl;
    std::cout << "  Passed: 2 - Caxpy, CaxpyNegative" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}