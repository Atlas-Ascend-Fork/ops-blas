/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file cdot_test.cpp
 * \brief Test for complex dot product (cdotu and cdotc)
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

uint32_t VerifyResult(const aclblasComplex& output, const aclblasComplex& golden)
{
    std::cout << "Output: (" << output.real << ", " << output.imag << ")" << std::endl;
    std::cout << "Golden: (" << golden.real << ", " << golden.imag << ")" << std::endl;

    if (!std::isfinite(output.real) || !std::isfinite(output.imag) || !std::isfinite(golden.real) ||
        !std::isfinite(golden.imag)) {
        std::cout << "[Failed] Case accuracy is verification failed!" << std::endl;
        return 1;
    }

    bool pass = true;
    float diffReal = std::abs(output.real - golden.real);
    float diffImag = std::abs(output.imag - golden.imag);
    float maxReal = std::max(std::abs(output.real), std::abs(golden.real));
    float maxImag = std::max(std::abs(output.imag), std::abs(golden.imag));
    if (maxReal > 0 && diffReal / maxReal > 1e-5) {
        pass = false;
    }
    if (maxImag > 0 && diffImag / maxImag > 1e-5) {
        pass = false;
    }

    if (pass) {
        std::cout << "[Success] Case accuracy is verification passed." << std::endl;
        return 0;
    } else {
        std::cout << "[Failed] Case accuracy is verification failed!" << std::endl;
        return 1;
    }
}

void ComputeCdotuGolden(
    const std::vector<aclblasComplex>& x, const std::vector<aclblasComplex>& y, uint32_t n, aclblasComplex* golden)
{
    float goldenReal = 0.0f;
    float goldenImag = 0.0f;

    for (uint32_t i = 0; i < n; i++) {
        goldenReal += x[i].real * y[i].real - x[i].imag * y[i].imag;
        goldenImag += x[i].real * y[i].imag + x[i].imag * y[i].real;
    }

    golden->real = goldenReal;
    golden->imag = goldenImag;
}

void ComputeCdotcGolden(
    const std::vector<aclblasComplex>& x, const std::vector<aclblasComplex>& y, uint32_t n, aclblasComplex* golden)
{
    float goldenReal = 0.0f;
    float goldenImag = 0.0f;

    for (uint32_t i = 0; i < n; i++) {
        goldenReal += x[i].real * y[i].real + x[i].imag * y[i].imag;
        goldenImag += x[i].real * y[i].imag - x[i].imag * y[i].real;
    }

    golden->real = goldenReal;
    golden->imag = goldenImag;
}

int32_t main(int32_t argc, char* argv[])
{
    int32_t deviceId = 0;

    constexpr uint32_t complexNum = 128;
    int incx = 1;
    int incy = 1;

    std::vector<aclblasComplex> x(complexNum);
    std::vector<aclblasComplex> y(complexNum);
    aclblasComplex result = {0.0f, 0.0f};
    aclblasComplex golden = {0.0f, 0.0f};

    for (uint32_t i = 0; i < complexNum; i++) {
        x[i] = {1.0f, 0.5f};
        y[i] = {3.0f, 2.0f};
    }

    aclInit(nullptr);
    aclrtSetDevice(deviceId);

    aclblasHandle_t handle = nullptr;
    auto ret = aclblasCreate(&handle);
    CHECK_RET(ret == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasCreate failed. ERROR: %d\n", ret); return ret);

    aclrtStream stream = nullptr;
    aclError aclRet = aclrtCreateStream(&stream);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", aclRet); aclblasDestroy(handle);
              return aclRet);
    ret = aclblasSetStream(handle, stream);
    CHECK_RET(ret == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasSetStream failed. ERROR: %d\n", ret); return ret);

    aclblasComplex* xDevice = nullptr;
    aclblasComplex* yDevice = nullptr;
    aclblasComplex* resultDevice = nullptr;
    size_t inputByteSize = complexNum * sizeof(aclblasComplex);
    size_t outputByteSize = sizeof(aclblasComplex);

    aclRet = aclrtMalloc((void**)&xDevice, inputByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc xDevice failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMalloc((void**)&yDevice, inputByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc yDevice failed. ERROR: %d\n", aclRet); aclrtFree(xDevice);
              return aclRet);
    aclRet = aclrtMalloc((void**)&resultDevice, outputByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc resultDevice failed. ERROR: %d\n", aclRet);
              aclrtFree(yDevice); aclrtFree(xDevice); return aclRet);
    aclRet = aclrtMemcpy(xDevice, inputByteSize, x.data(), inputByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy xDevice failed. ERROR: %d\n", aclRet);
              aclrtFree(resultDevice); aclrtFree(yDevice); aclrtFree(xDevice); return aclRet);
    aclRet = aclrtMemcpy(yDevice, inputByteSize, y.data(), inputByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy yDevice failed. ERROR: %d\n", aclRet);
              aclrtFree(resultDevice); aclrtFree(yDevice); aclrtFree(xDevice); return aclRet);

    std::cout << "=== Testing aclblasCdotu ===" << std::endl;
    ComputeCdotuGolden(x, y, complexNum, &golden);

    ret = aclblasCdotu(handle, complexNum, xDevice, incx, yDevice, incy, resultDevice);
    CHECK_RET(ret == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasCdotu failed. ERROR: %d\n", ret); return ret);

    aclRet = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet);
              aclrtFree(resultDevice); aclrtFree(yDevice); aclrtFree(xDevice); return aclRet);
    aclRet = aclrtMemcpy(&result, outputByteSize, resultDevice, outputByteSize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy result failed. ERROR: %d\n", aclRet);
              aclrtFree(resultDevice); aclrtFree(yDevice); aclrtFree(xDevice); return aclRet);

    uint32_t cdotuResult = VerifyResult(result, golden);

    std::cout << "\n=== Testing aclblasCdotc ===" << std::endl;
    ComputeCdotcGolden(x, y, complexNum, &golden);

    ret = aclblasCdotc(handle, complexNum, xDevice, incx, yDevice, incy, resultDevice);
    CHECK_RET(ret == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasCdotc failed. ERROR: %d\n", ret); return ret);

    aclRet = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet);
              aclrtFree(resultDevice); aclrtFree(yDevice); aclrtFree(xDevice); return aclRet);
    aclRet = aclrtMemcpy(&result, outputByteSize, resultDevice, outputByteSize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy result failed. ERROR: %d\n", aclRet);
              aclrtFree(resultDevice); aclrtFree(yDevice); aclrtFree(xDevice); return aclRet);

    uint32_t cdotcResult = VerifyResult(result, golden);

    aclrtFree(xDevice);
    aclrtFree(yDevice);
    aclrtFree(resultDevice);

    aclblasDestroy(handle);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return cdotuResult + cdotcResult;
}
