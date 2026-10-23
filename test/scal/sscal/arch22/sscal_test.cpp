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
 * \file sscal_test.cpp
 * \brief Test for aclblasSscal and aclblasCsscal interfaces
 */

#include <cstdint>
#include <iostream>
#include <vector>
#include <algorithm>
#include <iterator>
#include <cmath>
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

uint32_t VerifySscalResult(std::vector<float>& output, std::vector<float>& golden)
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
    constexpr float EPSILON = 1e-5f;
    for (size_t i = 0; i < output.size(); i++) {
        if (std::abs(output[i] - golden[i]) > EPSILON) {
            std::cout << "[Failed] Sscal accuracy verification failed!" << std::endl;
            return 1;
        }
    }
    std::cout << "[Success] Sscal accuracy verification passed." << std::endl;
    return 0;
}

uint32_t VerifyCsscalResult(std::vector<aclblasComplex>& output, std::vector<aclblasComplex>& golden)
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
            std::cout << "[Failed] Csscal Index " << i << ": output=(" << output[i].real << "," << output[i].imag
                      << ") golden=(" << golden[i].real << "," << golden[i].imag << ") diff=" << diff << std::endl;
            return 1;
        }
    }
    std::cout << "[Success] Csscal accuracy verification passed." << std::endl;
    return 0;
}

int32_t TestSscal(aclblasHandle handle, aclrtStream stream)
{
    constexpr uint32_t totalLength = 8 * 2048;
    constexpr float valueX = 1.2f;
    constexpr float alpha = 2.5f;
    std::vector<float> x(totalLength, valueX);
    int64_t incx = 1;

    uint8_t* xDevice = nullptr;
    size_t totalByteSize = totalLength * sizeof(float);
    aclError aclRet = aclrtMalloc((void**)&xDevice, totalByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc xDevice failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMemcpy(xDevice, totalByteSize, x.data(), totalByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(
        aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy xDevice failed. ERROR: %d\n", aclRet); aclrtFree(xDevice);
        return aclRet);

    std::cout << "========== Testing aclblasSscal ==========" << std::endl;

    // issue #493: 无效入参应在解引用与 tiling 转换之前拦截
    auto ret = aclblasSscal(nullptr, totalLength, &alpha, (float*)xDevice, incx);
    CHECK_RET(
        ret == ACLBLAS_STATUS_HANDLE_IS_NULLPTR, LOG_PRINT("aclblasSscal null-handle check failed. ERROR: %d\n", ret);
        aclrtFree(xDevice);
        return 1);
    ret = aclblasSscal(handle, -1, &alpha, (float*)xDevice, incx);
    CHECK_RET(
        ret == ACLBLAS_STATUS_INVALID_VALUE, LOG_PRINT("aclblasSscal negative n validation failed. ERROR: %d\n", ret);
        aclrtFree(xDevice);
        return 1);
    ret = aclblasSscal(handle, totalLength, nullptr, (float*)xDevice, incx);
    CHECK_RET(
        ret == ACLBLAS_STATUS_INVALID_VALUE, LOG_PRINT("aclblasSscal null-alpha check failed. ERROR: %d\n", ret);
        aclrtFree(xDevice);
        return 1);
    ret = aclblasSscal(handle, totalLength, &alpha, nullptr, incx);
    CHECK_RET(
        ret == ACLBLAS_STATUS_INVALID_VALUE, LOG_PRINT("aclblasSscal null-x check failed. ERROR: %d\n", ret);
        aclrtFree(xDevice);
        return 1);
    std::cout << "[PASS] Sscal invalid-input validation passed" << std::endl;

    ret = aclblasSscal(handle, totalLength, &alpha, (float*)xDevice, incx);
    CHECK_RET(
        ret == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasSscal failed. ERROR: %d\n", ret); aclrtFree(xDevice);
        return ret);

    aclRet = aclrtSynchronizeStream(stream);
    CHECK_RET(
        aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet); aclrtFree(xDevice);
        return aclRet);
    aclRet = aclrtMemcpy(x.data(), totalByteSize, xDevice, totalByteSize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(
        aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy x failed. ERROR: %d\n", aclRet); aclrtFree(xDevice);
        return aclRet);
    aclrtFree(xDevice);

    std::vector<float> golden(totalLength, valueX * alpha);
    return VerifySscalResult(x, golden);
}

int32_t TestCsscal(aclblasHandle handle, aclrtStream stream)
{
    constexpr uint32_t totalLength = 8 * 2048;
    constexpr aclblasComplex valueX{1.2f, 0.5f};
    constexpr float alpha = 2.5f;
    std::vector<aclblasComplex> x(totalLength, valueX);
    int incx = 1;

    aclblasComplex* xDevice = nullptr;
    size_t totalByteSize = totalLength * sizeof(aclblasComplex);
    aclError aclRet = aclrtMalloc((void**)&xDevice, totalByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc xDevice failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMemcpy(xDevice, totalByteSize, x.data(), totalByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(
        aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy xDevice failed. ERROR: %d\n", aclRet); aclrtFree(xDevice);
        return aclRet);

    std::cout << "========== Testing aclblasCsscal ==========" << std::endl;
    auto ret = aclblasCsscal(handle, totalLength, &alpha, xDevice, incx);
    CHECK_RET(
        ret == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasCsscal failed. ERROR: %d\n", ret); aclrtFree(xDevice);
        return ret);

    aclRet = aclrtSynchronizeStream(stream);
    CHECK_RET(
        aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet); aclrtFree(xDevice);
        return aclRet);
    aclRet = aclrtMemcpy(x.data(), totalByteSize, xDevice, totalByteSize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(
        aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy x failed. ERROR: %d\n", aclRet); aclrtFree(xDevice);
        return aclRet);
    aclrtFree(xDevice);

    std::vector<aclblasComplex> golden(totalLength, valueX * alpha);
    return VerifyCsscalResult(x, golden);
}

int32_t TestSscalArgumentValidation(aclblasHandle handle)
{
    const float alpha = 2.0f;
    float value = 1.0f;
    CHECK_RET(aclblasSscal(nullptr, 1, &alpha, &value, 1) == ACLBLAS_STATUS_HANDLE_IS_NULLPTR,
        LOG_PRINT("Sscal null-handle validation failed\n"); return 1);
    CHECK_RET(aclblasSscal(handle, -1, &alpha, &value, 1) == ACLBLAS_STATUS_INVALID_VALUE,
        LOG_PRINT("Sscal negative-length validation failed\n"); return 1);
    CHECK_RET(aclblasSscal(handle, 0, nullptr, nullptr, 1) == ACLBLAS_STATUS_SUCCESS,
        LOG_PRINT("Sscal zero-length quick return failed\n"); return 1);
    CHECK_RET(aclblasSscal(handle, 1, nullptr, &value, 1) == ACLBLAS_STATUS_INVALID_VALUE,
        LOG_PRINT("Sscal null-alpha validation failed\n"); return 1);
    CHECK_RET(aclblasSscal(handle, 1, &alpha, nullptr, 1) == ACLBLAS_STATUS_INVALID_VALUE,
        LOG_PRINT("Sscal null-x validation failed\n"); return 1);
    return 0;
}

void CleanupSscalTestResources(aclblasHandle handle, aclrtStream stream, int32_t deviceId)
{
    aclblasDestroy(handle);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
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

    CHECK_RET(TestSscalArgumentValidation(handle) == 0,
        LOG_PRINT("Sscal argument validation test failed\n");
        CleanupSscalTestResources(handle, stream, deviceId);
        return 1);

    int32_t result = 0;

    result = TestSscal(handle, stream);
    if (result != 0) {
        std::cout << "[FAIL] Sscal test failed" << std::endl;
        CleanupSscalTestResources(handle, stream, deviceId);
        return result;
    }
    std::cout << "[PASS] Sscal test passed" << std::endl;

    result = TestCsscal(handle, stream);
    if (result != 0) {
        std::cout << "[FAIL] Csscal test failed" << std::endl;
        CleanupSscalTestResources(handle, stream, deviceId);
        return result;
    }
    std::cout << "[PASS] Csscal test passed" << std::endl;

    CleanupSscalTestResources(handle, stream, deviceId);

    std::cout << "========================================" << std::endl;
    std::cout << "Test Summary:" << std::endl;
    std::cout << "  Passed: 2 - Sscal Csscal" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
