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

// G.FMT.04（一行一条语句）合规的断言辅助：期望不满足时打印并释放可选缓冲
static int32_t ExpectStatus(aclblasStatus_t got, aclblasStatus_t want, const char* what, void* buf)
{
    if (got == want) {
        return 0;
    }
    LOG_PRINT("%s failed. got %d want %d\n", what, static_cast<int>(got), static_cast<int>(want));
    if (buf != nullptr) {
        aclrtFree(buf);
    }
    return 1;
}

// ACL 返回码辅助：非 ACL_SUCCESS 时打印并释放可选缓冲
static int32_t ExpectAclOk(aclError got, const char* what, void* buf)
{
    if (got == ACL_SUCCESS) {
        return 0;
    }
    LOG_PRINT("%s failed. ERROR: %d\n", what, static_cast<int>(got));
    if (buf != nullptr) {
        aclrtFree(buf);
    }
    return 1;
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
    if (aclRet != ACL_SUCCESS) {
        LOG_PRINT("aclrtMalloc xDevice failed. ERROR: %d\n", aclRet);
        return aclRet;
    }
    aclRet = aclrtMemcpy(xDevice, totalByteSize, x.data(), totalByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_SUCCESS) {
        LOG_PRINT("aclrtMemcpy xDevice failed. ERROR: %d\n", aclRet);
        aclrtFree(xDevice);
        return aclRet;
    }

    std::cout << "========== Testing aclblasSscal ==========" << std::endl;

    // issue #493: 无效入参应在解引用与 tiling 转换之前拦截
    auto ret = aclblasSscal(nullptr, totalLength, &alpha, (float*)xDevice, incx);
    int32_t st = ExpectStatus(ret, ACLBLAS_STATUS_HANDLE_IS_NULLPTR, "aclblasSscal null-handle check", xDevice);
    CHECK_RET(st == 0, return st);
    ret = aclblasSscal(handle, -1, &alpha, (float*)xDevice, incx);
    st = ExpectStatus(ret, ACLBLAS_STATUS_INVALID_VALUE, "aclblasSscal negative n validation", xDevice);
    CHECK_RET(st == 0, return st);
    ret = aclblasSscal(handle, totalLength, nullptr, (float*)xDevice, incx);
    st = ExpectStatus(ret, ACLBLAS_STATUS_INVALID_VALUE, "aclblasSscal null-alpha check", xDevice);
    CHECK_RET(st == 0, return st);
    ret = aclblasSscal(handle, totalLength, &alpha, nullptr, incx);
    st = ExpectStatus(ret, ACLBLAS_STATUS_INVALID_VALUE, "aclblasSscal null-x check", xDevice);
    CHECK_RET(st == 0, return st);
    std::cout << "[PASS] Sscal invalid-input validation passed" << std::endl;

    ret = aclblasSscal(handle, totalLength, &alpha, (float*)xDevice, incx);
    st = ExpectStatus(ret, ACLBLAS_STATUS_SUCCESS, "aclblasSscal", xDevice);
    CHECK_RET(st == 0, return st);

    aclRet = aclrtSynchronizeStream(stream);
    st = ExpectAclOk(aclRet, "aclrtSynchronizeStream", xDevice);
    CHECK_RET(st == 0, return st);
    aclRet = aclrtMemcpy(x.data(), totalByteSize, xDevice, totalByteSize, ACL_MEMCPY_DEVICE_TO_HOST);
    st = ExpectAclOk(aclRet, "aclrtMemcpy x", xDevice);
    CHECK_RET(st == 0, return st);
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
    if (aclRet != ACL_SUCCESS) {
        LOG_PRINT("aclrtMalloc xDevice failed. ERROR: %d\n", aclRet);
        return aclRet;
    }
    aclRet = aclrtMemcpy(xDevice, totalByteSize, x.data(), totalByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_SUCCESS) {
        LOG_PRINT("aclrtMemcpy xDevice failed. ERROR: %d\n", aclRet);
        aclrtFree(xDevice);
        return aclRet;
    }

    std::cout << "========== Testing aclblasCsscal ==========" << std::endl;
    auto ret = aclblasCsscal(handle, totalLength, &alpha, xDevice, incx);
    int32_t st = ExpectStatus(ret, ACLBLAS_STATUS_SUCCESS, "aclblasCsscal", xDevice);
    CHECK_RET(st == 0, return st);

    aclRet = aclrtSynchronizeStream(stream);
    st = ExpectAclOk(aclRet, "aclrtSynchronizeStream", xDevice);
    CHECK_RET(st == 0, return st);
    aclRet = aclrtMemcpy(x.data(), totalByteSize, xDevice, totalByteSize, ACL_MEMCPY_DEVICE_TO_HOST);
    st = ExpectAclOk(aclRet, "aclrtMemcpy x", xDevice);
    CHECK_RET(st == 0, return st);
    aclrtFree(xDevice);

    std::vector<aclblasComplex> golden(totalLength, valueX * alpha);
    return VerifyCsscalResult(x, golden);
}

int32_t TestSscalArgumentValidation(aclblasHandle handle)
{
    const float alpha = 2.0f;
    float value = 1.0f;
    CHECK_RET(aclblasSscal(nullptr, 1, &alpha, &value, 1) == ACLBLAS_STATUS_HANDLE_IS_NULLPTR,
              LOG_PRINT("Sscal null-handle validation failed\n");
              return 1);
    CHECK_RET(aclblasSscal(handle, -1, &alpha, &value, 1) == ACLBLAS_STATUS_INVALID_VALUE,
              LOG_PRINT("Sscal negative-length validation failed\n");
              return 1);
    CHECK_RET(aclblasSscal(handle, 0, nullptr, nullptr, 1) == ACLBLAS_STATUS_SUCCESS,
              LOG_PRINT("Sscal zero-length quick return failed\n");
              return 1);
    CHECK_RET(aclblasSscal(handle, 1, nullptr, &value, 1) == ACLBLAS_STATUS_INVALID_VALUE,
              LOG_PRINT("Sscal null-alpha validation failed\n");
              return 1);
    CHECK_RET(aclblasSscal(handle, 1, &alpha, nullptr, 1) == ACLBLAS_STATUS_INVALID_VALUE,
              LOG_PRINT("Sscal null-x validation failed\n");
              return 1);
    return 0;
}

int32_t TestCsscalArgumentValidation(aclblasHandle handle)
{
    // issue #1323: Csscal 校验序列对齐同文件 aclblasSscal（README arch22 契约）
    const float alpha = 2.0f;
    aclblasComplex value{1.0f, 1.0f};
    CHECK_RET(aclblasCsscal(nullptr, 1, &alpha, &value, 1) == ACLBLAS_STATUS_HANDLE_IS_NULLPTR,
              LOG_PRINT("Csscal null-handle validation failed\n");
              return 1);
    CHECK_RET(aclblasCsscal(handle, -1, &alpha, &value, 1) == ACLBLAS_STATUS_INVALID_VALUE,
              LOG_PRINT("Csscal negative-length validation failed\n");
              return 1);
    CHECK_RET(aclblasCsscal(handle, 0, nullptr, nullptr, 1) == ACLBLAS_STATUS_SUCCESS,
              LOG_PRINT("Csscal zero-length quick return failed\n");
              return 1);
    CHECK_RET(aclblasCsscal(handle, 1, nullptr, &value, 1) == ACLBLAS_STATUS_INVALID_VALUE,
              LOG_PRINT("Csscal null-alpha validation failed\n");
              return 1);
    CHECK_RET(aclblasCsscal(handle, 1, &alpha, nullptr, 1) == ACLBLAS_STATUS_INVALID_VALUE,
              LOG_PRINT("Csscal null-x validation failed\n");
              return 1);
    return 0;
}

// 共享小工具：分配 device 缓冲并拷入主机数据，失败返回 nullptr
static void* AllocDeviceAndCopy(const void* host, size_t bytes)
{
    void* dPtr = nullptr;
    if (aclrtMalloc(&dPtr, bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        LOG_PRINT("aclrtMalloc failed\n");
        return nullptr;
    }
    if (aclrtMemcpy(dPtr, bytes, host, bytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
        LOG_PRINT("aclrtMemcpy H2D failed\n");
        aclrtFree(dPtr);
        return nullptr;
    }
    return dPtr;
}

static int32_t SyncAndVerifyDeviceAlpha(aclrtStream stream, void* xDevice, size_t bytes, void* hostDst)
{
    auto ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
        return ret;
    }
    ret = aclrtMemcpy(hostDst, bytes, xDevice, bytes, ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclrtMemcpy D2H failed. ERROR: %d\n", ret);
        return ret;
    }
    return 0;
}

static int32_t RunSscalDeviceAlpha(
    aclblasHandle handle, aclrtStream stream, const float* alphaDevice, uint8_t* xDevice, std::vector<float>& x,
    uint32_t totalLength)
{
    constexpr float deviceAlpha = 3.0f;
    auto ret = aclblasSscal(handle, totalLength, alphaDevice, reinterpret_cast<float*>(xDevice), 1);
    CHECK_RET(ret == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasSscal device-alpha failed. ERROR: %d\n", ret);
              return ret);
    size_t bytes = totalLength * sizeof(float);
    int32_t status = SyncAndVerifyDeviceAlpha(stream, xDevice, bytes, x.data());
    CHECK_RET(status == 0, return status);
    for (uint32_t i = 0; i < totalLength; i++) {
        if (std::abs(x[i] - 2.0f * deviceAlpha) > 1e-5f) {
            LOG_PRINT("Sscal device-alpha wrong result at %u: %f\n", i, x[i]);
            return 1;
        }
    }
    return 0;
}

static int32_t RunCsscalDeviceAlpha(
    aclblasHandle handle, aclrtStream stream, const float* alphaDevice, uint8_t* cxDevice,
    std::vector<aclblasComplex>& cx, uint32_t totalLength)
{
    constexpr float deviceAlpha = 3.0f;
    auto ret = aclblasCsscal(handle, totalLength, alphaDevice, reinterpret_cast<aclblasComplex*>(cxDevice), 1);
    CHECK_RET(ret == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasCsscal device-alpha failed. ERROR: %d\n", ret);
              return ret);
    size_t bytes = totalLength * sizeof(aclblasComplex);
    int32_t status = SyncAndVerifyDeviceAlpha(stream, cxDevice, bytes, cx.data());
    CHECK_RET(status == 0, return status);
    for (uint32_t i = 0; i < totalLength; i++) {
        float diffRe = std::abs(cx[i].real - 2.0f * deviceAlpha);
        float diffIm = std::abs(cx[i].imag - 1.0f * deviceAlpha);
        if (diffRe > 1e-3f || diffIm > 1e-3f) {
            LOG_PRINT("Csscal device-alpha wrong result at %u: (%f,%f)\n", i, cx[i].real, cx[i].imag);
            return 1;
        }
    }
    return 0;
}

int32_t TestAlphaDevicePointer(aclblasHandle handle, aclrtStream stream)
{
    // issue #1324: alpha 支持 Host/Device 双位置，device 指针经同步 D2H 读取而非裸解引用
    constexpr uint32_t totalLength = 8;
    constexpr float deviceAlpha = 3.0f;
    std::vector<float> x(totalLength, 2.0f);
    std::vector<aclblasComplex> cx(totalLength, aclblasComplex{2.0f, 1.0f});
    size_t xBytes = totalLength * sizeof(float);
    size_t cxBytes = totalLength * sizeof(aclblasComplex);

    float* alphaDevice = static_cast<float*>(AllocDeviceAndCopy(&deviceAlpha, sizeof(float)));
    uint8_t* xDevice = static_cast<uint8_t*>(AllocDeviceAndCopy(x.data(), xBytes));
    uint8_t* cxDevice = static_cast<uint8_t*>(AllocDeviceAndCopy(cx.data(), cxBytes));
    if (alphaDevice == nullptr || xDevice == nullptr || cxDevice == nullptr) {
        if (alphaDevice != nullptr)
            aclrtFree(alphaDevice);
        if (xDevice != nullptr)
            aclrtFree(xDevice);
        if (cxDevice != nullptr)
            aclrtFree(cxDevice);
        return 1;
    }

    int32_t status = RunSscalDeviceAlpha(handle, stream, alphaDevice, xDevice, x, totalLength);
    if (status == 0) {
        status = RunCsscalDeviceAlpha(handle, stream, alphaDevice, cxDevice, cx, totalLength);
    }

    aclrtFree(alphaDevice);
    aclrtFree(xDevice);
    aclrtFree(cxDevice);
    if (status == 0) {
        std::cout << "[PASS] alpha device-pointer dual-location test passed" << std::endl;
    }
    return status;
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
    if (ret != ACLBLAS_STATUS_SUCCESS) {
        LOG_PRINT("aclblasCreate failed. ERROR: %d\n", ret);
        return ret;
    }

    aclrtStream stream = nullptr;
    aclrtCreateStream(&stream);
    ret = aclblasSetStream(handle, stream);
    if (ret != ACLBLAS_STATUS_SUCCESS) {
        LOG_PRINT("aclblasSetStream failed. ERROR: %d\n", ret);
        return ret;
    }

    int32_t result = 0;
    if (TestSscalArgumentValidation(handle) != 0) {
        LOG_PRINT("Sscal argument validation test failed\n");
        result = 1;
    } else if (TestCsscalArgumentValidation(handle) != 0) {
        LOG_PRINT("Csscal argument validation test failed\n");
        result = 1;
    } else if (TestAlphaDevicePointer(handle, stream) != 0) {
        LOG_PRINT("alpha device-pointer test failed\n");
        result = 1;
    } else if (TestSscal(handle, stream) != 0) {
        std::cout << "[FAIL] Sscal test failed" << std::endl;
        result = 1;
    } else if (TestCsscal(handle, stream) != 0) {
        std::cout << "[FAIL] Csscal test failed" << std::endl;
        result = 1;
    }

    CleanupSscalTestResources(handle, stream, deviceId);

    if (result == 0) {
        std::cout << "========================================" << std::endl;
        std::cout << "Test Summary:" << std::endl;
        std::cout << "  Passed: 2 - Sscal Csscal" << std::endl;
        std::cout << "========================================" << std::endl;
    }

    return result;
}
