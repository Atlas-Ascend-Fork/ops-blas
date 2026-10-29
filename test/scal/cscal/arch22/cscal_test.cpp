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
 * \file cscal_test.cpp
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

namespace {

constexpr aclblasComplex kAlpha{2.5f, 1.0f};
constexpr float kEps = 1e-3f;

aclblasComplex CscalGoldenElem(const aclblasComplex& v)
{
    return aclblasComplex{kAlpha.real * v.real - kAlpha.imag * v.imag, kAlpha.real * v.imag + kAlpha.imag * v.real};
}

uint32_t VerifyResult(std::vector<aclblasComplex>& output, std::vector<aclblasComplex>& golden)
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
    for (size_t i = 0; i < output.size(); i++) {
        float diff = blasComplexAbs(output[i] - golden[i]);
        if (diff > kEps) {
            std::cout << "[Failed] Index " << i << ": output=(" << output[i].real << "," << output[i].imag
                      << ") golden=(" << golden[i].real << "," << golden[i].imag << ") diff=" << diff << std::endl;
            return 1;
        }
    }
    std::cout << "[Success] Case accuracy is verification passed." << std::endl;
    return 0;
}

// Run one device call on a byte buffer and copy the result back.
bool RunCscalOnDevice(
    aclblasHandle_t handle, aclrtStream stream, int n, int incx, std::vector<aclblasComplex>& xHost,
    aclblasComplex* xDevice)
{
    const size_t bytes = xHost.size() * sizeof(aclblasComplex);
    if (aclrtMemcpy(xDevice, bytes, xHost.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
        return false;
    }
    if (aclblasCscal(handle, n, &kAlpha, xDevice, incx) != ACLBLAS_STATUS_SUCCESS) {
        return false;
    }
    if (aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
        return false;
    }
    return aclrtMemcpy(xHost.data(), bytes, xDevice, bytes, ACL_MEMCPY_DEVICE_TO_HOST) == ACL_SUCCESS;
}

} // namespace

// 负向检查（issue #675/#676）：空句柄/空 alpha/空 x、n<=0 no-op、incx<=0 no-op
static int32_t RunCscalNegativeChecks(aclblasHandle_t handle)
{
    aclblasComplex sx{1.0f, 1.0f};
    const aclblasComplex sa{2.0f, 1.0f};
    int32_t negativeResult = 0;
    if (aclblasCscal(nullptr, 8, &sa, &sx, 1) != ACLBLAS_STATUS_HANDLE_IS_NULLPTR) {
        std::cout << "[Failed] cscal null handle not rejected" << std::endl;
        negativeResult = 1;
    }
    if (aclblasCscal(handle, 8, nullptr, &sx, 1) != ACLBLAS_STATUS_INVALID_VALUE ||
        aclblasCscal(handle, 8, &sa, nullptr, 1) != ACLBLAS_STATUS_INVALID_VALUE) {
        std::cout << "[Failed] cscal null alpha/x not rejected" << std::endl;
        negativeResult = 1;
    }
    if (aclblasCscal(handle, 0, &sa, &sx, 1) != ACLBLAS_STATUS_SUCCESS ||
        aclblasCscal(handle, -1, &sa, &sx, 1) != ACLBLAS_STATUS_SUCCESS) {
        std::cout << "[Failed] cscal n<=0 not fast-returned" << std::endl;
        negativeResult = 1;
    }
    if (aclblasCscal(handle, 8, &sa, &sx, 0) != ACLBLAS_STATUS_SUCCESS ||
        aclblasCscal(handle, 8, &sa, &sx, -2) != ACLBLAS_STATUS_SUCCESS) {
        std::cout << "[Failed] cscal incx<=0 not fast-returned" << std::endl;
        negativeResult = 1;
    }
    if (negativeResult == 0) {
        std::cout << "[Success] cscal negative checks passed." << std::endl;
    }
    return negativeResult;
}

// 连续路径用例（issue #674）：非 4 倍数 n 的残部由标量路径处理，golden 为全量复数缩放
static int32_t RunCscalContinuousCases(aclblasHandle_t handle, aclrtStream stream)
{
    const int caseLens[] = {8 * 2048, 5, 6, 9, 10, 13, 17, 4097};
    for (int n : caseLens) {
        const aclblasComplex valueX{1.2f, 0.5f};
        std::vector<aclblasComplex> xHost(n, valueX);
        std::vector<aclblasComplex> golden(n);
        for (int i = 0; i < n; i++) {
            golden[i] = CscalGoldenElem(valueX);
        }

        aclblasComplex* xDevice = nullptr;
        CHECK_RET(aclrtMalloc((void**)&xDevice, xHost.size() * sizeof(aclblasComplex), ACL_MEM_MALLOC_HUGE_FIRST) ==
                      ACL_SUCCESS,
                  LOG_PRINT("cscal xDev alloc failed\n");
                  return 1);
        if (!RunCscalOnDevice(handle, stream, n, 1, xHost, xDevice)) {
            std::cout << "[Failed] cscal continuous case n=" << n << " device run failed" << std::endl;
            aclrtFree(xDevice);
            return 1;
        }
        aclrtFree(xDevice);
        if (VerifyResult(xHost, golden) != 0) {
            std::cout << "[Failed] cscal continuous case n=" << n << std::endl;
            return 1;
        }
    }
    std::cout << "[Success] cscal continuous cases passed." << std::endl;
    return 0;
}

// 步长用例（issue #676）：incx>1 按 x[i*incx] 缩放，未引用槽位（哨兵）保持不变
static int32_t RunCscalStrideCases(aclblasHandle_t handle, aclrtStream stream)
{
    const int caseShapes[][2] = {{16, 2}, {16, 3}, {33, 2}};
    for (const auto& shape : caseShapes) {
        const int n = shape[0];
        const int incx = shape[1];
        const size_t slots = static_cast<size_t>((n - 1) * incx + 1);
        const aclblasComplex valueX{1.2f, 0.5f};
        std::vector<aclblasComplex> xHost(slots, valueX);
        std::vector<aclblasComplex> golden(slots, valueX);
        for (int i = 0; i < n; i++) {
            golden[static_cast<size_t>(i) * incx] = CscalGoldenElem(valueX);
        }

        aclblasComplex* xDevice = nullptr;
        CHECK_RET(aclrtMalloc((void**)&xDevice, xHost.size() * sizeof(aclblasComplex), ACL_MEM_MALLOC_HUGE_FIRST) ==
                      ACL_SUCCESS,
                  LOG_PRINT("cscal xDev alloc failed\n");
                  return 1);
        if (!RunCscalOnDevice(handle, stream, n, incx, xHost, xDevice)) {
            std::cout << "[Failed] cscal stride case n=" << n << " incx=" << incx << " device run failed" << std::endl;
            aclrtFree(xDevice);
            return 1;
        }
        aclrtFree(xDevice);
        for (size_t j = 0; j < slots; j++) {
            if (blasComplexAbs(xHost[j] - golden[j]) > kEps) {
                std::cout << "[Failed] cscal stride case n=" << n << " incx=" << incx << " slot " << j << std::endl;
                aclrtFree(xDevice);
                return 1;
            }
        }
    }
    std::cout << "[Success] cscal stride cases passed." << std::endl;
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

    const int32_t negativeResult = RunCscalNegativeChecks(handle);
    const int32_t continuousResult = RunCscalContinuousCases(handle, stream);
    const int32_t strideResult = RunCscalStrideCases(handle, stream);

    aclblasDestroy(handle);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    if (negativeResult != 0) {
        return negativeResult;
    }
    if (continuousResult != 0) {
        return continuousResult;
    }
    return strideResult;
}
