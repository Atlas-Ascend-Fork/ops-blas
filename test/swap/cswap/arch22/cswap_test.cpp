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
 * \file cswap_test.cpp
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

uint32_t VerifyResult(float* outputX, float* outputY, float* goldenX, float* goldenY, size_t totalFloats)
{
    auto printTensor = [](float* tensor, size_t size, const char* name) {
        constexpr size_t maxPrintSize = 10;
        std::cout << name << ": ";
        for (size_t i = 0; i < std::min(size, maxPrintSize); i++) {
            std::cout << tensor[i] << " ";
        }
        if (size > maxPrintSize) {
            std::cout << "...";
        }
        std::cout << std::endl;
    };
    printTensor(outputX, totalFloats, "OutputX");
    printTensor(goldenX, totalFloats, "GoldenX");
    printTensor(outputY, totalFloats, "OutputY");
    printTensor(goldenY, totalFloats, "GoldenY");
    constexpr float EPSILON = 1e-3f;
    for (size_t i = 0; i < totalFloats; i++) {
        float diff = std::abs(outputX[i] - goldenX[i]);
        if (diff > EPSILON) {
            std::cout << "[Failed] X Index " << i << ": output=" << outputX[i] << " golden=" << goldenX[i]
                      << " diff=" << diff << std::endl;
            return 1;
        }
    }
    for (size_t i = 0; i < totalFloats; i++) {
        float diff = std::abs(outputY[i] - goldenY[i]);
        if (diff > EPSILON) {
            std::cout << "[Failed] Y Index " << i << ": output=" << outputY[i] << " golden=" << goldenY[i]
                      << " diff=" << diff << std::endl;
            return 1;
        }
    }
    std::cout << "[Success] Case accuracy is verification passed." << std::endl;
    return 0;
}

// Single-side accuracy comparison: stride cases allocate x/y with different
// slot counts, so each side must be checked with its own float length.
static uint32_t VerifyOneSide(
    const aclblasComplex* output, const aclblasComplex* golden, size_t totalFloats, const char* name)
{
    constexpr float EPSILON = 1e-3f;
    const float* o = reinterpret_cast<const float*>(output);
    const float* g = reinterpret_cast<const float*>(golden);
    for (size_t i = 0; i < totalFloats; i++) {
        const float diff = std::abs(o[i] - g[i]);
        if (diff > EPSILON) {
            std::cout << "[Failed] " << name << " float index " << i << ": output=" << o[i] << " golden=" << g[i]
                      << " diff=" << diff << std::endl;
            return 1;
        }
    }
    return 0;
}

// negative checks (issue #635/#637): null handle, invalid n, null x/y, zero stride
static int32_t RunCswapNegativeChecks(aclblasHandle_t handle)
{
    aclblasComplex sx{1.0f, 0.0f};
    aclblasComplex sy{2.0f, 0.0f};
    int32_t negativeResult = 0;
    if (aclblasCswap(nullptr, 4, &sx, 1, &sy, 1) != ACLBLAS_STATUS_HANDLE_IS_NULLPTR) {
        std::cout << "[Failed] cswap null handle not rejected" << std::endl;
        negativeResult = 1;
    }
    if (aclblasCswap(handle, -1, &sx, 1, &sy, 1) != ACLBLAS_STATUS_INVALID_VALUE) {
        std::cout << "[Failed] cswap negative n not rejected" << std::endl;
        negativeResult = 1;
    }
    if (aclblasCswap(handle, 0, &sx, 1, &sy, 1) != ACLBLAS_STATUS_SUCCESS) {
        std::cout << "[Failed] cswap zero n not fast-returned" << std::endl;
        negativeResult = 1;
    }
    if (aclblasCswap(handle, 4, nullptr, 1, &sy, 1) != ACLBLAS_STATUS_INVALID_VALUE ||
        aclblasCswap(handle, 4, &sx, 1, nullptr, 1) != ACLBLAS_STATUS_INVALID_VALUE) {
        std::cout << "[Failed] cswap null x/y not rejected" << std::endl;
        negativeResult = 1;
    }
    if (aclblasCswap(handle, 4, &sx, 0, &sy, 1) != ACLBLAS_STATUS_INVALID_VALUE ||
        aclblasCswap(handle, 4, &sx, 1, &sy, 0) != ACLBLAS_STATUS_INVALID_VALUE) {
        std::cout << "[Failed] cswap zero stride not rejected" << std::endl;
        negativeResult = 1;
    }
    if (negativeResult == 0) {
        std::cout << "[Success] cswap negative checks passed." << std::endl;
    }
    return negativeResult;
}

// Build host data and the swap golden for one stride case (each side keeps its
// own slot count; untouched slots act as sentinels).
static void BuildCswapStrideGolden(
    int nElems, int incx, int incy, std::vector<aclblasComplex>& xHost, std::vector<aclblasComplex>& yHost,
    std::vector<aclblasComplex>& goldenX, std::vector<aclblasComplex>& goldenY)
{
    auto physIdx = [](int i, int inc, int n) -> int { return inc > 0 ? i * inc : (n - 1 - i) * (-inc); };
    const size_t xSlots = static_cast<size_t>((nElems - 1) * (incx < 0 ? -incx : incx) + 1);
    const size_t ySlots = static_cast<size_t>((nElems - 1) * (incy < 0 ? -incy : incy) + 1);
    xHost.assign(xSlots, aclblasComplex{0.0f, 0.0f});
    yHost.assign(ySlots, aclblasComplex{0.0f, 0.0f});
    for (size_t j = 0; j < xSlots; j++) {
        xHost[j] = aclblasComplex{10.0f + static_cast<float>(j), 20.0f + static_cast<float>(j)};
    }
    for (size_t j = 0; j < ySlots; j++) {
        yHost[j] = aclblasComplex{30.0f + static_cast<float>(j), 40.0f + static_cast<float>(j)};
    }
    goldenX = xHost;
    goldenY = yHost;
    for (int i = 0; i < nElems; i++) {
        const int px = physIdx(i, incx, nElems);
        const int py = physIdx(i, incy, nElems);
        goldenX[px] = yHost[py];
        goldenY[py] = xHost[px];
    }
}

// One stride case: build host data + golden, run on device, compare each side
// with its own slot count (x and y buffers have different slot counts).
static int32_t RunCswapStrideCase(aclblasHandle_t handle, aclrtStream stream, int nElems, int incx, int incy)
{
    const size_t xSlots = static_cast<size_t>((nElems - 1) * (incx < 0 ? -incx : incx) + 1);
    const size_t ySlots = static_cast<size_t>((nElems - 1) * (incy < 0 ? -incy : incy) + 1);
    std::vector<aclblasComplex> xHost;
    std::vector<aclblasComplex> yHost;
    std::vector<aclblasComplex> goldenX;
    std::vector<aclblasComplex> goldenY;
    BuildCswapStrideGolden(nElems, incx, incy, xHost, yHost, goldenX, goldenY);

    const size_t xBytes = xSlots * sizeof(aclblasComplex);
    const size_t yBytes = ySlots * sizeof(aclblasComplex);
    aclblasComplex* xDev = nullptr;
    aclblasComplex* yDev = nullptr;
    bool caseOk = aclrtMalloc((void**)&xDev, xBytes, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS &&
                  aclrtMalloc((void**)&yDev, yBytes, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS &&
                  aclrtMemcpy(xDev, xBytes, xHost.data(), xBytes, ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS &&
                  aclrtMemcpy(yDev, yBytes, yHost.data(), yBytes, ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS;
    if (caseOk) {
        const aclblasStatus_t callRet = aclblasCswap(handle, nElems, xDev, incx, yDev, incy);
        caseOk = callRet == ACLBLAS_STATUS_SUCCESS;
        if (caseOk) {
            caseOk = aclrtSynchronizeStream(stream) == ACL_SUCCESS;
        }
        if (caseOk) {
            caseOk = aclrtMemcpy(xHost.data(), xBytes, xDev, xBytes, ACL_MEMCPY_DEVICE_TO_HOST) == ACL_SUCCESS &&
                     aclrtMemcpy(yHost.data(), yBytes, yDev, yBytes, ACL_MEMCPY_DEVICE_TO_HOST) == ACL_SUCCESS;
        }
    }
    if (caseOk) {
        uint32_t verifyRet = VerifyOneSide(xHost.data(), goldenX.data(), xSlots * 2, "OutputX");
        if (verifyRet == 0) {
            verifyRet = VerifyOneSide(yHost.data(), goldenY.data(), ySlots * 2, "OutputY");
        }
        caseOk = verifyRet == 0;
    }
    if (xDev != nullptr) {
        aclrtFree(xDev);
    }
    if (yDev != nullptr) {
        aclrtFree(yDev);
    }
    if (!caseOk) {
        std::cout << "[Failed] cswap stride case incx=" << incx << " incy=" << incy << " n=" << nElems << std::endl;
        return 1;
    }
    return 0;
}

// stride regression (issue #637): positive/negative/mixed strides against
// host-side golden with sentinel slots that must stay untouched. The last
// case uses n > cores*pass-cap so the strided kernel's multi-element and
// multi-pass splitting is actually exercised (!720 review).
static int32_t RunCswapStrideCases(aclblasHandle_t handle, aclrtStream stream)
{
    const int caseShapes[][3] = {{2, 1, 4}, {2, -2, 4}, {-1, -1, 4}, {1, -2, 4}, {2, -2, 25000}};
    for (const auto& shape : caseShapes) {
        if (RunCswapStrideCase(handle, stream, shape[2], shape[0], shape[1]) != 0) {
            return 1;
        }
    }
    std::cout << "[Success] cswap stride cases passed." << std::endl;
    return 0;
}

// Basic unit-stride functional case: swap two full vectors and verify.
// x/y 以非常量引用传入（D2H 回拷覆写宿主数据，CI 编译器拒绝 const 指针去限定符转换）
static int32_t RunCswapBasicCase(
    aclblasHandle_t handle, aclrtStream stream, uint32_t totalLength, std::vector<aclblasComplex>& x,
    std::vector<aclblasComplex>& y)
{
    aclblasComplex* xDevice = nullptr;
    aclblasComplex* yDevice = nullptr;
    size_t totalByteSize = totalLength * sizeof(aclblasComplex);
    aclError aclRet = aclrtMalloc((void**)&xDevice, totalByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc xDevice failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMalloc((void**)&yDevice, totalByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc yDevice failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMemcpy(xDevice, totalByteSize, x.data(), totalByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy xDevice failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMemcpy(yDevice, totalByteSize, y.data(), totalByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy yDevice failed. ERROR: %d\n", aclRet); return aclRet);

    const aclblasStatus_t ret = aclblasCswap(handle, static_cast<int32_t>(totalLength), xDevice, 1, yDevice, 1);
    CHECK_RET(ret == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasCswap failed. ERROR: %d\n", ret); return ret);

    aclRet = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMemcpy(x.data(), totalByteSize, xDevice, totalByteSize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy x failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMemcpy(y.data(), totalByteSize, yDevice, totalByteSize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy y failed. ERROR: %d\n", aclRet); return aclRet);

    aclrtFree(xDevice);
    aclrtFree(yDevice);

    std::vector<aclblasComplex> goldenX(totalLength, aclblasComplex{2.5f, 1.5f});
    std::vector<aclblasComplex> goldenY(totalLength, aclblasComplex{1.5f, 0.5f});
    return VerifyResult(
        reinterpret_cast<float*>(x.data()), reinterpret_cast<float*>(y.data()),
        reinterpret_cast<float*>(goldenX.data()), reinterpret_cast<float*>(goldenY.data()), totalLength * 2);
}

int32_t main(int32_t argc, char* argv[])
{
    int32_t deviceId = 0;

    constexpr uint32_t totalLength = 8 * 2048;
    constexpr aclblasComplex valueX{1.5f, 0.5f};
    constexpr aclblasComplex valueY{2.5f, 1.5f};
    std::vector<aclblasComplex> x(totalLength, valueX);
    std::vector<aclblasComplex> y(totalLength, valueY);

    aclInit(nullptr);
    aclrtSetDevice(deviceId);

    aclblasHandle_t handle = nullptr;
    auto ret = aclblasCreate(&handle);
    CHECK_RET(ret == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasCreate failed. ERROR: %d\n", ret); return ret);

    aclrtStream stream = nullptr;
    aclrtCreateStream(&stream);
    ret = aclblasSetStream(handle, stream);
    CHECK_RET(ret == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasSetStream failed. ERROR: %d\n", ret); return ret);

    const int32_t negativeResult = RunCswapNegativeChecks(handle);
    const int32_t basicResult = RunCswapBasicCase(handle, stream, totalLength, x, y);
    const int32_t strideResult = RunCswapStrideCases(handle, stream);

    aclblasDestroy(handle);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    if (basicResult != 0) {
        return basicResult;
    }
    if (negativeResult != 0) {
        return negativeResult;
    }
    return strideResult;
}