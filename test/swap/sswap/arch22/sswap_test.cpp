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
    if (aclblasSswap(handle, 8, &sx, 0, &sy, 1) != ACLBLAS_STATUS_INVALID_VALUE ||
        aclblasSswap(handle, 8, &sx, 1, &sy, 0) != ACLBLAS_STATUS_INVALID_VALUE) {
        std::cout << "[Failed] sswap zero increment not rejected" << std::endl;
        negativeResult = 1;
    }
    if (negativeResult == 0) {
        std::cout << "[Success] sswap negative checks passed." << std::endl;
    }
    return negativeResult;
}

// Upload x/y, run aclblasSswap(n, incx, incy) and copy the results back into x/y.
static aclblasStatus_t RunSswapOnDevice(
    aclblasHandle_t handle, aclrtStream stream, int n, int incx, int incy, std::vector<float>& x, std::vector<float>& y)
{
    const size_t xBytes = x.size() * sizeof(float);
    const size_t yBytes = y.size() * sizeof(float);
    float* xDevice = nullptr;
    float* yDevice = nullptr;
    aclError aclRet = aclrtMalloc((void**)&xDevice, xBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc xDevice failed. ERROR: %d\n", aclRet);
              return ACLBLAS_STATUS_ALLOC_FAILED);
    aclRet = aclrtMalloc((void**)&yDevice, yBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc yDevice failed. ERROR: %d\n", aclRet); aclrtFree(xDevice);
              return ACLBLAS_STATUS_ALLOC_FAILED);
    aclRet = aclrtMemcpy(xDevice, xBytes, x.data(), xBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy xDevice failed. ERROR: %d\n", aclRet); aclrtFree(xDevice);
              aclrtFree(yDevice); return ACLBLAS_STATUS_INTERNAL_ERROR);
    aclRet = aclrtMemcpy(yDevice, yBytes, y.data(), yBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy yDevice failed. ERROR: %d\n", aclRet); aclrtFree(xDevice);
              aclrtFree(yDevice); return ACLBLAS_STATUS_INTERNAL_ERROR);

    const aclblasStatus_t ret = aclblasSswap(handle, n, xDevice, incx, yDevice, incy);
    if (ret == ACLBLAS_STATUS_SUCCESS) {
        aclRet = aclrtSynchronizeStream(stream);
        CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet);
                  aclrtFree(xDevice); aclrtFree(yDevice); return ACLBLAS_STATUS_INTERNAL_ERROR);
        aclRet = aclrtMemcpy(x.data(), xBytes, xDevice, xBytes, ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy x failed. ERROR: %d\n", aclRet); aclrtFree(xDevice);
                  aclrtFree(yDevice); return ACLBLAS_STATUS_INTERNAL_ERROR);
        aclRet = aclrtMemcpy(y.data(), yBytes, yDevice, yBytes, ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy y failed. ERROR: %d\n", aclRet); aclrtFree(xDevice);
                  aclrtFree(yDevice); return ACLBLAS_STATUS_INTERNAL_ERROR);
    }
    aclrtFree(xDevice);
    aclrtFree(yDevice);
    return ret;
}

// Basic unit-stride functional case: swap two full vectors and verify.
static int32_t RunSswapBasicCase(
    aclblasHandle_t handle, aclrtStream stream, uint32_t totalLength, std::vector<float>& x, std::vector<float>& y)
{
    const aclblasStatus_t ret = RunSswapOnDevice(handle, stream, static_cast<int>(totalLength), 1, 1, x, y);
    CHECK_RET(ret == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasSswap failed. ERROR: %d\n", ret); return ret);

    const float valueX = 1.5f;
    const float valueY = 2.5f;
    std::vector<float> goldenX(totalLength, valueY);
    std::vector<float> goldenY(totalLength, valueX);
    return VerifyResult(x, y, goldenX, goldenY);
}

// Storage length of a vector holding n logical elements with stride inc.
static size_t SswapStorageLength(int n, int inc)
{
    const int64_t absInc = inc < 0 ? -static_cast<int64_t>(inc) : inc;
    return static_cast<size_t>(1 + (static_cast<int64_t>(n) - 1) * absInc);
}

// Fill the whole storage with distinct sentinel values so writes outside the
// logical index set (stride gaps) stay visible.
static void FillSswapSentinel(std::vector<float>& v, int base)
{
    for (size_t i = 0; i < v.size(); ++i) {
        v[i] = static_cast<float>(base + static_cast<int>(i));
    }
}

// One strided case: sentinel-filled storage, independent logical-index oracle, exact compare.
static int32_t RunSswapStrideCase(aclblasHandle_t handle, aclrtStream stream, int n, int incx, int incy)
{
    std::vector<float> x(SswapStorageLength(n, incx));
    std::vector<float> y(SswapStorageLength(n, incy));
    FillSswapSentinel(x, 10);
    FillSswapSentinel(y, 100);
    std::vector<float> expectedX = x;
    std::vector<float> expectedY = y;
    int64_t xIndex = incx < 0 ? static_cast<int64_t>(n - 1) * -static_cast<int64_t>(incx) : 0;
    int64_t yIndex = incy < 0 ? static_cast<int64_t>(n - 1) * -static_cast<int64_t>(incy) : 0;
    for (int i = 0; i < n; ++i, xIndex += incx, yIndex += incy) {
        std::swap(expectedX[xIndex], expectedY[yIndex]);
    }

    const aclblasStatus_t ret = RunSswapOnDevice(handle, stream, n, incx, incy, x, y);
    CHECK_RET(ret == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasSswap failed. ERROR: %d\n", ret); return ret);
    if (x != expectedX || y != expectedY) {
        std::cout << "[Failed] strided swap mismatch (n=" << n << ", incx=" << incx << ", incy=" << incy << ")"
                  << std::endl;
        return 1;
    }
    std::cout << "[Success] strided swap passed (n=" << n << ", incx=" << incx << ", incy=" << incy << ")."
              << std::endl;
    return 0;
}

// Strided regression: positive, negative and mixed strides incl. a multi-pass
// length crossing the per-pass batching boundary (issue #509).
static int32_t RunSswapStrideCases(aclblasHandle_t handle, aclrtStream stream)
{
    static const int32_t cases[][3] = {{7, 2, 2},   {7, -1, 1},  {7, 1, -2},  {7, -2, 3},
                                       {5, -1, -1}, {33, 4, -3}, {1025, 2, 2}};
    int32_t failed = 0;
    for (const auto& c : cases) {
        failed |= RunSswapStrideCase(handle, stream, c[0], c[1], c[2]);
    }
    if (failed == 0) {
        std::cout << "[Success] sswap stride cases passed." << std::endl;
    }
    return failed;
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
    const int32_t strideResult = RunSswapStrideCases(handle, stream);

    aclblasDestroy(handle);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    if (basicResult != 0) {
        return basicResult;
    }
    if (strideResult != 0) {
        return strideResult;
    }
    return negativeResult;
}