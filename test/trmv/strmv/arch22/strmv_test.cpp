/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <iostream>
#include <vector>
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

constexpr float EPSILON = 1e-3f;

uint32_t VerifyResult(std::vector<float>& output, std::vector<float>& golden)
{
    constexpr size_t maxPrintSize = 10;
    std::cout << "Output: ";
    for (size_t i = 0; i < std::min(output.size(), maxPrintSize); i++) {
        std::cout << output[i] << " ";
    }
    if (output.size() > maxPrintSize) {
        std::cout << "...";
    }
    std::cout << std::endl;

    std::cout << "Golden: ";
    for (size_t i = 0; i < std::min(golden.size(), maxPrintSize); i++) {
        std::cout << golden[i] << " ";
    }
    if (golden.size() > maxPrintSize) {
        std::cout << "...";
    }
    std::cout << std::endl;

    for (size_t i = 0; i < output.size(); i++) {
        float diff = std::abs(output[i] - golden[i]);
        if (diff > EPSILON) {
            std::cout << "[Failed] Index " << i << ": output=" << output[i] << " golden=" << golden[i]
                      << " diff=" << diff << std::endl;
            return 1;
        }
    }
    std::cout << "[Success] Case accuracy is verification passed." << std::endl;
    return 0;
}

void ComputeGoldenStrmvUpperN(const float* A, float* x, int64_t n, int64_t lda, int64_t incx, int64_t diag)
{
    std::vector<float> temp(n);
    for (int64_t i = 0; i < n; i++) {
        temp[i] = x[i * incx];
    }
    for (int64_t i = n - 1; i >= 0; i--) {
        float sum = 0.0f;
        for (int64_t j = i; j < n; j++) {
            sum += A[i + j * lda] * temp[j];
        }
        if (diag == 1) {
            sum = temp[i] + sum - A[i + i * lda] * temp[i];
        }
        x[i * incx] = sum;
    }
}

void ComputeGoldenStrmvLowerN(const float* A, float* x, int64_t n, int64_t lda, int64_t incx, int64_t diag)
{
    std::vector<float> temp(n);
    for (int64_t i = 0; i < n; i++) {
        temp[i] = x[i * incx];
    }
    for (int64_t i = 0; i < n; i++) {
        float sum = 0.0f;
        for (int64_t j = 0; j <= i; j++) {
            sum += A[i + j * lda] * temp[j];
        }
        if (diag == 1) {
            sum = temp[i] + sum - A[i + i * lda] * temp[i];
        }
        x[i * incx] = sum;
    }
}

void ComputeGoldenStrmvUpperT(const float* A, float* x, int64_t n, int64_t lda, int64_t incx, int64_t diag)
{
    std::vector<float> temp(n);
    for (int64_t i = 0; i < n; i++) {
        temp[i] = x[i * incx];
    }
    for (int64_t j = 0; j < n; j++) {
        float sum = 0.0f;
        for (int64_t i = 0; i <= j; i++) {
            sum += A[i + j * lda] * temp[i];
        }
        if (diag == 1) {
            sum = temp[j] + sum - A[j + j * lda] * temp[j];
        }
        x[j * incx] = sum;
    }
}

void ComputeGoldenStrmvLowerT(const float* A, float* x, int64_t n, int64_t lda, int64_t incx, int64_t diag)
{
    std::vector<float> temp(n);
    for (int64_t i = 0; i < n; i++) {
        temp[i] = x[i * incx];
    }
    for (int64_t j = n - 1; j >= 0; j--) {
        float sum = 0.0f;
        for (int64_t i = j; i < n; i++) {
            sum += A[i + j * lda] * temp[i];
        }
        if (diag == 1) {
            sum = temp[j] + sum - A[j + j * lda] * temp[j];
        }
        x[j * incx] = sum;
    }
}

int TestStrmvUpperN()
{
    std::cout << "Testing strmv (uplo=U, trans=N, diag=N):" << std::endl;

    int32_t deviceId = 0;
    constexpr int64_t N = 128;
    constexpr int64_t lda = N;
    constexpr int64_t incx = 1;

    std::vector<float> A(N * N, 1.0f);
    for (int64_t i = 0; i < N; i++) {
        for (int64_t j = 0; j < i; j++) {
            A[i + j * lda] = 0.0f;
        }
        A[i + i * lda] = 2.0f;
    }
    std::vector<float> x(N, 1.0f);
    std::vector<float> xGolden = x;

    ComputeGoldenStrmvUpperN(A.data(), xGolden.data(), N, lda, incx, 0);

    aclInit(nullptr);
    aclrtSetDevice(deviceId);

    aclblasHandle_t handle = nullptr;
    auto ret = aclblasCreate(&handle);
    CHECK_RET(ret == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasCreate failed. ERROR: %d\n", ret); return ret);

    aclrtStream stream = nullptr;
    aclrtCreateStream(&stream);
    ret = aclblasSetStream(handle, stream);
    CHECK_RET(ret == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasSetStream failed. ERROR: %d\n", ret); return ret);

    float* aDevice = nullptr;
    float* xDevice = nullptr;
    size_t aByteSize = N * N * sizeof(float);
    size_t xByteSize = N * sizeof(float);
    aclError aclRet = aclrtMalloc((void**)&aDevice, aByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc aDevice failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMalloc((void**)&xDevice, xByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc xDevice failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMemcpy(aDevice, aByteSize, A.data(), aByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy aDevice failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMemcpy(xDevice, xByteSize, x.data(), xByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy xDevice failed. ERROR: %d\n", aclRet); return aclRet);

    ret = aclblasStrmv(handle, ACLBLAS_UPPER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT, N, aDevice, lda, xDevice, incx);
    CHECK_RET(ret == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasStrmv failed. ERROR: %d\n", ret); return ret);

    aclRet = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMemcpy(x.data(), xByteSize, xDevice, xByteSize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy x failed. ERROR: %d\n", aclRet); return aclRet);

    aclrtFree(aDevice);
    aclrtFree(xDevice);
    aclblasDestroy(handle);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return VerifyResult(x, xGolden);
}

int TestStrmvLowerN()
{
    std::cout << "Testing strmv (uplo=L, trans=N, diag=N):" << std::endl;

    int32_t deviceId = 0;
    constexpr int64_t N = 128;
    constexpr int64_t lda = N;
    constexpr int64_t incx = 1;

    std::vector<float> A(N * N, 1.0f);
    for (int64_t i = 0; i < N; i++) {
        for (int64_t j = i + 1; j < N; j++) {
            A[i + j * lda] = 0.0f;
        }
        A[i + i * lda] = 2.0f;
    }
    std::vector<float> x(N, 1.0f);
    std::vector<float> xGolden = x;

    ComputeGoldenStrmvLowerN(A.data(), xGolden.data(), N, lda, incx, 0);

    aclInit(nullptr);
    aclrtSetDevice(deviceId);

    aclblasHandle_t handle = nullptr;
    auto ret = aclblasCreate(&handle);
    CHECK_RET(ret == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasCreate failed. ERROR: %d\n", ret); return ret);

    aclrtStream stream = nullptr;
    aclrtCreateStream(&stream);
    ret = aclblasSetStream(handle, stream);
    CHECK_RET(ret == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasSetStream failed. ERROR: %d\n", ret); return ret);

    float* aDevice = nullptr;
    float* xDevice = nullptr;
    size_t aByteSize = N * N * sizeof(float);
    size_t xByteSize = N * sizeof(float);
    aclError aclRet = aclrtMalloc((void**)&aDevice, aByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc aDevice failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMalloc((void**)&xDevice, xByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc xDevice failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMemcpy(aDevice, aByteSize, A.data(), aByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy aDevice failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMemcpy(xDevice, xByteSize, x.data(), xByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy xDevice failed. ERROR: %d\n", aclRet); return aclRet);

    ret = aclblasStrmv(handle, ACLBLAS_LOWER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT, N, aDevice, lda, xDevice, incx);
    CHECK_RET(ret == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasStrmv failed. ERROR: %d\n", ret); return ret);

    aclRet = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMemcpy(x.data(), xByteSize, xDevice, xByteSize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy x failed. ERROR: %d\n", aclRet); return aclRet);

    aclrtFree(aDevice);
    aclrtFree(xDevice);
    aclblasDestroy(handle);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return VerifyResult(x, xGolden);
}

int TestStrmvUpperT()
{
    std::cout << "Testing strmv (uplo=U, trans=T, diag=N):" << std::endl;

    int32_t deviceId = 0;
    constexpr int64_t N = 128;
    constexpr int64_t lda = N;
    constexpr int64_t incx = 1;

    std::vector<float> A(N * N, 1.0f);
    for (int64_t i = 0; i < N; i++) {
        for (int64_t j = 0; j < i; j++) {
            A[i + j * lda] = 0.0f;
        }
        A[i + i * lda] = 2.0f;
    }
    std::vector<float> x(N, 1.0f);
    std::vector<float> xGolden = x;

    ComputeGoldenStrmvUpperT(A.data(), xGolden.data(), N, lda, incx, 0);

    aclInit(nullptr);
    aclrtSetDevice(deviceId);

    aclblasHandle_t handle = nullptr;
    auto ret = aclblasCreate(&handle);
    CHECK_RET(ret == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasCreate failed. ERROR: %d\n", ret); return ret);

    aclrtStream stream = nullptr;
    aclrtCreateStream(&stream);
    ret = aclblasSetStream(handle, stream);
    CHECK_RET(ret == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasSetStream failed. ERROR: %d\n", ret); return ret);

    float* aDevice = nullptr;
    float* xDevice = nullptr;
    size_t aByteSize = N * N * sizeof(float);
    size_t xByteSize = N * sizeof(float);
    aclError aclRet = aclrtMalloc((void**)&aDevice, aByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc aDevice failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMalloc((void**)&xDevice, xByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc xDevice failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMemcpy(aDevice, aByteSize, A.data(), aByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy aDevice failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMemcpy(xDevice, xByteSize, x.data(), xByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy xDevice failed. ERROR: %d\n", aclRet); return aclRet);

    ret = aclblasStrmv(handle, ACLBLAS_UPPER, ACLBLAS_OP_T, ACLBLAS_NON_UNIT, N, aDevice, lda, xDevice, incx);
    CHECK_RET(ret == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasStrmv failed. ERROR: %d\n", ret); return ret);

    aclRet = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMemcpy(x.data(), xByteSize, xDevice, xByteSize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy x failed. ERROR: %d\n", aclRet); return aclRet);

    aclrtFree(aDevice);
    aclrtFree(xDevice);
    aclblasDestroy(handle);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return VerifyResult(x, xGolden);
}

int TestStrmvLowerT()
{
    std::cout << "Testing strmv (uplo=L, trans=T, diag=N):" << std::endl;

    int32_t deviceId = 0;
    constexpr int64_t N = 128;
    constexpr int64_t lda = N;
    constexpr int64_t incx = 1;

    std::vector<float> A(N * N, 1.0f);
    for (int64_t i = 0; i < N; i++) {
        for (int64_t j = i + 1; j < N; j++) {
            A[i + j * lda] = 0.0f;
        }
        A[i + i * lda] = 2.0f;
    }
    std::vector<float> x(N, 1.0f);
    std::vector<float> xGolden = x;

    ComputeGoldenStrmvLowerT(A.data(), xGolden.data(), N, lda, incx, 0);

    aclInit(nullptr);
    aclrtSetDevice(deviceId);

    aclblasHandle_t handle = nullptr;
    auto ret = aclblasCreate(&handle);
    CHECK_RET(ret == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasCreate failed. ERROR: %d\n", ret); return ret);

    aclrtStream stream = nullptr;
    aclrtCreateStream(&stream);
    ret = aclblasSetStream(handle, stream);
    CHECK_RET(ret == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasSetStream failed. ERROR: %d\n", ret); return ret);

    float* aDevice = nullptr;
    float* xDevice = nullptr;
    size_t aByteSize = N * N * sizeof(float);
    size_t xByteSize = N * sizeof(float);
    aclError aclRet = aclrtMalloc((void**)&aDevice, aByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc aDevice failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMalloc((void**)&xDevice, xByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc xDevice failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMemcpy(aDevice, aByteSize, A.data(), aByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy aDevice failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMemcpy(xDevice, xByteSize, x.data(), xByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy xDevice failed. ERROR: %d\n", aclRet); return aclRet);

    ret = aclblasStrmv(handle, ACLBLAS_LOWER, ACLBLAS_OP_T, ACLBLAS_NON_UNIT, N, aDevice, lda, xDevice, incx);
    CHECK_RET(ret == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasStrmv failed. ERROR: %d\n", ret); return ret);

    aclRet = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMemcpy(x.data(), xByteSize, xDevice, xByteSize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy x failed. ERROR: %d\n", aclRet); return aclRet);

    aclrtFree(aDevice);
    aclrtFree(xDevice);
    aclblasDestroy(handle);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return VerifyResult(x, xGolden);
}

// issue #580: handle 为空与非法参数须返回对应错误码，而非解引用崩溃或按垃圾参数下发内核
int TestStrmvNegative()
{
    std::cout << "Testing strmv negative cases:" << std::endl;
    int failed = 0;
    float dummy = 0.0f;

    int32_t deviceId = 0;
    aclInit(nullptr);
    aclrtSetDevice(deviceId);
    aclblasHandle_t handle = nullptr;
    if (aclblasCreate(&handle) != ACLBLAS_STATUS_SUCCESS) {
        std::cout << "[FAIL] aclblasCreate failed" << std::endl;
        return 1;
    }

    // handle 为空：期望 HANDLE_IS_NULLPTR
    auto ret = aclblasStrmv(nullptr, ACLBLAS_UPPER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT, 8, &dummy, 8, &dummy, 1);
    if (ret != ACLBLAS_STATUS_HANDLE_IS_NULLPTR) {
        std::cout << "[FAIL] null handle: expected " << ACLBLAS_STATUS_HANDLE_IS_NULLPTR << " got " << ret << std::endl;
        ++failed;
    }
    // 非法 uplo
    ret = aclblasStrmv(
        handle, static_cast<aclblasFillMode_t>(0), ACLBLAS_OP_N, ACLBLAS_NON_UNIT, 8, &dummy, 8, &dummy, 1);
    if (ret != ACLBLAS_STATUS_INVALID_VALUE) {
        std::cout << "[FAIL] invalid uplo: expected " << ACLBLAS_STATUS_INVALID_VALUE << " got " << ret << std::endl;
        ++failed;
    }
    // 负 n
    ret = aclblasStrmv(handle, ACLBLAS_UPPER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT, -1, &dummy, 8, &dummy, 1);
    if (ret != ACLBLAS_STATUS_INVALID_VALUE) {
        std::cout << "[FAIL] negative n: expected " << ACLBLAS_STATUS_INVALID_VALUE << " got " << ret << std::endl;
        ++failed;
    }
    // lda 过小
    ret = aclblasStrmv(handle, ACLBLAS_UPPER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT, 8, &dummy, 4, &dummy, 1);
    if (ret != ACLBLAS_STATUS_INVALID_VALUE) {
        std::cout << "[FAIL] small lda: expected " << ACLBLAS_STATUS_INVALID_VALUE << " got " << ret << std::endl;
        ++failed;
    }
    // incx 为 0
    ret = aclblasStrmv(handle, ACLBLAS_UPPER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT, 8, &dummy, 8, &dummy, 0);
    if (ret != ACLBLAS_STATUS_INVALID_VALUE) {
        std::cout << "[FAIL] zero incx: expected " << ACLBLAS_STATUS_INVALID_VALUE << " got " << ret << std::endl;
        ++failed;
    }
    // A 为空
    ret = aclblasStrmv(handle, ACLBLAS_UPPER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT, 8, nullptr, 8, &dummy, 1);
    if (ret != ACLBLAS_STATUS_INVALID_VALUE) {
        std::cout << "[FAIL] null A: expected " << ACLBLAS_STATUS_INVALID_VALUE << " got " << ret << std::endl;
        ++failed;
    }
    aclblasDestroy(handle);
    aclrtResetDevice(deviceId);
    aclFinalize();

    if (failed == 0) {
        std::cout << "[PASS] strmv negative cases" << std::endl;
    }
    return failed == 0 ? 0 : 1;
}

int32_t main(int32_t argc, char* argv[])
{
    int ret1 = TestStrmvUpperN();
    int ret2 = TestStrmvLowerN();
    int ret3 = TestStrmvUpperT();
    int ret4 = TestStrmvLowerT();
    int ret5 = TestStrmvNegative();

    if (ret1 == 0 && ret2 == 0 && ret3 == 0 && ret4 == 0 && ret5 == 0) {
        std::cout << "[PASS] strmv_test" << std::endl;
        return 0;
    } else {
        std::cout << "[FAIL] strmv_test" << std::endl;
        return 1;
    }
}