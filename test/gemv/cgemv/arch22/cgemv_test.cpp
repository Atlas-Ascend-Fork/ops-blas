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

constexpr float EPSILON = 1e-3f;

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
        if (diff > EPSILON) {
            std::cout << "[Failed] Index " << i << ": output=(" << output[i].real << "," << output[i].imag
                      << ") golden=(" << golden[i].real << "," << golden[i].imag << ") diff=" << diff << std::endl;
            return 1;
        }
    }
    std::cout << "[Success] Case accuracy is verification passed." << std::endl;
    return 0;
}

void ComputeGoldenCgemvN(
    const aclblasComplex* A, const aclblasComplex* x, aclblasComplex* y,
    const aclblasComplex& alpha, const aclblasComplex& beta, int64_t m, int64_t n, int64_t lda, int64_t incx,
    int64_t incy)
{
    for (int64_t i = 0; i < m; i++) {
        aclblasComplex sum{0.0f, 0.0f};
        for (int64_t j = 0; j < n; j++) {
            sum += A[i + j * lda] * x[j * incx];
        }
        y[i * incy] = alpha * sum + beta * y[i * incy];
    }
}

void ComputeGoldenCgemvT(
    const aclblasComplex* A, const aclblasComplex* x, aclblasComplex* y,
    const aclblasComplex& alpha, const aclblasComplex& beta, int64_t m, int64_t n, int64_t lda, int64_t incx,
    int64_t incy)
{
    for (int64_t j = 0; j < n; j++) {
        aclblasComplex sum{0.0f, 0.0f};
        for (int64_t i = 0; i < m; i++) {
            sum += A[i + j * lda] * x[i * incx];
        }
        y[j * incy] = alpha * sum + beta * y[j * incy];
    }
}

void ComputeGoldenCgemvC(
    const aclblasComplex* A, const aclblasComplex* x, aclblasComplex* y,
    const aclblasComplex& alpha, const aclblasComplex& beta, int64_t m, int64_t n, int64_t lda, int64_t incx,
    int64_t incy)
{
    for (int64_t j = 0; j < n; j++) {
        aclblasComplex sum{0.0f, 0.0f};
        for (int64_t i = 0; i < m; i++) {
            const aclblasComplex aConj{A[i + j * lda].real, -A[i + j * lda].imag};
            sum += aConj * x[i * incx];
        }
        y[j * incy] = alpha * sum + beta * y[j * incy];
    }
}

int TestCgemvNoTrans()
{
    std::cout << "Testing cgemv (trans=N):" << std::endl;

    int32_t deviceId = 0;
    constexpr int64_t M = 512;
    constexpr int64_t N = 256;
    constexpr int64_t lda = M;
    constexpr int64_t incx = 1;
    constexpr int64_t incy = 1;

    const aclblasComplex alpha{1.5f, 0.5f};
    const aclblasComplex beta{0.5f, 0.25f};

    std::vector<aclblasComplex> A(M * N, aclblasComplex{1.0f, 0.5f});
    std::vector<aclblasComplex> x(N, aclblasComplex{2.0f, 1.0f});
    std::vector<aclblasComplex> y(M, aclblasComplex{3.0f, 1.5f});
    std::vector<aclblasComplex> yGolden = y;

    ComputeGoldenCgemvN(A.data(), x.data(), yGolden.data(), alpha, beta, M, N, lda, incx, incy);

    aclInit(nullptr);
    aclrtSetDevice(deviceId);

    aclblasHandle_t handle = nullptr;
    auto ret = aclblasCreate(&handle);
    CHECK_RET(ret == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasCreate failed. ERROR: %d\n", ret); return ret);

    aclrtStream stream = nullptr;
    aclrtCreateStream(&stream);
    ret = aclblasSetStream(handle, stream);
    CHECK_RET(ret == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasSetStream failed. ERROR: %d\n", ret); return ret);

    aclblasComplex* aDevice = nullptr;
    aclblasComplex* xDevice = nullptr;
    aclblasComplex* yDevice = nullptr;
    size_t aByteSize = M * N * sizeof(aclblasComplex);
    size_t xByteSize = N * sizeof(aclblasComplex);
    size_t yByteSize = M * sizeof(aclblasComplex);
    aclError aclRet = aclrtMalloc((void**)&aDevice, aByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc aDevice failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMalloc((void**)&xDevice, xByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc xDevice failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMalloc((void**)&yDevice, yByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc yDevice failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMemcpy(aDevice, aByteSize, A.data(), aByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy aDevice failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMemcpy(xDevice, xByteSize, x.data(), xByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy xDevice failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMemcpy(yDevice, yByteSize, y.data(), yByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy yDevice failed. ERROR: %d\n", aclRet); return aclRet);

    ret = aclblasCgemv(handle, ACLBLAS_OP_N, M, N, &alpha, aDevice, lda, xDevice, incx, &beta, yDevice, incy);
    CHECK_RET(ret == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasCgemv failed. ERROR: %d\n", ret); return ret);

    aclRet = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMemcpy(y.data(), yByteSize, yDevice, yByteSize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy y failed. ERROR: %d\n", aclRet); return aclRet);

    aclrtFree(aDevice);
    aclrtFree(xDevice);
    aclrtFree(yDevice);
    aclblasDestroy(handle);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return VerifyResult(y, yGolden);
}

int TestCgemvTrans()
{
    std::cout << "Testing cgemv (trans=T/C):" << std::endl;

    int32_t deviceId = 0;
    constexpr int64_t M = 512;
    constexpr int64_t N = 256;
    constexpr int64_t lda = M;
    constexpr int64_t incx = 1;
    constexpr int64_t incy = 1;

    const aclblasComplex alpha{1.5f, 0.5f};
    const aclblasComplex beta{0.5f, 0.25f};

    std::vector<aclblasComplex> A(M * N, aclblasComplex{1.0f, 0.5f});
    std::vector<aclblasComplex> x(M, aclblasComplex{2.0f, 1.0f});
    std::vector<aclblasComplex> yInitial(N, aclblasComplex{3.0f, 1.5f});
    std::vector<aclblasComplex> y = yInitial;
    std::vector<aclblasComplex> yGolden = yInitial;

    ComputeGoldenCgemvT(A.data(), x.data(), yGolden.data(), alpha, beta, M, N, lda, incx, incy);

    aclInit(nullptr);
    aclrtSetDevice(deviceId);

    aclblasHandle_t handle = nullptr;
    auto ret = aclblasCreate(&handle);
    CHECK_RET(ret == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasCreate failed. ERROR: %d\n", ret); return ret);

    aclrtStream stream = nullptr;
    aclrtCreateStream(&stream);
    ret = aclblasSetStream(handle, stream);
    CHECK_RET(ret == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasSetStream failed. ERROR: %d\n", ret); return ret);

    aclblasComplex* aDevice = nullptr;
    aclblasComplex* xDevice = nullptr;
    aclblasComplex* yDevice = nullptr;
    size_t aByteSize = M * N * sizeof(aclblasComplex);
    size_t xByteSize = M * sizeof(aclblasComplex);
    size_t yByteSize = N * sizeof(aclblasComplex);
    aclError aclRet = aclrtMalloc((void**)&aDevice, aByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc aDevice failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMalloc((void**)&xDevice, xByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc xDevice failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMalloc((void**)&yDevice, yByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc yDevice failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMemcpy(aDevice, aByteSize, A.data(), aByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy aDevice failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMemcpy(xDevice, xByteSize, x.data(), xByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy xDevice failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMemcpy(yDevice, yByteSize, y.data(), yByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy yDevice failed. ERROR: %d\n", aclRet); return aclRet);

    ret = aclblasCgemv(handle, ACLBLAS_OP_T, M, N, &alpha, aDevice, lda, xDevice, incx, &beta, yDevice, incy);
    CHECK_RET(ret == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasCgemv failed. ERROR: %d\n", ret); return ret);

    aclRet = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMemcpy(y.data(), yByteSize, yDevice, yByteSize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy y failed. ERROR: %d\n", aclRet); return aclRet);
    const int transResult = VerifyResult(y, yGolden);

    y = yInitial;
    yGolden = yInitial;
    ComputeGoldenCgemvC(A.data(), x.data(), yGolden.data(), alpha, beta, M, N, lda, incx, incy);
    aclRet = aclrtMemcpy(yDevice, yByteSize, y.data(), yByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy initial y failed. ERROR: %d\n", aclRet); return aclRet);

    ret = aclblasCgemv(handle, ACLBLAS_OP_C, M, N, &alpha, aDevice, lda, xDevice, incx, &beta, yDevice, incy);
    CHECK_RET(ret == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasCgemv OP_C failed. ERROR: %d\n", ret); return ret);
    aclRet = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream OP_C failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMemcpy(y.data(), yByteSize, yDevice, yByteSize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy OP_C result failed. ERROR: %d\n", aclRet); return aclRet);
    const int conjResult = VerifyResult(y, yGolden);

    aclrtFree(aDevice);
    aclrtFree(xDevice);
    aclrtFree(yDevice);
    aclblasDestroy(handle);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return (transResult == 0 && conjResult == 0) ? 0 : 1;
}

int32_t main(int32_t argc, char* argv[])
{
    int ret1 = TestCgemvNoTrans();
    int ret2 = TestCgemvTrans();

    if (ret1 == 0 && ret2 == 0) {
        std::cout << "[PASS] cgemv_test" << std::endl;
        return 0;
    } else {
        std::cout << "[FAIL] cgemv_test" << std::endl;
        return 1;
    }
}
