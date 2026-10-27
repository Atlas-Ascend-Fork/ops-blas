/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iterator>
#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "complex.h"

using namespace std;

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

aclblasComplex complexMul(const aclblasComplex& a, const aclblasComplex& b) { return a * b; }

void computeGolden(
    const vector<aclblasComplex>& A, const vector<aclblasComplex>& x, vector<aclblasComplex>& y, int m, int n,
    int trans, const aclblasComplex& alpha, const aclblasComplex& beta)
{
    vector<aclblasComplex> yTemp(m);
    for (int i = 0; i < m; i++) {
        aclblasComplex sum{0.0f, 0.0f};
        for (int j = 0; j < n; j++) {
            if (trans == 0) {
                sum += A[i * n + j] * x[j];
            } else if (trans == 1) {
                sum += A[j * m + i] * x[j];
            } else {
                aclblasComplex conjA{A[j * m + i].real, -A[j * m + i].imag};
                sum += conjA * x[j];
            }
        }
        yTemp[i] = sum;
    }

    for (int i = 0; i < m; i++) {
        y[i] = alpha * yTemp[i] + beta * y[i];
    }
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

    constexpr float epsilon = 1e-3f;
    size_t errorCount = 0;
    float maxError = 0.0f;

    for (size_t i = 0; i < output.size(); i++) {
        float error = blasComplexAbs(output[i] - golden[i]);
        if (error > maxError) {
            maxError = error;
        }
        if (error > epsilon) {
            errorCount++;
        }
    }

    std::cout << "Max error: " << maxError << std::endl;

    if (errorCount == 0) {
        std::cout << "[Success] Case accuracy is verification passed." << std::endl;
        return 0;
    } else {
        std::cout << "[Failed] Case accuracy verification failed." << std::endl;
        return 1;
    }
}

int main()
{
    aclError ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);

    int32_t deviceId = 0;
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); aclFinalize(); return ret);

    aclblasHandle_t handle = nullptr;
    auto blasRet = aclblasCreate(&handle);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasCreate failed. ERROR: %d\n", blasRet); aclFinalize();
              return blasRet);

    aclrtStream stream = nullptr;
    ret = aclrtCreateStream(&stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); aclblasDestroy(handle);
              aclFinalize(); return ret);

    blasRet = aclblasSetStream(handle, stream);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasSetStream failed. ERROR: %d\n", blasRet);
              aclblasDestroy(handle); aclrtDestroyStream(stream); aclFinalize(); return blasRet);

    const int64_t batchCount = 3;
    const int64_t m = 32;
    const int64_t n = 32;
    const aclblasOperation_t trans = ACLBLAS_OP_N;

    aclblasComplex alpha{1.0f, 0.0f};
    aclblasComplex beta{0.0f, 0.0f};

    const int64_t lda = m;
    const int64_t incx = 1;
    const int64_t incy = 1;

    vector<aclblasComplex> AHost(batchCount * m * n);
    vector<aclblasComplex> xHost(batchCount * n);
    vector<aclblasComplex> yHost(batchCount * m);
    vector<aclblasComplex> yGolden(batchCount * m);

    for (int b = 0; b < batchCount; b++) {
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                int idx = b * m * n + i * n + j;
                AHost[idx] = aclblasComplex{(i + j + b) * 0.1f, (i - j + b) * 0.1f};
            }
        }

        for (int i = 0; i < n; i++) {
            int idx = b * n + i;
            xHost[idx] = aclblasComplex{(i + b) * 0.2f, (i - b) * 0.2f};
        }
    }

    auto computeBatchGolden = [&](int transMode) {
        for (int b = 0; b < batchCount; b++) {
            vector<aclblasComplex> AComplex(m * n);
            vector<aclblasComplex> xComplex(n);
            vector<aclblasComplex> yComplex(m);
            for (int i = 0; i < m * n; i++) {
                AComplex[i] = AHost[b * m * n + i];
            }
            for (int i = 0; i < n; i++) {
                xComplex[i] = xHost[b * n + i];
            }
            computeGolden(AComplex, xComplex, yComplex, m, n, transMode, alpha, beta);
            for (int i = 0; i < m; i++) {
                yGolden[b * m + i] = yComplex[i];
            }
        }
    };
    computeBatchGolden(0);

    aclblasComplex* aDevice = nullptr;
    aclblasComplex* xDevice = nullptr;
    aclblasComplex* yDevice = nullptr;
    size_t aByteSize = batchCount * m * n * sizeof(aclblasComplex);
    size_t xByteSize = batchCount * n * sizeof(aclblasComplex);
    size_t yByteSize = batchCount * m * sizeof(aclblasComplex);
    ret = aclrtMalloc((void**)&aDevice, aByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc aDevice failed. ERROR: %d\n", ret); aclrtFree(aDevice);
              aclblasDestroy(handle); aclrtDestroyStream(stream); aclFinalize(); return ret);
    ret = aclrtMalloc((void**)&xDevice, xByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc xDevice failed. ERROR: %d\n", ret); aclrtFree(aDevice);
              aclrtFree(xDevice); aclblasDestroy(handle); aclrtDestroyStream(stream); aclFinalize(); return ret);
    ret = aclrtMalloc((void**)&yDevice, yByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc yDevice failed. ERROR: %d\n", ret); aclrtFree(aDevice);
              aclrtFree(xDevice); aclrtFree(yDevice); aclblasDestroy(handle); aclrtDestroyStream(stream); aclFinalize();
              return ret);
    ret = aclrtMemcpy(aDevice, aByteSize, AHost.data(), aByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy aDevice failed. ERROR: %d\n", ret); aclrtFree(aDevice);
              aclrtFree(xDevice); aclrtFree(yDevice); aclblasDestroy(handle); aclrtDestroyStream(stream); aclFinalize();
              return ret);
    ret = aclrtMemcpy(xDevice, xByteSize, xHost.data(), xByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy xDevice failed. ERROR: %d\n", ret); aclrtFree(aDevice);
              aclrtFree(xDevice); aclrtFree(yDevice); aclblasDestroy(handle); aclrtDestroyStream(stream); aclFinalize();
              return ret);
    ret = aclrtMemcpy(yDevice, yByteSize, yHost.data(), yByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy yDevice failed. ERROR: %d\n", ret); aclrtFree(aDevice);
              aclrtFree(xDevice); aclrtFree(yDevice); aclblasDestroy(handle); aclrtDestroyStream(stream); aclFinalize();
              return ret);

    blasRet =
        aclblasCgemvBatched(handle, trans, m, n, &alpha, aDevice, lda, xDevice, incx, &beta, yDevice, incy, batchCount);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasCgemvBatched failed. ERROR: %d\n", blasRet);
              aclrtFree(aDevice); aclrtFree(xDevice); aclrtFree(yDevice); aclblasDestroy(handle);
              aclrtDestroyStream(stream); aclFinalize(); return blasRet);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); aclrtFree(aDevice);
              aclrtFree(xDevice); aclrtFree(yDevice); aclblasDestroy(handle); aclrtDestroyStream(stream); aclFinalize();
              return ret);
    ret = aclrtMemcpy(yHost.data(), yByteSize, yDevice, yByteSize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy yHost failed. ERROR: %d\n", ret); aclrtFree(aDevice);
              aclrtFree(xDevice); aclrtFree(yDevice); aclblasDestroy(handle); aclrtDestroyStream(stream); aclFinalize();
              return ret);

    vector<aclblasComplex> yInitial(yHost.size(), aclblasComplex{0.0f, 0.0f});
    const bool normalResultPassed = VerifyResult(yHost, yGolden) == 0;
    ret = aclrtMemcpy(yDevice, yByteSize, yInitial.data(), yByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy y reset failed. ERROR: %d\n", ret); aclrtFree(aDevice);
              aclrtFree(xDevice); aclrtFree(yDevice); aclblasDestroy(handle); aclrtDestroyStream(stream); aclFinalize();
              return ret);

    blasRet = aclblasCgemvBatched(
        handle, ACLBLAS_OP_T, m, n, &alpha, aDevice, lda, xDevice, incx, &beta, yDevice, incy, batchCount);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasCgemvBatched(OP_T) failed. ERROR: %d\n", blasRet);
              aclrtFree(aDevice); aclrtFree(xDevice); aclrtFree(yDevice); aclblasDestroy(handle);
              aclrtDestroyStream(stream); aclFinalize(); return blasRet);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream after OP_T failed. ERROR: %d\n", ret);
              aclrtFree(aDevice); aclrtFree(xDevice); aclrtFree(yDevice); aclblasDestroy(handle);
              aclrtDestroyStream(stream); aclFinalize(); return ret);
    ret = aclrtMemcpy(yHost.data(), yByteSize, yDevice, yByteSize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy OP_T result failed. ERROR: %d\n", ret); aclrtFree(aDevice);
              aclrtFree(xDevice); aclrtFree(yDevice); aclblasDestroy(handle); aclrtDestroyStream(stream); aclFinalize();
              return ret);
    computeBatchGolden(1);
    const bool transposeResultPassed = VerifyResult(yHost, yGolden) == 0;

    ret = aclrtMemcpy(yDevice, yByteSize, yInitial.data(), yByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy y reset failed. ERROR: %d\n", ret); aclrtFree(aDevice);
              aclrtFree(xDevice); aclrtFree(yDevice); aclblasDestroy(handle); aclrtDestroyStream(stream); aclFinalize();
              return ret);

    blasRet = aclblasCgemvBatched(
        handle, ACLBLAS_OP_C, m, n, &alpha, aDevice, lda, xDevice, incx, &beta, yDevice, incy, batchCount);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasCgemvBatched(OP_C) failed. ERROR: %d\n", blasRet);
              aclrtFree(aDevice); aclrtFree(xDevice); aclrtFree(yDevice); aclblasDestroy(handle);
              aclrtDestroyStream(stream); aclFinalize(); return blasRet);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream after OP_C failed. ERROR: %d\n", ret);
              aclrtFree(aDevice); aclrtFree(xDevice); aclrtFree(yDevice); aclblasDestroy(handle);
              aclrtDestroyStream(stream); aclFinalize(); return ret);
    ret = aclrtMemcpy(yHost.data(), yByteSize, yDevice, yByteSize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy OP_C result failed. ERROR: %d\n", ret); aclrtFree(aDevice);
              aclrtFree(xDevice); aclrtFree(yDevice); aclblasDestroy(handle); aclrtDestroyStream(stream); aclFinalize();
              return ret);
    computeBatchGolden(2);
    const bool conjugateResultPassed = VerifyResult(yHost, yGolden) == 0;

    aclrtFree(aDevice);
    aclrtFree(xDevice);
    aclrtFree(yDevice);
    aclblasDestroy(handle);
    aclrtDestroyStream(stream);
    aclFinalize();

    return (normalResultPassed && transposeResultPassed && conjugateResultPassed) ? ACL_SUCCESS : 1;
}
