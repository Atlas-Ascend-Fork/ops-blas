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
 * \file cgerc_test.cpp
 * \brief Test for cgerc operator
 */

#include <cstdint>
#include <iostream>
#include <vector>
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

uint32_t VerifyResult(
    const std::vector<aclblasComplex>& output, const std::vector<aclblasComplex>& golden, const char* test_name)
{
    std::cout << "\n========== " << test_name << " ==========" << std::endl;

    // Print first few results for comparison (horizontal format)
    constexpr size_t maxPrintCount = 5;
    std::cout << "\n--- First " << maxPrintCount << " results ---" << std::endl;
    std::cout << "Index  | Output (real, imag)       | Golden (real, imag)        | Diff" << std::endl;
    std::cout << "-------|---------------------------|----------------------------|--------" << std::endl;
    for (size_t i = 0; i < std::min(output.size(), maxPrintCount); i++) {
        printf(
            "%-6zu | (%8.4f, %8.4f) | (%8.4f, %8.4f)  | %8.6f\n", i, output[i].real, output[i].imag, golden[i].real,
            golden[i].imag, blasComplexAbs(output[i] - golden[i]));
    }

    const float epsilon = 1e-3;
    uint32_t errors = 0;
    float maxError = 0.0f;
    size_t maxErrorIndex = 0;

    for (size_t i = 0; i < output.size(); i++) {
        float error = blasComplexAbs(output[i] - golden[i]);
        if (error > maxError) {
            maxError = error;
            maxErrorIndex = i;
        }
        if (error > epsilon) {
            if (errors < 5) { // Only print first 5 errors
                printf(
                    "Mismatch[%zu]: out=(%.4f,%.4f) gold=(%.4f,%.4f) diff=%.6f\n", i, output[i].real, output[i].imag,
                    golden[i].real, golden[i].imag, error);
            }
            errors++;
        }
    }

    std::cout << "\n--- Statistics ---" << std::endl;
    printf(
        "Total: %zu, MaxErr: %.6f @ idx %zu, Threshold: %.6f, Errors: %u\n", output.size(), maxError, maxErrorIndex,
        epsilon, errors);

    if (errors == 0) {
        std::cout << "[Success] " << test_name << " verification passed." << std::endl;
        return 0;
    } else {
        std::cout << "[Failed] " << test_name << " verification failed!" << std::endl;
        return 1;
    }
}

// Compute golden result for cgerc: A = alpha * x * conj(y^T) + A
void computeGolden(
    std::vector<aclblasComplex>& A, const std::vector<aclblasComplex>& x, const std::vector<aclblasComplex>& y,
    const aclblasComplex& alpha, int64_t m, int64_t n)
{
    // A = alpha * x * conj(y^T) + A
    // For each element A[i][j] = alpha * x[i] * conj(y[j]) + A[i][j]
    for (int64_t i = 0; i < m; i++) {
        for (int64_t j = 0; j < n; j++) {
            aclblasComplex conjY{y[j].real, -y[j].imag};
            A[i * n + j] = alpha * x[i] * conjY + A[i * n + j];
        }
    }
}

struct MinimalCgercBuffers {
    aclblasComplex* xDevice = nullptr;
    aclblasComplex* yDevice = nullptr;
    aclblasComplex* ADevice = nullptr;

    void Release()
    {
        if (ADevice != nullptr) {
            aclrtFree(ADevice);
        }
        if (yDevice != nullptr) {
            aclrtFree(yDevice);
        }
        if (xDevice != nullptr) {
            aclrtFree(xDevice);
        }
    }

    uint32_t Fail()
    {
        Release();
        return 1;
    }

    uint32_t Prepare()
    {
        constexpr aclblasComplex xHost{2.0f, 3.0f};
        constexpr aclblasComplex yHost{4.0f, 5.0f};
        constexpr aclblasComplex initialA{7.0f, 11.0f};
        constexpr size_t bytes = sizeof(aclblasComplex);
        aclError aclRet = aclrtMalloc(reinterpret_cast<void**>(&xDevice), bytes, ACL_MEM_MALLOC_HUGE_FIRST);
        if (aclRet != ACL_SUCCESS) {
            LOG_PRINT("minimal cgerc x allocation failed: %d\n", aclRet);
            return Fail();
        }
        aclRet = aclrtMalloc(reinterpret_cast<void**>(&yDevice), bytes, ACL_MEM_MALLOC_HUGE_FIRST);
        if (aclRet != ACL_SUCCESS) {
            LOG_PRINT("minimal cgerc y allocation failed: %d\n", aclRet);
            return Fail();
        }
        aclRet = aclrtMalloc(reinterpret_cast<void**>(&ADevice), bytes, ACL_MEM_MALLOC_HUGE_FIRST);
        if (aclRet != ACL_SUCCESS) {
            LOG_PRINT("minimal cgerc A allocation failed: %d\n", aclRet);
            return Fail();
        }
        aclRet = aclrtMemcpy(xDevice, bytes, &xHost, bytes, ACL_MEMCPY_HOST_TO_DEVICE);
        if (aclRet == ACL_SUCCESS) {
            aclRet = aclrtMemcpy(yDevice, bytes, &yHost, bytes, ACL_MEMCPY_HOST_TO_DEVICE);
        }
        if (aclRet == ACL_SUCCESS) {
            aclRet = aclrtMemcpy(ADevice, bytes, &initialA, bytes, ACL_MEMCPY_HOST_TO_DEVICE);
        }
        if (aclRet != ACL_SUCCESS) {
            LOG_PRINT("minimal cgerc input copy failed: %d\n", aclRet);
            return Fail();
        }
        return 0;
    }
};

uint32_t RunMinimalValidCase(aclblasHandle handle, aclrtStream stream)
{
    constexpr aclblasComplex alpha{1.0f, 0.0f};
    constexpr aclblasComplex expected{30.0f, 13.0f};
    constexpr size_t bytes = sizeof(aclblasComplex);
    aclblasComplex actual{};
    MinimalCgercBuffers buffers;
    if (buffers.Prepare() != 0) {
        return 1;
    }

    const aclblasStatus_t status = aclblasCgerc(handle, 1, 1, &alpha, buffers.xDevice, 1, buffers.yDevice, 1, buffers.ADevice, 1);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        LOG_PRINT("minimal cgerc call failed: %d\n", status);
        return buffers.Fail();
    }
    aclError aclRet = aclrtSynchronizeStream(stream);
    if (aclRet == ACL_SUCCESS) {
        aclRet = aclrtMemcpy(&actual, bytes, buffers.ADevice, bytes, ACL_MEMCPY_DEVICE_TO_HOST);
    }
    if (aclRet != ACL_SUCCESS) {
        LOG_PRINT("minimal cgerc output copy failed: %d\n", aclRet);
        return buffers.Fail();
    }

    const float error = std::hypot(actual.real - expected.real, actual.imag - expected.imag);
    buffers.Release();
    if (error > 1e-3f) {
        LOG_PRINT(
            "minimal cgerc mismatch: got=(%.4f, %.4f), expected=(%.4f, %.4f)\n", actual.real, actual.imag,
            expected.real, expected.imag);
        return 1;
    }
    std::cout << "[Success] minimal valid m=n=1 cgerc verification passed." << std::endl;
    return 0;
}

int32_t main(int32_t argc, char* argv[])
{
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    aclblasHandle handle = nullptr;

    aclError ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); aclFinalize(); return ret);

    ret = aclrtCreateStream(&stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); aclFinalize(); return ret);

    aclblasCreate(&handle);
    aclblasSetStream(handle, stream);

    constexpr int64_t m = 4;
    constexpr int64_t n = 4;
    constexpr int64_t incx = 1;
    constexpr int64_t incy = 1;
    constexpr int64_t lda = m;

    aclblasComplex alpha{1.0f, 0.0f};

    std::vector<aclblasComplex> xHost(m);
    for (int64_t i = 0; i < m; i++) {
        xHost[i] = aclblasComplex{i + 1.0f, i + 0.5f};
    }

    std::vector<aclblasComplex> yHost(n);
    for (int64_t i = 0; i < n; i++) {
        yHost[i] = aclblasComplex{i + 2.0f, i + 1.5f};
    }

    std::vector<aclblasComplex> AHost(m * n);
    for (int64_t i = 0; i < m * n; i++) {
        AHost[i] = aclblasComplex{i * 0.1f, i * 0.2f};
    }

    std::vector<aclblasComplex> goldenHost = AHost;
    computeGolden(goldenHost, xHost, yHost, alpha, m, n);

    size_t xSize = m * sizeof(aclblasComplex);
    size_t ySize = n * sizeof(aclblasComplex);
    size_t aSize = m * n * sizeof(aclblasComplex);

    aclblasComplex* xDevice = nullptr;
    aclblasComplex* yDevice = nullptr;
    aclblasComplex* ADevice = nullptr;

    aclError aclRet = aclrtMalloc((void**)&xDevice, xSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc xDevice failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMalloc((void**)&yDevice, ySize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc yDevice failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMalloc((void**)&ADevice, aSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc ADevice failed. ERROR: %d\n", aclRet); return aclRet);

    aclRet = aclrtMemcpy(xDevice, xSize, xHost.data(), xSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy xDevice failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMemcpy(yDevice, ySize, yHost.data(), ySize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy yDevice failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMemcpy(ADevice, aSize, AHost.data(), aSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy ADevice failed. ERROR: %d\n", aclRet); return aclRet);

    ret = aclblasCgerc(handle, m, n, &alpha, xDevice, incx, yDevice, incy, ADevice, lda);
    CHECK_RET(ret == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasCgerc failed. ERROR: %d\n", ret); return ret);

    aclRet = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet); return aclRet);

    aclRet = aclrtMemcpy(AHost.data(), aSize, ADevice, aSize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy AHost failed. ERROR: %d\n", aclRet); return aclRet);

    aclrtFree(xDevice);
    aclrtFree(yDevice);
    aclrtFree(ADevice);

    const uint32_t minimalCaseErrors = RunMinimalValidCase(handle, stream);

    aclblasDestroy(handle);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return VerifyResult(AHost, goldenHost, "cgerc_test") + minimalCaseErrors;
}
