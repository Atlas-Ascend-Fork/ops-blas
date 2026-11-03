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
// issue #732: 支持步长与 lda；负步长按本仓惯例从物理向量尾端起算
void computeGolden(
    std::vector<aclblasComplex>& A, const std::vector<aclblasComplex>& x, const std::vector<aclblasComplex>& y,
    const aclblasComplex& alpha, int64_t m, int64_t n, int64_t incx = 1, int64_t incy = 1, int64_t lda = 0)
{
    const int64_t ld = (lda > 0) ? lda : n;
    const int64_t absIncx = (incx > 0) ? incx : -incx;
    const int64_t absIncy = (incy > 0) ? incy : -incy;
    for (int64_t i = 0; i < m; i++) {
        const int64_t xPhys = (incx > 0) ? i * absIncx : (m - 1 - i) * absIncx;
        for (int64_t j = 0; j < n; j++) {
            const int64_t yPhys = (incy > 0) ? j * absIncy : (n - 1 - j) * absIncy;
            aclblasComplex conjY{y[yPhys].real, -y[yPhys].imag};
            A[i * ld + j] = alpha * x[xPhys] * conjY + A[i * ld + j];
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

    const aclblasStatus_t status =
        aclblasCgerc(handle, 1, 1, &alpha, buffers.xDevice, 1, buffers.yDevice, 1, buffers.ADevice, 1);
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

// issue #779: 步长回归提取为独立函数，控制 main() 圈复杂度
// issue #809: 测试设备缓冲分配/拷贝/释放共享 helper
static aclblasComplex* AllocAndCopyComplex(const std::vector<aclblasComplex>& host)
{
    aclblasComplex* dev = nullptr;
    aclError ret = aclrtMalloc((void**)&dev, host.size() * sizeof(aclblasComplex), ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        return nullptr;
    }
    ret = aclrtMemcpy(
        dev, host.size() * sizeof(aclblasComplex), host.data(), host.size() * sizeof(aclblasComplex),
        ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        aclrtFree(dev);
        return nullptr;
    }
    return dev;
}

static void FreeCgercDevice(aclblasComplex* sxDev, aclblasComplex* syDev, aclblasComplex* sADev)
{
    if (sxDev != nullptr)
        aclrtFree(sxDev);
    if (syDev != nullptr)
        aclrtFree(syDev);
    if (sADev != nullptr)
        aclrtFree(sADev);
}

// issue #809: 单个步长用例执行独立成函数
static uint32_t RunCgercStrideCase(
    aclblasHandle_t handle, aclrtStream stream, const aclblasComplex& alpha, const char* name, int64_t incx,
    int64_t incy, int64_t lda)
{
    constexpr int64_t m = 8;
    constexpr int64_t n = 8;
    const int64_t ld = (lda > 0) ? lda : m;
    const int64_t xPhysLen = 1 + (m - 1) * ((incx > 0) ? incx : -incx);
    const int64_t yPhysLen = 1 + (n - 1) * ((incy > 0) ? incy : -incy);
    std::vector<aclblasComplex> sx(xPhysLen);
    std::vector<aclblasComplex> sy(yPhysLen);
    for (int64_t i = 0; i < xPhysLen; i++) {
        sx[i] = aclblasComplex{i + 1.0f, i + 0.5f};
    }
    for (int64_t i = 0; i < yPhysLen; i++) {
        sy[i] = aclblasComplex{i + 2.0f, i + 1.5f};
    }
    std::vector<aclblasComplex> sA(m * ld);
    for (int64_t i = 0; i < m * ld; i++) {
        sA[i] = aclblasComplex{i * 0.1f, i * 0.2f};
    }
    std::vector<aclblasComplex> sGolden = sA;
    computeGolden(sGolden, sx, sy, alpha, m, n, incx, incy, ld);
    aclblasComplex* sxDev = AllocAndCopyComplex(sx);
    aclblasComplex* syDev = AllocAndCopyComplex(sy);
    aclblasComplex* sADev = AllocAndCopyComplex(sA);
    if (sxDev == nullptr || syDev == nullptr || sADev == nullptr) {
        std::cout << "[Failed] stride case " << name << " setup failed" << std::endl;
        FreeCgercDevice(sxDev, syDev, sADev);
        return 1;
    }
    const aclblasStatus_t sStatus = aclblasCgerc(handle, m, n, &alpha, sxDev, incx, syDev, incy, sADev, ld);
    const aclError syncRet = aclrtSynchronizeStream(stream);
    uint32_t errors = 0;
    if (sStatus != ACLBLAS_STATUS_SUCCESS || syncRet != ACL_SUCCESS) {
        std::cout << "[Failed] stride case " << name << " status=" << static_cast<int>(sStatus) << std::endl;
        errors = 1;
    } else {
        std::vector<aclblasComplex> sOut(m * ld);
        aclrtMemcpy(
            sOut.data(), sA.size() * sizeof(aclblasComplex), sADev, sA.size() * sizeof(aclblasComplex),
            ACL_MEMCPY_DEVICE_TO_HOST);
        errors = VerifyResult(sOut, sGolden, name);
    }
    FreeCgercDevice(sxDev, syDev, sADev);
    return errors;
}
// issue #809: 步长回归提取为独立函数（循环调度到 RunCgercStrideCase）
uint32_t TestCgercStrided(aclblasHandle_t handle, aclrtStream stream, const aclblasComplex& alpha)
{
    uint32_t strideErrors = 0;
    strideErrors += RunCgercStrideCase(handle, stream, alpha, "incx2", 2, 1, 0);
    strideErrors += RunCgercStrideCase(handle, stream, alpha, "negx", -2, 1, 0);
    strideErrors += RunCgercStrideCase(handle, stream, alpha, "lda", 2, 1, 9);
    return strideErrors;
}

// issue #779: 负向用例提取为独立函数
uint32_t TestCgercNegative(aclblasHandle_t handle, const aclblasComplex& alpha)
{
    uint32_t negativeErrors = 0;
    aclblasComplex dummy{1.0f, 1.0f};
    struct NegCase {
        const char* name;
        aclblasStatus_t want;
    };
    const NegCase negCases[] = {
        {"null handle", ACLBLAS_STATUS_HANDLE_IS_NULLPTR},
        {"negative m", ACLBLAS_STATUS_INVALID_VALUE},
        {"zero incx", ACLBLAS_STATUS_INVALID_VALUE},
        {"small lda", ACLBLAS_STATUS_INVALID_VALUE},
        {"null x", ACLBLAS_STATUS_INVALID_VALUE}};
    for (const auto& nc : negCases) {
        aclblasStatus_t got = ACLBLAS_STATUS_SUCCESS;
        if (std::string(nc.name) == "null handle") {
            got = aclblasCgerc(nullptr, 4, 4, &alpha, &dummy, 1, &dummy, 1, &dummy, 4);
        } else if (std::string(nc.name) == "negative m") {
            got = aclblasCgerc(handle, -1, 4, &alpha, &dummy, 1, &dummy, 1, &dummy, 4);
        } else if (std::string(nc.name) == "zero incx") {
            got = aclblasCgerc(handle, 4, 4, &alpha, &dummy, 0, &dummy, 1, &dummy, 4);
        } else if (std::string(nc.name) == "small lda") {
            got = aclblasCgerc(handle, 4, 4, &alpha, &dummy, 1, &dummy, 1, &dummy, 2);
        } else {
            got = aclblasCgerc(handle, 4, 4, &alpha, nullptr, 1, &dummy, 1, &dummy, 4);
        }
        if (got != nc.want) {
            std::cout << "[Failed] negative case " << nc.name << ": expected " << static_cast<int>(nc.want) << " got "
                      << static_cast<int>(got) << std::endl;
            negativeErrors++;
        }
    }
    return negativeErrors;
}

// issue #809: 测试数据初始化独立成函数，控制 TestCgercContiguous 规模
static void InitCgercTestData(
    std::vector<aclblasComplex>& xHost, std::vector<aclblasComplex>& yHost, std::vector<aclblasComplex>& AHost,
    std::vector<aclblasComplex>& goldenHost, const aclblasComplex& alpha, int64_t m, int64_t n)
{
    for (int64_t i = 0; i < m; i++) {
        xHost[i] = aclblasComplex{i + 1.0f, i + 0.5f};
    }
    for (int64_t i = 0; i < n; i++) {
        yHost[i] = aclblasComplex{i + 2.0f, i + 1.5f};
    }
    for (int64_t i = 0; i < m * n; i++) {
        AHost[i] = aclblasComplex{i * 0.1f, i * 0.2f};
    }
    goldenHost = AHost;
    computeGolden(goldenHost, xHost, yHost, alpha, m, n);
}

// issue #779: 连续主用例提取为独立函数，控制 main() 规模
uint32_t TestCgercContiguous(aclblasHandle_t handle, aclrtStream stream)
{
    constexpr int64_t m = 4;
    constexpr int64_t n = 4;
    constexpr int64_t incx = 1;
    constexpr int64_t incy = 1;
    constexpr int64_t lda = m;
    aclblasComplex alpha{1.0f, 0.0f};
    std::vector<aclblasComplex> xHost(m), yHost(n), AHost(m * n), goldenHost;
    InitCgercTestData(xHost, yHost, AHost, goldenHost, alpha, m, n);

    aclblasComplex* xDevice = AllocAndCopyComplex(xHost);
    aclblasComplex* yDevice = AllocAndCopyComplex(yHost);
    aclblasComplex* ADevice = AllocAndCopyComplex(AHost);
    if (xDevice == nullptr || yDevice == nullptr || ADevice == nullptr) {
        std::cout << "[Failed] device buffer setup failed" << std::endl;
        FreeCgercDevice(xDevice, yDevice, ADevice);
        return 1;
    }

    const aclblasStatus_t status = aclblasCgerc(handle, m, n, &alpha, xDevice, incx, yDevice, incy, ADevice, lda);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        std::cout << "[Failed] aclblasCgerc failed. ERROR: " << static_cast<int>(status) << std::endl;
        FreeCgercDevice(xDevice, yDevice, ADevice);
        return 1;
    }

    const aclError syncRet = aclrtSynchronizeStream(stream);
    if (syncRet != ACL_SUCCESS) {
        std::cout << "[Failed] aclrtSynchronizeStream failed. ERROR: " << static_cast<int>(syncRet) << std::endl;
        FreeCgercDevice(xDevice, yDevice, ADevice);
        return 1;
    }

    aclError aclRet = aclrtMemcpy(
        AHost.data(), AHost.size() * sizeof(aclblasComplex), ADevice, AHost.size() * sizeof(aclblasComplex),
        ACL_MEMCPY_DEVICE_TO_HOST);
    if (aclRet != ACL_SUCCESS) {
        std::cout << "[Failed] aclrtMemcpy AHost failed. ERROR: " << static_cast<int>(aclRet) << std::endl;
        FreeCgercDevice(xDevice, yDevice, ADevice);
        return 1;
    }

    FreeCgercDevice(xDevice, yDevice, ADevice);

    return VerifyResult(AHost, goldenHost, "cgerc_test");
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

    const aclblasComplex alpha{1.0f, 0.0f};
    const uint32_t contiguousResult = TestCgercContiguous(handle, stream);

    const uint32_t strideErrors = TestCgercStrided(handle, stream, alpha);
    const uint32_t negativeErrors = TestCgercNegative(handle, alpha);

    const uint32_t minimalCaseErrors = RunMinimalValidCase(handle, stream);

    aclblasDestroy(handle);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return contiguousResult + strideErrors + negativeErrors + minimalCaseErrors;
}
