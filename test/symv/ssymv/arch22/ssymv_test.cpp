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
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "cann_ops_blas_common.h"

struct TestCase {
    const char* name;
    aclblasFillMode_t uplo;
    int n;
    int lda;
    float alpha;
    float beta;
    int incx;
    int incy;
    aclblasStatus_t expectResult;
    float tol;
};

static const std::vector<TestCase> gSsymvTests = {
    {"UPPER n=32 lda=32 incx=1 incy=1", ACLBLAS_UPPER, 32, 32, 0.8f, 1.2f, 1, 1, ACLBLAS_STATUS_SUCCESS, 1e-5f},
    {"LOWER n=32 lda=32 incx=1 incy=1", ACLBLAS_LOWER, 32, 32, 0.8f, 1.2f, 1, 1, ACLBLAS_STATUS_SUCCESS, 1e-5f},
    {"UPPER n=64 lda=64 incx=1 incy=1", ACLBLAS_UPPER, 64, 64, 0.5f, 1.5f, 1, 1, ACLBLAS_STATUS_SUCCESS, 1e-5f},
    {"LOWER n=64 lda=64 incx=1 incy=1", ACLBLAS_LOWER, 64, 64, 0.5f, 1.5f, 1, 1, ACLBLAS_STATUS_SUCCESS, 1e-5f},
    {"UPPER n=128 lda=128 incx=1 incy=1", ACLBLAS_UPPER, 128, 128, 1.0f, 1.0f, 1, 1, ACLBLAS_STATUS_SUCCESS, 1e-5f},
    {"LOWER n=128 lda=128 incx=1 incy=1", ACLBLAS_LOWER, 128, 128, 1.0f, 1.0f, 1, 1, ACLBLAS_STATUS_SUCCESS, 1e-5f},
    {"UPPER n=256 lda=256 incx=1 incy=1", ACLBLAS_UPPER, 256, 256, 2.0f, 0.5f, 1, 1, ACLBLAS_STATUS_SUCCESS, 1e-5f},
    {"LOWER n=256 lda=256 incx=1 incy=1", ACLBLAS_LOWER, 256, 256, 2.0f, 0.5f, 1, 1, ACLBLAS_STATUS_SUCCESS, 1e-5f},
    {"UPPER n=512 lda=512 incx=1 incy=1", ACLBLAS_UPPER, 512, 512, 1.2f, 0.8f, 1, 1, ACLBLAS_STATUS_SUCCESS, 1e-4f},
    {"LOWER n=512 lda=512 incx=1 incy=1", ACLBLAS_LOWER, 512, 512, 1.2f, 0.8f, 1, 1, ACLBLAS_STATUS_SUCCESS, 1e-4f},
    {"UPPER n=1024 lda=1024 incx=1 incy=1", ACLBLAS_UPPER, 1024, 1024, 1.0f, 0.5f, 1, 1, ACLBLAS_STATUS_SUCCESS, 1e-4f},
    {"LOWER n=1024 lda=1024 incx=1 incy=1", ACLBLAS_LOWER, 1024, 1024, 1.0f, 0.5f, 1, 1, ACLBLAS_STATUS_SUCCESS, 1e-4f},
    {"UPPER n=2048 lda=2048 incx=1 incy=1", ACLBLAS_UPPER, 2048, 2048, 1.2f, 0.8f, 1, 1, ACLBLAS_STATUS_SUCCESS, 1e-3f},
    {"LOWER n=2048 lda=2048 incx=1 incy=1", ACLBLAS_LOWER, 2048, 2048, 1.2f, 0.8f, 1, 1, ACLBLAS_STATUS_SUCCESS, 1e-3f},
    {"LOWER n=32 lda=40 incx=1 incy=1", ACLBLAS_LOWER, 32, 40, 1.0f, 0.5f, 1, 1, ACLBLAS_STATUS_SUCCESS, 1e-5f},
    {"UPPER n=64 lda=80 incx=1 incy=1", ACLBLAS_UPPER, 64, 80, 1.0f, 1.0f, 1, 1, ACLBLAS_STATUS_SUCCESS, 1e-5f},
    {"UPPER n=32 incx=2 incy=1", ACLBLAS_UPPER, 32, 32, 1.0f, 0.0f, 2, 1, ACLBLAS_STATUS_SUCCESS, 1e-5f},
    {"LOWER n=64 incx=-2 incy=1", ACLBLAS_LOWER, 64, 64, 1.0f, 0.0f, -2, 1, ACLBLAS_STATUS_SUCCESS, 1e-5f},
    {"LOWER n=32 incx=-1 incy=1", ACLBLAS_LOWER, 32, 32, 0.5f, 0.0f, -1, 1, ACLBLAS_STATUS_SUCCESS, 1e-5f},
    {"UPPER n=32 incx=1 incy=2", ACLBLAS_UPPER, 32, 32, 1.0f, 0.0f, 1, 2, ACLBLAS_STATUS_SUCCESS, 1e-5f},
    {"LOWER n=32 incx=1 incy=-1", ACLBLAS_LOWER, 32, 32, 1.0f, 0.0f, 1, -1, ACLBLAS_STATUS_SUCCESS, 1e-5f},
    {"n=0", ACLBLAS_LOWER, 0, 1, 1.0f, 0.0f, 1, 1, ACLBLAS_STATUS_SUCCESS, 0.0f},
    {"n=1 UPPER", ACLBLAS_UPPER, 1, 1, 1.2f, 0.5f, 1, 1, ACLBLAS_STATUS_SUCCESS, 1e-5f},
    {"n=1 LOWER", ACLBLAS_LOWER, 1, 1, 1.2f, 0.5f, 1, 1, ACLBLAS_STATUS_SUCCESS, 1e-5f},
    {"n=2 UPPER", ACLBLAS_UPPER, 2, 2, 1.0f, 0.5f, 1, 1, ACLBLAS_STATUS_SUCCESS, 1e-5f},
    {"n=2 LOWER", ACLBLAS_LOWER, 2, 2, 1.0f, 0.5f, 1, 1, ACLBLAS_STATUS_SUCCESS, 1e-5f},
    {"n=3 LOWER", ACLBLAS_LOWER, 3, 3, 1.5f, 0.2f, 1, 1, ACLBLAS_STATUS_SUCCESS, 1e-5f},
    {"n=63 UPPER", ACLBLAS_UPPER, 63, 63, 0.8f, 0.3f, 1, 1, ACLBLAS_STATUS_SUCCESS, 1e-5f},
    {"n=63 LOWER", ACLBLAS_LOWER, 63, 63, 0.8f, 0.3f, 1, 1, ACLBLAS_STATUS_SUCCESS, 1e-5f},
    {"n=127 UPPER", ACLBLAS_UPPER, 127, 127, 1.0f, 0.0f, 1, 1, ACLBLAS_STATUS_SUCCESS, 1e-4f},
    {"n=129 LOWER", ACLBLAS_LOWER, 129, 129, 1.0f, 0.0f, 1, 1, ACLBLAS_STATUS_SUCCESS, 1e-4f},
    {"n=255 UPPER", ACLBLAS_UPPER, 255, 255, 1.2f, 0.5f, 1, 1, ACLBLAS_STATUS_SUCCESS, 1e-4f},
    {"n=257 LOWER", ACLBLAS_LOWER, 257, 257, 1.2f, 0.5f, 1, 1, ACLBLAS_STATUS_SUCCESS, 1e-4f},
    {"n=7681 UPPER full-chunk scalar writeback", ACLBLAS_UPPER, 7681, 7681, 0.5f, 0.25f, 1, 1, ACLBLAS_STATUS_SUCCESS,
     1e-3f},
    {"beta=0 LOWER n=32", ACLBLAS_LOWER, 32, 32, 1.5f, 0.0f, 1, 1, ACLBLAS_STATUS_SUCCESS, 1e-5f},
    {"alpha=0 UPPER n=32", ACLBLAS_UPPER, 32, 32, 0.0f, 0.8f, 1, 1, ACLBLAS_STATUS_SUCCESS, 1e-5f},
    {"error: invalid uplo", static_cast<aclblasFillMode_t>(0), 10, 10, 1.0f, 0.0f, 1, 1, ACLBLAS_STATUS_INVALID_VALUE,
     0.f},
    {"error: incx=0", ACLBLAS_LOWER, 10, 10, 1.0f, 0.0f, 0, 1, ACLBLAS_STATUS_INVALID_VALUE, 0.f},
    {"error: incy=0", ACLBLAS_LOWER, 10, 10, 1.0f, 0.0f, 1, 0, ACLBLAS_STATUS_INVALID_VALUE, 0.f},
    {"error: lda < max(1,n)", ACLBLAS_LOWER, 10, 5, 1.0f, 0.0f, 1, 1, ACLBLAS_STATUS_INVALID_VALUE, 0.f},
    {"error: handle is nullptr", ACLBLAS_LOWER, 10, 10, 1.0f, 0.0f, 1, 1, ACLBLAS_STATUS_HANDLE_IS_NULLPTR, 0.f},
};

static inline uint32_t PhysicalPos(uint32_t logical, uint32_t n, int64_t inc, uint32_t absInc)
{
    return (inc >= 0) ? (logical * absInc) : ((n - 1U - logical) * absInc);
}

static void GenerateSymMatrix(std::vector<float>& a, uint32_t n, uint32_t lda, uint32_t seed)
{
    a.assign(static_cast<size_t>(n) * lda, 0.0f);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    for (uint32_t i = 0; i < n; ++i) {
        for (uint32_t j = 0; j <= i; ++j) {
            float v = dist(rng);
            a[static_cast<size_t>(i) * lda + j] = v;
            a[static_cast<size_t>(j) * lda + i] = v;
        }
    }
}

static void GenerateStrided(std::vector<float>& v, uint32_t n, int inc, uint32_t seed)
{
    uint32_t absInc = static_cast<uint32_t>(std::abs(inc));
    size_t bufSize = static_cast<size_t>(n - 1U) * absInc + 1U;
    v.assign(bufSize, 0.0f);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    for (uint32_t i = 0; i < n; ++i) {
        uint32_t pos = PhysicalPos(i, n, inc, absInc);
        v[pos] = dist(rng);
    }
}

static void ComputeSsymvGolden(
    const std::vector<float>& a, const std::vector<float>& x, const std::vector<float>& y, uint32_t n, uint32_t lda,
    int incx, int incy, float alpha, float beta, std::vector<float>& out)
{
    uint32_t absIncx = static_cast<uint32_t>(std::abs(incx));
    uint32_t absIncy = static_cast<uint32_t>(std::abs(incy));
    out.assign(y.size(), 0.0f);
    for (uint32_t row = 0; row < n; ++row) {
        float sum = 0.0f;
        for (uint32_t col = 0; col < n; ++col) {
            uint32_t r = std::max(row, col);
            uint32_t c = std::min(row, col);
            sum += a[static_cast<size_t>(r) * lda + c] * x[PhysicalPos(col, n, incx, absIncx)];
        }
        uint32_t yPos = PhysicalPos(row, n, incy, absIncy);
        out[yPos] = alpha * sum + beta * y[yPos];
    }
}

static int RunSsymvCase(const TestCase& tc, aclblasHandle_t handle, aclrtStream stream)
{
    std::cout << "  " << tc.name << " ... " << std::flush;
    if (tc.n == 0 || tc.expectResult != ACLBLAS_STATUS_SUCCESS) {
        aclblasHandle_t useHandle = (tc.expectResult == ACLBLAS_STATUS_HANDLE_IS_NULLPTR) ? nullptr : handle;
        aclblasStatus_t ret = aclblasSsymv(
            useHandle, tc.uplo, tc.n, &tc.alpha, nullptr, tc.lda, nullptr, tc.incx, &tc.beta, nullptr, tc.incy);
        if (ret != tc.expectResult) {
            std::cout << "FAILED: expected " << static_cast<int>(tc.expectResult) << " got " << static_cast<int>(ret)
                      << std::endl;
            return 1;
        }
        if (tc.expectResult == ACLBLAS_STATUS_SUCCESS) {
            std::cout << "PASSED" << std::endl;
        } else {
            std::cout << "PASSED" << std::endl;
        }
        return 0;
    }
    uint32_t nU32 = static_cast<uint32_t>(tc.n);
    uint32_t ldaU32 = static_cast<uint32_t>(tc.lda);
    uint32_t seed = static_cast<uint32_t>(tc.n + tc.lda * 100);

    std::vector<float> aHost;
    GenerateSymMatrix(aHost, nU32, ldaU32, seed);
    std::vector<float> xHost;
    GenerateStrided(xHost, nU32, tc.incx, seed + 1);
    std::vector<float> yHost;
    GenerateStrided(yHost, nU32, tc.incy, seed + 2);

    size_t aBytes = aHost.size() * sizeof(float);
    size_t xBytes = xHost.size() * sizeof(float);
    size_t yBytes = yHost.size() * sizeof(float);

    void* dA = nullptr;
    void* dX = nullptr;
    void* dY = nullptr;
    // issue #590: 分配/拷贝失败时报错退出，不得对空设备指针继续拷贝
    aclError aclRet = aclrtMalloc(&dA, aBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclRet != ACL_SUCCESS) {
        std::cout << "FAILED: aclrtMalloc dA ERROR " << aclRet << std::endl;
        return 1;
    }
    aclRet = aclrtMemcpy(dA, aBytes, aHost.data(), aBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_SUCCESS) {
        aclrtFree(dA);
        std::cout << "FAILED: aclrtMemcpy dA ERROR " << aclRet << std::endl;
        return 1;
    }
    aclRet = aclrtMalloc(&dX, xBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclRet != ACL_SUCCESS) {
        aclrtFree(dA);
        std::cout << "FAILED: aclrtMalloc dX ERROR " << aclRet << std::endl;
        return 1;
    }
    aclRet = aclrtMemcpy(dX, xBytes, xHost.data(), xBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_SUCCESS) {
        aclrtFree(dA);
        aclrtFree(dX);
        std::cout << "FAILED: aclrtMemcpy dX ERROR " << aclRet << std::endl;
        return 1;
    }
    aclRet = aclrtMalloc(&dY, yBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclRet != ACL_SUCCESS) {
        aclrtFree(dA);
        aclrtFree(dX);
        std::cout << "FAILED: aclrtMalloc dY ERROR " << aclRet << std::endl;
        return 1;
    }
    aclRet = aclrtMemcpy(dY, yBytes, yHost.data(), yBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_SUCCESS) {
        aclrtFree(dA);
        aclrtFree(dX);
        aclrtFree(dY);
        std::cout << "FAILED: aclrtMemcpy dY ERROR " << aclRet << std::endl;
        return 1;
    }

    aclblasStatus_t blasRet = aclblasSsymv(
        handle, tc.uplo, tc.n, &tc.alpha, static_cast<const float*>(dA), tc.lda, static_cast<const float*>(dX), tc.incx,
        &tc.beta, static_cast<float*>(dY), tc.incy);
    if (blasRet != tc.expectResult) {
        aclrtFree(dA);
        aclrtFree(dX);
        aclrtFree(dY);
        std::cout << "FAILED: expected " << static_cast<int>(tc.expectResult) << " got " << static_cast<int>(blasRet)
                  << std::endl;
        return 1;
    }
    // issue #590: 同步失败时内核结果不可信，报错退出而非基于陈旧数据比较
    aclRet = aclrtSynchronizeStream(stream);
    if (aclRet != ACL_SUCCESS) {
        aclrtFree(dA);
        aclrtFree(dX);
        aclrtFree(dY);
        std::cout << "FAILED: aclrtSynchronizeStream ERROR " << aclRet << std::endl;
        return 1;
    }
    std::vector<float> yNpu(yHost.size());
    aclRet = aclrtMemcpy(yNpu.data(), yBytes, dY, yBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    if (aclRet != ACL_SUCCESS) {
        aclrtFree(dA);
        aclrtFree(dX);
        aclrtFree(dY);
        std::cout << "FAILED: aclrtMemcpy D2H ERROR " << aclRet << std::endl;
        return 1;
    }
    std::vector<float> yGolden;
    ComputeSsymvGolden(aHost, xHost, yHost, nU32, ldaU32, tc.incx, tc.incy, tc.alpha, tc.beta, yGolden);
    aclrtFree(dA);
    aclrtFree(dX);
    aclrtFree(dY);

    float absTol = tc.tol, relTol = tc.tol, maxDiff = 0.0f;
    for (size_t i = 0; i < yNpu.size(); ++i) {
        float diff = std::abs(yNpu[i] - yGolden[i]);
        if (diff > maxDiff)
            maxDiff = diff;
        float scale = std::max(std::abs(yNpu[i]), std::abs(yGolden[i]));
        if (diff > absTol && diff > relTol * scale) {
            std::cout << "FAILED at " << i << " (" << yNpu[i] << " vs " << yGolden[i] << " diff=" << diff << ")"
                      << std::endl;
            return 1;
        }
    }
    std::cout << "PASSED (maxDiff=" << maxDiff << ")" << std::endl;
    return 0;
}

int32_t main(int32_t argc, char* argv[])
{
    (void)argc;
    (void)argv;
    int32_t deviceId = 0;
    // issue #590: 初始化链路逐一检查返回值，失败时明确报错并以非零码退出
    aclError aclRet = aclInit(nullptr);
    if (aclRet != ACL_SUCCESS) {
        std::cout << "FAILED: aclInit ERROR " << aclRet << std::endl;
        return 1;
    }
    aclRet = aclrtSetDevice(deviceId);
    if (aclRet != ACL_SUCCESS) {
        std::cout << "FAILED: aclrtSetDevice ERROR " << aclRet << std::endl;
        return 1;
    }
    aclrtStream stream = nullptr;
    aclRet = aclrtCreateStream(&stream);
    if (aclRet != ACL_SUCCESS) {
        std::cout << "FAILED: aclrtCreateStream ERROR " << aclRet << std::endl;
        return 1;
    }
    aclblasHandle_t handle = nullptr;
    aclblasStatus_t blasRet = aclblasCreate(&handle);
    if (blasRet != ACLBLAS_STATUS_SUCCESS) {
        std::cout << "FAILED: aclblasCreate ERROR " << blasRet << std::endl;
        return 1;
    }
    blasRet = aclblasSetStream(handle, stream);
    if (blasRet != ACLBLAS_STATUS_SUCCESS) {
        std::cout << "FAILED: aclblasSetStream ERROR " << blasRet << std::endl;
        return 1;
    }

    int failed = 0;
    int total = static_cast<int>(gSsymvTests.size());
    std::cout << "Running " << total << " ssymv arch22 test cases..." << std::endl;
    for (const auto& tc : gSsymvTests) {
        if (RunSsymvCase(tc, handle, stream) != 0)
            ++failed;
    }
    std::cout << "\nResults: " << (total - failed) << "/" << total << " passed" << std::endl;

    aclblasDestroy(handle);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return (failed == 0) ? 0 : 1;
}
