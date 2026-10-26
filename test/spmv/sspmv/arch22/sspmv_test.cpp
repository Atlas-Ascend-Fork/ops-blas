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
    float alpha;
    float beta;
    int incx;
    int incy;
    aclblasStatus_t expectResult;
    float tol;
};

static const std::vector<TestCase> gSspmvTests = {
    {"UPPER n=32 incx=1 incy=1", ACLBLAS_UPPER, 32, 1.2f, 0.8f, 1, 1, ACLBLAS_STATUS_SUCCESS, 1e-5f},
    {"LOWER n=32 incx=1 incy=1", ACLBLAS_LOWER, 32, 1.2f, 0.8f, 1, 1, ACLBLAS_STATUS_SUCCESS, 1e-5f},
    {"UPPER n=32 beta=1.5", ACLBLAS_UPPER, 32, 0.5f, 1.5f, 1, 1, ACLBLAS_STATUS_SUCCESS, 1e-5f},
    {"LOWER n=32 beta=1.5", ACLBLAS_LOWER, 32, 0.5f, 1.5f, 1, 1, ACLBLAS_STATUS_SUCCESS, 1e-5f},
    {"UPPER n=64 incx=1 incy=1", ACLBLAS_UPPER, 64, 0.5f, 1.5f, 1, 1, ACLBLAS_STATUS_SUCCESS, 1e-5f},
    {"LOWER n=64 incx=1 incy=1", ACLBLAS_LOWER, 64, 0.5f, 1.5f, 1, 1, ACLBLAS_STATUS_SUCCESS, 1e-5f},
    {"UPPER n=128 incx=1 incy=1", ACLBLAS_UPPER, 128, 1.0f, 1.0f, 1, 1, ACLBLAS_STATUS_SUCCESS, 1e-5f},
    {"LOWER n=128 incx=1 incy=1", ACLBLAS_LOWER, 128, 1.0f, 1.0f, 1, 1, ACLBLAS_STATUS_SUCCESS, 1e-5f},
    {"UPPER n=128 beta=1.5", ACLBLAS_UPPER, 128, 0.5f, 1.5f, 1, 1, ACLBLAS_STATUS_SUCCESS, 1e-5f},
    {"LOWER n=128 beta=1.5", ACLBLAS_LOWER, 128, 0.5f, 1.5f, 1, 1, ACLBLAS_STATUS_SUCCESS, 1e-5f},
    {"UPPER n=256 incx=1 incy=1", ACLBLAS_UPPER, 256, 2.0f, 0.5f, 1, 1, ACLBLAS_STATUS_SUCCESS, 1e-5f},
    {"LOWER n=256 incx=1 incy=1", ACLBLAS_LOWER, 256, 2.0f, 0.5f, 1, 1, ACLBLAS_STATUS_SUCCESS, 1e-5f},
    {"UPPER n=256 beta=1.5", ACLBLAS_UPPER, 256, 0.5f, 1.5f, 1, 1, ACLBLAS_STATUS_SUCCESS, 1e-5f},
    {"LOWER n=256 beta=1.5", ACLBLAS_LOWER, 256, 0.5f, 1.5f, 1, 1, ACLBLAS_STATUS_SUCCESS, 1e-5f},
    {"UPPER n=512 incx=1 incy=1", ACLBLAS_UPPER, 512, 1.2f, 0.8f, 1, 1, ACLBLAS_STATUS_SUCCESS, 1e-4f},
    {"LOWER n=512 incx=1 incy=1", ACLBLAS_LOWER, 512, 1.2f, 0.8f, 1, 1, ACLBLAS_STATUS_SUCCESS, 1e-4f},
    {"UPPER n=1024 incx=1 incy=1", ACLBLAS_UPPER, 1024, 1.0f, 0.5f, 1, 1, ACLBLAS_STATUS_SUCCESS, 1e-4f},
    {"LOWER n=1024 incx=1 incy=1", ACLBLAS_LOWER, 1024, 1.0f, 0.5f, 1, 1, ACLBLAS_STATUS_SUCCESS, 1e-4f},
    {"UPPER n=2048 incx=1 incy=1", ACLBLAS_UPPER, 2048, 1.2f, 0.8f, 1, 1, ACLBLAS_STATUS_SUCCESS, 1e-4f},
    {"LOWER n=2048 incx=1 incy=1", ACLBLAS_LOWER, 2048, 1.2f, 0.8f, 1, 1, ACLBLAS_STATUS_SUCCESS, 1e-4f},
    {"UPPER n=32 incx=2 incy=1", ACLBLAS_UPPER, 32, 1.0f, 0.0f, 2, 1, ACLBLAS_STATUS_SUCCESS, 1e-5f},
    {"LOWER n=64 incx=-2 incy=1", ACLBLAS_LOWER, 64, 1.0f, 0.0f, -2, 1, ACLBLAS_STATUS_SUCCESS, 1e-5f},
    {"LOWER n=32 incx=-1 incy=1", ACLBLAS_LOWER, 32, 0.5f, 0.0f, -1, 1, ACLBLAS_STATUS_SUCCESS, 1e-5f},
    {"UPPER n=32 incx=1 incy=2", ACLBLAS_UPPER, 32, 1.0f, 0.0f, 1, 2, ACLBLAS_STATUS_SUCCESS, 1e-5f},
    {"LOWER n=64 incx=1 incy=-1", ACLBLAS_LOWER, 64, 0.5f, 0.0f, 1, -1, ACLBLAS_STATUS_SUCCESS, 1e-5f},
    {"UPPER n=32 incx=1 incy=3", ACLBLAS_UPPER, 32, 1.0f, 0.5f, 1, 3, ACLBLAS_STATUS_SUCCESS, 1e-5f},
    {"n=0", ACLBLAS_LOWER, 0, 1.0f, 0.0f, 1, 1, ACLBLAS_STATUS_SUCCESS, 0.0f},
    {"n=1 UPPER", ACLBLAS_UPPER, 1, 1.2f, 0.5f, 1, 1, ACLBLAS_STATUS_SUCCESS, 1e-5f},
    {"n=1 LOWER", ACLBLAS_LOWER, 1, 1.2f, 0.5f, 1, 1, ACLBLAS_STATUS_SUCCESS, 1e-5f},
    {"n=2 UPPER", ACLBLAS_UPPER, 2, 1.0f, 0.5f, 1, 1, ACLBLAS_STATUS_SUCCESS, 1e-5f},
    {"n=2 LOWER", ACLBLAS_LOWER, 2, 1.0f, 0.5f, 1, 1, ACLBLAS_STATUS_SUCCESS, 1e-5f},
    {"n=3 UPPER", ACLBLAS_UPPER, 3, 1.5f, 0.2f, 1, 1, ACLBLAS_STATUS_SUCCESS, 1e-5f},
    {"n=3 LOWER", ACLBLAS_LOWER, 3, 1.5f, 0.2f, 1, 1, ACLBLAS_STATUS_SUCCESS, 1e-5f},
    {"n=63 UPPER", ACLBLAS_UPPER, 63, 0.8f, 0.3f, 1, 1, ACLBLAS_STATUS_SUCCESS, 1e-5f},
    {"n=63 LOWER", ACLBLAS_LOWER, 63, 0.8f, 0.3f, 1, 1, ACLBLAS_STATUS_SUCCESS, 1e-5f},
    {"n=127 UPPER", ACLBLAS_UPPER, 127, 1.0f, 0.0f, 1, 1, ACLBLAS_STATUS_SUCCESS, 1e-4f},
    {"n=129 LOWER", ACLBLAS_LOWER, 129, 1.0f, 0.0f, 1, 1, ACLBLAS_STATUS_SUCCESS, 1e-4f},
    {"n=255 UPPER", ACLBLAS_UPPER, 255, 1.2f, 0.5f, 1, 1, ACLBLAS_STATUS_SUCCESS, 1e-4f},
    {"n=257 LOWER", ACLBLAS_LOWER, 257, 1.2f, 0.5f, 1, 1, ACLBLAS_STATUS_SUCCESS, 1e-4f},
    {"beta=0 LOWER n=32", ACLBLAS_LOWER, 32, 1.5f, 0.0f, 1, 1, ACLBLAS_STATUS_SUCCESS, 1e-5f},
    {"alpha=0 UPPER n=32", ACLBLAS_UPPER, 32, 0.0f, 0.8f, 1, 1, ACLBLAS_STATUS_SUCCESS, 1e-5f},
    {"error: invalid uplo", static_cast<aclblasFillMode_t>(0), 10, 1.0f, 0.0f, 1, 1, ACLBLAS_STATUS_INVALID_VALUE, 0.f},
    {"error: incx=0", ACLBLAS_LOWER, 10, 1.0f, 0.0f, 0, 1, ACLBLAS_STATUS_INVALID_VALUE, 0.f},
    {"error: incy=0", ACLBLAS_LOWER, 10, 1.0f, 0.0f, 1, 0, ACLBLAS_STATUS_INVALID_VALUE, 0.f},
    {"error: handle is nullptr", ACLBLAS_LOWER, 10, 1.0f, 0.0f, 1, 1, ACLBLAS_STATUS_HANDLE_IS_NULLPTR, 0.f},
};

static inline uint32_t PhysicalPos(uint32_t logical, uint32_t n, int64_t inc, uint32_t absInc)
{
    return (inc >= 0) ? (logical * absInc) : ((n - 1U - logical) * absInc);
}

static void GeneratePacked(std::vector<float>& ap, uint32_t n, aclblasFillMode_t uplo, uint32_t seed)
{
    size_t packedSize = static_cast<size_t>(n) * (n + 1U) / 2U;
    ap.assign(packedSize, 0.0f);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    size_t idx = 0;
    for (uint32_t col = 0; col < n; ++col) {
        uint32_t rowStart = (uplo == ACLBLAS_UPPER) ? 0 : col;
        uint32_t rowEnd = (uplo == ACLBLAS_UPPER) ? col + 1U : n;
        for (uint32_t row = rowStart; row < rowEnd; ++row) {
            ap[idx++] = dist(rng);
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

static void ComputeSspmvGolden(
    const std::vector<float>& ap, const std::vector<float>& x, const std::vector<float>& y, uint32_t n,
    aclblasFillMode_t uplo, int incx, int incy, float alpha, float beta, std::vector<float>& out)
{
    uint32_t absIncx = static_cast<uint32_t>(std::abs(incx));
    uint32_t absIncy = static_cast<uint32_t>(std::abs(incy));

    out.assign(y.size(), 0.0f);
    for (uint32_t row = 0; row < n; ++row) {
        float sum = 0.0f;
        for (uint32_t col = 0; col < n; ++col) {
            uint32_t packedRow = (uplo == ACLBLAS_UPPER) ? std::min(row, col) : std::max(row, col);
            uint32_t packedCol = (uplo == ACLBLAS_UPPER) ? std::max(row, col) : std::min(row, col);
            uint64_t idx;
            if (uplo == ACLBLAS_UPPER) {
                idx = static_cast<uint64_t>(packedCol) * (packedCol + 1ULL) / 2ULL + packedRow;
            } else {
                uint64_t columnStart =
                    static_cast<uint64_t>(packedCol) * (2ULL * static_cast<uint64_t>(n) - packedCol + 1ULL) / 2ULL;
                idx = columnStart + packedRow - packedCol;
            }
            uint32_t xPos = PhysicalPos(col, n, incx, absIncx);
            sum += ap[idx] * x[xPos];
        }
        uint32_t yPos = PhysicalPos(row, n, incy, absIncy);
        out[yPos] = alpha * sum + beta * y[yPos];
    }
}

static int RunSspmvCase(const TestCase& tc, aclblasHandle_t handle, aclrtStream stream)
{
    std::cout << "  " << tc.name << " ... " << std::flush;

    if (tc.n == 0 || tc.expectResult != ACLBLAS_STATUS_SUCCESS) {
        aclblasHandle_t useHandle = (tc.expectResult == ACLBLAS_STATUS_HANDLE_IS_NULLPTR) ? nullptr : handle;
        aclblasStatus_t ret =
            aclblasSspmv(useHandle, tc.uplo, tc.n, &tc.alpha, nullptr, nullptr, tc.incx, &tc.beta, nullptr, tc.incy);
        if (ret != tc.expectResult) {
            std::cout << "FAILED: expected " << static_cast<int>(tc.expectResult) << " got " << static_cast<int>(ret)
                      << std::endl;
            return 1;
        }
        if (tc.expectResult == ACLBLAS_STATUS_NOT_SUPPORTED) {
            std::cout << "PASSED (NOT_SUPPORTED)" << std::endl;
        } else {
            std::cout << "PASSED" << std::endl;
        }
        return 0;
    }

    uint32_t nU32 = static_cast<uint32_t>(tc.n);
    uint32_t seed = static_cast<uint32_t>(tc.n + std::abs(tc.incx + tc.incy) * 1000);

    std::vector<float> apHost;
    GeneratePacked(apHost, nU32, tc.uplo, seed);
    std::vector<float> xHost;
    GenerateStrided(xHost, nU32, tc.incx, seed + 1);
    std::vector<float> yHost;
    GenerateStrided(yHost, nU32, tc.incy, seed + 2);

    size_t apBytes = apHost.size() * sizeof(float);
    size_t xBytes = xHost.size() * sizeof(float);
    size_t yBytes = yHost.size() * sizeof(float);

    void* dAP = nullptr;
    void* dX = nullptr;
    void* dY = nullptr;
    // issue #592: 分配/拷贝失败时报错退出（删除原先从未读取的 aclRet 死赋值）
    aclError aclRet = aclrtMalloc(&dAP, apBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclRet != ACL_SUCCESS) {
        std::cout << "FAILED: aclrtMalloc dAP ERROR " << aclRet << std::endl;
        return 1;
    }
    aclRet = aclrtMemcpy(dAP, apBytes, apHost.data(), apBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_SUCCESS) {
        aclrtFree(dAP);
        std::cout << "FAILED: aclrtMemcpy dAP ERROR " << aclRet << std::endl;
        return 1;
    }
    aclRet = aclrtMalloc(&dX, xBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclRet != ACL_SUCCESS) {
        aclrtFree(dAP);
        std::cout << "FAILED: aclrtMalloc dX ERROR " << aclRet << std::endl;
        return 1;
    }
    aclRet = aclrtMemcpy(dX, xBytes, xHost.data(), xBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_SUCCESS) {
        aclrtFree(dAP);
        aclrtFree(dX);
        std::cout << "FAILED: aclrtMemcpy dX ERROR " << aclRet << std::endl;
        return 1;
    }
    aclRet = aclrtMalloc(&dY, yBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclRet != ACL_SUCCESS) {
        aclrtFree(dAP);
        aclrtFree(dX);
        std::cout << "FAILED: aclrtMalloc dY ERROR " << aclRet << std::endl;
        return 1;
    }
    aclRet = aclrtMemcpy(dY, yBytes, yHost.data(), yBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_SUCCESS) {
        aclrtFree(dAP);
        aclrtFree(dX);
        aclrtFree(dY);
        std::cout << "FAILED: aclrtMemcpy dY ERROR " << aclRet << std::endl;
        return 1;
    }
    aclblasStatus_t blasRet = aclblasSspmv(
        handle, tc.uplo, tc.n, &tc.alpha, static_cast<const float*>(dAP), static_cast<const float*>(dX), tc.incx,
        &tc.beta, static_cast<float*>(dY), tc.incy);

    if (blasRet != tc.expectResult) {
        aclrtFree(dAP);
        aclrtFree(dX);
        aclrtFree(dY);
        std::cout << "FAILED: expected " << static_cast<int>(tc.expectResult) << " got " << static_cast<int>(blasRet)
                  << std::endl;
        return 1;
    }

    // issue #592: 同步/D2H 失败时报错退出，不得基于陈旧数据比较
    aclRet = aclrtSynchronizeStream(stream);
    if (aclRet != ACL_SUCCESS) {
        aclrtFree(dAP);
        aclrtFree(dX);
        aclrtFree(dY);
        std::cout << "FAILED: aclrtSynchronizeStream ERROR " << aclRet << std::endl;
        return 1;
    }

    std::vector<float> yNpu(yHost.size());
    aclRet = aclrtMemcpy(yNpu.data(), yBytes, dY, yBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    if (aclRet != ACL_SUCCESS) {
        aclrtFree(dAP);
        aclrtFree(dX);
        aclrtFree(dY);
        std::cout << "FAILED: aclrtMemcpy D2H ERROR " << aclRet << std::endl;
        return 1;
    }

    std::vector<float> yGolden;
    ComputeSspmvGolden(apHost, xHost, yHost, nU32, tc.uplo, tc.incx, tc.incy, tc.alpha, tc.beta, yGolden);

    aclrtFree(dAP);
    aclrtFree(dX);
    aclrtFree(dY);

    float absTol = tc.tol;
    float relTol = tc.tol;
    float maxDiff = 0.0f;
    for (size_t i = 0; i < yNpu.size(); ++i) {
        float diff = std::abs(yNpu[i] - yGolden[i]);
        if (diff > maxDiff)
            maxDiff = diff;
        float scale = std::max(std::abs(yNpu[i]), std::abs(yGolden[i]));
        if (diff > absTol && diff > relTol * scale) {
            std::cout << "FAILED at index " << i << " (" << yNpu[i] << " vs " << yGolden[i] << " diff=" << diff << ")"
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
    aclError ret = aclInit(nullptr);
    if (ret != ACL_SUCCESS) {
        fprintf(stderr, "aclInit failed\n");
        return ret;
    }
    // issue #592: main 初始化链路逐一检查返回值，失败时明确报错并以非零码退出
    aclError devRet = aclrtSetDevice(deviceId);
    if (devRet != ACL_SUCCESS) {
        fprintf(stderr, "aclrtSetDevice failed. ERROR: %d\n", devRet);
        return devRet;
    }
    aclrtStream stream = nullptr;
    devRet = aclrtCreateStream(&stream);
    if (devRet != ACL_SUCCESS) {
        fprintf(stderr, "aclrtCreateStream failed. ERROR: %d\n", devRet);
        return devRet;
    }
    aclblasHandle_t handle = nullptr;
    aclblasStatus_t blasRet = aclblasCreate(&handle);
    if (blasRet != ACLBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "aclblasCreate failed. ERROR: %d\n", blasRet);
        return blasRet;
    }
    blasRet = aclblasSetStream(handle, stream);
    if (blasRet != ACLBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "aclblasSetStream failed. ERROR: %d\n", blasRet);
        return blasRet;
    }

    int failed = 0;
    int total = static_cast<int>(gSspmvTests.size());
    std::cout << "Running " << total << " sspmv arch22 test cases..." << std::endl;
    for (const auto& tc : gSspmvTests) {
        if (RunSspmvCase(tc, handle, stream) != 0)
            ++failed;
    }
    std::cout << "\nResults: " << (total - failed) << "/" << total << " passed" << std::endl;

    aclblasDestroy(handle);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return (failed == 0) ? 0 : 1;
}
