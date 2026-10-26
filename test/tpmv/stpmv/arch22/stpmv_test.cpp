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
    aclblasFillMode uplo;
    aclblasOperation trans;
    aclblasDiagType diag;
    int64_t n;
    int64_t incx;
    aclblasStatus_t expectResult;
    float tol;
};

static const std::vector<TestCase> gTests = {
    {"LOWER+N NON_UNIT n=32 incx=1", ACLBLAS_LOWER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT, 32, 1, ACLBLAS_STATUS_SUCCESS,
     1e-5f},
    {"UPPER+N NON_UNIT n=32 incx=1", ACLBLAS_UPPER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT, 32, 1, ACLBLAS_STATUS_SUCCESS,
     1e-5f},
    {"LOWER+T UNIT n=32 incx=1", ACLBLAS_LOWER, ACLBLAS_OP_T, ACLBLAS_UNIT, 32, 1, ACLBLAS_STATUS_SUCCESS, 1e-5f},
    {"UPPER+T UNIT n=32 incx=1", ACLBLAS_UPPER, ACLBLAS_OP_T, ACLBLAS_UNIT, 32, 1, ACLBLAS_STATUS_SUCCESS, 1e-5f},
    {"UPPER+C NON_UNIT n=63 incx=1", ACLBLAS_UPPER, ACLBLAS_OP_C, ACLBLAS_NON_UNIT, 63, 1, ACLBLAS_STATUS_SUCCESS,
     1e-4f},
    {"LOWER+N NON_UNIT n=64 incx=2", ACLBLAS_LOWER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT, 64, 2, ACLBLAS_STATUS_SUCCESS,
     1e-4f},
    {"UPPER+T NON_UNIT n=65 incx=-1", ACLBLAS_UPPER, ACLBLAS_OP_T, ACLBLAS_NON_UNIT, 65, -1, ACLBLAS_STATUS_SUCCESS,
     1e-4f},
    {"LOWER+C UNIT n=129 incx=-2", ACLBLAS_LOWER, ACLBLAS_OP_C, ACLBLAS_UNIT, 129, -2, ACLBLAS_STATUS_SUCCESS, 1e-3f},
    {"LOWER+T NON_UNIT n=32 incx=1", ACLBLAS_LOWER, ACLBLAS_OP_T, ACLBLAS_NON_UNIT, 32, 1, ACLBLAS_STATUS_SUCCESS,
     1e-5f},
    {"UPPER+T NON_UNIT n=32 incx=1", ACLBLAS_UPPER, ACLBLAS_OP_T, ACLBLAS_NON_UNIT, 32, 1, ACLBLAS_STATUS_SUCCESS,
     1e-5f},
    {"LOWER+C NON_UNIT n=32 incx=1", ACLBLAS_LOWER, ACLBLAS_OP_C, ACLBLAS_NON_UNIT, 32, 1, ACLBLAS_STATUS_SUCCESS,
     1e-5f},
    {"UPPER+C NON_UNIT n=32 incx=1", ACLBLAS_UPPER, ACLBLAS_OP_C, ACLBLAS_NON_UNIT, 32, 1, ACLBLAS_STATUS_SUCCESS,
     1e-5f},
    {"LOWER+N NON_UNIT n=32 incx=-1", ACLBLAS_LOWER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT, 32, -1, ACLBLAS_STATUS_SUCCESS,
     1e-5f},
    {"UPPER+N NON_UNIT n=32 incx=2", ACLBLAS_UPPER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT, 32, 2, ACLBLAS_STATUS_SUCCESS,
     1e-5f},
    {"UPPER+N UNIT n=32 incx=1", ACLBLAS_UPPER, ACLBLAS_OP_N, ACLBLAS_UNIT, 32, 1, ACLBLAS_STATUS_SUCCESS, 1e-5f},
    {"LOWER+N UNIT n=32 incx=1", ACLBLAS_LOWER, ACLBLAS_OP_N, ACLBLAS_UNIT, 32, 1, ACLBLAS_STATUS_SUCCESS, 1e-5f},
    {"UPPER+C UNIT n=32 incx=1", ACLBLAS_UPPER, ACLBLAS_OP_C, ACLBLAS_UNIT, 32, 1, ACLBLAS_STATUS_SUCCESS, 1e-5f},
    {"n=0", ACLBLAS_LOWER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT, 0, 1, ACLBLAS_STATUS_SUCCESS, 0.f},
    {"n=1 LOWER+N NON_UNIT", ACLBLAS_LOWER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT, 1, 1, ACLBLAS_STATUS_SUCCESS, 1e-5f},
    {"n=1 UPPER+N UNIT", ACLBLAS_UPPER, ACLBLAS_OP_N, ACLBLAS_UNIT, 1, 1, ACLBLAS_STATUS_SUCCESS, 1e-5f},
    {"n=1 LOWER+T NON_UNIT", ACLBLAS_LOWER, ACLBLAS_OP_T, ACLBLAS_NON_UNIT, 1, 1, ACLBLAS_STATUS_SUCCESS, 1e-5f},
    {"n=2 UPPER+N NON_UNIT", ACLBLAS_UPPER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT, 2, 1, ACLBLAS_STATUS_SUCCESS, 1e-5f},
    {"n=2 LOWER+T UNIT", ACLBLAS_LOWER, ACLBLAS_OP_T, ACLBLAS_UNIT, 2, 1, ACLBLAS_STATUS_SUCCESS, 1e-5f},
    {"n=3 UPPER+N NON_UNIT", ACLBLAS_UPPER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT, 3, 1, ACLBLAS_STATUS_SUCCESS, 1e-5f},
    {"n=127 UPPER+N NON_UNIT", ACLBLAS_UPPER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT, 127, 1, ACLBLAS_STATUS_SUCCESS, 1e-4f},
    {"n=128 LOWER+N NON_UNIT", ACLBLAS_LOWER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT, 128, 1, ACLBLAS_STATUS_SUCCESS, 1e-4f},
    {"n=255 UPPER+T NON_UNIT", ACLBLAS_UPPER, ACLBLAS_OP_T, ACLBLAS_NON_UNIT, 255, 1, ACLBLAS_STATUS_SUCCESS, 1e-4f},
    {"n=257 LOWER+N NON_UNIT", ACLBLAS_LOWER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT, 257, 1, ACLBLAS_STATUS_SUCCESS, 1e-4f},
    {"n=512 LOWER+N UNIT incx=3", ACLBLAS_LOWER, ACLBLAS_OP_N, ACLBLAS_UNIT, 512, 3, ACLBLAS_STATUS_SUCCESS, 1e-4f},
    {"n=512 UPPER+N NON_UNIT incx=-1", ACLBLAS_UPPER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT, 512, -1, ACLBLAS_STATUS_SUCCESS,
     1e-4f},
    {"n=1024 LOWER+N NON_UNIT", ACLBLAS_LOWER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT, 1024, 1, ACLBLAS_STATUS_SUCCESS, 1e-4f},
    {"n=1024 UPPER+N NON_UNIT", ACLBLAS_UPPER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT, 1024, 1, ACLBLAS_STATUS_SUCCESS, 1e-4f},
    {"n=2048 LOWER+T NON_UNIT", ACLBLAS_LOWER, ACLBLAS_OP_T, ACLBLAS_NON_UNIT, 2048, 1, ACLBLAS_STATUS_SUCCESS, 1e-4f},
    {"n=2048 UPPER+T NON_UNIT", ACLBLAS_UPPER, ACLBLAS_OP_T, ACLBLAS_NON_UNIT, 2048, 1, ACLBLAS_STATUS_SUCCESS, 1e-4f},
    {"INVALID uplo", static_cast<aclblasFillMode>(0), ACLBLAS_OP_N, ACLBLAS_NON_UNIT, 10, 1,
     ACLBLAS_STATUS_INVALID_VALUE, 0.f},
    {"INVALID trans", ACLBLAS_LOWER, static_cast<aclblasOperation>(0), ACLBLAS_NON_UNIT, 10, 1,
     ACLBLAS_STATUS_INVALID_VALUE, 0.f},
    {"INVALID diag", ACLBLAS_UPPER, ACLBLAS_OP_N, static_cast<aclblasDiagType>(0), 10, 1, ACLBLAS_STATUS_INVALID_VALUE,
     0.f},
    {"incx=0", ACLBLAS_LOWER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT, 10, 0, ACLBLAS_STATUS_INVALID_VALUE, 0.f},
    {"n<0", ACLBLAS_UPPER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT, -1, 1, ACLBLAS_STATUS_SUCCESS, 0.f},
    {"handle is nullptr", ACLBLAS_LOWER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT, 10, 1, ACLBLAS_STATUS_HANDLE_IS_NULLPTR, 0.f},
};

static inline bool IsFormulaA(uint32_t uplo, uint32_t trans)
{
    return (
        (uplo == ACLBLAS_LOWER && trans == ACLBLAS_OP_N) ||
        (uplo == ACLBLAS_UPPER && (trans == ACLBLAS_OP_T || trans == ACLBLAS_OP_C)));
}

static inline uint32_t PackedIndex(uint32_t row, uint32_t col, bool isFormulaA)
{
    if (isFormulaA) {
        return static_cast<size_t>(row) * (row + 1ULL) / 2ULL + col;
    }
    return row + col * (col + 1U) / 2U;
}

static inline uint32_t PhysicalPos(uint32_t logical, uint32_t n, int64_t incx, uint32_t absIncx)
{
    return (incx >= 0) ? (logical * absIncx) : ((n - 1U - logical) * absIncx);
}

static void GeneratePacked(std::vector<float>& ap, uint32_t n, bool isUpper, uint32_t seed)
{
    size_t packedSize = static_cast<size_t>(n) * (n + 1U) / 2U;
    ap.assign(packedSize, 0.0f);
    std::mt19937 rng(seed + 100);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    if (isUpper) {
        for (uint32_t col = 0; col < n; ++col) {
            for (uint32_t row = 0; row <= col; ++row) {
                ap[row + col * (col + 1U) / 2U] = dist(rng);
            }
        }
    } else {
        for (uint32_t row = 0; row < n; ++row) {
            for (uint32_t col = 0; col <= row; ++col) {
                ap[static_cast<size_t>(row) * (row + 1ULL) / 2ULL + col] = dist(rng);
            }
        }
    }
}

static void GenerateX(std::vector<float>& x, uint32_t n, int64_t incx, uint32_t seed)
{
    uint32_t absInc = static_cast<uint32_t>(std::abs(incx));
    size_t bufSize = static_cast<size_t>(n - 1U) * absInc + 1U;
    x.assign(bufSize, 0.0f);
    std::mt19937 rng(seed + 200);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    for (uint32_t i = 0; i < n; ++i) {
        uint32_t pos = PhysicalPos(i, n, incx, absInc);
        x[pos] = dist(rng);
    }
}

static void ComputeGolden(
    const std::vector<float>& ap, const std::vector<float>& xInput, std::vector<float>& y, uint32_t n, uint32_t uplo,
    uint32_t trans, uint32_t diag, int64_t incx)
{
    bool isFormulaA = IsFormulaA(uplo, trans);
    bool isUnit = (diag == ACLBLAS_UNIT);
    uint32_t absInc = static_cast<uint32_t>(std::abs(incx));

    y.assign(xInput.size(), 0.0f);

    for (uint32_t row = 0; row < n; ++row) {
        float sum = 0.0f;
        uint32_t colStart, colEnd;
        if (isFormulaA) {
            colStart = 0;
            colEnd = row;
        } else {
            colStart = row;
            colEnd = n - 1U;
        }
        for (uint32_t col = colStart; col <= colEnd; ++col) {
            float aVal;
            if (isUnit && col == row) {
                aVal = 1.0f;
            } else {
                aVal = ap[PackedIndex(row, col, isFormulaA)];
            }
            uint32_t xPos = PhysicalPos(col, n, incx, absInc);
            sum += aVal * xInput[xPos];
        }
        uint32_t yPos = PhysicalPos(row, n, incx, absInc);
        y[yPos] = sum;
    }
}

static int RunCase(const TestCase& tc, aclblasHandle_t handle, aclrtStream stream)
{
    auto toLegacyUplo = static_cast<aclblasFillMode>(tc.uplo);
    auto toLegacyTrans = static_cast<aclblasOperation>(tc.trans);
    auto toLegacyDiag = static_cast<aclblasDiagType>(tc.diag);

    std::cout << "  " << tc.name << " ... " << std::flush;

    if (tc.n == 0 || tc.n < 0 || tc.expectResult != ACLBLAS_STATUS_SUCCESS) {
        aclblasHandle_t useHandle = (tc.expectResult == ACLBLAS_STATUS_HANDLE_IS_NULLPTR) ? nullptr : handle;
        std::vector<float> dummy;
        float* yDummy = dummy.empty() ? nullptr : dummy.data();
        aclblasStatus_t ret = aclblasStpmv_legacy(
            useHandle, toLegacyUplo, toLegacyTrans, toLegacyDiag, tc.n, nullptr, nullptr, yDummy, tc.incx);
        if (ret != tc.expectResult) {
            std::cout << "FAILED: expected " << static_cast<int>(tc.expectResult) << ", got " << static_cast<int>(ret)
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
    bool isUpper = (tc.uplo == ACLBLAS_UPPER);

    uint32_t seed = static_cast<uint32_t>(tc.n + std::abs(tc.incx) * 1000);

    std::vector<float> apHost;
    GeneratePacked(apHost, nU32, isUpper, seed);
    std::vector<float> xHost;
    GenerateX(xHost, nU32, tc.incx, seed);

    size_t packedBytes = apHost.size() * sizeof(float);
    size_t xBytes = xHost.size() * sizeof(float);

    void* dAP = nullptr;
    aclError aclRet = aclrtMalloc(&dAP, packedBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclRet != ACL_SUCCESS) {
        std::cout << "FAILED: aclrtMalloc dAP" << std::endl;
        return 1;
    }
    // issue #593: H2D 拷贝失败时报错退出，不得在未初始化设备内存上执行内核
    aclRet = aclrtMemcpy(dAP, packedBytes, apHost.data(), packedBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_SUCCESS) {
        aclrtFree(dAP);
        std::cout << "FAILED: aclrtMemcpy dAP ERROR " << aclRet << std::endl;
        return 1;
    }

    void* dX = nullptr;
    aclRet = aclrtMalloc(&dX, xBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclRet != ACL_SUCCESS) {
        aclrtFree(dAP);
        std::cout << "FAILED: aclrtMalloc dX" << std::endl;
        return 1;
    }
    aclRet = aclrtMemcpy(dX, xBytes, xHost.data(), xBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_SUCCESS) {
        aclrtFree(dAP);
        aclrtFree(dX);
        std::cout << "FAILED: aclrtMemcpy dX ERROR " << aclRet << std::endl;
        return 1;
    }

    void* dY = nullptr;
    aclRet = aclrtMalloc(&dY, xBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclRet != ACL_SUCCESS) {
        aclrtFree(dAP);
        aclrtFree(dX);
        std::cout << "FAILED: aclrtMalloc dY" << std::endl;
        return 1;
    }
    {
        std::vector<float> yZero(xHost.size(), 0.0f);
        aclRet = aclrtMemcpy(dY, xBytes, yZero.data(), xBytes, ACL_MEMCPY_HOST_TO_DEVICE);
        if (aclRet != ACL_SUCCESS) {
            aclrtFree(dAP);
            aclrtFree(dX);
            aclrtFree(dY);
            std::cout << "FAILED: aclrtMemcpy dY ERROR " << aclRet << std::endl;
            return 1;
        }
    }

    aclblasStatus_t blasRet = aclblasStpmv_legacy(
        handle, toLegacyUplo, toLegacyTrans, toLegacyDiag, tc.n, static_cast<const float*>(dAP),
        static_cast<const float*>(dX), static_cast<float*>(dY), tc.incx);

    if (blasRet != tc.expectResult) {
        aclrtFree(dAP);
        aclrtFree(dX);
        aclrtFree(dY);
        std::cout << "FAILED: expected " << static_cast<int>(tc.expectResult) << ", got " << static_cast<int>(blasRet)
                  << std::endl;
        return 1;
    }

    // issue #593: 同步/D2H 失败时报错退出，不得以全零或陈旧数据参与比较
    aclRet = aclrtSynchronizeStream(stream);
    if (aclRet != ACL_SUCCESS) {
        aclrtFree(dAP);
        aclrtFree(dX);
        aclrtFree(dY);
        std::cout << "FAILED: aclrtSynchronizeStream ERROR " << aclRet << std::endl;
        return 1;
    }

    std::vector<float> yNPU(xHost.size());
    aclRet = aclrtMemcpy(yNPU.data(), xBytes, dY, xBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    if (aclRet != ACL_SUCCESS) {
        aclrtFree(dAP);
        aclrtFree(dX);
        aclrtFree(dY);
        std::cout << "FAILED: aclrtMemcpy D2H ERROR " << aclRet << std::endl;
        return 1;
    }

    std::vector<float> yGolden;
    ComputeGolden(
        apHost, xHost, yGolden, nU32, static_cast<uint32_t>(tc.uplo), static_cast<uint32_t>(tc.trans),
        static_cast<uint32_t>(tc.diag), tc.incx);

    aclrtFree(dAP);
    aclrtFree(dX);
    aclrtFree(dY);

    float maxDiff = 0.0f;
    for (size_t i = 0; i < yNPU.size(); ++i) {
        float diff = std::abs(yNPU[i] - yGolden[i]);
        if (diff > maxDiff)
            maxDiff = diff;
    }

    float absTol = tc.tol;
    float relTol = tc.tol;
    for (size_t i = 0; i < yNPU.size(); ++i) {
        float diff = std::abs(yNPU[i] - yGolden[i]);
        float scale = std::max(std::abs(yNPU[i]), std::abs(yGolden[i]));
        if (diff > absTol && diff > relTol * scale) {
            std::cout << "FAILED at index " << i << " (" << yNPU[i] << " vs " << yGolden[i] << " diff=" << diff << ")"
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
        fprintf(stderr, "aclInit failed. ERROR: %d\n", ret);
        return ret;
    }

    ret = aclrtSetDevice(deviceId);
    if (ret != ACL_SUCCESS) {
        fprintf(stderr, "aclrtSetDevice failed. ERROR: %d\n", ret);
        aclFinalize();
        return ret;
    }

    aclrtStream stream = nullptr;
    ret = aclrtCreateStream(&stream);
    if (ret != ACL_SUCCESS) {
        fprintf(stderr, "aclrtCreateStream failed. ERROR: %d\n", ret);
        aclrtResetDevice(deviceId);
        aclFinalize();
        return ret;
    }

    aclblasHandle_t handle = nullptr;
    // issue #593: 句柄创建/绑定失败时不得以未初始化句柄继续
    aclblasStatus_t blasRet = aclblasCreate(&handle);
    if (blasRet != ACLBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "aclblasCreate failed. ERROR: %d\n", blasRet);
        aclrtDestroyStream(stream);
        aclrtResetDevice(deviceId);
        aclFinalize();
        return blasRet;
    }
    blasRet = aclblasSetStream(handle, stream);
    if (blasRet != ACLBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "aclblasSetStream failed. ERROR: %d\n", blasRet);
        aclblasDestroy(handle);
        aclrtDestroyStream(stream);
        aclrtResetDevice(deviceId);
        aclFinalize();
        return blasRet;
    }

    int failed = 0;
    int total = static_cast<int>(gTests.size());

    std::cout << "Running " << total << " stpmv arch22 test cases..." << std::endl;

    for (const auto& tc : gTests) {
        if (RunCase(tc, handle, stream) != 0) {
            ++failed;
        }
    }

    std::cout << "\nResults: " << (total - failed) << "/" << total << " passed" << std::endl;

    aclblasDestroy(handle);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return (failed == 0) ? 0 : 1;
}
