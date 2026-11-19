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
#include "stbmv_npu_wrapper.h"

struct TestCase {
    const char* name;
    aclblasFillMode uplo;
    aclblasOperation trans;
    aclblasDiagType diag;
    int64_t n;
    int64_t k;
    int64_t lda;
    int64_t incx;
    aclblasStatus_t expectResult;
    float tol;
    bool allocBuffers;
};

static const std::vector<TestCase> gStbmvTests = {
    {"LOWER+N NON_UNIT n=32 k=8", ACLBLAS_LOWER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT, 32, 8, 32, 1, ACLBLAS_STATUS_SUCCESS,
     1e-5f},
    {"LOWER+N NON_UNIT n=64 k=16", ACLBLAS_LOWER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT, 64, 16, 64, 1, ACLBLAS_STATUS_SUCCESS,
     1e-5f},
    {"LOWER+N UNIT n=32 k=8", ACLBLAS_LOWER, ACLBLAS_OP_N, ACLBLAS_UNIT, 32, 8, 32, 1, ACLBLAS_STATUS_SUCCESS, 1e-5f},
    {"LOWER+T NON_UNIT n=32 k=8", ACLBLAS_LOWER, ACLBLAS_OP_T, ACLBLAS_NON_UNIT, 32, 8, 32, 1, ACLBLAS_STATUS_SUCCESS,
     1e-5f},
    {"LOWER+C NON_UNIT n=32 k=8", ACLBLAS_LOWER, ACLBLAS_OP_C, ACLBLAS_NON_UNIT, 32, 8, 32, 1, ACLBLAS_STATUS_SUCCESS,
     1e-5f},
    {"LOWER+T UNIT n=32 k=8", ACLBLAS_LOWER, ACLBLAS_OP_T, ACLBLAS_UNIT, 32, 8, 32, 1, ACLBLAS_STATUS_SUCCESS, 1e-5f},
    {"UPPER+N NON_UNIT n=32 k=8", ACLBLAS_UPPER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT, 32, 8, 32, 1, ACLBLAS_STATUS_SUCCESS,
     1e-5f},
    {"UPPER+T NON_UNIT n=32 k=8", ACLBLAS_UPPER, ACLBLAS_OP_T, ACLBLAS_NON_UNIT, 32, 8, 32, 1, ACLBLAS_STATUS_SUCCESS,
     1e-5f},
    {"UPPER+N UNIT n=32 k=8", ACLBLAS_UPPER, ACLBLAS_OP_N, ACLBLAS_UNIT, 32, 8, 32, 1, ACLBLAS_STATUS_SUCCESS, 1e-5f},
    {"UPPER+C NON_UNIT n=32 k=8", ACLBLAS_UPPER, ACLBLAS_OP_C, ACLBLAS_NON_UNIT, 32, 8, 32, 1, ACLBLAS_STATUS_SUCCESS,
     1e-5f},
    {"UPPER+T UNIT n=32 k=8", ACLBLAS_UPPER, ACLBLAS_OP_T, ACLBLAS_UNIT, 32, 8, 32, 1, ACLBLAS_STATUS_SUCCESS, 1e-5f},
    {"UPPER+C UNIT n=32 k=8", ACLBLAS_UPPER, ACLBLAS_OP_C, ACLBLAS_UNIT, 32, 8, 32, 1, ACLBLAS_STATUS_SUCCESS, 1e-5f},
    {"LOWER+C UNIT n=32 k=8", ACLBLAS_LOWER, ACLBLAS_OP_C, ACLBLAS_UNIT, 32, 8, 32, 1, ACLBLAS_STATUS_SUCCESS, 1e-5f},
    {"LOWER+N n=63 k=16", ACLBLAS_LOWER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT, 63, 16, 63, 1, ACLBLAS_STATUS_SUCCESS, 1e-5f},
    {"UPPER+N n=63 k=16", ACLBLAS_UPPER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT, 63, 16, 63, 1, ACLBLAS_STATUS_SUCCESS, 1e-5f},
    {"LOWER+N n=128 k=32", ACLBLAS_LOWER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT, 128, 32, 128, 1, ACLBLAS_STATUS_SUCCESS,
     1e-5f},
    {"LOWER+T n=128 k=32", ACLBLAS_LOWER, ACLBLAS_OP_T, ACLBLAS_NON_UNIT, 128, 32, 128, 1, ACLBLAS_STATUS_SUCCESS,
     1e-5f},
    {"UPPER+T n=128 k=32", ACLBLAS_UPPER, ACLBLAS_OP_T, ACLBLAS_NON_UNIT, 128, 32, 128, 1, ACLBLAS_STATUS_SUCCESS,
     1e-5f},
    {"LOWER+N n=255 k=64", ACLBLAS_LOWER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT, 255, 64, 255, 1, ACLBLAS_STATUS_SUCCESS,
     1e-4f},
    {"LOWER+N n=256 k=64", ACLBLAS_LOWER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT, 256, 64, 256, 1, ACLBLAS_STATUS_SUCCESS,
     1e-4f},
    {"UPPER+N n=256 k=64", ACLBLAS_UPPER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT, 256, 64, 256, 1, ACLBLAS_STATUS_SUCCESS,
     1e-4f},
    {"LOWER+N n=512 k=128", ACLBLAS_LOWER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT, 512, 128, 512, 1, ACLBLAS_STATUS_SUCCESS,
     1e-3f},
    {"UPPER+N n=512 k=128", ACLBLAS_UPPER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT, 512, 128, 512, 1, ACLBLAS_STATUS_SUCCESS,
     1e-3f},
    {"LOWER+T n=512 k=128", ACLBLAS_LOWER, ACLBLAS_OP_T, ACLBLAS_NON_UNIT, 512, 128, 512, 1, ACLBLAS_STATUS_SUCCESS,
     1e-3f},
    {"LOWER+N n=1024 k=256", ACLBLAS_LOWER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT, 1024, 256, 1024, 1, ACLBLAS_STATUS_SUCCESS,
     1e-3f},
    {"UPPER+T n=1024 k=256", ACLBLAS_UPPER, ACLBLAS_OP_T, ACLBLAS_NON_UNIT, 1024, 256, 1024, 1, ACLBLAS_STATUS_SUCCESS,
     1e-3f},
    {"LOWER+N incx=2 n=64 k=16", ACLBLAS_LOWER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT, 64, 16, 64, 2, ACLBLAS_STATUS_SUCCESS,
     1e-5f},
    {"LOWER+N incx=-1 n=64 k=16", ACLBLAS_LOWER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT, 64, 16, 64, -1, ACLBLAS_STATUS_SUCCESS,
     1e-5f},
    {"UPPER+N incx=-2 n=64 k=16", ACLBLAS_UPPER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT, 64, 16, 64, -2, ACLBLAS_STATUS_SUCCESS,
     1e-5f},
    {"n=0", ACLBLAS_LOWER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT, 0, 0, 1, 1, ACLBLAS_STATUS_SUCCESS, 0.0f},
    {"n=1 LOWER+N NON_UNIT", ACLBLAS_LOWER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT, 1, 1, 2, 1, ACLBLAS_STATUS_SUCCESS, 1e-5f},
    {"n=1 UPPER+N UNIT", ACLBLAS_UPPER, ACLBLAS_OP_N, ACLBLAS_UNIT, 1, 1, 2, 1, ACLBLAS_STATUS_SUCCESS, 1e-5f},
    {"n=2 LOWER+T NON_UNIT", ACLBLAS_LOWER, ACLBLAS_OP_T, ACLBLAS_NON_UNIT, 2, 1, 2, 1, ACLBLAS_STATUS_SUCCESS, 1e-5f},
    {"n=3 LOWER+N NON_UNIT", ACLBLAS_LOWER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT, 3, 2, 3, 1, ACLBLAS_STATUS_SUCCESS, 1e-5f},
    {"error: incx=0", ACLBLAS_LOWER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT, 10, 1, 10, 0, ACLBLAS_STATUS_INVALID_VALUE, 0.0f},
    // k/lda 标量非法用例走完整构造路径（分配非空 a/x/y 设备缓冲），保证 INVALID_VALUE
    // 来自新增的 k/lda 校验而非既有空指针校验——否则用例在修复前的 master 上同样通过，
    // 无法守护 #964（对齐 #591 检视口径）。
    {"error: k<0", ACLBLAS_LOWER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT, 10, -1, 10, 1, ACLBLAS_STATUS_INVALID_VALUE, 0.0f,
     true},
    {"error: lda<k+1", ACLBLAS_LOWER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT, 10, 4, 4, 1, ACLBLAS_STATUS_INVALID_VALUE, 0.0f,
     true},
    {"error: invalid uplo", static_cast<aclblasFillMode>(0), ACLBLAS_OP_N, ACLBLAS_NON_UNIT, 10, 1, 10, 1,
     ACLBLAS_STATUS_INVALID_VALUE, 0.0f},
    {"error: invalid trans", ACLBLAS_LOWER, static_cast<aclblasOperation>(0), ACLBLAS_NON_UNIT, 10, 1, 10, 1,
     ACLBLAS_STATUS_INVALID_VALUE, 0.0f},
    {"error: invalid diag", ACLBLAS_LOWER, ACLBLAS_OP_N, static_cast<aclblasDiagType>(0), 10, 1, 10, 1,
     ACLBLAS_STATUS_INVALID_VALUE, 0.0f},
    {"error: n<0", ACLBLAS_LOWER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT, -1, 1, 1, 1, ACLBLAS_STATUS_SUCCESS, 0.0f},
    {"error: handle is nullptr", ACLBLAS_LOWER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT, 10, 1, 10, 1,
     ACLBLAS_STATUS_HANDLE_IS_NULLPTR, 0.0f},
};

static inline uint32_t PhysicalPos(uint32_t logical, uint32_t n, int64_t inc, uint32_t absInc)
{
    return (inc >= 0) ? (logical * absInc) : ((n - 1U - logical) * absInc);
}

static void GenerateBanded(std::vector<float>& a, uint32_t n, uint32_t k, uint32_t lda, uint32_t seed, bool isUpper)
{
    a.assign(static_cast<size_t>(k + 1U) * lda, 0.0f);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    if (isUpper) {
        for (uint32_t col = 0; col < n; ++col) {
            uint32_t maxDist = std::min(k, col);
            for (uint32_t d = 0; d <= maxDist; ++d) {
                uint32_t bandIdx = k - d;
                a[static_cast<size_t>(bandIdx) * lda + col] = dist(rng);
            }
        }
    } else {
        for (uint32_t col = 0; col < n; ++col) {
            uint32_t maxBandRow = std::min(k, n - 1U - col);
            for (uint32_t bandRow = 0; bandRow <= maxBandRow; ++bandRow) {
                a[static_cast<size_t>(bandRow) * lda + col] = dist(rng);
            }
        }
    }
}

static void GenerateStrided(std::vector<float>& v, uint32_t n, int64_t inc, uint32_t seed)
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

// 计算当前行在带状矩阵中的列范围 [colStart, colEnd]：转置交换上下三角语义
static void StbmvBandRange(
    uint32_t row, uint32_t n, uint32_t k, bool isUpper, bool isTrans, uint32_t& colStart, uint32_t& colEnd)
{
    if (!isTrans) {
        if (isUpper) {
            colStart = row;
            colEnd = std::min(n - 1U, row + k);
        } else {
            colStart = (row > k) ? (row - k) : 0U;
            colEnd = row;
        }
    } else {
        if (isUpper) {
            colStart = (row > k) ? (row - k) : 0U;
            colEnd = row;
        } else {
            colStart = row;
            colEnd = std::min(n - 1U, row + k);
        }
    }
}

static void ComputeStbmvGolden(
    const std::vector<float>& a, const std::vector<float>& x, uint32_t n, uint32_t k, uint32_t lda, int64_t incx,
    uint32_t uplo, uint32_t trans, uint32_t diag, std::vector<float>& out)
{
    uint32_t absIncx = static_cast<uint32_t>(std::abs(incx));
    bool isUpper = (uplo == ACLBLAS_UPPER);
    bool isTrans = (trans == ACLBLAS_OP_T || trans == ACLBLAS_OP_C);
    bool isUnit = (diag == ACLBLAS_UNIT);

    out.assign(x.size(), 0.0f);

    for (uint32_t row = 0; row < n; ++row) {
        float sum = 0.0f;
        uint32_t colStart, colEnd;
        StbmvBandRange(row, n, k, isUpper, isTrans, colStart, colEnd);

        for (uint32_t col = colStart; col <= colEnd; ++col) {
            float aVal;
            if (isUnit && col == row) {
                aVal = 1.0f;
            } else {
                uint32_t bandIdx;
                uint32_t aCol;
                if (!isTrans) {
                    aCol = col;
                    bandIdx = isUpper ? (k - (col - row)) : (row - col);
                } else {
                    aCol = row;
                    bandIdx = isUpper ? (k - (row - col)) : (col - row);
                }
                aVal = a[static_cast<size_t>(bandIdx) * lda + aCol];
            }
            uint32_t xPos = PhysicalPos(col, n, incx, absIncx);
            sum += aVal * x[xPos];
        }
        uint32_t yPos = PhysicalPos(row, n, incx, absIncx);
        out[yPos] = sum;
    }
}

static void FreeStbmvBuffers(void* dA, void* dX, void* dY)
{
    aclrtFree(dA);
    aclrtFree(dX);
    aclrtFree(dY);
}

// 分配 dY 并写入全零初值，失败时就地释放自身缓冲；调用方负责释放此前已分配的 dA/dX
static int AllocAndUploadStbmvYBuffer(void** dY, size_t xBytes)
{
    aclError aclRet = aclrtMalloc(dY, xBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclRet != ACL_SUCCESS) {
        std::cout << "FAILED: aclrtMalloc dY ERROR " << aclRet << std::endl;
        return 1;
    }
    std::vector<float> yZero(xBytes / sizeof(float), 0.0f);
    aclRet = aclrtMemcpy(*dY, xBytes, yZero.data(), xBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_SUCCESS) {
        aclrtFree(*dY);
        *dY = nullptr;
        std::cout << "FAILED: aclrtMemcpy dY ERROR " << aclRet << std::endl;
        return 1;
    }
    return 0;
}

// issue #591: 分配/拷贝失败时报错退出，不得对空设备指针继续拷贝；任一步失败时
// 就地释放已分配缓冲，出参仅在全部成功后有效
static int AllocAndUploadStbmvBuffers(
    const std::vector<float>& aHost, const std::vector<float>& xHost, void** dA, void** dX, void** dY)
{
    const size_t aBytes = aHost.size() * sizeof(float);
    const size_t xBytes = xHost.size() * sizeof(float);
    aclError aclRet = aclrtMalloc(dA, aBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclRet != ACL_SUCCESS) {
        std::cout << "FAILED: aclrtMalloc dA ERROR " << aclRet << std::endl;
        return 1;
    }
    aclRet = aclrtMemcpy(*dA, aBytes, aHost.data(), aBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_SUCCESS) {
        aclrtFree(*dA);
        *dA = nullptr;
        std::cout << "FAILED: aclrtMemcpy dA ERROR " << aclRet << std::endl;
        return 1;
    }
    aclRet = aclrtMalloc(dX, xBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclRet != ACL_SUCCESS) {
        aclrtFree(*dA);
        *dA = nullptr;
        std::cout << "FAILED: aclrtMalloc dX ERROR " << aclRet << std::endl;
        return 1;
    }
    aclRet = aclrtMemcpy(*dX, xBytes, xHost.data(), xBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_SUCCESS) {
        aclrtFree(*dA);
        aclrtFree(*dX);
        *dA = nullptr;
        *dX = nullptr;
        std::cout << "FAILED: aclrtMemcpy dX ERROR " << aclRet << std::endl;
        return 1;
    }
    if (AllocAndUploadStbmvYBuffer(dY, xBytes) != 0) {
        aclrtFree(*dA);
        aclrtFree(*dX);
        *dA = nullptr;
        *dX = nullptr;
        return 1;
    }
    return 0;
}

// 同步 + D2H 取回 + golden 比对，成功返回 0；退出前释放设备缓冲
static int VerifyStbmvOutput(
    const TestCase& tc, aclrtStream stream, const std::vector<float>& aHost, const std::vector<float>& xHost, void* dA,
    void* dX, void* dY, uint32_t nU32, uint32_t kU32, uint32_t ldaU32)
{
    const size_t xBytes = xHost.size() * sizeof(float);
    // issue #591: 同步/D2H 失败时报错退出，不得基于陈旧或未初始化数据比较
    aclError aclRet = aclrtSynchronizeStream(stream);
    if (aclRet != ACL_SUCCESS) {
        FreeStbmvBuffers(dA, dX, dY);
        std::cout << "FAILED: aclrtSynchronizeStream ERROR " << aclRet << std::endl;
        return 1;
    }

    std::vector<float> yNpu(xHost.size());
    aclRet = aclrtMemcpy(yNpu.data(), xBytes, dY, xBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    if (aclRet != ACL_SUCCESS) {
        FreeStbmvBuffers(dA, dX, dY);
        std::cout << "FAILED: aclrtMemcpy D2H ERROR " << aclRet << std::endl;
        return 1;
    }

    std::vector<float> yGolden;
    ComputeStbmvGolden(
        aHost, xHost, nU32, kU32, ldaU32, tc.incx, static_cast<uint32_t>(tc.uplo), static_cast<uint32_t>(tc.trans),
        static_cast<uint32_t>(tc.diag), yGolden);

    FreeStbmvBuffers(dA, dX, dY);

    float absTol = tc.tol, relTol = tc.tol, maxDiff = 0.0f;
    for (size_t i = 0; i < yNpu.size(); ++i) {
        float diff = std::abs(yNpu[i] - yGolden[i]);
        if (diff > maxDiff) {
            maxDiff = diff;
        }
        float scale = std::max(std::abs(yNpu[i]), std::abs(yGolden[i]));
        if (diff > absTol && diff > relTol * scale) {
            std::cout << "FAILED at " << i << " (" << yNpu[i] << " vs " << yGolden[i] << ")" << std::endl;
            return 1;
        }
    }
    std::cout << "PASSED (maxDiff=" << maxDiff << ")" << std::endl;
    return 0;
}

// n<=0 或标量非法（非 allocBuffers）用例：不分配设备缓冲，直接断言期望状态码
static int RunStbmvErrorCase(const TestCase& tc, aclblasHandle_t handle)
{
    auto toLegacyUplo = static_cast<aclblasFillMode>(tc.uplo);
    auto toLegacyTrans = static_cast<aclblasOperation>(tc.trans);
    auto toLegacyDiag = static_cast<aclblasDiagType>(tc.diag);
    aclblasHandle_t useHandle = (tc.expectResult == ACLBLAS_STATUS_HANDLE_IS_NULLPTR) ? nullptr : handle;
    std::vector<float> dummy;
    aclblasStatus_t ret = aclblasStbmv_legacy(
        useHandle, toLegacyUplo, toLegacyTrans, toLegacyDiag, nullptr, tc.lda, nullptr,
        dummy.empty() ? nullptr : dummy.data(), tc.n, tc.k, tc.incx);
    if (ret != tc.expectResult) {
        std::cout << "FAILED: expected " << static_cast<int>(tc.expectResult) << " got " << static_cast<int>(ret)
                  << std::endl;
        return 1;
    }
    if (tc.expectResult == ACLBLAS_STATUS_SUCCESS) {
        std::cout << "PASSED (NOT_SUPPORTED)" << std::endl;
    } else {
        std::cout << "PASSED" << std::endl;
    }
    return 0;
}

static int RunStbmvCase(const TestCase& tc, aclblasHandle_t handle, aclrtStream stream)
{
    std::cout << "  " << tc.name << " ... " << std::flush;

    if (tc.n <= 0 || (tc.expectResult != ACLBLAS_STATUS_SUCCESS && !tc.allocBuffers)) {
        return RunStbmvErrorCase(tc, handle);
    }

    const uint32_t nU32 = static_cast<uint32_t>(tc.n);
    // allocBuffers 的标量非法用例（k<0/lda<k+1）仅要求非空设备缓冲，内核不会下发；
    // 生成阶段将 k 钳到非负，避免 uint32 回绕产生超大分配。
    const uint32_t kU32 = static_cast<uint32_t>(tc.k < 0 ? 0 : tc.k);
    const uint32_t ldaU32 = static_cast<uint32_t>(tc.lda);
    const uint32_t seed = static_cast<uint32_t>(tc.n + tc.k + tc.lda * 100);
    const bool isUpper = (tc.uplo == ACLBLAS_UPPER);

    std::vector<float> aHost;
    GenerateBanded(aHost, nU32, kU32, ldaU32, seed, isUpper);
    std::vector<float> xHost;
    GenerateStrided(xHost, nU32, tc.incx, seed + 1);

    void* dA = nullptr;
    void* dX = nullptr;
    void* dY = nullptr;
    if (AllocAndUploadStbmvBuffers(aHost, xHost, &dA, &dX, &dY) != 0) {
        return 1;
    }

    auto toLegacyUplo = static_cast<aclblasFillMode>(tc.uplo);
    auto toLegacyTrans = static_cast<aclblasOperation>(tc.trans);
    auto toLegacyDiag = static_cast<aclblasDiagType>(tc.diag);
    aclblasStatus_t blasRet = aclblasStbmv_legacy(
        handle, toLegacyUplo, toLegacyTrans, toLegacyDiag, static_cast<const float*>(dA), tc.lda,
        static_cast<const float*>(dX), static_cast<float*>(dY), tc.n, tc.k, tc.incx);
    if (blasRet != tc.expectResult) {
        FreeStbmvBuffers(dA, dX, dY);
        std::cout << "FAILED: expected " << static_cast<int>(tc.expectResult) << " got " << static_cast<int>(blasRet)
                  << std::endl;
        return 1;
    }
    if (blasRet != ACLBLAS_STATUS_SUCCESS) {
        FreeStbmvBuffers(dA, dX, dY);
        std::cout << "PASSED" << std::endl;
        return 0;
    }
    return VerifyStbmvOutput(tc, stream, aHost, xHost, dA, dX, dY, nU32, kU32, ldaU32);
}

int32_t main(int32_t argc, char* argv[])
{
    (void)argc;
    (void)argv;
    int32_t deviceId = 0;
    // issue #591: main 初始化链路逐一检查返回值，失败时明确报错并以非零码退出
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
    int total = static_cast<int>(gStbmvTests.size());
    std::cout << "Running " << total << " stbmv arch22 test cases..." << std::endl;
    for (const auto& tc : gStbmvTests) {
        if (RunStbmvCase(tc, handle, stream) != 0)
            ++failed;
    }
    std::cout << "\nResults: " << (total - failed) << "/" << total << " passed" << std::endl;

    aclblasDestroy(handle);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return (failed == 0) ? 0 : 1;
}
