/**
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <vector>
#include <algorithm>
#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "cann_ops_blas_common.h"
#include "common/helper/aclblas_handle_internal.h"

void strsv_kernel_do(
    uint8_t* A, uint8_t* x, uint8_t* tilingGm, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag,
    int n, int lda, uint32_t numBlocks, void* stream);

constexpr uint32_t MAX_CORE_NUM = 40;

struct StrsvTilingData {
    uint32_t n;
    uint32_t lda;
    uint32_t useCoreNum;
    uint32_t startRow[MAX_CORE_NUM];
    uint32_t rowCount[MAX_CORE_NUM];
};

constexpr uint64_t BYTENUM_PER_FLOAT32_TILING = 4;
constexpr uint64_t UB_BYTENUM_PER_BLOCK_TILING = 32;
constexpr uint64_t ELEMENTS_PER_BLOCK_TILING = UB_BYTENUM_PER_BLOCK_TILING / BYTENUM_PER_FLOAT32_TILING;

static StrsvTilingData CalStrsvTilingData(int n, int lda, uint32_t coreNum)
{
    StrsvTilingData tiling;
    tiling.n = static_cast<uint32_t>(n);
    tiling.lda = static_cast<uint32_t>(lda);
    tiling.useCoreNum = 0;

    for (uint32_t i = 0; i < MAX_CORE_NUM; i++) {
        tiling.startRow[i] = 0;
        tiling.rowCount[i] = 0;
    }

    if (coreNum == 0) {
        coreNum = 1;
    }

    uint32_t totalBlockNum = coreNum;
    uint32_t rowsPerBlock = (n + totalBlockNum - 1) / totalBlockNum;

    for (uint32_t i = 0; i < totalBlockNum && i < MAX_CORE_NUM; i++) {
        tiling.startRow[i] = i * rowsPerBlock;
        uint32_t rowEnd = (i + 1) * rowsPerBlock;
        if (rowEnd > n)
            rowEnd = n;
        tiling.rowCount[i] = (rowEnd > tiling.startRow[i]) ? (rowEnd - tiling.startRow[i]) : 0;

        if (tiling.rowCount[i] > 0) {
            tiling.useCoreNum = i + 1;
        }
    }

    return tiling;
}

static void PackStrsvVector(
    const std::vector<float>& xHost, int n, int incx, size_t absIncx, std::vector<float>& xPacked)
{
    for (int i = 0; i < n; i++) {
        const size_t offset = incx > 0 ? static_cast<size_t>(i) * absIncx : static_cast<size_t>(n - 1 - i) * absIncx;
        xPacked[i] = xHost[offset];
    }
}

static void ScatterStrsvVector(
    const std::vector<float>& xPacked, int n, int incx, size_t absIncx, std::vector<float>& xHost)
{
    for (int i = 0; i < n; i++) {
        const size_t offset = incx > 0 ? static_cast<size_t>(i) * absIncx : static_cast<size_t>(n - 1 - i) * absIncx;
        xHost[offset] = xPacked[i];
    }
}

aclblasStatus_t aclblasStrsv(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, int n,
    const float* A, int lda, float* x, int incx)
{
    if (n <= 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    if (incx == 0) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    if (A == nullptr || x == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    aclrtContext currentCtx = nullptr;
    aclError aclRet = aclrtGetCurrentContext(&currentCtx);
    if (aclRet != ACL_SUCCESS || currentCtx == nullptr) {
        return ACLBLAS_STATUS_NOT_INITIALIZED;
    }

    // 从 handle 获取 stream
    aclrtStream useStream = nullptr;
    if (handle != nullptr) {
        auto* h = handle;
        useStream = h->stream;
    }

    const size_t absIncx = static_cast<size_t>(incx < 0 ? -static_cast<int64_t>(incx) : incx);
    const size_t xElements = (static_cast<size_t>(n) - 1) * absIncx + 1;
    const size_t xSize = static_cast<size_t>(n) * sizeof(float);
    const size_t xSpanSize = xElements * sizeof(float);

    uint8_t* xDevice = nullptr;
    aclRet = aclrtMalloc(reinterpret_cast<void**>(&xDevice), xSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclRet != ACL_SUCCESS) {
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }

    // A and x are device pointers. Stage the strided x span on the host,
    // then pack the logical vector for the contiguous Arch22 kernel.
    std::vector<float> xHost(xElements);
    std::vector<float> xPacked(n);
    aclRet = aclrtMemcpy(xHost.data(), xSpanSize, x, xSpanSize, ACL_MEMCPY_DEVICE_TO_HOST);
    if (aclRet != ACL_SUCCESS) {
        aclrtFree(xDevice);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    PackStrsvVector(xHost, n, incx, absIncx, xPacked);

    aclRet = aclrtMemcpy(xDevice, xSize, xPacked.data(), xSize, ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_SUCCESS) {
        aclrtFree(xDevice);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    constexpr uint32_t numBlocks = 1;

    StrsvTilingData tiling = CalStrsvTilingData(n, lda, numBlocks);

    uint8_t* tilingDevice = nullptr;
    aclRet = aclrtMalloc(reinterpret_cast<void**>(&tilingDevice), sizeof(StrsvTilingData), ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclRet != ACL_SUCCESS) {
        aclrtFree(xDevice);
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }

    aclRet =
        aclrtMemcpy(tilingDevice, sizeof(StrsvTilingData), &tiling, sizeof(StrsvTilingData), ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_SUCCESS) {
        aclrtFree(xDevice);
        aclrtFree(tilingDevice);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    strsv_kernel_do(
        reinterpret_cast<uint8_t*>(const_cast<float*>(A)), xDevice, tilingDevice, uplo, trans, diag, n, lda,
        tiling.useCoreNum, useStream);

    aclRet = aclrtSynchronizeStream(useStream);
    if (aclRet != ACL_SUCCESS) {
        aclrtFree(xDevice);
        aclrtFree(tilingDevice);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    aclRet = aclrtMemcpy(xPacked.data(), xSize, xDevice, xSize, ACL_MEMCPY_DEVICE_TO_HOST);
    if (aclRet != ACL_SUCCESS) {
        aclrtFree(xDevice);
        aclrtFree(tilingDevice);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    ScatterStrsvVector(xPacked, n, incx, absIncx, xHost);
    aclRet = aclrtMemcpy(x, xSpanSize, xHost.data(), xSpanSize, ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_SUCCESS) {
        aclrtFree(xDevice);
        aclrtFree(tilingDevice);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    aclrtFree(xDevice);
    aclrtFree(tilingDevice);

    return ACLBLAS_STATUS_SUCCESS;
}
