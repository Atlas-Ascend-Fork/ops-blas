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
#include <algorithm>
#include "log/log.h"
#include "cann_ops_blas.h"
#include "saxpy_tiling_data.h"
#include "saxpy_kernel.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"

static aclblasStatus_t ValidateSaxpyParams(const float* alpha, const float* x, int incx, float* y, int incy)
{
    if (alpha == nullptr) {
        OP_LOGE("aclblasSaxpy", "alpha must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (x == nullptr) {
        OP_LOGE("aclblasSaxpy", "x must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (y == nullptr) {
        OP_LOGE("aclblasSaxpy", "y must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (incx == 0) {
        OP_LOGE("aclblasSaxpy", "incx must not be zero, got %d", incx);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (incy == 0) {
        OP_LOGE("aclblasSaxpy", "incy must not be zero, got %d", incy);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

static SaxpyTilingData CalSaxpyTilingDataContiguous(uint32_t totalN, uint32_t coreNum, float alpha)
{
    SaxpyTilingData tiling{};
    tiling.totalN = totalN;
    tiling.incx = 1;
    tiling.incy = 1;

    constexpr uint32_t alignUnit = SAXPY_ELEMENTS_PER_BLOCK;
    if (coreNum == 0) {
        OP_LOGE("aclblasSaxpy", "coreNum is 0, skip tiling calculation");
        return tiling;
    }
    uint32_t rawPerCore = totalN / coreNum;
    tiling.perCoreN = (rawPerCore / alignUnit) * alignUnit;
    tiling.remainder = totalN - tiling.perCoreN * coreNum;

    constexpr uint32_t queueCount = 2;
    // 连续路径单轮搬运 tile 的元素数上限（2^15）：kernel 按 tileLoop/tileTail 分轮搬运，
    // 该上限约束单轮 DataCopy 规模；UB 容量推导值低于上限时此封顶不生效
    constexpr uint32_t SAXPY_CONTIG_TILE_MAX_ELEMENTS = 32768;
    uint32_t maxElements = UB_SIZE / (queueCount * sizeof(float));
    maxElements = (maxElements / alignUnit) * alignUnit;
    if (maxElements > SAXPY_CONTIG_TILE_MAX_ELEMENTS) {
        maxElements = SAXPY_CONTIG_TILE_MAX_ELEMENTS;
    }
    tiling.tileSize = maxElements;

    tiling.alpha = alpha;

    return tiling;
}

static SaxpyTilingData CalSaxpyTilingDataStrided(
    int64_t n, int64_t incx, int64_t incy, uint32_t aivCoreNum, float alpha)
{
    SaxpyTilingData tiling{};
    tiling.totalN = static_cast<uint32_t>(n);
    tiling.incx = incx;
    tiling.incy = incy;
    tiling.alpha = alpha;

    uint32_t useCoreNum = std::min(aivCoreNum, static_cast<uint32_t>(n));
    if (useCoreNum == 0) {
        useCoreNum = 1;
    }
    if (useCoreNum > SAXPY_MAX_CORE_NUM) {
        useCoreNum = SAXPY_MAX_CORE_NUM;
    }
    tiling.useCoreNum = useCoreNum;

    uint32_t baseCount = static_cast<uint32_t>(n) / useCoreNum;
    uint32_t remain = static_cast<uint32_t>(n) % useCoreNum;
    uint32_t offset = 0;
    for (uint32_t i = 0; i < useCoreNum; i++) {
        tiling.startOffset[i] = offset;
        tiling.calCount[i] = baseCount + (i < remain ? 1 : 0);
        offset += tiling.calCount[i];
    }
    for (uint32_t i = useCoreNum; i < SAXPY_MAX_CORE_NUM; i++) {
        tiling.startOffset[i] = 0;
        tiling.calCount[i] = 0;
    }

    tiling.nthreads = std::min(
        CeilAlign<uint32_t>(CeilDiv<uint32_t>(static_cast<uint32_t>(n), useCoreNum), SIMT_MIN_THREAD_NUM),
        SIMT_MAX_THREAD_NUM);

    return tiling;
}

static aclblasStatus_t LaunchSaxpyKernel(
    aclblasHandle_t handle, int n, int incx, int incy, const float* alpha, float* x, float* y)
{
    uint32_t coreNum = GetAivCoreCount();
    if (coreNum == 0) {
        OP_LOGE("aclblasSaxpy", "GetAivCoreCount failed");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    uint32_t totalN = static_cast<uint32_t>(n);
    SaxpyTilingData tiling{};
    uint32_t numBlocks;

    if (incx == 1 && incy == 1) {
        numBlocks = (totalN < coreNum) ? totalN : coreNum;
        tiling = CalSaxpyTilingDataContiguous(totalN, numBlocks, *alpha);
    } else {
        numBlocks = std::min(CeilDiv<uint32_t>(totalN, SIMT_MIN_THREAD_NUM), coreNum);
        // kernel 以 blockIdx 直接索引 tiling.calCount[SAXPY_MAX_CORE_NUM] 定长数组，
        // 启动块数必须同步按该上限封顶（对照 sasum/scasum 的封顶口径），
        // 否则核数大于 SAXPY_MAX_CORE_NUM 的硬件上 blockIdx 越界读取 calCount
        if (numBlocks > SAXPY_MAX_CORE_NUM) {
            numBlocks = SAXPY_MAX_CORE_NUM;
        }
        tiling = CalSaxpyTilingDataStrided(n, incx, incy, numBlocks, *alpha);
    }

    OP_LOGD(
        "aclblasSaxpy", "tiling: n=%d incx=%d incy=%d numBlocks=%u alpha=%.6f", n, incx, incy, numBlocks,
        static_cast<double>(*alpha));
    OP_LOGI("aclblasSaxpy", "launching kernel: blocks=%u, cores=%u", numBlocks, coreNum);

    auto* h = handle;
    saxpy_kernel_do(
        reinterpret_cast<uint8_t*>(x), reinterpret_cast<uint8_t*>(y), nullptr, tiling, numBlocks, h->stream);

    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t aclblasSaxpy(aclblasHandle_t handle, int n, const float* alpha, float* x, int incx, float* y, int incy)
{
    if (handle == nullptr) {
        OP_LOGE("aclblasSaxpy", "handle is nullptr");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }

    // BLAS Level 1 standard: n <= 0 is a no-op, not an error
    if (n <= 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    aclblasStatus_t validRet = ValidateSaxpyParams(alpha, x, incx, y, incy);
    if (validRet != ACLBLAS_STATUS_SUCCESS) {
        return validRet;
    }

    OP_LOGI("aclblasSaxpy", "n=%d, incx=%d, incy=%d, alpha=%.6f", n, incx, incy, static_cast<double>(*alpha));

    return LaunchSaxpyKernel(handle, n, incx, incy, alpha, x, y);
}
