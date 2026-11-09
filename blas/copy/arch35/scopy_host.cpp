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
#include <vector>
#include "log/log.h"
#include "cann_ops_blas.h"
#include "scopy_tiling_data.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"

void scopy_kernel_do(
    uint8_t* x, uint8_t* y, uint8_t* workSpace, const ScopyTilingData& tiling, uint32_t numBlocks, void* stream);

static aclblasStatus_t ValidateScopyParams(const float* x, const float* y, int incx, int incy)
{
    if (x == nullptr || y == nullptr) {
        OP_LOGE("aclblasScopy", "x/y must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (incx == 0) {
        OP_LOGE("aclblasScopy", "incx must not be zero");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (incy == 0) {
        OP_LOGE("aclblasScopy", "incy must not be zero");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

static ScopyTilingData CalScopyTilingData(uint32_t n, uint32_t coreNum, int incx, int incy)
{
    ScopyTilingData tiling;
    tiling.totalN = n;
    tiling.incx = incx;
    tiling.incy = incy;

    constexpr uint32_t alignUnit = ELEMENTS_PER_BLOCK;
    uint32_t rawPerCore = n / coreNum;
    tiling.perCoreN = (rawPerCore / alignUnit) * alignUnit;
    uint32_t leftover = n - tiling.perCoreN * coreNum;
    tiling.extraBlockCores = leftover / alignUnit;
    tiling.tailElements = leftover % alignUnit;

    auto* platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint64_t ubSize = 248 * 1024; // fallback
    if (platform != nullptr) {
        platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    }
    // BUFFER_NUM=2 in kernel: continuous path uses double-buffer Prime-Pump-Drain
    // Each slot is tileSize*sizeof(float), total = 2*tileSize*sizeof(float) for inQueueX_
    constexpr uint32_t bufferCount = 2;
    constexpr uint32_t ubSafetyMargin = 256;

    if (incx == 1 && incy == 1) {
        // Pure continuous: large tiles using DataCopy, double buffer
        uint32_t availableUB = ubSize - ubSafetyMargin;
        uint32_t maxElements = availableUB / (bufferCount * sizeof(float));
        tiling.tileSize = (maxElements / alignUnit) * alignUnit;
    } else if (incx == 1 || incy == 1) {
        // Mixed: one side continuous (DataCopy, unlimited), one side discrete (Compact, batched).
        // UB layout: inQueueX_ (2×tileSize×4B) + tmpBuf_ (tileSize×8B for Gather).
        // Worst-case (reorder=true): tileSize ≤ UB/(16B) ≈ 15872.
        constexpr uint32_t bytesPerElemInOut = 2 * sizeof(float) + sizeof(float) + sizeof(uint32_t); // 16B
        uint32_t availableUB = ubSize - ubSafetyMargin;
        uint32_t maxElements = availableUB / bytesPerElemInOut;
        tiling.tileSize = (maxElements / alignUnit) * alignUnit;
    } else {
        // Pure discrete: both sides use Compact, limited by blockCount ≤ 4095
        tiling.tileSize = (4095 / alignUnit) * alignUnit; // = 4088
    }

    return tiling;
}

// Prepare reverse-index offset table in handle's workspace for kernel DataCopy.
// Returns workspace pointer on success, nullptr on failure or if not needed.
static uint8_t* PrepareOffsetWorkspace(_aclblas_handle* h, int incx, int incy, uint32_t tileSize)
{
    bool needReorder = (incx < 0) != (incy < 0);
    if (!needReorder)
        return nullptr;

    void* ws = GetEffectiveWorkspace(h);
    size_t wsSize = GetEffectiveWorkspaceSize(h);
    uint32_t offsetBytes = tileSize * sizeof(uint32_t);
    if (ws == nullptr || wsSize < offsetBytes) {
        OP_LOGD(
            "aclblasScopy", "workspace unavailable or too small (%zu < %u), fallback to scalar loop", wsSize,
            offsetBytes);
        return nullptr;
    }

    std::vector<uint32_t> offsetHost(tileSize);
    for (uint32_t i = 0; i < tileSize; i++) {
        offsetHost[i] = (tileSize - 1 - i) * sizeof(float);
    }
    aclError aclRet = aclrtMemcpy(ws, offsetBytes, offsetHost.data(), offsetBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_SUCCESS) {
        OP_LOGE("aclblasScopy", "aclrtMemcpy to workspace failed, fallback to scalar loop");
        return nullptr;
    }

    OP_LOGD("aclblasScopy", "pre-generated offset table to workspace, size=%u bytes", offsetBytes);
    return static_cast<uint8_t*>(ws);
}

static void LaunchScopyKernel(
    const ScopyTilingData& tiling, const float* x, float* y, uint8_t* workSpace, uint32_t numBlocks,
    aclrtStream useStream)
{
    scopy_kernel_do(
        reinterpret_cast<uint8_t*>(const_cast<float*>(x)), reinterpret_cast<uint8_t*>(y), workSpace, tiling, numBlocks,
        useStream);
}

aclblasStatus_t aclblasScopy(aclblasHandle_t handle, int n, const float* x, int incx, float* y, int incy)
{
    if (n < 0) {
        OP_LOGE("aclblasScopy", "n must be >= 0, got %d", n);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    if (n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    if (handle == nullptr) {
        OP_LOGE("aclblasScopy", "handle is nullptr");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }

    aclblasStatus_t st = ValidateScopyParams(x, y, incx, incy);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }

    // issue #866: (n-1)*|inc| 以 uint64 计算并拒绝超过 uint32 索引域的组合（对齐 #804/#806
    // sasum/sscal 拒绝式口径），超限输入不再下发 kernel
    const int64_t absIncx64 = (incx < 0) ? -static_cast<int64_t>(incx) : static_cast<int64_t>(incx);
    const int64_t absIncy64 = (incy < 0) ? -static_cast<int64_t>(incy) : static_cast<int64_t>(incy);
    const int64_t spanX = static_cast<int64_t>(n - 1) * absIncx64;
    const int64_t spanY = static_cast<int64_t>(n - 1) * absIncy64;
    if (spanX > static_cast<int64_t>(UINT32_MAX) || spanY > static_cast<int64_t>(UINT32_MAX)) {
        OP_LOGE(
            "aclblasScopy",
            "strided offset overflow: n=%d incx=%d incy=%d spanX=%lld spanY=%lld exceeds uint32 index domain", n, incx,
            incy, static_cast<long long>(spanX), static_cast<long long>(spanY));
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    uint32_t aivCoreNum = GetAivCoreCount();
    if (aivCoreNum == 0) {
        OP_LOGE("aclblasScopy", "vector core count is 0");
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }

    auto* h = handle;
    aclrtStream useStream = h->stream;

    uint32_t totalN = static_cast<uint32_t>(n);
    uint32_t numBlocks = (totalN < aivCoreNum) ? totalN : aivCoreNum;

    ScopyTilingData tiling = CalScopyTilingData(totalN, numBlocks, incx, incy);

    OP_LOGD(
        "aclblasScopy", "tiling: totalN=%u perCoreN=%u extra=%u tail=%u tileSize=%u numBlocks=%u incx=%d incy=%d",
        tiling.totalN, tiling.perCoreN, tiling.extraBlockCores, tiling.tailElements, tiling.tileSize, numBlocks,
        tiling.incx, tiling.incy);
    OP_LOGD("aclblasScopy", "launching kernel: blocks=%u n=%u incx=%d incy=%d", numBlocks, totalN, incx, incy);

    uint8_t* wsPtr = PrepareOffsetWorkspace(h, incx, incy, tiling.tileSize);

    LaunchScopyKernel(tiling, x, y, wsPtr, numBlocks, useStream);
    return ACLBLAS_STATUS_SUCCESS;
}
