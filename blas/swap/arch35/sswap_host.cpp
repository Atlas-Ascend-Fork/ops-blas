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
#include "log/log.h"
#include "cann_ops_blas.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"
#include "sswap_tiling_data.h"

void sswap_kernel_do(
    uint8_t* x, uint8_t* y, uint8_t* workSpace, const SswapTilingData& tiling, uint32_t numBlocks, void* stream);

static aclblasStatus_t ValidateSswapParams(const float* x, const float* y, int incx, int incy)
{
    if (x == nullptr || y == nullptr) {
        OP_LOGE("aclblasSswap", "x/y must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (incx == 0 || incy == 0) {
        OP_LOGE("aclblasSswap", "incx and incy must not be zero");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

static SswapTilingData CalSswapTilingData(uint32_t n, uint32_t coreNum, int32_t incx, int32_t incy)
{
    SswapTilingData tiling{};
    tiling.totalN = n;
    tiling.incx = incx;
    tiling.incy = incy;
    tiling.kx = incx < 0 ? (1LL - static_cast<int64_t>(n)) * incx : 0;
    tiling.ky = incy < 0 ? (1LL - static_cast<int64_t>(n)) * incy : 0;
    tiling.xLength =
        1 + (static_cast<uint64_t>(n) - 1) * static_cast<uint64_t>(incx < 0 ? -static_cast<int64_t>(incx) : incx);
    tiling.yLength =
        1 + (static_cast<uint64_t>(n) - 1) * static_cast<uint64_t>(incy < 0 ? -static_cast<int64_t>(incy) : incy);

    constexpr uint32_t alignUnit = ELEMENTS_PER_BLOCK;
    uint32_t rawPerCore = n / coreNum;
    tiling.perCoreN = (rawPerCore / alignUnit) * alignUnit;
    tiling.remainder = n - tiling.perCoreN * coreNum;

    constexpr uint32_t ubSize = 248 * 1024;
    constexpr uint32_t bufferCount = 2;
    uint32_t maxElements = ubSize / (bufferCount * sizeof(float));
    tiling.tileSize = (maxElements / alignUnit) * alignUnit;

    return tiling;
}

aclblasStatus_t aclblasSswap(aclblasHandle_t handle, int n, float* x, int incx, float* y, int incy)
{
    if (n <= 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    if (handle == nullptr) {
        OP_LOGE("aclblasSswap", "handle is nullptr");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }

    aclblasStatus_t st = ValidateSswapParams(x, y, incx, incy);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }

    uint32_t aivCoreNum = GetAivCoreCount();
    if (aivCoreNum == 0) {
        OP_LOGE("aclblasSswap", "GetAivCoreCount failed");
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }

    auto* h = handle;
    aclrtStream useStream = h->stream;

    uint32_t totalN = static_cast<uint32_t>(n);
    uint32_t numBlocks = (totalN < aivCoreNum) ? totalN : aivCoreNum;

    SswapTilingData tiling = CalSswapTilingData(totalN, numBlocks, incx, incy);

    OP_LOGD(
        "aclblasSswap", "tiling: totalN=%u perCoreN=%u remainder=%u tileSize=%u numBlocks=%u", tiling.totalN,
        tiling.perCoreN, tiling.remainder, tiling.tileSize, numBlocks);
    OP_LOGI("aclblasSswap", "launching kernel");

    sswap_kernel_do(
        reinterpret_cast<uint8_t*>(x), reinterpret_cast<uint8_t*>(y), nullptr, tiling, numBlocks, useStream);

    return ACLBLAS_STATUS_SUCCESS;
}
