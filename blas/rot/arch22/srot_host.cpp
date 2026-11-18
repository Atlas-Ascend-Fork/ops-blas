/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <climits>
#include "cann_ops_blas.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"
#include "log/log.h"
#include "srot_kernel.h"

namespace {
aclblasStatus_t ValidateSrotInputs(
    int n, const float* x, int incx, const float* y, int incy, const float* c, const float* s)
{
    if (x == nullptr || y == nullptr || c == nullptr || s == nullptr) {
        OP_LOGE("aclblasSrot", "x, y, c and s must not be nullptr when n is positive");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (incx == INT_MIN || incy == INT_MIN) {
        OP_LOGE("aclblasSrot", "INT_MIN stride is invalid: incx=%d, incy=%d", incx, incy);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    const int64_t absX = incx < 0 ? -static_cast<int64_t>(incx) : incx;
    const int64_t absY = incy < 0 ? -static_cast<int64_t>(incy) : incy;
    if ((n - 1) * absX > INT32_MAX || (n - 1) * absY > INT32_MAX) {
        OP_LOGE("aclblasSrot", "vector offset exceeds INT32_MAX: n=%d, incx=%d, incy=%d", n, incx, incy);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t PrepareSrotScalars(const float* c, const float* s, SrotTilingData& tiling)
{
    bool cDevice = false;
    bool sDevice = false;
    if (CheckPtrLocation(c, &cDevice) != ACLBLAS_STATUS_SUCCESS) {
        OP_LOGE("aclblasSrot", "failed to query the memory location of c");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (CheckPtrLocation(s, &sDevice) != ACLBLAS_STATUS_SUCCESS) {
        OP_LOGE("aclblasSrot", "failed to query the memory location of s");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    tiling.cIsDevice = cDevice;
    tiling.sIsDevice = sDevice;
    tiling.c = cDevice ? 0.0f : *c;
    tiling.s = sDevice ? 0.0f : *s;
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t BuildSrotTiling(
    int n, int incx, int incy, const float* c, const float* s, SrotTilingData& tiling, uint32_t& blocks)
{
    const auto status = PrepareSrotScalars(c, s, tiling);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }
    auto* platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    if (platform == nullptr || platform->GetCoreNumAiv() == 0) {
        OP_LOGE("aclblasSrot", "platform information or AIV cores are unavailable");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    uint64_t ubBytes = 0;
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytes);
    const bool contiguous = incx == 1 && incy == 1;
    const uint32_t tileCap = contiguous ? 4096U : 256U;
    const uint32_t alignment = contiguous ? 64U : 8U;
    // Four double-buffered queues, one scratch tile, and (strided only) indices.
    const uint32_t bytesPerElement = contiguous ? 36U : 320U;
    const uint32_t tile = std::min<uint64_t>(tileCap, ubBytes / bytesPerElement) / alignment * alignment;
    if (tile == 0) {
        OP_LOGE("aclblasSrot", "UB capacity is insufficient: bytes=%llu", static_cast<unsigned long long>(ubBytes));
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    const uint32_t count = static_cast<uint32_t>(n);
    blocks = (incx == 0 || incy == 0) ? 1U : std::min(platform->GetCoreNumAiv(), CeilDiv(count, tileCap));
    tiling.totalN = count;
    tiling.blockLength = CeilAlign(CeilDiv(count, blocks), 128U);
    tiling.tileLength = tile;
    tiling.incx = incx;
    tiling.incy = incy;
    return ACLBLAS_STATUS_SUCCESS;
}
} // namespace

extern "C" aclblasStatus_t aclblasSrot(
    aclblasHandle_t handle, int n, float* x, int incx, float* y, int incy, const float* c, const float* s)
{
    if (handle == nullptr) {
        OP_LOGE("aclblasSrot", "handle must not be nullptr");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (n < 0) {
        OP_LOGE("aclblasSrot", "n must be nonnegative: n=%d", n);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    auto status = ValidateSrotInputs(n, x, incx, y, incy, c, s);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }
    SrotTilingData tiling{};
    uint32_t blocks = 0;
    status = BuildSrotTiling(n, incx, incy, c, s, tiling, blocks);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }
    srot_kernel_do(
        x, y, tiling.cIsDevice ? c : nullptr, tiling.sIsDevice ? s : nullptr, blocks, tiling, handle->stream);
    return ACLBLAS_STATUS_SUCCESS;
}
