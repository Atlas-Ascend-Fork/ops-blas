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
#include "cann_ops_blas.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"
#include "log/log.h"
#include "common/helper/kernel_constant.h"
#include "isamax_kernel.h"
#include "isamax_tiling_data.h"

namespace {

static aclblasStatus_t ValidateIsamaxParams(
    aclblasHandle_t handle, int n, int incx, const float* x, int* result, bool& quickReturn)
{
    quickReturn = false;
    if (handle == nullptr) {
        OP_LOGE("aclblasIsamax", "handle is nullptr");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (n < 0) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (result == nullptr) {
        OP_LOGE("aclblasIsamax", "result must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (n == 0 || incx < 1) {
        // issue #738: result 语义为 device 地址，快速返回按 icamax 惯例由调用方
        // aclrtMemsetAsync 写回，不得在 host 直接解引用
        quickReturn = true;
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (x == nullptr) {
        OP_LOGE("aclblasIsamax", "x must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    // issue #745: 与 icamin 对齐的步长跨度守卫——kernel 以 int32 寻址 float 元素
    const int64_t lastElement = static_cast<int64_t>(n - 1) * static_cast<int64_t>(incx);
    if (lastElement < 0 || lastElement > static_cast<int64_t>(INT32_MAX)) {
        OP_LOGE(
            "aclblasIsamax", "strided offset overflow: n=%d incx=%d lastElement=%lld", n, incx,
            static_cast<long long>(lastElement));
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

static IsamaxTilingData CalcIsamaxTiling(int n, uint32_t numBlocks, int incx)
{
    IsamaxTilingData tiling{};
    uint32_t totalN = static_cast<uint32_t>(n);
    tiling.totalN = totalN;
    tiling.useCoreNum = numBlocks;
    tiling.incx = static_cast<uint32_t>(incx);

    tiling.perCoreN = totalN / numBlocks;
    tiling.lastCoreN = tiling.perCoreN + (totalN % numBlocks);

    tiling.tileSize = FP32_MAX_DATA_COUNT;

    if (incx != 1) {
        tiling.nthreads = std::min(
            CeilAlign<uint32_t>(CeilDiv<uint32_t>(tiling.perCoreN, SIMT_MIN_THREAD_NUM), SIMT_MIN_THREAD_NUM),
            SIMT_MAX_THREAD_NUM);
    } else {
        tiling.nthreads = 0;
    }

    return tiling;
}

static aclblasStatus_t LaunchIsamaxKernel(aclblasHandle_t handle, int n, const float* x, int incx, int* result)
{
    bool quickReturn = false;
    aclblasStatus_t vStatus = ValidateIsamaxParams(handle, n, incx, x, result, quickReturn);
    if (vStatus != ACLBLAS_STATUS_SUCCESS || n <= 0 || incx < 1) {
        if (quickReturn && result != nullptr) {
            // issue #738: 快速返回按 icamax 惯例经 aclrtMemsetAsync 设备写回
            const aclError memRet = aclrtMemsetAsync(result, sizeof(int), 0, sizeof(int), handle->stream);
            if (memRet != ACL_SUCCESS) {
                OP_LOGE("aclblasIsamax", "failed to write quick-return result, ret=%d", static_cast<int>(memRet));
                return ACLBLAS_STATUS_EXECUTION_FAILED;
            }
        }
        return vStatus;
    }

    auto* h = handle;

    uint32_t aivCoreNum = GetAivCoreCount();
    if (aivCoreNum == 0) {
        OP_LOGE("aclblasIsamax", "GetAivCoreCount failed");
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    uint32_t numBlocks = std::min(static_cast<uint32_t>(n), aivCoreNum);

    IsamaxTilingData tiling = CalcIsamaxTiling(n, numBlocks, incx);

    OP_LOGD(
        "aclblasIsamax", "tiling: totalN=%u perCoreN=%u lastCoreN=%u useCoreNum=%u tileSize=%u nthreads=%u",
        tiling.totalN, tiling.perCoreN, tiling.lastCoreN, tiling.useCoreNum, tiling.tileSize, tiling.nthreads);

    void* workSpace = GetEffectiveWorkspace(h);
    size_t workspaceBytes = GetEffectiveWorkspaceSize(h);
    constexpr uint32_t ALIGN_FLOATS = 64;
    uint32_t totalFloats = numBlocks * 2;
    uint32_t alignedFloats = ((totalFloats + ALIGN_FLOATS - 1) / ALIGN_FLOATS) * ALIGN_FLOATS;
    size_t requiredBytes = static_cast<size_t>(alignedFloats) * sizeof(float);
    if (workspaceBytes < requiredBytes) {
        OP_LOGE("aclblasIsamax", "workspace too small: need %zu, got %zu", requiredBytes, workspaceBytes);
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }

    OP_LOGI(
        "aclblasIsamax", "launching kernel: blocks=%u, useCoreNum=%u, nthreads=%u, incx=%d", numBlocks,
        tiling.useCoreNum, tiling.nthreads, incx);

    isamax_kernel_do(
        reinterpret_cast<GM_ADDR>(const_cast<float*>(x)), reinterpret_cast<GM_ADDR>(result),
        reinterpret_cast<GM_ADDR>(workSpace), tiling, numBlocks, reinterpret_cast<void*>(h->stream));

    return ACLBLAS_STATUS_SUCCESS;
}

} // namespace

aclblasStatus_t aclblasIsamax(aclblasHandle_t handle, int n, const float* x, int incx, int* result)
{
    return LaunchIsamaxKernel(handle, n, x, incx, result);
}
