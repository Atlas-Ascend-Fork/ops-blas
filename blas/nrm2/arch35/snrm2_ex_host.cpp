/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file snrm2_ex_host.cpp
 * \brief aclblasSnrm2Ex 算子 Host 侧分发实现。
 */

#include <algorithm>
#include <cstdint>
#include <climits>
#include "acl/acl.h"
#include "log/log.h"
#include "cann_ops_blas.h"
#include "cann_ops_blas_common.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"
#include "common/helper/kernel_constant.h"
#include "snrm2_ex_tiling_data.h"

// tiling 按值传递
aclError snrm2_ex_kernel_do(
    uint8_t* x, uint8_t* result, uint8_t* workSpace, const Snrm2ExTilingData& tiling, uint32_t numBlocks, void* stream);

static aclblasStatus_t ValidateSnrm2ExParams(
    aclblasHandle_t handle, aclDataType xtype, int64_t n, int64_t incx, const void* x, const void* result)
{
    if (handle == nullptr) {
        OP_LOGE("aclblasSnrm2Ex", "handle is nullptr");
        return ACLBLAS_STATUS_NOT_INITIALIZED;
    }
    if (n < 0) {
        OP_LOGE("aclblasSnrm2Ex", "n must be >= 0, got %lld", static_cast<long long>(n));
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (n > static_cast<int64_t>(UINT32_MAX)) {
        OP_LOGE("aclblasSnrm2Ex", "n exceeds uint32_t limit, got %lld", static_cast<long long>(n));
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (incx == 0) {
        OP_LOGE("aclblasSnrm2Ex", "incx must not be zero");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    // 防止 abs(incx) 在计算偏移时溢出。
    if (incx == static_cast<int64_t>(INT32_MIN) || incx == INT64_MIN) {
        OP_LOGE("aclblasSnrm2Ex", "incx must not be INT32_MIN or INT64_MIN");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    // 防止 SIMT 路径 GM 索引计算时 uint64_t 溢出。
    // kernel 中 gmIdx = elemIdx * absInc (或 (n-1-elemIdx) * absInc)，均为 uint64_t。
    // n <= UINT32_MAX，absInc 最大约 9.22e18，乘积可能溢出 uint64_t。
    if (n > 0) {
        int64_t absInc = (incx > 0) ? incx : -incx;
        uint64_t unAbsInc = static_cast<uint64_t>(absInc);
        if (unAbsInc > 0 && static_cast<uint64_t>(n - 1) > UINT64_MAX / unAbsInc) {
            OP_LOGE(
                "aclblasSnrm2Ex", "(n-1)*abs(incx) exceeds UINT64_MAX, n=%lld incx=%lld", static_cast<long long>(n),
                static_cast<long long>(incx));
            return ACLBLAS_STATUS_INVALID_VALUE;
        }
    }
    if (xtype != ACL_FLOAT && xtype != ACL_FLOAT16) {
        OP_LOGE("aclblasSnrm2Ex", "invalid xtype=%d", static_cast<int>(xtype));
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (result == nullptr) {
        OP_LOGE("aclblasSnrm2Ex", "result must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

static Snrm2ExTilingData CalcSnrm2ExTilingData(int64_t n, int64_t incx, aclDataType xtype, uint32_t aivCoreNum)
{
    Snrm2ExTilingData tiling{};
    tiling.n = n;
    tiling.incx = incx;
    tiling.xtype = static_cast<uint32_t>(xtype);
    tiling.maxDataCount = SNRM2_EX_MAX_DATA_COUNT;

    uint32_t totalN = static_cast<uint32_t>(n);
    uint32_t useCoreNum = (totalN < aivCoreNum) ? totalN : aivCoreNum;
    if (useCoreNum == 0) {
        useCoreNum = 1;
    }
    if (useCoreNum > SNRM2_EX_MAX_CORE_NUM) {
        useCoreNum = SNRM2_EX_MAX_CORE_NUM;
    }
    tiling.useCoreNum = useCoreNum;
    tiling.batchPerCore = totalN / useCoreNum;
    tiling.remain = totalN % useCoreNum;

    // SIMT 线程数计算（incx != 1 路径）：每个 core 最多 calNumMax 个元素，
    // 每线程处理 1 个，对齐到 SIMT_MIN_THREAD_NUM，上限 SIMT_MAX_THREAD_NUM。
    tiling.nthreads = 0;
    if (incx != 1) {
        uint32_t calNumMax = tiling.batchPerCore + (tiling.remain > 0 ? 1 : 0);
        tiling.nthreads = std::min(CeilAlign<uint32_t>(calNumMax, SIMT_MIN_THREAD_NUM), SIMT_MAX_THREAD_NUM);
    }
    return tiling;
}

static aclblasStatus_t LaunchSnrm2ExKernel(
    _aclblas_handle* h, aclDataType xtype, const void* x, int64_t n, int64_t incx, void* result, uint32_t aivCoreNum)
{
    Snrm2ExTilingData tiling = CalcSnrm2ExTilingData(n, incx, xtype, aivCoreNum);

    // 复用 handle workspace，每个 core 2 个 FP32（scale_local, ssq_local）。
    size_t workspaceBytes = static_cast<size_t>(tiling.useCoreNum) * 2 * sizeof(float);
    CHECK_RET(workspaceBytes <= GetEffectiveWorkspaceSize(h),
              OP_LOGE("aclblasSnrm2Ex", "workspace %zu > handle %zu", workspaceBytes, GetEffectiveWorkspaceSize(h));
              return ACLBLAS_STATUS_ALLOC_FAILED);
    auto* workspaceDevice = reinterpret_cast<uint8_t*>(GetEffectiveWorkspace(h));

    aclError aclRet = aclrtMemsetAsync(workspaceDevice, workspaceBytes, 0, workspaceBytes, h->stream);
    if (aclRet != ACL_SUCCESS) {
        OP_LOGE("aclblasSnrm2Ex", "aclrtMemsetAsync workspace failed, ret=%d", aclRet);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    OP_LOGD(
        "aclblasSnrm2Ex",
        "tiling: n=%lld incx=%lld xtype=%u useCoreNum=%u maxDataCount=%u batchPerCore=%u remain=%u nthreads=%u",
        static_cast<long long>(tiling.n), static_cast<long long>(tiling.incx), tiling.xtype, tiling.useCoreNum,
        tiling.maxDataCount, tiling.batchPerCore, tiling.remain, tiling.nthreads);
    OP_LOGI("aclblasSnrm2Ex", "launching kernel: blocks=%u, cores=%u", tiling.useCoreNum, aivCoreNum);

    const aclError launchErr = snrm2_ex_kernel_do(
        reinterpret_cast<uint8_t*>(const_cast<void*>(x)), reinterpret_cast<uint8_t*>(result), workspaceDevice, tiling,
        tiling.useCoreNum, h->stream);
    if (launchErr != ACL_SUCCESS) {
        OP_LOGE("aclblasSnrm2Ex", "snrm2_ex_kernel_do launch failed: %d", static_cast<int>(launchErr));
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t aclblasSnrm2Ex(
    aclblasHandle_t handle, aclDataType xtype, const void* x, const int64_t n, const int64_t incx, void* result)
{
    aclblasStatus_t status = ValidateSnrm2ExParams(handle, xtype, n, incx, x, result);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }

    // n == 0：直接返回 result = 0.0f，不触发 kernel。
    if (n == 0) {
        float zero = 0.0f;
        aclError aclRet = aclrtMemcpy(result, sizeof(float), &zero, sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(aclRet == ACL_SUCCESS, OP_LOGE("aclblasSnrm2Ex", "aclrtMemcpy zero result failed, ret=%d", aclRet);
                  return ACLBLAS_STATUS_INTERNAL_ERROR);
        return ACLBLAS_STATUS_SUCCESS;
    }

    if (x == nullptr) {
        OP_LOGE("aclblasSnrm2Ex", "x must not be nullptr when n > 0");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    auto* h = handle;

    uint32_t aivCoreNum = GetAivCoreCount();
    if (aivCoreNum == 0) {
        OP_LOGE("aclblasSnrm2Ex", "vector core count is 0");
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }

    return LaunchSnrm2ExKernel(h, xtype, x, n, incx, result, aivCoreNum);
}
