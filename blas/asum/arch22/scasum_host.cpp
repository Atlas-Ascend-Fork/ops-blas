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
 * \file scasum_host.cpp
 * \brief scasum Host-side dispatch for Atlas A2/A3 (arch22, complex64 asum)
 */

#include <cstdint>
#include <algorithm>
#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "cann_ops_blas_common.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"
#include "log/log.h"
#include "scasum_tiling_data.h"
#include "scasum_kernel.h"

namespace {

static ScasumTilingData CalcScasumTilingData(int64_t totalComplexNum, uint32_t vecCoreNum)
{
    ScasumTilingData tiling;
    tiling.n = static_cast<uint32_t>(totalComplexNum);
    tiling.incx = 0;
    tiling.useCoreNum = 1;
    tiling.perCoreN = 0;
    tiling.remainder = 0;

    uint32_t useCoreNum = std::min(vecCoreNum, static_cast<uint32_t>(totalComplexNum));
    if (useCoreNum == 0) {
        useCoreNum = 1;
    }
    tiling.useCoreNum = useCoreNum;
    tiling.perCoreN = static_cast<uint32_t>(totalComplexNum) / useCoreNum;
    tiling.remainder = static_cast<uint32_t>(totalComplexNum) % useCoreNum;

    return tiling;
}

static aclblasStatus_t ValidateScasumParams(
    aclblasHandle_t handle, int n, int incx, const aclblasComplex* x, float* result)
{
    if (handle == nullptr) {
        OP_LOGE("aclblasScasum", "handle is nullptr");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (result == nullptr) {
        OP_LOGE("aclblasScasum", "result must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (n > 0 && incx > 0 && x == nullptr) {
        OP_LOGE("aclblasScasum", "x must not be nullptr when n > 0 and incx > 0");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

static aclblasStatus_t ScasumExecuteKernel(
    _aclblas_handle* h, const aclblasComplex* x, float* result, uint32_t numBlocks, const ScasumTilingData& tiling)
{
    return scasum_kernel_do(
        reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(x)), reinterpret_cast<uint8_t*>(result), nullptr,
        tiling, numBlocks, h->stream);
}

} // namespace

aclblasStatus_t aclblasScasum(aclblasHandle_t handle, int n, const aclblasComplex* x, int incx, float* result)
{
    aclblasStatus_t status = ValidateScasumParams(handle, n, incx, x, result);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }

    // n <= 0 或 incx <= 0：result 写 0.0f 后返回成功，不触发 kernel。
    if (n <= 0 || incx <= 0) {
        float zero = 0.0f;
        aclError memRet = aclrtMemcpy(result, sizeof(float), &zero, sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
        if (memRet != ACL_SUCCESS) {
            OP_LOGE("aclblasScasum", "aclrtMemcpy for early-return zero failed: %d", memRet);
            return ACLBLAS_STATUS_EXECUTION_FAILED;
        }
        return ACLBLAS_STATUS_SUCCESS;
    }

    // float 视图最大索引 2*(n-1)*incx + 1 须落在 uint32 索引域内。
    const int64_t floatSpan = static_cast<int64_t>(n - 1) * static_cast<int64_t>(incx) * 2;
    if (floatSpan > static_cast<int64_t>(UINT32_MAX)) {
        OP_LOGE(
            "aclblasScasum", "strided offset overflow: n=%d incx=%d floatSpan=%lld exceeds uint32 float index domain",
            n, incx, static_cast<long long>(floatSpan));
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    uint32_t aivCoreNum = GetAivCoreCount();
    if (aivCoreNum == 0) {
        OP_LOGE("aclblasScasum", "GetAivCoreCount failed");
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    uint32_t numBlocks = std::min(aivCoreNum, static_cast<uint32_t>(n));
    if (numBlocks == 0) {
        numBlocks = 1;
    }

    ScasumTilingData tiling = CalcScasumTilingData(n, numBlocks);
    tiling.incx = incx;

    OP_LOGD(
        "aclblasScasum", "tiling: n=%ld incx=%ld useCoreNum=%u perCoreN=%u remainder=%u",
        static_cast<long>(tiling.n), static_cast<long>(tiling.incx), tiling.useCoreNum, tiling.perCoreN,
        tiling.remainder);

    auto* h = handle;
    return ScasumExecuteKernel(h, x, result, numBlocks, tiling);
}
