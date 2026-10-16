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
#include <cstdint>

#include "cann_ops_blas.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"
#include "common/helper/kernel_constant.h"
#include "cswap_kernel.h"
#include "log/log.h"

namespace {

aclblasStatus_t ValidateCswapParams(const aclblasComplex* x, const aclblasComplex* y, int incx, int incy)
{
    if (x == nullptr || y == nullptr) {
        OP_LOGE("aclblasCswap", "x/y must not be nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (incx == 0) {
        OP_LOGE("aclblasCswap", "incx must not be zero");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (incy == 0) {
        OP_LOGE("aclblasCswap", "incy must not be zero");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

} // namespace

extern "C" aclblasStatus_t aclblasCswap(
    aclblasHandle_t handle, int n, aclblasComplex* x, int incx, aclblasComplex* y, int incy)
{
    if (n <= 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (handle == nullptr) {
        OP_LOGE("aclblasCswap", "handle is nullptr");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }

    aclblasStatus_t status = ValidateCswapParams(x, y, incx, incy);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }

    uint32_t aivCoreNum = GetAivCoreCount();
    if (aivCoreNum == 0) {
        OP_LOGE("aclblasCswap", "vector core count is 0");
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }

    uint32_t totalN = static_cast<uint32_t>(n);
    uint32_t numBlocks = std::min(CeilDiv<uint32_t>(totalN, SIMT_MIN_THREAD_NUM), aivCoreNum);
    numBlocks = std::max(numBlocks, 1U);
    uint32_t threadCount =
        std::min(CeilAlign<uint32_t>(CeilDiv<uint32_t>(totalN, numBlocks), SIMT_MIN_THREAD_NUM), SIMT_MAX_THREAD_NUM);

    OP_LOGD(
        "aclblasCswap", "launching SIMT kernel: n=%u blocks=%u threads=%u incx=%d incy=%d", totalN, numBlocks,
        threadCount, incx, incy);
    cswap_kernel_do(
        reinterpret_cast<uint8_t*>(x), reinterpret_cast<uint8_t*>(y), totalN, incx, incy, threadCount, numBlocks,
        handle->stream);
    return ACLBLAS_STATUS_SUCCESS;
}
