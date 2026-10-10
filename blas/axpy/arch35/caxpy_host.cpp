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
#include "caxpy_kernel.h"
#include "caxpy_tiling_data.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"
#include "common/helper/kernel_constant.h"
#include "log/log.h"

namespace {

aclblasStatus_t ValidateCaxpyParams(
    const aclblasComplex* alpha, const aclblasComplex* x, int incx, const aclblasComplex* y, int incy)
{
    if (alpha == nullptr || x == nullptr || y == nullptr) {
        OP_LOGE("aclblasCaxpy", "alpha, x and y must not be nullptr when n is positive");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (incx == 0 || incy == 0) {
        OP_LOGE("aclblasCaxpy", "incx and incy must not be zero, got incx=%d incy=%d", incx, incy);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

aclblasStatus_t LaunchCaxpyKernel(
    aclblasHandle_t handle, int n, const aclblasComplex* alpha, const aclblasComplex* x, int incx,
    aclblasComplex* y, int incy)
{
    uint32_t aivCoreNum = GetAivCoreCount();
    if (aivCoreNum == 0) {
        OP_LOGE("aclblasCaxpy", "GetAivCoreCount failed");
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }

    uint32_t totalN = static_cast<uint32_t>(n);
    uint32_t numBlocks = std::min(CeilDiv<uint32_t>(totalN, SIMT_MIN_THREAD_NUM), aivCoreNum);
    numBlocks = std::max(numBlocks, 1U);
    uint32_t threadCount =
        std::min(CeilAlign<uint32_t>(CeilDiv<uint32_t>(totalN, numBlocks), SIMT_MIN_THREAD_NUM), SIMT_MAX_THREAD_NUM);

    CaxpyTilingData tiling{};
    tiling.totalN = totalN;
    tiling.incx = incx;
    tiling.incy = incy;
    tiling.threadCount = threadCount;
    tiling.numBlocks = numBlocks;
    tiling.alphaReal = alpha->real;
    tiling.alphaImag = alpha->imag;

    OP_LOGD(
        "aclblasCaxpy", "launch: n=%u incx=%d incy=%d blocks=%u threads=%u", totalN, incx, incy, numBlocks,
        threadCount);
    caxpy_kernel_do(
        reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(x)), reinterpret_cast<uint8_t*>(y), tiling, numBlocks,
        handle->stream);
    return ACLBLAS_STATUS_SUCCESS;
}

} // namespace

extern "C" aclblasStatus_t aclblasCaxpy(
    aclblasHandle_t handle, int n, const aclblasComplex* alpha, const aclblasComplex* x, int incx,
    aclblasComplex* y, int incy)
{
    if (handle == nullptr) {
        OP_LOGE("aclblasCaxpy", "handle is nullptr");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (n < 0) {
        OP_LOGE("aclblasCaxpy", "n must be non-negative, got %d", n);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }

    aclblasStatus_t status = ValidateCaxpyParams(alpha, x, incx, y, incy);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }
    return LaunchCaxpyKernel(handle, n, alpha, x, incx, y, incy);
}
