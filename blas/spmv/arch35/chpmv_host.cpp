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
#include "cann_ops_blas.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"
#include "common/helper/kernel_constant.h"
#include "chpmv_kernel.h"
#include "log/log.h"

namespace {
aclblasStatus_t ChpmvValidateInput(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n, const aclblasComplex* alpha, const aclblasComplex* ap,
    const aclblasComplex* x, int incx, const aclblasComplex* beta, aclblasComplex* y, int incy)
{
    if (handle == nullptr) {
        OP_LOGE("aclblasChpmv", "handle is nullptr.");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (n < 0) {
        OP_LOGE("aclblasChpmv", "n must be nonnegative, got %d.", n);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (uplo != ACLBLAS_UPPER && uplo != ACLBLAS_LOWER) {
        OP_LOGE("aclblasChpmv", "uplo is invalid, got %d.", static_cast<int>(uplo));
        return ACLBLAS_STATUS_INVALID_ENUM;
    }
    if (incx == 0 || incy == 0 || alpha == nullptr || beta == nullptr || ap == nullptr || x == nullptr ||
        y == nullptr) {
        OP_LOGE("aclblasChpmv", "incx/incy must be nonzero and alpha/beta/ap/x/y must be nonnull.");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}

inline bool ChpmvIsNoOp(const aclblasComplex& alpha, const aclblasComplex& beta)
{
    return alpha.real == 0.0f && alpha.imag == 0.0f && beta.real == 1.0f && beta.imag == 0.0f;
}

inline bool ChpmvUseSimd(uint32_t size, const aclblasComplex& alpha, uint64_t ubBytes)
{
    return size >= 512 && (alpha.real != 0.0f || alpha.imag != 0.0f) &&
           ubBytes >= CHPMV_STAGE_BYTES + CHPMV_X_BYTES + CHPMV_ACC_BYTES + 64U * 1024U;
}
} // namespace

aclblasStatus_t aclblasChpmv(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n, const aclblasComplex* alpha, const aclblasComplex* ap,
    const aclblasComplex* x, int incx, const aclblasComplex* beta, aclblasComplex* y, int incy)
{
    const auto inputStatus = ChpmvValidateInput(handle, uplo, n, alpha, ap, x, incx, beta, y, incy);
    if (inputStatus != ACLBLAS_STATUS_SUCCESS || n == 0) {
        return inputStatus;
    }
    const aclblasComplex a = *alpha;
    const aclblasComplex b = *beta;
    if (ChpmvIsNoOp(a, b)) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    const uint32_t cores = GetAivCoreCount();
    if (cores == 0) {
        OP_LOGE("aclblasChpmv", "AIV core count is zero.");
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    const uint32_t size = static_cast<uint32_t>(n);
    const bool groupOrdered = size >= 16;
    auto* platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    if (platform == nullptr) {
        OP_LOGE("aclblasChpmv", "PlatformAscendCManager is nullptr.");
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    uint64_t ubBytes = 0;
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytes);
    const bool useSimd = ChpmvUseSimd(size, a, ubBytes);
    constexpr uint32_t groupsPerBlock = CHPMV_THREADS / CHPMV_GROUP_SIZE;
    const uint32_t rowsPerBlock = useSimd ? CHPMV_TILE_ROWS : (groupOrdered ? groupsPerBlock : SIMT_MIN_THREAD_NUM);
    const uint32_t blocks = std::min(CeilDiv<uint32_t>(size, rowsPerBlock), cores);
    ChpmvTilingData tiling{};
    tiling.n = size;
    tiling.uplo = static_cast<uint32_t>(uplo);
    tiling.threads = CHPMV_THREADS;
    tiling.strategy = useSimd ? 2U : (groupOrdered ? 1U : 0U);
    tiling.incx = static_cast<int64_t>(incx);
    tiling.incy = static_cast<int64_t>(incy);
    tiling.alpha = a;
    tiling.beta = b;
    const aclError status = chpmv_kernel_do(
        reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(ap)),
        reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(x)), reinterpret_cast<uint8_t*>(y), tiling, blocks,
        handle->stream);
    if (status != ACL_SUCCESS) {
        OP_LOGE("aclblasChpmv", "kernel launch failed, ACL error %d.", status);
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    return ACLBLAS_STATUS_SUCCESS;
}
