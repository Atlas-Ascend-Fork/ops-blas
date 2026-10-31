/*
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
#include <limits>
#include "cann_ops_blas.h"
#include "common/helper/aclblas_handle_internal.h"
#include "ctrsv_kernel.h"
#include "log/log.h"

aclblasStatus_t aclblasCtrsv(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, int n,
    const aclblasComplex* a, int lda, aclblasComplex* x, int incx)
{
    if (handle == nullptr) {
        OP_LOGE("aclblasCtrsv", "handle is nullptr");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if ((uplo != ACLBLAS_UPPER && uplo != ACLBLAS_LOWER) ||
        (trans != ACLBLAS_OP_N && trans != ACLBLAS_OP_T && trans != ACLBLAS_OP_C) ||
        (diag != ACLBLAS_UNIT && diag != ACLBLAS_NON_UNIT) || n < 0 || lda < std::max(1, n) || incx == 0) {
        OP_LOGE("aclblasCtrsv", "invalid enum, n, lda or incx");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (a == nullptr || x == nullptr) {
        OP_LOGE("aclblasCtrsv", "A or x is nullptr for nonempty input");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    const int64_t step = incx;
    const uint64_t stride = step < 0 ? -step : step;
    const uint64_t extent = uint64_t(n - 1) * stride + 1;
    const uint64_t limit = std::numeric_limits<int64_t>::max() / sizeof(aclblasComplex);
    if (uint64_t(n) * lda > limit || extent > limit) {
        OP_LOGE("aclblasCtrsv", "matrix or vector byte extent exceeds int64 range");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    CtrsvTilingData tiling{uint32_t(n),    uint32_t(lda),   step,          step < 0 ? extent - 1 : 0,
                           uint32_t(uplo), uint32_t(trans), uint32_t(diag)};
    ctrsv_kernel_do(
        reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(a)), reinterpret_cast<uint8_t*>(x), tiling,
        handle->stream,
        GetEffectiveWorkspaceSize(handle) >= CTRSV_WORKSPACE_BYTES ? GetEffectiveWorkspace(handle) : nullptr);
    return ACLBLAS_STATUS_SUCCESS;
}
