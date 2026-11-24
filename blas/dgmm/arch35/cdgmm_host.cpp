/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cstdint>

#include "cann_ops_blas.h"
#include "cdgmm_kernel.h"
#include "common/helper/host_utils.h"

static_assert(sizeof(aclblasComplex) == 2U * sizeof(float), "complex64 ABI mismatch");

extern "C" aclblasStatus_t aclblasCdgmm(
    aclblasHandle_t handle, aclblasSideMode_t mode, int m, int n, const aclblasComplex* A, int lda,
    const aclblasComplex* x, int incx, aclblasComplex* C, int ldc)
{
    if (handle == nullptr) {
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (mode != ACLBLAS_SIDE_LEFT && mode != ACLBLAS_SIDE_RIGHT) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (m < 0 || n < 0 || incx == 0) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (lda < std::max(1, m) || ldc < std::max(1, m)) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (m == 0 || n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (A == nullptr || x == nullptr || C == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (A == C && lda != ldc) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }

    aclrtStream stream = nullptr;
    const aclblasStatus_t streamStatus = aclblasGetStream(handle, &stream);
    if (streamStatus != ACLBLAS_STATUS_SUCCESS) {
        return streamStatus;
    }
    const uint32_t coreCount = GetAivCoreCount();
    if (coreCount == 0U) {
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    const CdgmmTilingData tiling = MakeCdgmmTiling(
        mode == ACLBLAS_SIDE_LEFT, static_cast<uint32_t>(m), static_cast<uint32_t>(n), static_cast<uint32_t>(lda),
        static_cast<uint32_t>(ldc), incx, coreCount);
    cdgmm_kernel_do(
        reinterpret_cast<const uint8_t*>(A), reinterpret_cast<const uint8_t*>(x), reinterpret_cast<uint8_t*>(C), tiling,
        stream);
    return ACLBLAS_STATUS_SUCCESS;
}
