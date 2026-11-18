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
#include "ctpttr_kernel.h"
#include "common/helper/host_utils.h"

static_assert(sizeof(aclblasComplex) == 8U, "complex64 ABI mismatch");

extern "C" aclblasStatus_t aclblasCtpttr(
    aclblasHandle_t handle, aclblasFillMode_t uplo, const int n, const aclblasComplex* AP, aclblasComplex* A,
    const int lda)
{
    if (handle == nullptr) {
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (n < 0) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (uplo != ACLBLAS_UPPER && uplo != ACLBLAS_LOWER) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (lda < std::max(1, n)) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (A == nullptr || AP == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    // Non-overlap is a caller precondition, not an in-place operation.
    aclrtStream stream = nullptr;
    const auto status = aclblasGetStream(handle, &stream);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }
    const uint32_t cores = GetAivCoreCount();
    if (cores == 0U) {
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    const auto tiling =
        MakeCtpttrTiling(static_cast<uint32_t>(n), static_cast<uint32_t>(lda), uplo == ACLBLAS_UPPER, cores);
    ctpttr_kernel_do(reinterpret_cast<const uint8_t*>(AP), reinterpret_cast<uint8_t*>(A), tiling, stream);
    return ACLBLAS_STATUS_SUCCESS;
}
