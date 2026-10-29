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
#include <limits>
#include "cann_ops_blas.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"
#include "cher2_kernel.h"

aclblasStatus_t aclblasCher2(
    aclblasHandle_t handle, aclblasFillMode_t uplo, const int n, const aclblasComplex* alpha, const aclblasComplex* x,
    const int incx, const aclblasComplex* y, const int incy, aclblasComplex* A, const int lda)
{
    if (handle == nullptr) {
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (n < 0) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (uplo != ACLBLAS_UPPER && uplo != ACLBLAS_LOWER) {
        return ACLBLAS_STATUS_INVALID_ENUM;
    }
    if (lda < std::max(1, n) || incx == 0 || incy == 0 || incx == std::numeric_limits<int>::min() ||
        incy == std::numeric_limits<int>::min() || alpha == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    // Structural validation precedes both quick returns. Alpha is host-resident.
    if (n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    const aclblasComplex alphaValue = *alpha;
    if (alphaValue.real == 0.0f && alphaValue.imag == 0.0f) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (x == nullptr || y == nullptr || A == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    const uint32_t cores = GetAivCoreCount();
    if (cores == 0) {
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    const uint64_t side = (static_cast<uint64_t>(n) + 63) / 64;
    const uint32_t blocks = static_cast<uint32_t>(std::min<uint64_t>(cores, side * (side + 1) / 2));
    const Cher2TilingData tiling{
        static_cast<uint32_t>(n), static_cast<uint32_t>(lda), uplo == ACLBLAS_UPPER,     alphaValue.real,
        alphaValue.imag,          static_cast<int64_t>(incx), static_cast<int64_t>(incy)};
    cher2_kernel_do(
        reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(x)),
        reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(y)), reinterpret_cast<uint8_t*>(A), tiling, blocks,
        handle->stream);
    return ACLBLAS_STATUS_SUCCESS;
}
