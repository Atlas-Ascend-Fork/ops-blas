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
#include <climits>
#include <limits>

#include "cann_ops_blas.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"
#include "cher2_tiling_data.h"

namespace {
bool HasZeroStride(int incx, int incy) { return incx == 0 || incy == 0; }
} // namespace

extern "C" aclblasStatus_t aclblasCher2(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n, const aclblasComplex* alpha, const aclblasComplex* x,
    int incx, const aclblasComplex* y, int incy, aclblasComplex* a, int lda)
{
    if (handle == nullptr) {
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    // The supplied negative-dimension case also has an invalid enum: reject
    // the dimension first. All structural checks precede either quick return.
    if (n < 0) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (uplo != ACLBLAS_UPPER && uplo != ACLBLAS_LOWER) {
        return ACLBLAS_STATUS_INVALID_ENUM;
    }
    if (HasZeroStride(incx, incy) || incx == INT_MIN || incy == INT_MIN || lda < std::max(1, n) || alpha == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (n == 0 || (alpha->real == 0.0f && alpha->imag == 0.0f)) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (x == nullptr || y == nullptr || a == nullptr) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    const uint64_t maxElements = std::numeric_limits<size_t>::max() / sizeof(aclblasComplex);
    if (static_cast<uint64_t>(n) * static_cast<uint64_t>(lda) > maxElements) {
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    const uint32_t cores = GetAivCoreCount();
    if (cores == 0) {
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    const bool useAiv = incx == 1 && incy == 1 && n >= 256 && n <= 4096 && cores >= 8;
    const uint32_t blocks = useAiv ? cores / 8U * 8U : std::min(cores, static_cast<uint32_t>(n));
    const uint32_t threads = std::min(256U, std::max(32U, (static_cast<uint32_t>(n) + 31U) / 32U * 32U));
    const Cher2TilingData tiling{
        static_cast<uint32_t>(n),
        static_cast<uint32_t>(lda),
        static_cast<uint32_t>(uplo),
        threads,
        alpha->real,
        alpha->imag,
        incx,
        incy};
    auto launch = useAiv ? cher2_aiv_kernel_do : cher2_kernel_do;
    launch(
        reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(x)),
        reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(y)), reinterpret_cast<uint8_t*>(a), tiling, blocks,
        handle->stream);
    return ACLBLAS_STATUS_SUCCESS;
}
