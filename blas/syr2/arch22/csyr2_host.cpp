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
#include <limits>
#include "acl/acl.h"
#include "log/log.h"
#include "cann_ops_blas.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"
#include "csyr2_kernel.h"

static aclblasStatus_t ValidateCsyr2Params(
    aclblasFillMode_t uplo, int n, const aclblasComplex* alpha, int incx, int incy, int lda)
{
    if (uplo != ACLBLAS_UPPER && uplo != ACLBLAS_LOWER) {
        OP_LOGE("aclblasCsyr2", "invalid uplo %d", (int)uplo);
        return ACLBLAS_STATUS_INVALID_ENUM;
    }
    if (n < 0 || lda < (n > 0 ? n : 1) || incx == 0 || incy == 0) {
        OP_LOGE("aclblasCsyr2", "invalid dimensions or increments");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (!alpha) {
        OP_LOGE("aclblasCsyr2", "null pointer");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}
static void LaunchCsyr2Unit(
    aclblasHandle_t handle, int n, int lda, uint32_t availableCores, uint8_t* xGm, uint8_t* yGm, uint8_t* aGm,
    const Csyr2TilingData& t)
{
    // Unit alpha and unit strides always use a dedicated vector implementation.
    // Exact-length transfers also support partial bands and lda padding.
    if (t.uplo == ACLBLAS_UPPER && n <= 16) {
        // Columns shorter than one DMA block share a line; serialize them.
        const uint32_t cores = lda < 4 ? 1U : std::min(availableCores, static_cast<uint32_t>(n));
        csyr2_small_upper_do(cores, handle->stream, xGm, yGm, aGm, t);
    } else if (n <= 2048) {
        const uint32_t cores = std::min(availableCores, (static_cast<uint32_t>(n) + 3U) / 4U);
        if (t.uplo == ACLBLAS_UPPER) {
            csyr2_banded_vector_do(cores, handle->stream, xGm, yGm, aGm, t);
        } else if (n >= 64 && availableCores >= 16U) {
            csyr2_lower_columns_do((cores / 16U) * 16U, handle->stream, xGm, yGm, aGm, t);
        } else {
            csyr2_lower_vector_do(cores, handle->stream, xGm, yGm, aGm, t);
        }
    } else if (t.uplo == ACLBLAS_UPPER && n <= 3072 && availableCores >= 16U) {
        // Four resident vectors plus two two-column batches fit in 192 KiB.
        csyr2_upper_atomic_do(availableCores, handle->stream, xGm, yGm, aGm, t);
    } else if (t.uplo == ACLBLAS_LOWER && n <= 3072 && availableCores >= 16U) {
        // Experimental batch2 atomic path: 8 resident/output vectors fit 192 KiB.
        csyr2_lower_columns_do((std::min(availableCores, 48U) / 16U) * 16U, handle->stream, xGm, yGm, aGm, t);
    } else if (t.uplo == ACLBLAS_LOWER && n <= 4096) {
        csyr2_lower_general_do(availableCores, handle->stream, xGm, yGm, aGm, t);
    } else if (t.uplo == ACLBLAS_UPPER && n <= 6144) {
        csyr2_direct_vector_do(availableCores, handle->stream, xGm, yGm, aGm, t);
    } else {
        csyr2_unit_tiled_do(availableCores, handle->stream, xGm, yGm, aGm, t);
    }
}

static void LaunchCsyr2Prepared(
    aclblasHandle_t handle, uint32_t availableCores, uint8_t* xGm, uint8_t* yGm, uint8_t* aGm, const Csyr2TilingData& t)
{
    // Scalar stores need a single core for small matrices sharing cache lines.
    // Larger alpha amplifies vector rounding beyond the absolute-error cap.
    // Vector DMA column gaps are uint32 byte counts, even though GM offsets
    // are uint64. Preserve large-lda inputs through the 64-bit scalar path.
    const bool wideColumnStride =
        static_cast<uint64_t>(t.lda) * sizeof(aclblasComplex) > std::numeric_limits<uint32_t>::max();
    if (wideColumnStride || t.incx != 1 || t.incy != 1 || t.ar > 1.0F || t.ar < -1.0F || t.ai > 1.0F || t.ai < -1.0F) {
        csyr2_kernel_do(xGm, yGm, aGm, t, t.n < 64 ? 1U : availableCores, handle->stream);
        return;
    }

    // General alpha uses bulk transfers with grouped complex products.
    if (t.ar != 1.0F || t.ai != 0.0F) {
        const uint32_t cores = std::min(availableCores, (t.n + 3U) / 4U);
        csyr2_alpha_tiled_do(cores, handle->stream, xGm, yGm, aGm, t);
        return;
    }

    LaunchCsyr2Unit(handle, t.n, t.lda, availableCores, xGm, yGm, aGm, t);
}

aclblasStatus_t aclblasCsyr2(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n, const aclblasComplex* alpha, const aclblasComplex* x,
    int incx, const aclblasComplex* y, int incy, aclblasComplex* A, int lda)
{
    if (!handle) {
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    auto st = ValidateCsyr2Params(uplo, n, alpha, incx, incy, lda);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }
    if (n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    aclblasComplex alphaValue{};
    st = ReadScalarFromHostOrDevice(alpha, handle->stream, alphaValue);
    if (st != ACLBLAS_STATUS_SUCCESS) {
        return st;
    }
    if (alphaValue.real == 0.0F && alphaValue.imag == 0.0F) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (!x || !y || !A) {
        OP_LOGE("aclblasCsyr2", "null data pointer");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    const uint32_t availableCores = GetAivCoreCount();
    if (!availableCores) {
        OP_LOGE("aclblasCsyr2", "no AIV cores");
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    const Csyr2TilingData t{(uint32_t)n, (uint32_t)uplo, (uint32_t)lda, alphaValue.real, alphaValue.imag, incx, incy};
    auto* xGm = reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(x));
    auto* yGm = reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(y));
    auto* aGm = reinterpret_cast<uint8_t*>(A);

    LaunchCsyr2Prepared(handle, availableCores, xGm, yGm, aGm, t);
    return ACLBLAS_STATUS_SUCCESS;
}
