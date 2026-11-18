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
#include "cann_ops_blas.h"
#include "log/log.h"
#include "common/helper/aclblas_handle_internal.h"
#include "common/helper/host_utils.h"
#include "common/helper/kernel_constant.h"
#include "chemv_tiling_data.h"

// Keep equal to CHEMV_REG_ROWS in the scratch kernel (up to64 register rows supported).
constexpr uint32_t CHEMV_REG_ROWS = 64U;
static_assert(CHEMV_REG_ROWS > 0U && CHEMV_REG_ROWS <= 64U, "invalid row tile");

void chemv_kernel_do(uint8_t* a, uint8_t* x, uint8_t* y, uint32_t blocks, const ChemvTilingData& tiling, void* stream);

namespace {
aclblasStatus_t ValidateChemvAttributes(
    aclblasFillMode_t uplo, int n, int lda, int incx, int incy, const aclblasComplex* alpha, const aclblasComplex* beta)
{
    if (uplo != ACLBLAS_UPPER && uplo != ACLBLAS_LOWER) {
        OP_LOGE("aclblasChemv", "Invalid uplo: %d", static_cast<int>(uplo));
        return ACLBLAS_STATUS_INVALID_ENUM;
    }
    if (lda < n || incx == 0 || incy == 0 || incx == INT_MIN || incy == INT_MIN || alpha == nullptr ||
        beta == nullptr) {
        OP_LOGE("aclblasChemv", "Invalid dimensions, increments or scalar pointers");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    return ACLBLAS_STATUS_SUCCESS;
}
uint32_t ChemvRowsPerBlock(bool regFast, int n, uint32_t cores)
{
    uint32_t rowsPerBlock = 32U;
    if (regFast) {
        rowsPerBlock = 16U;
        const uint32_t averageRows = CeilDiv<uint32_t>(static_cast<uint32_t>(n), cores);
        while (rowsPerBlock < CHEMV_REG_ROWS && rowsPerBlock < averageRows) {
            rowsPerBlock *= 2U;
        }
    }
    return rowsPerBlock;
}
ChemvTilingData ChemvMakeTiling(
    int n, int lda, aclblasFillMode_t uplo, int incx, int incy, const aclblasComplex& a, const aclblasComplex& b,
    uint32_t rowsPerBlock)
{
    ChemvTilingData t{};
    t.n = static_cast<uint32_t>(n);
    t.lda = static_cast<uint32_t>(lda);
    t.uplo = static_cast<uint32_t>(uplo);
    t.nthreads = 32U;
    t.rowsPerBlock = rowsPerBlock;
    t.incx = incx;
    t.incy = incy;
    t.alphaReal = a.real;
    t.alphaImag = a.imag;
    t.betaReal = b.real;
    t.betaImag = b.imag;
    return t;
}
} // namespace

aclblasStatus_t aclblasChemv(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n, const aclblasComplex* alpha, const aclblasComplex* A,
    int lda, const aclblasComplex* x, int incx, const aclblasComplex* beta, aclblasComplex* y, int incy)
{
    // n=0 precedes all attribute/scalar checks, but never the handle check.
    if (handle == nullptr) {
        OP_LOGE("aclblasChemv", "Handle is nullptr");
        return ACLBLAS_STATUS_HANDLE_IS_NULLPTR;
    }
    if (n < 0) {
        OP_LOGE("aclblasChemv", "Invalid n: %d", n);
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    if (n == 0) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    const auto status = ValidateChemvAttributes(uplo, n, lda, incx, incy, alpha, beta);
    if (status != ACLBLAS_STATUS_SUCCESS) {
        return status;
    }
    // This API uses host scalar pointers. Snapshot them before asynchronous launch.
    const aclblasComplex a = *alpha;
    const aclblasComplex b = *beta;
    const bool alphaZero = a.real == 0.0f && a.imag == 0.0f;
    // No-op contract: A, x and y may be null because no device buffer is accessed.
    // Scalar, dimension and stride validation above still applies.
    if (alphaZero && b.real == 1.0f && b.imag == 0.0f) {
        return ACLBLAS_STATUS_SUCCESS;
    }
    if (y == nullptr || (!alphaZero && (A == nullptr || x == nullptr))) {
        OP_LOGE("aclblasChemv", "Required device buffer is nullptr");
        return ACLBLAS_STATUS_INVALID_VALUE;
    }
    const uint32_t cores = GetAivCoreCount();
    if (cores == 0) {
        OP_LOGE("aclblasChemv", "No available AIV cores");
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    const bool regFast = a.real == 1.0f && a.imag == 0.0f && b.real == 0.0f && b.imag == 0.0f && incx == 1 && incy == 1;
    const uint32_t rowsPerBlock = ChemvRowsPerBlock(regFast, n, cores);
    const uint32_t blocks = std::min(CeilDiv<uint32_t>(n, rowsPerBlock), cores);
    const auto t = ChemvMakeTiling(n, lda, uplo, incx, incy, a, b, rowsPerBlock);
    chemv_kernel_do(
        reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(A)),
        reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(x)), reinterpret_cast<uint8_t*>(y), blocks, t,
        handle->stream);
    return ACLBLAS_STATUS_SUCCESS;
}
