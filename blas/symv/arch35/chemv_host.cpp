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
// Resolve alpha/beta by memory location: host scalars are dereferenced directly,
// device scalars are snapshotted via D2H on the handle stream. Same contract as
// the arch22 chemv and the 950PR cherk/csyrk family (issues #849/#628); a device
// pointer must never be silently dereferenced as host memory.
aclblasStatus_t ResolveChemvScalars(
    const aclblasComplex* alpha, const aclblasComplex* beta, aclrtStream stream, aclblasComplex& a, aclblasComplex& b)
{
    aclblasStatus_t ret = ReadScalarFromHostOrDevice(alpha, stream, a);
    if (ret != ACLBLAS_STATUS_SUCCESS) {
        OP_LOGE("aclblasChemv", "Failed to resolve alpha scalar location, st=%d", static_cast<int>(ret));
        return ret;
    }
    ret = ReadScalarFromHostOrDevice(beta, stream, b);
    if (ret != ACLBLAS_STATUS_SUCCESS) {
        OP_LOGE("aclblasChemv", "Failed to resolve beta scalar location, st=%d", static_cast<int>(ret));
        return ret;
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
// Core-count guard, rowsPerBlock selection and the kernel launch. The host-side
// gate keeps an out-of-set rowsPerBlock from reaching the tiling, where the
// kernel fast path would silently skip every block and leave y unwritten.
aclblasStatus_t LaunchChemvKernel(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n, int lda, int incx, int incy, const aclblasComplex* A,
    const aclblasComplex* x, aclblasComplex* y, const aclblasComplex& a, const aclblasComplex& b)
{
    const uint32_t cores = GetAivCoreCount();
    if (cores == 0) {
        OP_LOGE("aclblasChemv", "No available AIV cores");
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    const bool regFast = a.real == 1.0f && a.imag == 0.0f && b.real == 0.0f && b.imag == 0.0f && incx == 1 && incy == 1;
    const uint32_t rowsPerBlock = ChemvRowsPerBlock(regFast, n, cores);
    if (!CHEMV_ROWS_PER_BLOCK_VALID(rowsPerBlock)) {
        OP_LOGE("aclblasChemv", "rowsPerBlock outside supported set {16,32,64}: %u", rowsPerBlock);
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    const uint32_t blocks = std::min(CeilDiv<uint32_t>(n, rowsPerBlock), cores);
    const auto t = ChemvMakeTiling(n, lda, uplo, incx, incy, a, b, rowsPerBlock);
    chemv_kernel_do(
        reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(A)),
        reinterpret_cast<uint8_t*>(const_cast<aclblasComplex*>(x)), reinterpret_cast<uint8_t*>(y), blocks, t,
        handle->stream);
    return ACLBLAS_STATUS_SUCCESS;
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
    // Resolve alpha/beta by memory location (see ResolveChemvScalars).
    aclblasComplex a{};
    aclblasComplex b{};
    const aclblasStatus_t scalarRet = ResolveChemvScalars(alpha, beta, handle->stream, a, b);
    if (scalarRet != ACLBLAS_STATUS_SUCCESS) {
        return scalarRet;
    }
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
    return LaunchChemvKernel(handle, uplo, n, lda, incx, incy, A, x, y, a, b);
}
